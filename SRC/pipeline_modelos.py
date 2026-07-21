"""
Pipeline reproducible de entrenamiento para los dos modelos elegidos del
sistema de recomendación (ver Notebooks/05_recommender_model.ipynb):

  - Warm-start: LightGBM (hay historial de interacciones de usuario/producto)
  - Cold-start: Content-Based Filtering (no hay historial, solo atributos)

Replica la metodología anti-leakage del notebook: split temporal (no
aleatorio), features de usuario/producto recalculados usando solo datos de
train, negativos "duros" muestreados por popularidad, y calibración de
umbral ajustada solo con datos de train.

Uso:
    python SRC/recommender_pipeline.py

Guarda los artefactos entrenados en Models/warm_start_lightgbm.joblib y
Models/cold_start_content_based.joblib.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_engineering import crear_features_producto, crear_features_usuario, generar_features

RAW_DIR = PROJECT_DIR / "Data" / "Raw"
MODELS_DIR = PROJECT_DIR / "Models"

TOP_K = 10
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 30
EVENT_WEIGHTS = {"page_view": 1, "add_to_cart": 3, "purchase": 5}

FEATURE_COLS = [
    "age", "country", "marketing_opt_in", "n_sessions", "n_purchases_user", "ticket_promedio",
    "n_products_viewed", "n_products_carted", "rating_promedio_usr",
    "price_usd", "cost_usd", "margin_usd", "popularidad", "rating_promedio", "n_views",
    "n_cart", "n_purchases_product", "n_ratings", "category",
]
NUMERIC_COLS_CONTENT = [
    "price_usd", "cost_usd", "margin_usd", "popularidad",
    "rating_promedio", "n_ratings", "n_views", "n_cart", "n_purchases",
]


def rank_metrics_por_usuario(df_scores, top_k=TOP_K):
    """MAP@K y NDCG@K rankeando los candidatos DE CADA usuario por separado."""
    aps, ndcgs = [], []
    for _, grupo in df_scores.groupby("customer_id"):
        grupo = grupo.sort_values("score", ascending=False)
        relevantes = grupo["label"].to_numpy()
        n_relevantes = relevantes.sum()
        if n_relevantes == 0:
            continue

        hits, ap, dcg = 0, 0.0, 0.0
        for idx, es_relevante in enumerate(relevantes[:top_k]):
            if es_relevante:
                hits += 1
                ap += hits / (idx + 1)
                dcg += 1 / np.log2(idx + 2)
        ap /= min(top_k, n_relevantes)

        ideal_hits = min(top_k, n_relevantes)
        idcg = sum(1 / np.log2(i + 2) for i in range(int(ideal_hits)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        aps.append(ap)

    return {
        "map@k": float(np.mean(aps)) if aps else 0.0,
        "ndcg@k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_usuarios_evaluados": len(aps),
    }


def _cargar_datos_crudos():
    products_raw = pd.read_csv(RAW_DIR / "products.csv")
    customers_raw = pd.read_csv(RAW_DIR / "customers.csv")
    sessions_raw = pd.read_csv(RAW_DIR / "sessions.csv", parse_dates=["start_time"])
    orders_raw = pd.read_csv(RAW_DIR / "orders.csv", parse_dates=["order_time"])
    order_items_raw = pd.read_csv(RAW_DIR / "order_items.csv")
    reviews_raw = pd.read_csv(RAW_DIR / "reviews.csv", parse_dates=["review_time"])
    return products_raw, customers_raw, sessions_raw, orders_raw, order_items_raw, reviews_raw


def _split_temporal(user_item_df, events_con_user, train_frac=0.8):
    """Split 80/20 por fecha de última interacción de cada par usuario-producto."""
    positive_pairs = user_item_df[user_item_df["score"] > 0][["customer_id", "product_id"]].copy()
    positive_pairs["label"] = 1

    interaction_times = (
        events_con_user
        .dropna(subset=["customer_id", "product_id"])
        .assign(
            customer_id=lambda d: d["customer_id"].astype(int),
            product_id=lambda d: d["product_id"].astype(int),
        )
        .groupby(["customer_id", "product_id"])["timestamp"]
        .max()
        .reset_index()
    )
    positive_pairs = positive_pairs.merge(interaction_times, on=["customer_id", "product_id"], how="left")
    positive_pairs = positive_pairs.dropna(subset=["timestamp"])

    cutoff = positive_pairs["timestamp"].quantile(train_frac)
    train_positive = positive_pairs[positive_pairs["timestamp"] <= cutoff].drop(columns="timestamp")
    test_positive = positive_pairs[positive_pairs["timestamp"] > cutoff].drop(columns="timestamp")

    print(f"Corte temporal ({int(train_frac * 100)}% train / {int((1 - train_frac) * 100)}% test): {cutoff}")
    print(f"Positivos train: {len(train_positive)} | Positivos test: {len(test_positive)}")

    return positive_pairs, train_positive, test_positive, cutoff


def _samplear_negativos_duros(n, customer_pool, product_pool, product_probs, observed, rng):
    """Negativos muestreados con probabilidad proporcional a la popularidad del producto."""
    negativos = set()
    while len(negativos) < n:
        faltan = n - len(negativos)
        candidatos_customer = rng.choice(customer_pool, size=faltan * 2)
        candidatos_product = rng.choice(product_pool, size=faltan * 2, p=product_probs)
        for cid, pid in zip(candidatos_customer, candidatos_product):
            par = (int(cid), int(pid))
            if par not in observed and par not in negativos:
                negativos.add(par)
                if len(negativos) >= n:
                    break
    return list(negativos)


def preparar_datos_modelado():
    """
    Reproduce la preparación de datos del notebook: split temporal, features
    de usuario/producto calculados solo con datos de train (evita leakage) y
    negativos "duros" muestreados por popularidad.
    """
    interaction_matrix, product_features, user_features, user_item_df, events_con_user = generar_features()
    events_con_user["timestamp"] = pd.to_datetime(events_con_user["timestamp"])

    products_raw, customers_raw, sessions_raw, orders_raw, order_items_raw, reviews_raw = _cargar_datos_crudos()

    positive_pairs, train_positive, test_positive, cutoff = _split_temporal(user_item_df, events_con_user)

    events_train = events_con_user[events_con_user["timestamp"] <= cutoff]
    sessions_train = sessions_raw[sessions_raw["start_time"] <= cutoff]
    orders_train = orders_raw[orders_raw["order_time"] <= cutoff]
    order_items_train = order_items_raw[order_items_raw["order_id"].isin(orders_train["order_id"])]
    reviews_train = reviews_raw[reviews_raw["review_time"] <= cutoff]

    product_features_train = crear_features_producto(
        products_raw, events_train, reviews_train, order_items_train, orders_train
    )
    user_features_train = crear_features_usuario(
        customers_raw, sessions_train, events_train, orders_train, order_items_train, reviews_train
    )

    rng = np.random.default_rng(RANDOM_STATE)
    customer_ids_unique = customers_raw["customer_id"].to_numpy()
    product_ids_unique = products_raw["product_id"].to_numpy()

    popularidad_train = product_features_train.set_index("product_id")["popularidad"]
    pesos_producto = popularidad_train.reindex(product_ids_unique).fillna(0).to_numpy() + 1.0
    probs_producto = pesos_producto / pesos_producto.sum()

    observed_pairs = set(map(tuple, positive_pairs[["customer_id", "product_id"]].to_numpy()))

    n_neg_train = min(len(train_positive), 100_000)
    n_neg_test = min(len(test_positive), 30_000)

    neg_train = _samplear_negativos_duros(
        n_neg_train, customer_ids_unique, product_ids_unique, probs_producto, observed_pairs, rng
    )
    observed_pairs.update(neg_train)
    neg_test = _samplear_negativos_duros(
        n_neg_test, customer_ids_unique, product_ids_unique, probs_producto, observed_pairs, rng
    )

    train_negative = pd.DataFrame(neg_train, columns=["customer_id", "product_id"])
    train_negative["label"] = 0
    test_negative = pd.DataFrame(neg_test, columns=["customer_id", "product_id"])
    test_negative["label"] = 0

    train_pairs = pd.concat([train_positive, train_negative], axis=0, ignore_index=True)
    test_pairs = pd.concat([test_positive, test_negative], axis=0, ignore_index=True)

    print("Train (label -> conteo):", train_pairs["label"].value_counts().to_dict())
    print("Test (label -> conteo):", test_pairs["label"].value_counts().to_dict())

    user_features_model = user_features_train.rename(columns={"n_purchases": "n_purchases_user"})
    product_features_model = product_features_train.rename(columns={"n_purchases": "n_purchases_product"})

    def construir_features(pares_df):
        df = pares_df.merge(user_features_model, on="customer_id", how="left")
        df = df.merge(product_features_model, on="product_id", how="left")
        return df

    train_df = construir_features(train_pairs)
    test_df = construir_features(test_pairs)

    feature_cols = [col for col in FEATURE_COLS if col in train_df.columns]

    # Codificación de categóricas ajustada solo con train (una categoría no
    # vista en train se mapea a una clase "desconocida" aparte en test).
    category_mappings = {}
    for col in ["country", "category"]:
        train_df[col] = train_df[col].fillna("missing").astype(str)
        test_df[col] = test_df[col].fillna("missing").astype(str)

        categorias = sorted(train_df[col].unique())
        mapping = {cat: i for i, cat in enumerate(categorias)}
        desconocida = len(categorias)
        category_mappings[col] = {"mapping": mapping, "desconocida": desconocida}

        train_df[col] = train_df[col].map(mapping)
        test_df[col] = test_df[col].map(mapping).fillna(desconocida).astype(int)

    X_train = train_df[feature_cols].copy()
    y_train = train_df["label"]
    X_test = test_df[feature_cols].copy()
    y_test = test_df["label"]

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_cols, index=X_test.index)

    sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)

    print("Tamaño train:", X_train.shape, "| Tamaño test:", X_test.shape)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "sample_weight_train": sample_weight_train,
        "feature_cols": feature_cols,
        "imputer": imputer,
        "category_mappings": category_mappings,
        "train_df": train_df,
        "test_df": test_df,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "product_features_train": product_features_train,
        "customer_ids_unique": customer_ids_unique,
        "product_ids_unique": product_ids_unique,
        "events_train": events_train,

        # ==================================================
        # Datos adicionales para la API
        # ==================================================

         "products": products_raw,
        "customers": customers_raw,
        "product_features": product_features_train,
        "user_features": user_features_train,
    }
    


def entrenar_warm_start(datos):
    """
    Modelo elegido para warm-start: LightGBM con early stopping (250
    iteraciones máx., corta apenas deja de mejorar en un 15% de train
    reservado como validación interna).
    """
    X_train, y_train = datos["X_train"], datos["y_train"]
    X_test, y_test = datos["X_test"], datos["y_test"]
    sample_weight_train = datos["sample_weight_train"]

    X_train_fit, X_train_val, y_train_fit, y_train_val, sample_weight_fit, sample_weight_val = train_test_split(
        X_train, y_train, sample_weight_train,
        test_size=0.15, stratify=y_train, random_state=RANDOM_STATE,
    )

    modelo = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    modelo.fit(
        X_train_fit, y_train_fit,
        sample_weight=sample_weight_fit,
        eval_set=[(X_train_val, y_train_val)],
        eval_sample_weight=[sample_weight_val],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    pred = modelo.predict(X_test)
    proba = modelo.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }
    df_scores = pd.DataFrame({
        "customer_id": datos["test_df"]["customer_id"].to_numpy(),
        "score": proba,
        "label": y_test.to_numpy(),
    })
    metrics.update(rank_metrics_por_usuario(df_scores))

    print(f"\n=== LightGBM (warm-start) | best_iteration={modelo.best_iteration_} ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    return modelo, metrics


def entrenar_cold_start(datos):
    """
    Modelo elegido para cold-start: Content-Based Filtering. Vectoriza cada
    producto (features numéricos escalados + categoría one-hot) y construye
    el perfil de cada usuario como el promedio ponderado de los vectores de
    los productos con los que interactuó, comparados con similitud coseno.
    """
    product_features_train = datos["product_features_train"]
    product_ids_unique = datos["product_ids_unique"]
    customer_ids_unique = datos["customer_ids_unique"]
    events_train = datos["events_train"]
    train_pairs = datos["train_pairs"].copy()
    test_pairs = datos["test_pairs"].copy()

    product_feats_content = product_features_train.set_index("product_id").reindex(product_ids_unique)
    product_feats_content["category"] = product_feats_content["category"].fillna("missing")

    category_dummies = pd.get_dummies(product_feats_content["category"], prefix="cat")

    scaler_content = StandardScaler()
    numeric_scaled = scaler_content.fit_transform(product_feats_content[NUMERIC_COLS_CONTENT].fillna(0))
    product_vectors = np.hstack([numeric_scaled, category_dummies.to_numpy()])

    customer_index = {cid: i for i, cid in enumerate(customer_ids_unique)}
    product_index = {pid: j for j, pid in enumerate(product_ids_unique)}

    events_train_scored = events_train.copy()
    events_train_scored["score"] = events_train_scored["event_type"].map(EVENT_WEIGHTS).fillna(1)

    interacciones_content = (
        events_train_scored
        .dropna(subset=["customer_id", "product_id"])
        .assign(
            customer_id=lambda d: d["customer_id"].astype(int),
            product_id=lambda d: d["product_id"].astype(int),
        )
    )
    interacciones_content = interacciones_content[interacciones_content["product_id"].isin(product_index)]

    filas = interacciones_content["customer_id"].map(customer_index).to_numpy()
    columnas = interacciones_content["product_id"].map(product_index).to_numpy()
    pesos = interacciones_content["score"].to_numpy()

    n_users = len(customer_ids_unique)
    n_features = product_vectors.shape[1]
    perfil_usuario_sum = np.zeros((n_users, n_features))
    peso_usuario_sum = np.zeros(n_users)
    np.add.at(perfil_usuario_sum, filas, product_vectors[columnas] * pesos[:, None])
    np.add.at(peso_usuario_sum, filas, pesos)

    # Usuarios sin interacciones en train quedan con perfil en cero (similitud
    # 0 con todo, ya que no hay información de contenido para ellos).
    peso_usuario_sum_safe = np.where(peso_usuario_sum == 0, 1.0, peso_usuario_sum)
    perfil_usuario = perfil_usuario_sum / peso_usuario_sum_safe[:, None]

    content_scores_matrix = cosine_similarity(perfil_usuario, product_vectors)

    def score_content(customer_id, product_id):
        i, j = customer_index.get(customer_id), product_index.get(product_id)
        if i is None or j is None:
            return 0.0
        return content_scores_matrix[i, j]

    train_pairs["content_score"] = [
        score_content(c, p) for c, p in zip(train_pairs["customer_id"], train_pairs["product_id"])
    ]
    test_pairs["content_score"] = [
        score_content(c, p) for c, p in zip(test_pairs["customer_id"], test_pairs["product_id"])
    ]

    calibrador_content = LogisticRegression(class_weight="balanced")
    calibrador_content.fit(train_pairs[["content_score"]], train_pairs["label"])
    pred = calibrador_content.predict(test_pairs[["content_score"]])

    metrics = {
        "accuracy": accuracy_score(test_pairs["label"], pred),
        "precision": precision_score(test_pairs["label"], pred, zero_division=0),
        "recall": recall_score(test_pairs["label"], pred, zero_division=0),
        "f1": f1_score(test_pairs["label"], pred, zero_division=0),
    }
    metrics.update(rank_metrics_por_usuario(
        test_pairs.rename(columns={"content_score": "score"})[["customer_id", "score", "label"]]
    ))

    print("\n=== Content-Based Filtering (cold-start) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    artefactos = {
        "scaler": scaler_content,
        "category_columns": list(category_dummies.columns),
        "numeric_cols": NUMERIC_COLS_CONTENT,
        "product_vectors": product_vectors,
        "product_ids": product_ids_unique,
        "calibrador": calibrador_content,
    }
    return artefactos, metrics


def main():
    print("Preparando datos (split temporal + features sin leakage + negativos duros)...")
    datos = preparar_datos_modelado()

    print("\nEntrenando modelo warm-start (LightGBM)...")
    modelo_warm, metrics_warm = entrenar_warm_start(datos)

    print("\nEntrenando modelo cold-start (Content-Based Filtering)...")
    artefactos_cold, metrics_cold = entrenar_cold_start(datos)

    # ============================================================
    # CARGAR DATOS CRUDOS PARA LA INFERENCIA
    # ============================================================

    products_raw = pd.read_csv(RAW_DIR / "products.csv")
    customers_raw = pd.read_csv(RAW_DIR / "customers.csv")

    # Renombrar para que coincidan con el modelo
    user_features_api = (
        datos["user_features"]
        .rename(columns={"n_purchases": "n_purchases_user"})
        .copy()
    )

    product_features_api = (
        datos["product_features"]
        .rename(columns={"n_purchases": "n_purchases_product"})
        .copy()
    )

    # ============================================================
    # CREAR CATÁLOGO DE PRODUCTOS
    # ============================================================

    catalogo_productos = products_raw[
        [
            "product_id",
            "name",
            "category",
            "price_usd",
            "cost_usd",
            "margin_usd",
        ]
    ].copy()

    catalogo_productos.rename(
        columns={"name": "product_name"},
        inplace=True
    )

    # ============================================================
    # CREAR BASE DE USUARIOS
    # ============================================================

    usuarios_db = customers_raw.merge(
        user_features_api,
        on="customer_id",
        how="left"
    )


    # ============================================================
    # HISTORIAL DE PRODUCTOS POR USUARIO
    # ============================================================

    historial_usuario = (
        datos["events_train"]
        .dropna(subset=["customer_id", "product_id"])
        .groupby("customer_id")["product_id"]
        .unique()
        .apply(list)
        .to_dict()
    )


    # ============================================================
    # CATEGORÍA FAVORITA DEL USUARIO
    # ============================================================

    eventos_categoria = (
        datos["events_train"]
        .merge(
            products_raw[["product_id", "category"]],
            on="product_id",
            how="left"
        )
    )

    categoria_favorita = (
        eventos_categoria
        .groupby("customer_id")["category"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .to_dict()
    )


    
    
    

    # ============================================================
    # GUARDAR MODELO WARM START
    # ============================================================

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    warm_artifacts = {

        # Modelo
        "modelo": modelo_warm,

        # Preprocesamiento
        "feature_cols": datos["feature_cols"],
        "imputer": datos["imputer"],
        "category_mappings": datos["category_mappings"],

        # Datos para inferencia
        "products": catalogo_productos,
        "customers": usuarios_db,
        "product_features": product_features_api,
        "user_features": user_features_api,

        "historial_usuario": historial_usuario,
        "categoria_favorita": categoria_favorita,


        # Métricas
        "metrics": metrics_warm,
    }

    joblib.dump(
    warm_artifacts,
    MODELS_DIR / "warm_start_lightgbm.joblib"
)


    # ============================================================
    # GUARDAR MODELO COLD START
    # ============================================================

    cold_artifacts = {

        **artefactos_cold,

        "products": catalogo_productos,
        "product_features": product_features_api,

        "metrics": metrics_cold,
    }

    joblib.dump(
        cold_artifacts,
        MODELS_DIR / "cold_start_content_based.joblib"
    )

    # ============================================================
    # INFORMACIÓN FINAL
    # ============================================================

    print("\n" + "=" * 60)
    print("MODELOS ENTRENADOS CORRECTAMENTE")
    print("=" * 60)

    print(f"\nModelos guardados en:\n{MODELS_DIR}")

    print("\nArchivos generados:")

    print("   warm_start_lightgbm.joblib")
    print("      • Modelo LightGBM")
    print("      • Feature Columns")
    print("      • Imputer")
    print("      • Category Mappings")
    print("      • User Features")
    print("      • Product Features")
    print("      • Customers")
    print("      • Products")
    print("      • Métricas")

    print()

    print("   cold_start_content_based.joblib")
    print("      • Product Vectors")
    print("      • Product Features")
    print("      • Products")
    print("      • Calibrador")
    print("      • Métricas")

    print("\nPipeline finalizado correctamente.")


if __name__ == "__main__":
    main()
