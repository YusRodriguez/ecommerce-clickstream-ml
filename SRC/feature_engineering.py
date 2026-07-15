"""
Módulo de Feature Engineering para Sistema de Recomendación.

Crea tres estructuras clave:
  1. Matriz usuario-item dispersa (para collaborative filtering SVD/ALS)
  2. Features de producto (para content-based y cold start)
  3. Features de usuario (para cold start de clientes nuevos)

Función principal: generar_features() ejecuta todo el pipeline.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_clean import limpiar_tablas


# =============================================================================
# 1. MATRIZ USUARIO-ITEM (para collaborative filtering)
# =============================================================================

def crear_matriz_usuario_item(events, sessions):
    """
    Construye la matriz de interacciones usuario-producto con scores implícitos.

    Pesos:
      - page_view:    1
      - add_to_cart:  3
      - purchase:     5

    Retorna:
      - matriz_usuario_item: DataFrame con columnas [customer_id, product_id, score]
      - events_con_user: events con customer_id unido (para uso posterior)
    """
    print("=" * 60)
    print("PASO 1: Matriz usuario-item")
    print("=" * 60)

    events_con_user = events.merge(
        sessions[["session_id", "customer_id"]],
        on="session_id",
        how="left",
    )

    event_weights = {"page_view": 1, "add_to_cart": 3, "purchase": 5}
    events_con_user["score"] = events_con_user["event_type"].map(event_weights).fillna(1)

    matriz = (
        events_con_user.groupby(["customer_id", "product_id"])["score"]
        .sum()
        .reset_index()
    )

    n_users = matriz["customer_id"].nunique()
    n_prods = matriz["product_id"].nunique()
    esparsidad = (1 - len(matriz) / (n_users * n_prods)) * 100

    print(f"  Usuarios: {n_users:,}")
    print(f"  Productos: {n_prods:,}")
    print(f"  Interacciones: {len(matriz):,}")
    print(f"  Esparsidad: {esparsidad:.2f}%")

    return matriz, events_con_user


# =============================================================================
# 2. FEATURES DE PRODUCTO (para content-based y cold start)
# =============================================================================

def crear_features_producto(products, events_con_user, reviews, order_items, orders):
    """
    Crea un DataFrame con una fila por producto y sus features derivados.

    Features:
      - category, price_usd, cost_usd, margin_usd (originales)
      - n_views, n_cart, n_purchases (conteos desde events)
      - popularidad (n_compradores desde order_items)
      - rating_promedio, n_ratings (desde reviews)
    """
    print("\n" + "=" * 60)
    print("PASO 2: Features de producto")
    print("=" * 60)

    # Conteos desde events
    pv = events_con_user[events_con_user["event_type"] == "page_view"]
    ac = events_con_user[events_con_user["event_type"] == "add_to_cart"]
    pu = events_con_user[events_con_user["event_type"] == "purchase"]

    n_views = pv.groupby("product_id").size().rename("n_views")
    n_cart = ac.groupby("product_id").size().rename("n_cart")
    n_purchases_events = pu.groupby("product_id").size().rename("n_purchases")

    # Popularidad desde order_items (n_compradores únicos)
    oi_con_order = order_items.merge(
        orders[["order_id", "customer_id"]], on="order_id", how="left"
    )
    popularidad = (
        oi_con_order.groupby("product_id")["customer_id"]
        .nunique()
        .rename("popularidad")
    )

    # Rating promedio desde reviews
    rating_stats = (
        reviews.groupby("product_id")["rating"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "rating_promedio", "count": "n_ratings"})
    )

    # Unir todo
    df_prod = products[["product_id", "category", "price_usd", "cost_usd", "margin_usd"]].copy()
    df_prod = df_prod.set_index("product_id")

    for feat in [n_views, n_cart, n_purchases_events, popularidad, rating_stats]:
        df_prod = df_prod.join(feat, how="left")

    df_prod = df_prod.fillna(0)

    print(f"  Productos: {len(df_prod):,}")
    print(f"  Features: {list(df_prod.columns)}")

    return df_prod.reset_index()


# =============================================================================
# 3. FEATURES DE USUARIO (para cold start y enriquecimiento)
# =============================================================================

def crear_features_usuario(
    customers, sessions, events_con_user, orders, order_items, reviews
):
    """
    Crea un DataFrame con una fila por usuario y sus features derivados.

    Features demográficas:
      - age, country, marketing_opt_in (originales)

    Features de comportamiento:
      - n_sessions (conteo de sesiones)
      - n_purchases (compras únicas)
      - ticket_promedio (total_usd promedio)
      - n_products_viewed (productos distintos vistos)
      - n_products_carted (productos distintos en carrito)
      - rating_promedio (rating promedio dado)
    """
    print("\n" + "=" * 60)
    print("PASO 3: Features de usuario")
    print("=" * 60)

    # Demografía
    df_usr = customers[["customer_id", "age", "country", "marketing_opt_in"]].copy()
    df_usr = df_usr.set_index("customer_id")

    # Sesiones por usuario
    n_sessions = sessions.groupby("customer_id").size().rename("n_sessions")
    df_usr = df_usr.join(n_sessions, how="left")

    # Compras (desde orders)
    n_purchases = orders.groupby("customer_id").size().rename("n_purchases")
    ticket_prom = (
        orders.groupby("customer_id")["total_usd"].mean().rename("ticket_promedio")
    )
    df_usr = df_usr.join(n_purchases, how="left").join(ticket_prom, how="left")

    # Productos vistos y en carrito (desde events)
    pv_user = events_con_user[events_con_user["event_type"] == "page_view"]
    ac_user = events_con_user[events_con_user["event_type"] == "add_to_cart"]

    n_viewed = (
        pv_user.groupby("customer_id")["product_id"]
        .nunique()
        .rename("n_products_viewed")
    )
    n_carted = (
        ac_user.groupby("customer_id")["product_id"]
        .nunique()
        .rename("n_products_carted")
    )
    df_usr = df_usr.join(n_viewed, how="left").join(n_carted, how="left")

    # Rating promedio dado por el usuario
    oi_con_order = order_items.merge(
        orders[["order_id", "customer_id"]], on="order_id", how="left"
    )
    rev_con_customer = reviews.merge(
        oi_con_order[["order_id", "customer_id"]].drop_duplicates("order_id"),
        on="order_id",
        how="left",
    )
    rating_usr = (
        rev_con_customer.groupby("customer_id")["rating"]
        .mean()
        .rename("rating_promedio_usr")
    )
    df_usr = df_usr.join(rating_usr, how="left")

    df_usr = df_usr.fillna(0)

    print(f"  Usuarios: {len(df_usr):,}")
    print(f"  Features: {list(df_usr.columns)}")

    return df_usr.reset_index()


# =============================================================================
# 4. PREPROCESAMIENTO (para modelado)
# =============================================================================

def preprocesar_para_modelado(matriz_usuario_item, features_producto, features_usuario):
    """
    Prepara los datos finales para los modelos de recomendación.

    Retorna:
      - interaction_matrix: sparse matrix usuario x producto (para SVD/ALS)
      - product_features: DataFrame de features de producto (para content-based)
      - user_features: DataFrame de features de usuario (para cold start)
      - user_item_df: DataFrame denso con scores (para modelos que lo necesiten)
    """
    print("\n" + "=" * 60)
    print("PASO 4: Preprocesamiento para modelado")
    print("=" * 60)

    from sklearn.preprocessing import StandardScaler, LabelEncoder

    # --- 4a. Matriz dispersa usuario x producto ---
    interaction_matrix = matriz_usuario_item.pivot_table(
        index="customer_id",
        columns="product_id",
        values="score",
        fill_value=0,
    )
    print(f"  Matriz dispersa: {interaction_matrix.shape[0]} usuarios x {interaction_matrix.shape[1]} productos")

    # --- 4b. Features de producto: encoding de category ---
    product_features = features_producto.copy()
    le_cat = LabelEncoder()
    product_features["category_encoded"] = le_cat.fit_transform(product_features["category"])

    scaler_prod = StandardScaler()
    num_cols_prod = ["price_usd", "cost_usd", "margin_usd", "popularidad",
                     "rating_promedio", "n_views", "n_cart", "n_purchases"]
    product_features[num_cols_prod] = scaler_prod.fit_transform(product_features[num_cols_prod])

    print(f"  Features producto: {product_features.shape[1]} columnas")

    # --- 4c. Features de usuario: encoding de country ---
    user_features = features_usuario.copy()
    le_country = LabelEncoder()
    user_features["country_encoded"] = le_country.fit_transform(user_features["country"])

    scaler_usr = StandardScaler()
    num_cols_usr = ["age", "n_sessions", "n_purchases", "ticket_promedio",
                    "n_products_viewed", "n_products_carted", "rating_promedio_usr"]
    user_features[num_cols_usr] = scaler_usr.fit_transform(user_features[num_cols_usr])

    print(f"  Features usuario: {user_features.shape[1]} columnas")

    # --- 4d. DataFrame denso usuario-item con scores ---
    user_item_df = matriz_usuario_item.copy()

    print(f"  DataFrame usuario-item: {user_item_df.shape[0]:,} filas")

    return interaction_matrix, product_features, user_features, user_item_df


# =============================================================================
# 5. PIPELINE PRINCIPAL
# =============================================================================

def generar_features(ruta_data="Data/Raw"):
    """
    Pipeline completo de feature engineering.

    Retorna:
      - interaction_matrix: sparse matrix usuario x producto
      - product_features: features de producto
      - user_features: features de usuario
      - user_item_df: DataFrame con scores
      - events_con_user: events con customer_id (para uso posterior)
    """
    print("\n" + "#" * 60)
    print("# PIPELINE DE FEATURE ENGINEERING")
    print("#" * 60)

    # 1. Limpiar datos
    events, sessions, reviews, orders, order_items, customers, products = limpiar_tablas()

    # 2. Matriz usuario-item
    matriz_usuario_item, events_con_user = crear_matriz_usuario_item(events, sessions)

    # 3. Features de producto
    features_producto = crear_features_producto(
        products, events_con_user, reviews, order_items, orders
    )

    # 4. Features de usuario
    features_usuario = crear_features_usuario(
        customers, sessions, events_con_user, orders, order_items, reviews
    )

    # 5. Preprocesamiento
    interaction_matrix, product_features, user_features, user_item_df = (
        preprocesar_para_modelado(matriz_usuario_item, features_producto, features_usuario)
    )

    print("\n" + "#" * 60)
    print("# FEATURE ENGINEERING COMPLETADO")
    print("#" * 60)
    print(f"\n  Matriz usuario-item: {interaction_matrix.shape}")
    print(f"  Features producto:   {product_features.shape}")
    print(f"  Features usuario:    {user_features.shape}")

    return interaction_matrix, product_features, user_features, user_item_df, events_con_user


if __name__ == "__main__":
    import os

    (
        interaction_matrix,
        product_features,
        user_features,
        user_item_df,
        events_con_user,
    ) = generar_features()

    # Guardar resultados
    os.makedirs("Data/Processed", exist_ok=True)

    interaction_matrix.to_csv("Data/Processed/interaction_matrix.csv")
    product_features.to_csv("Data/Processed/product_features.csv", index=False)
    user_features.to_csv("Data/Processed/user_features.csv", index=False)
    user_item_df.to_csv("Data/Processed/user_item_df.csv", index=False)

    print("\n✓ Archivos guardados en Data/Processed/")
