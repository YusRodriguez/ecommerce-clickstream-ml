import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI(
    title="🛒 E-commerce Recommender API",
    description="""
    API para la recomendación de productos basada en perfiles de usuario.
    
    ### Características:
    - **Cold Start**: Recomendación por popularidad para nuevos usuarios.
    - **Personalización**: Recomendación basada en historial para usuarios existentes.
    - **Monitoreo**: Endpoint de métricas para validación del modelo.
    """,
    version="1.0.0",
    contact={
        "name": "Equipo Data Science",
    }
)

# ==========================================================
# CARGA DE MODELOS
# ==========================================================

try:

    warm_model = joblib.load(
        "Models/warm_start_lightgbm.joblib"
    )

    cold_model = joblib.load(
        "Models/cold_start_content_based.joblib"
    )

except Exception as e:

    raise RuntimeError(
        f"No fue posible cargar los modelos.\n{e}"
    )

# ==========================================================
# HISTORIAL Y CATEGORÍA FAVORITA
# ==========================================================

HISTORIAL_USUARIO = warm_model["historial_usuario"]

HISTORIAL_DETALLADO = warm_model.get("historial_detallado", {})

CATEGORIA_FAVORITA = warm_model["categoria_favorita"]

PREFERENCIAS_CATEGORIA = warm_model.get("preferencias_categoria", {})

USER_CAT_STATS = warm_model.get("user_cat_stats", pd.DataFrame())

if not USER_CAT_STATS.empty and "category" in USER_CAT_STATS.columns:
    USER_CAT_STATS["category"] = USER_CAT_STATS["category"].astype(str).str.lower().str.strip()

ALPHA_BOOST = 0.3


# ==========================================================
# WARM START
# ==========================================================

modelo = warm_model["modelo"]

feature_cols = warm_model["feature_cols"]

imputer = warm_model["imputer"]

scaler_products = warm_model["scaler_products"]

product_numeric_cols = warm_model["product_numeric_cols"]

category_mappings = warm_model["category_mappings"]

for _col, _meta in category_mappings.items():
    _meta["mapping"] = {k.lower().strip() if isinstance(k, str) else k: v for k, v in _meta["mapping"].items()}

# ==========================================================
# DECODIFICACIÓN DE COUNTRY
# ==========================================================

REVERSE_COUNTRY_MAPPING = warm_model.get("reverse_country_mapping", {})

# ==========================================================
# DATOS NECESARIOS PARA LA INFERENCIA
# ==========================================================

products = warm_model["products"].copy()

# Compatibilidad con el resto de la API
if "name" in products.columns:
    products.rename(columns={"name": "product_name"}, inplace=True)

customers = warm_model["customers"]

product_features = warm_model["product_features"]

user_features = warm_model["user_features"]


# ==========================================================
# COLD START
# ==========================================================

scaler = cold_model["scaler"]

product_vectors = cold_model["product_vectors"]

product_ids = cold_model["product_ids"]

numeric_cols_content = cold_model["numeric_cols"]

category_columns = cold_model["category_columns"]

PERFILES_USUARIO = cold_model["perfiles_usuario"]

# ==========================================================
# POPULARIDAD POR PAÍS (segmentación del Cold Start)
# ==========================================================
# A partir de las órdenes reales, calcula por país cuántas veces
# se compró cada producto ahí. Un país vacío ("") significa
# "Todos los países": no se segmenta, se usa la popularidad
# general del catálogo completo.
#
# Nota: la gran mayoría de los productos se vendió alguna vez en
# casi todos los países (entre el 58% y el 96% del catálogo según
# el país), así que filtrar solo por "se vendió alguna vez ahí"
# casi no cambia el top-K frente a "todos los países" -> por eso
# se usa la CANTIDAD de compras por país como score de ranking,
# no una simple presencia sí/no.

_orders_pais = pd.read_csv(
    "Data/Raw/orders.csv",
    usecols=["order_id", "country"]
)

_order_items_pais = pd.read_csv(
    "Data/Raw/order_items.csv",
    usecols=["order_id", "product_id"]
)

_ventas_pais = _order_items_pais.merge(
    _orders_pais,
    on="order_id",
    how="inner"
)

POPULARIDAD_PAIS = {

    pais: grupo.set_index("product_id")["compras"].to_dict()

    for pais, grupo in (

        _ventas_pais
        .groupby(["country", "product_id"])
        .size()
        .rename("compras")
        .reset_index()
        .groupby("country")

    )

}

PRODUCTOS_POR_PAIS = {
    pais: set(int(pid) for pid in popularidad.keys())
    for pais, popularidad in POPULARIDAD_PAIS.items()
}

# ==========================================================
# CATÁLOGO
# ==========================================================
CATALOGO = products.copy()

CATALOGO["category"] = (
    CATALOGO["category"]
    .astype(str)
    .str.lower()
    .str.strip()
)


CATALOGO_NOMBRES = dict(

    zip(

        CATALOGO["product_id"],

        CATALOGO["product_name"]

    )

)

TODOS_LOS_PRODUCTOS = (

    CATALOGO["product_id"]

    .unique()

    .tolist()

)

# ==========================================================
# FEATURES DE PRODUCTOS
# ==========================================================

PRODUCTOS_DB = (

    product_features

    .merge(

        products.drop(columns=["category", "price_usd", "cost_usd", "margin_usd"], errors="ignore"),

        on="product_id",

        how="left"

    )

    .set_index("product_id")

    .to_dict("index")

)

# ==========================================================
# FEATURES DE USUARIOS
# ==========================================================

USUARIOS_DB = (

    user_features

    .set_index("customer_id")

    .to_dict("index")

)

# ==========================================================
# REQUEST
# ==========================================================

class RecommendationRequest(BaseModel):

    customer_id: int

    context: dict = {}

    country: str = None

    age: int = None

# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================

def usuario_existe(customer_id: int):

    return customer_id in USUARIOS_DB


def obtener_usuario(customer_id: int):

    usuario = USUARIOS_DB.get(customer_id)

    if usuario is not None:

        country_encoded = usuario.get("country")

        if country_encoded is not None and isinstance(country_encoded, (int, float)):

            usuario["country"] = REVERSE_COUNTRY_MAPPING.get(
                int(country_encoded),
                str(country_encoded)
            )

        fila = customers[customers["customer_id"] == customer_id]
        if not fila.empty:
            usuario["name"] = fila.iloc[0].get("name", "")
            usuario["email"] = fila.iloc[0].get("email", "")

    return usuario


def tiene_compras(customer_id: int):

    usuario = obtener_usuario(customer_id)

    if usuario is None:
        return False

    return usuario.get("n_purchases_user", 0) > 0


def obtener_producto(product_id: int):
    return PRODUCTOS_DB.get(product_id)

def obtener_candidatos(customer_id, context):

    vistos = set(HISTORIAL_USUARIO.get(customer_id, []))

    candidatos = CATALOGO.copy()

    candidatos = candidatos[
        ~candidatos.product_id.isin(vistos)
    ]

    prefs = PREFERENCIAS_CATEGORIA.get(customer_id, {})
    if prefs:
        top_cats = sorted(prefs.keys(), key=lambda c: prefs[c], reverse=True)[:4]
        candidatos_pref = candidatos[candidatos["category"].isin(top_cats)]
        if len(candidatos_pref) >= 10:
            candidatos = candidatos_pref
        else:
            candidatos = candidatos

    return candidatos


def obtener_nombre_producto(product_id):

    return CATALOGO_NOMBRES.get(

        product_id,

        "Producto"

    )


# ==========================================================
# COLD START
# ==========================================================

def recomendar_cold_start(customer_id, context, top_k=10):
    """
    Recomendación para usuarios sin compras utilizando
    Content-Based Filtering.

    Si el usuario tiene historial real de interacción (vistas/carrito) desde
    el período de entrenamiento, se usa su perfil real (PERFILES_USUARIO).
    Si no hay ningún registro de ese usuario, se arma un vector a mano con
    lo que venga en `context`.
    """

    perfil_real = PERFILES_USUARIO.get(customer_id)

    sin_categoria = False

    if perfil_real is not None:
        vector_usuario = perfil_real.reshape(1, -1)

    else:
        # -----------------------------
        # Categoría
        # -----------------------------
        categoria = context.get(
            "category",
            "missing"
        )

        categoria_vector = []

        for col in category_columns:

            if col == f"cat_{categoria}":
                categoria_vector.append(1)

            else:
                categoria_vector.append(0)

        # "Todas las categorías": no hay ninguna preferencia de
        # contenido para comparar (ni categoría ni datos numéricos
        # reales del usuario anónimo).
        sin_categoria = not any(categoria_vector)

        # -----------------------------
        # Numérico
        # -----------------------------
        # Un usuario anónimo no tiene señales numéricas reales
        # (país y categoría no son features del vector de contenido).
        # Se usa el perfil neutro/promedio del catálogo (0 en el
        # espacio ya escalado) para que sea la categoría la que
        # oriente la recomendación, en vez de escalar un 0 crudo
        # (que se traduce en un vector muy alejado del promedio
        # y termina dominando la similaridad).
        numeric_scaled = np.zeros((1, len(numeric_cols_content)))

        vector_usuario = np.hstack([

            numeric_scaled,

            np.array(categoria_vector).reshape(1, -1)

        ])

    # -----------------------------
    # Similaridad / Popularidad + Segmentación por país
    # -----------------------------
    # País vacío ("Todos los países") -> sin segmentar.
    pais = context.get("country")

    if sin_categoria:

        # Sin ninguna preferencia de contenido, la similaridad de
        # coseno contra un vector nulo no aporta orden real: se
        # recomienda directamente por popularidad.
        popularidad_pais = POPULARIDAD_PAIS.get(pais) if pais else None

        if popularidad_pais:

            # Popularidad real DE ESE país (cantidad de compras ahí),
            # no la popularidad general del catálogo.
            similitudes = np.array([
                popularidad_pais.get(int(pid), 0)
                for pid in product_ids
            ])

        else:
            idx_popularidad = numeric_cols_content.index("popularidad")
            similitudes = product_vectors[:, idx_popularidad]

        indices_candidatos = list(range(len(product_ids)))

    else:

        similitudes = cosine_similarity(
            vector_usuario,
            product_vectors
        )[0]

        productos_pais = PRODUCTOS_POR_PAIS.get(pais) if pais else None

        if productos_pais:

            indices_candidatos = [
                i for i, pid in enumerate(product_ids)
                if int(pid) in productos_pais
            ]

        else:
            indices_candidatos = list(range(len(product_ids)))

    similitudes_candidatas = similitudes[indices_candidatos]

    mejores_local = np.argsort(similitudes_candidatas)[::-1][:top_k]

    mejores = [indices_candidatos[i] for i in mejores_local]

    recomendaciones = []

    for indice in mejores:

        pid = int(product_ids[indice])

        fila_cat = CATALOGO[CATALOGO["product_id"] == pid]

        cat = fila_cat["category"].values[0] if not fila_cat.empty else ""

        precio = round(float(fila_cat["price_usd"].values[0]), 2) if not fila_cat.empty else 0.0

        recomendaciones.append({

            "product_id": pid,

            "product_name": obtener_nombre_producto(pid),

            "category": cat,

            "price": precio,

            "score": float(similitudes[indice])

        })

    return recomendaciones


# ==========================================================
# COLD START - DEMOGRAPHIC COLLABORATIVE FILTERING
# ==========================================================

def recomendar_cold_start_demographic(context, top_k=10):
    """
    Recomendación para usuarios anónimos (sin perfil en PERFILES_USUARIO)
    basada en usuarios similares por demographics.

    Busca usuarios del mismo país y rango etario (±5 años) que tengan
    compras, y recomienda los productos que más compraron entre ellos.
    Si se especifica una categoría, filtra solo productos de esa categoría.
    """
    age = context.get("age")
    country = context.get("country")
    category = context.get("category", "").strip().lower()

    if age is None or not country:
        return []

    age = int(age)

    usuarios_similares = []

    for uid, udata in USUARIOS_DB.items():

        if udata.get("n_purchases_user", 0) <= 0:
            continue

        if udata.get("country") != country:
            continue

        user_age = int(udata.get("age", 0))

        if abs(user_age - age) <= 5:
            usuarios_similares.append(uid)

    if not usuarios_similares:
        return []

    productos_frecuencia = {}

    for uid in usuarios_similares:

        productos_vistos = HISTORIAL_USUARIO.get(uid, [])

        for pid in productos_vistos:

            pid = int(pid)

            if category:
                cat_producto = CATALOGO[
                    CATALOGO["product_id"] == pid
                ]["category"].values

                if len(cat_producto) == 0 or cat_producto[0] != category:
                    continue

            if pid not in productos_frecuencia:
                productos_frecuencia[pid] = 0

            productos_frecuencia[pid] += 1

    productos_ordenados = sorted(
        productos_frecuencia.items(),
        key=lambda x: x[1],
        reverse=True
    )

    recomendaciones = []

    for pid, frecuencia in productos_ordenados[:top_k]:

        fila_cat = CATALOGO[CATALOGO["product_id"] == pid]

        precio = round(float(fila_cat["price_usd"].values[0]), 2) if not fila_cat.empty else 0.0

        recomendaciones.append({

            "product_id": pid,

            "product_name": obtener_nombre_producto(pid),

            "category": CATALOGO[
                CATALOGO["product_id"] == pid
            ]["category"].values[0] if pid in CATALOGO["product_id"].values else "",

            "price": precio,

            "score": round(frecuencia / len(usuarios_similares), 4),

            "reason": f"Popular entre {len(usuarios_similares)} usuarios de {country} de {age} años"
                      + (f" en {context.get('category', '')}" if category else "")

        })

    return recomendaciones


# ==========================================================
# WARM START
# ==========================================================

def recomendar_warm_start(customer_id, context, top_k=10):
    """
    Recomendación personalizada utilizando LightGBM.
    Para usuarios con historial.
    """

    usuario = obtener_usuario(customer_id)

    if usuario is None:
        raise HTTPException(
            status_code=404,
            detail="El usuario no posee historial."
        )

    # =====================================================
    # Obtener candidatos
    # =====================================================

    productos_candidatos = obtener_candidatos(
       customer_id,
       context
    )

    if len(productos_candidatos) == 0:

        productos_candidatos = CATALOGO.copy()

        productos_candidatos = productos_candidatos[
            ~productos_candidatos["product_id"].isin(
                HISTORIAL_USUARIO.get(customer_id, [])
            )
        ]

    filas = []
    ids_productos = []

    # =====================================================
    # Construcción del dataset para inferencia
    # =====================================================

    for _, producto_catalogo in productos_candidatos.iterrows():

        product_id = producto_catalogo["product_id"]

        producto = obtener_producto(product_id)

        if producto is None:
            continue

        fila = {}

        # Features del usuario
        fila.update(usuario)

        # Contexto recibido desde la API
        fila.update(context)

        # Features del producto
        fila.update(producto)

        filas.append(fila)

        ids_productos.append(product_id)

    if len(filas) == 0:
        return []

    # =====================================================
    # DataFrame
    # =====================================================

    X = pd.DataFrame(filas)

    # =====================================================
    # Merge de stats usuario-categoría
    # =====================================================

    if not USER_CAT_STATS.empty and "customer_id" in X.columns and "category" in X.columns:
        X["category"] = X["category"].astype(str).str.lower().str.strip()
        X = X.merge(
            USER_CAT_STATS,
            on=["customer_id", "category"],
            how="left"
        )

    # =====================================================
    # Codificación de variables categóricas
    # =====================================================

    for columna, meta in category_mappings.items():

        if columna in X.columns:

            X[columna] = (
                X[columna]
                .map(meta["mapping"])
                .fillna(meta["desconocida"])
            )

    # =====================================================
    # Reordenar columnas
    # =====================================================

    X = X.reindex(
        columns=feature_cols,
        fill_value=0
    )

    # =====================================================
    # Imputación
    # =====================================================

    X = pd.DataFrame(
        imputer.transform(X),
        columns=feature_cols
    )

    # =====================================================
    # Escalado de features numéricas de producto
    # =====================================================

    cols_to_scale = [c for c in product_numeric_cols if c in X.columns]

    if cols_to_scale:
        X[cols_to_scale] = scaler_products.transform(X[cols_to_scale])

    # =====================================================
    # Predicción LightGBM
    # =====================================================

    scores_lgbm = modelo.predict_proba(X)[:, 1]

    # =====================================================
    # Content-based similarity (perfil del usuario vs productos)
    # =====================================================

    perfil = PERFILES_USUARIO.get(customer_id)

    if perfil is not None:

        idx_map = {int(pid): i for i, pid in enumerate(product_ids)}

        vec_indices = [idx_map.get(int(pid)) for pid in ids_productos]

        valid_mask = [i is not None for i in vec_indices]
        valid_vectors = np.array([
            product_vectors[vec_indices[i]] for i in range(len(vec_indices)) if valid_mask[i]
        ])

        if len(valid_vectors) > 0:
            sims = cosine_similarity(
                perfil.reshape(1, -1), valid_vectors
            )[0]
        else:
            sims = np.zeros(len(ids_productos))

        content_scores = np.zeros(len(ids_productos))
        j = 0
        for i in range(len(ids_productos)):
            if valid_mask[i]:
                content_scores[i] = sims[j]
                j += 1

    else:

        content_scores = np.zeros(len(ids_productos))

    # =====================================================
    # Blend: LightGBM + Content-Based
    # =====================================================

    ALPHA = 0.5

    scores_final = ALPHA * scores_lgbm + (1 - ALPHA) * content_scores

    resultados = pd.DataFrame({

        "product_id": ids_productos,

        "score_lgbm": scores_lgbm,

        "score_content": content_scores,

        "score": scores_final

    })

    # =====================================================
    # Agregar información del catálogo
    # =====================================================

    resultados = resultados.merge(

        CATALOGO[
            [
                "product_id",
                "product_name",
                "category",
                "price_usd"
            ]
        ],

        on="product_id",

        how="left"

    )

    resultados = resultados.sort_values("score", ascending=False).head(top_k)

    seleccionados = [fila for _, fila in resultados.iterrows()]

    recomendaciones = []

    for fila in seleccionados:

        recomendaciones.append({

            "product_id": int(fila["product_id"]),

            "product_name": fila["product_name"],

            "category": fila["category"],

            "price": round(float(fila["price_usd"]), 2),

            "score": round(float(fila["score"]), 4),

            "reason": "Recommended by LightGBM + Content-Based"

        })

    return recomendaciones
# ==========================================================
# ENDPOINT PRINCIPAL
# ==========================================================

@app.post("/recommend")
def recommend(request: RecommendationRequest):

    """
    Genera recomendaciones personalizadas.

    Si el usuario existe en la base de usuarios entrenada:
        -> Warm Start (LightGBM)

    Si no existe:
        -> Cold Start (Content-Based o Demographic)
    """

    try:

        # ---------------------------------------
        # Usuario con historial
        # ---------------------------------------

        if usuario_existe(request.customer_id) and tiene_compras(request.customer_id):

            recomendaciones = recomendar_warm_start(

                customer_id=request.customer_id,

                context=request.context,

                top_k=10

            )

            usuario = obtener_usuario(request.customer_id)

            prefs_usuario = PREFERENCIAS_CATEGORIA.get(request.customer_id, {})

            categorias_ordenadas = sorted(
                prefs_usuario.items(), key=lambda x: x[1], reverse=True
            )

            usuario_info = {}
            if usuario:
                usuario_info = {
                    "customer_id": request.customer_id,
                    "name": usuario.get("name", ""),
                    "email": usuario.get("email", ""),
                    "age": usuario.get("age", 0),
                    "country": usuario.get("country", ""),
                    "n_sessions": int(usuario.get("n_sessions", 0)),
                    "n_purchases": int(usuario.get("n_purchases_user", 0)),
                    "ticket_promedio": round(float(usuario.get("ticket_promedio", 0)), 2),
                    "n_products_viewed": int(usuario.get("n_products_viewed", 0)),
                    "n_products_carted": int(usuario.get("n_products_carted", 0)),
                    "rating_promedio_usr": round(float(usuario.get("rating_promedio_usr", 0)), 2),
                }

            return {

                "customer_id": request.customer_id,

                "modelo": "Warm Start (LightGBM)",

                "usuario_con_historial": True,

                "usuario": usuario_info,

                "categorias_preferidas": [
                    {"category": cat, "peso": round(peso, 4)}
                    for cat, peso in categorias_ordenadas
                ],

                "recommendations": recomendaciones

            }

        # ---------------------------------------
        # Usuario nuevo
        # ---------------------------------------

        contexto_demografico = {}

        if request.age is not None:

            contexto_demografico["age"] = request.age

        if request.country is not None:

            contexto_demografico["country"] = request.country

        if request.context.get("category"):

            contexto_demografico["category"] = request.context["category"]

        if contexto_demografico.get("age") and contexto_demografico.get("country"):

            recomendaciones = recomendar_cold_start_demographic(

                contexto_demografico,

                top_k=10

            )

            modelo_usado = "Cold Start (Demographic Collaborative Filtering)"

        else:

            recomendaciones = recomendar_cold_start(

                request.customer_id,

                request.context,

                top_k=10

            )

            modelo_usado = "Cold Start (Content Based)"

        prefs_cold = PREFERENCIAS_CATEGORIA.get(request.customer_id, {})
        categorias_cold = sorted(prefs_cold.items(), key=lambda x: x[1], reverse=True)

        usuario_cold = obtener_usuario(request.customer_id)
        usuario_cold_info = {}
        if usuario_cold:
            usuario_cold_info = {
                "customer_id": request.customer_id,
                "name": usuario_cold.get("name", ""),
                "age": usuario_cold.get("age", 0),
                "country": usuario_cold.get("country", ""),
            }

        return {

            "customer_id": request.customer_id,

            "modelo": modelo_usado,

            "usuario_con_historial": False,

            "usuario": usuario_cold_info,

            "categorias_preferidas": [
                {"category": cat, "peso": round(peso, 4)}
                for cat, peso in categorias_cold
            ],

            "recommendations": recomendaciones

        }

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# ==========================================================
# MÉTRICAS
# ==========================================================

@app.get("/model-metrics")
def model_metrics():

    return {

        "warm_start": warm_model["metrics"],

        "cold_start": cold_model["metrics"]

    }


# ==========================================================
# CATÁLOGO
# ==========================================================

@app.get("/products")
def products():

    return CATALOGO.to_dict(

        orient="records"

    )


# ==========================================================
# USUARIOS
# ==========================================================

@app.get("/users/{customer_id}")
def user(customer_id: int):

    if not usuario_existe(customer_id):

        raise HTTPException(

            status_code=404,

            detail="Usuario no encontrado."

        )

    return obtener_usuario(customer_id)


@app.get("/users-list")
def users_list():

    """
    Devuelve los usuarios con historial de compras
    (customer_id, name, email) para poblar el selector
    de usuarios del frontend.
    """

    usuarios_con_compras = customers[customers["n_purchases_user"] > 0]

    resultado = []

    for _, fila in usuarios_con_compras.iterrows():

        resultado.append({

            "customer_id": int(fila["customer_id"]),

            "name": fila["name"],

            "email": fila["email"]

        })

    return resultado


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get("/")
def root():

    return {

        "status": "running",

        "api": "E-commerce Recommendation API",

        "warm_model": "LightGBM",

        "cold_model": "Content Based",

        "n_products": len(TODOS_LOS_PRODUCTOS),

        "n_users": len(USUARIOS_DB)

    }


@app.get("/health")
def health():

    return {

        "status": "healthy"

    }


