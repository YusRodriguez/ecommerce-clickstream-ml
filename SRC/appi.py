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

CATEGORIA_FAVORITA = warm_model["categoria_favorita"]


# ==========================================================
# WARM START
# ==========================================================

modelo = warm_model["modelo"]

feature_cols = warm_model["feature_cols"]

imputer = warm_model["imputer"]

category_mappings = warm_model["category_mappings"]

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

        products,

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

# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================

def usuario_existe(customer_id: int):

    return customer_id in USUARIOS_DB


def obtener_usuario(customer_id: int):

    return USUARIOS_DB.get(customer_id)


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

    categoria = context.get("category")

    if not categoria:
        categoria = CATEGORIA_FAVORITA.get(customer_id)

    if categoria:
        categoria = categoria.lower().strip()

        candidatos = candidatos[
            candidatos["category"] == categoria
        ]

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

        recomendaciones.append({

            "product_id": pid,

            "product_name": obtener_nombre_producto(pid),

            "score": float(similitudes[indice])

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
    print("Cantidad de candidatos:", len(productos_candidatos))
    print(productos_candidatos[["product_id", "category"]].head())

    # Si ya compró todo en su categoría favorita,
    # usar cualquier producto que todavía no haya visto
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
    # Predicción
    # =====================================================
    print("=" * 70)
    print("CLIENTE:", customer_id)
    print("USUARIO:")
    print(usuario)

    print("\nX antes de predecir:")
    print(X.head())

    print("=" * 70)

    scores = modelo.predict_proba(X)[:, 1]

    resultados = pd.DataFrame({

        "product_id": ids_productos,

        "score": scores

    })

    print("Primeros scores:")
    print(scores[:10])

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

    resultados = resultados.sort_values(

        "score",

        ascending=False

    ).head(top_k)

    recomendaciones = []

    for _, fila in resultados.iterrows():

        recomendaciones.append({

            "product_id": int(fila["product_id"]),

            "product_name": fila["product_name"],

            "category": fila["category"],

            "price": round(float(fila["price_usd"]), 2),

            "score": round(float(fila["score"]), 4),

            "reason": "Recommended by LightGBM"

        })

    return recomendaciones
# ==========================================================
# ENDPOINT PRINCIPAL
# ==========================================================

@app.post("/recommend")
def recommend(request: RecommendationRequest):

    print("CLIENTE:", request.customer_id)
    print("CONTEXTO RECIBIDO:", request.context)
    
    """
    Genera recomendaciones personalizadas.

    Si el usuario existe en la base de usuarios entrenada:
        -> Warm Start (LightGBM)

    Si no existe:
        -> Cold Start (Content-Based)
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

            return {

                "customer_id": request.customer_id,

                "modelo": "Warm Start (LightGBM)",

                "usuario_con_historial": True,

                "historial": usuario,

                "recommendations": recomendaciones

            }

        # ---------------------------------------
        # Usuario nuevo
        # ---------------------------------------

        recomendaciones = recomendar_cold_start(

            request.customer_id,

            request.context,

            top_k=10

        )

        return {

            "customer_id": request.customer_id,

            "modelo": "Cold Start (Content Based)",

            "usuario_con_historial": False,

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


