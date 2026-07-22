import logging
from typing import Any, Dict, List, Optional
 
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recommender_api")
 
app = FastAPI(
    title="🛒 E-commerce Recommender API",
    description="""
    API para la recomendación de productos basada en perfiles de usuario.
 
    ### Características:
    - **Cold Start**: Recomendación por similitud de contenido para nuevos usuarios.
    - **Personalización**: Recomendación basada en historial para usuarios existentes.
    - **Monitoreo**: Endpoint de métricas para validación del modelo.
    """,
    version="1.0.0",
    contact={"name": "Equipo Data Science"},
)
 
# ==========================================================
# CARGA DE MODELOS
# ==========================================================
 
try:
    warm_model = joblib.load("Models/warm_start_lightgbm.joblib")
    cold_model = joblib.load("Models/cold_start_content_based.joblib")
except Exception as exc:
    raise RuntimeError(f"No fue posible cargar los modelos.\n{exc}") from exc
 
# ==========================================================
# HISTORIAL Y CATEGORÍA FAVORITA
# ==========================================================
 
HISTORIAL_USUARIO: Dict[int, list] = warm_model["historial_usuario"]
CATEGORIA_FAVORITA: Dict[int, str] = warm_model["categoria_favorita"]
 
# ==========================================================
# WARM START — artefactos del modelo
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
# COLD START — artefactos del modelo
# ==========================================================
 
scaler = cold_model["scaler"]
product_vectors = cold_model["product_vectors"]
product_ids = cold_model["product_ids"]
numeric_cols_content = cold_model["numeric_cols"]
category_columns = cold_model["category_columns"]
 
# ==========================================================
# CATÁLOGO
# ==========================================================
 
CATALOGO = products.copy()
CATALOGO["category"] = CATALOGO["category"].astype(str).str.lower().str.strip()
 
CATALOGO_NOMBRES = dict(zip(CATALOGO["product_id"], CATALOGO["product_name"]))
TODOS_LOS_PRODUCTOS = CATALOGO["product_id"].unique().tolist()
 
# ==========================================================
# FEATURES DE PRODUCTOS Y USUARIOS (indexados para lookup O(1))
# ==========================================================
 
PRODUCTOS_DB: Dict[int, dict] = (
    product_features.merge(products, on="product_id", how="left")
    .set_index("product_id")
    .to_dict("index")
)
 
USUARIOS_DB: Dict[int, dict] = user_features.set_index("customer_id").to_dict("index")
 
# ==========================================================
# ESQUEMAS DE REQUEST
# ==========================================================
 
 
class RecommendationRequest(BaseModel):
    """Payload esperado por `POST /recommend`.
 
    `context` es intencionalmente flexible: para warm-start solo se
    espera `category` (opcional); para cold-start se esperan además
    `age` y `country`.
    """
 
    customer_id: int
    context: Dict[str, Any] = {}
 
 
# ==========================================================
# FUNCIONES AUXILIARES DE DOMINIO
# ==========================================================
 
 
def usuario_existe(customer_id: int) -> bool:
    """Indica si `customer_id` está en la base de usuarios entrenada."""
    return customer_id in USUARIOS_DB
 
 
def obtener_usuario(customer_id: int) -> Optional[dict]:
    """Devuelve las features del usuario, o `None` si no existe."""
    return USUARIOS_DB.get(customer_id)
 
 
def tiene_compras(customer_id: int) -> bool:
    """Indica si el usuario tiene al menos una compra registrada.
 
    Es, junto con `usuario_existe`, el criterio que decide si una
    solicitud se enruta a warm-start o a cold-start (ver `/recommend`).
    """
    usuario = obtener_usuario(customer_id)
    if usuario is None:
        return False
    return usuario.get("n_purchases_user", 0) > 0
 
 
def es_elegible_warm_start(customer_id: int) -> bool:
    """Criterio único de elegibilidad para warm-start.
 
    Centraliza la condición que usan tanto `/recommend` (para decidir
    qué modelo usar) como `/users-list` (para decidir qué usuarios
    ofrecer en el selector del frontend), evitando que ambos puntos
    del sistema puedan quedar desincronizados.
    """
    return usuario_existe(customer_id) and tiene_compras(customer_id)
 
 
def obtener_producto(product_id: int) -> Optional[dict]:
    """Devuelve las features de un producto, o `None` si no existe."""
    return PRODUCTOS_DB.get(product_id)
 
 
def obtener_candidatos(customer_id: int, context: Dict[str, Any]) -> pd.DataFrame:
    """Arma el conjunto de productos candidatos para un usuario.
 
    Excluye productos ya vistos por el usuario y, si hay una categoría
    (explícita en `context` o inferida de su categoría favorita),
    filtra el catálogo a esa categoría.
    """
    vistos = set(HISTORIAL_USUARIO.get(customer_id, []))
 
    candidatos = CATALOGO.copy()
    candidatos = candidatos[~candidatos.product_id.isin(vistos)]
 
    categoria = context.get("category")
    if not categoria:
        categoria = CATEGORIA_FAVORITA.get(customer_id)
 
    if categoria:
        categoria = categoria.lower().strip()
        candidatos = candidatos[candidatos["category"] == categoria]
 
    return candidatos
 
 
def obtener_nombre_producto(product_id: int) -> str:
    """Devuelve el nombre de un producto, con un valor por defecto."""
    return CATALOGO_NOMBRES.get(product_id, "Producto")
 
 
# ==========================================================
# COLD START
# ==========================================================
 
 
def recomendar_cold_start(context: Dict[str, Any], top_k: int = 10) -> List[dict]:
    """Recomienda productos para usuarios nuevos vía Content-Based Filtering.
 
    Construye un vector de perfil a partir de `context` (variables
    numéricas + categoría) y devuelve los `top_k` productos más
    similares por similitud coseno.
    """
    # -----------------------------
    # Variables numéricas
    # -----------------------------
    numeric_values = []
    for col in numeric_cols_content:
        numeric_values.append(context.get(col, 0))
 
    # -----------------------------
    # Categoría (one-hot)
    # -----------------------------
    categoria = context.get("category", "missing")
    categoria_vector = []
    for col in category_columns:
        if col == f"cat_{categoria}":
            categoria_vector.append(1)
        else:
            categoria_vector.append(0)
 
    # -----------------------------
    # Escalado y vector final
    # -----------------------------
    numeric_scaled = scaler.transform(np.array(numeric_values).reshape(1, -1))
    vector_usuario = np.hstack(
        [numeric_scaled, np.array(categoria_vector).reshape(1, -1)]
    )
 
    # -----------------------------
    # Similaridad coseno
    # -----------------------------
    similitudes = cosine_similarity(vector_usuario, product_vectors)[0]
    mejores = np.argsort(similitudes)[::-1][:top_k]
 
    recomendaciones = []
    for indice in mejores:
        pid = int(product_ids[indice])
        recomendaciones.append(
            {
                "product_id": pid,
                "product_name": obtener_nombre_producto(pid),
                "score": float(similitudes[indice]),
            }
        )
 
    return recomendaciones
 
 
# ==========================================================
# WARM START
# ==========================================================
 
 
def recomendar_warm_start(
    customer_id: int, context: Dict[str, Any], top_k: int = 10
) -> List[dict]:
    """Recomienda productos personalizados vía LightGBM.
 
    Solo debe invocarse para usuarios con historial (ver
    `es_elegible_warm_start`). Construye, para cada producto candidato,
    una fila con features de usuario + contexto + producto, y rankea
    por la probabilidad predicha por el modelo.
    """
    usuario = obtener_usuario(customer_id)
    if usuario is None:
        raise HTTPException(status_code=404, detail="El usuario no posee historial.")
 
    # =====================================================
    # Candidatos
    # =====================================================
    productos_candidatos = obtener_candidatos(customer_id, context)
    logger.info(
        "customer_id=%s candidatos=%d", customer_id, len(productos_candidatos)
    )
 
    # Si ya interactuó con todo en su categoría favorita, usar
    # cualquier producto que todavía no haya visto.
    if len(productos_candidatos) == 0:
        productos_candidatos = CATALOGO.copy()
        productos_candidatos = productos_candidatos[
            ~productos_candidatos["product_id"].isin(
                HISTORIAL_USUARIO.get(customer_id, [])
            )
        ]
 
    # =====================================================
    # Construcción del dataset para inferencia
    # =====================================================
    filas = []
    ids_productos = []
 
    for _, producto_catalogo in productos_candidatos.iterrows():
        product_id = producto_catalogo["product_id"]
        producto = obtener_producto(product_id)
 
        if producto is None:
            continue
 
        fila: Dict[str, Any] = {}
        fila.update(usuario)  # features del usuario
        fila.update(context)  # contexto recibido en el request
        fila.update(producto)  # features del producto
 
        filas.append(fila)
        ids_productos.append(product_id)
 
    if len(filas) == 0:
        return []
 
    X = pd.DataFrame(filas)
 
    # =====================================================
    # Codificación de variables categóricas
    # =====================================================
    for columna, meta in category_mappings.items():
        if columna in X.columns:
            X[columna] = X[columna].map(meta["mapping"]).fillna(meta["desconocida"])
 
    # =====================================================
    # Alinear columnas al orden esperado por el modelo e imputar
    # =====================================================
    X = X.reindex(columns=feature_cols, fill_value=0)
    X = pd.DataFrame(imputer.transform(X), columns=feature_cols)
 
    # =====================================================
    # Predicción
    # =====================================================
    scores = modelo.predict_proba(X)[:, 1]
 
    resultados = pd.DataFrame({"product_id": ids_productos, "score": scores})
 
    # =====================================================
    # Enriquecer con datos del catálogo y quedarnos con el top_k
    # =====================================================
    resultados = resultados.merge(
        CATALOGO[["product_id", "product_name", "category", "price_usd"]],
        on="product_id",
        how="left",
    )
    resultados = resultados.sort_values("score", ascending=False).head(top_k)
 
    recomendaciones = []
    for _, fila in resultados.iterrows():
        recomendaciones.append(
            {
                "product_id": int(fila["product_id"]),
                "product_name": fila["product_name"],
                "category": fila["category"],
                "price": round(float(fila["price_usd"]), 2),
                "score": round(float(fila["score"]), 4),
                "reason": "Recommended by LightGBM",
            }
        )
 
    return recomendaciones
 
 
# ==========================================================
# ENDPOINT PRINCIPAL
# ==========================================================
 
 
@app.post("/recommend")
def recommend(request: RecommendationRequest) -> dict:
    """Genera recomendaciones personalizadas para un cliente.
 
    - Si `customer_id` existe en la base entrenada y tiene compras
      registradas -> **Warm Start** (LightGBM).
    - En cualquier otro caso (usuario nuevo o sin compras) ->
      **Cold Start** (Content-Based).
    """
    logger.info(
        "POST /recommend customer_id=%s context=%s",
        request.customer_id,
        request.context,
    )
 
    try:
        if es_elegible_warm_start(request.customer_id):
            recomendaciones = recomendar_warm_start(
                customer_id=request.customer_id,
                context=request.context,
                top_k=10,
            )
            usuario = obtener_usuario(request.customer_id)
 
            return {
                "customer_id": request.customer_id,
                "modelo": "Warm Start (LightGBM)",
                "usuario_con_historial": True,
                "historial": usuario,
                "recommendations": recomendaciones,
            }
 
        recomendaciones = recomendar_cold_start(request.context, top_k=10)
 
        return {
            "customer_id": request.customer_id,
            "modelo": "Cold Start (Content Based)",
            "usuario_con_historial": False,
            "recommendations": recomendaciones,
        }
 
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error generando recomendaciones para %s", request.customer_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
 
 
# ==========================================================
# MÉTRICAS
# ==========================================================
 
 
@app.get("/model-metrics")
def model_metrics() -> dict:
    """Métricas de evaluación guardadas para ambos modelos."""
    return {
        "warm_start": warm_model["metrics"],
        "cold_start": cold_model["metrics"],
    }
 
 
# ==========================================================
# CATÁLOGO
# ==========================================================
 
 
@app.get("/products")
def get_products() -> list:
    """Catálogo completo de productos."""
    return CATALOGO.to_dict(orient="records")
 
 
# ==========================================================
# USUARIOS
# ==========================================================
 
 
@app.get("/users-list")
def users_list() -> list:
    """Usuarios elegibles para warm-start, para poblar un selector.
 
    Devuelve únicamente `customer_id`, `name` y `email` de los
    clientes que efectivamente recibirán recomendaciones Warm Start
    (mismo criterio que usa `/recommend`), para que el frontend nunca
    ofrezca en la lista a alguien que en la práctica caería en
    Cold Start.
    """
    usuarios = customers.copy()
    usuarios = usuarios[
        usuarios["customer_id"].apply(es_elegible_warm_start)
    ]
 
    return (
        usuarios[["customer_id", "name", "email"]]
        .sort_values("name")
        .to_dict(orient="records")
    )
 
 
@app.get("/users/{customer_id}")
def get_user(customer_id: int) -> dict:
    """Perfil (features) de un usuario existente.
 
    Devuelve 404 si el `customer_id` no está en la base entrenada.
    """
    usuario = obtener_usuario(customer_id)
    if usuario is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado.")
 
    return {"customer_id": customer_id, **usuario}
 
 
# ==========================================================
# HEALTH CHECK
# ==========================================================
 
 
@app.get("/")
def root() -> dict:
    """Estado general del servicio."""
    return {
        "status": "running",
        "api": "E-commerce Recommendation API",
        "warm_model": "LightGBM",
        "cold_model": "Content Based",
        "n_products": len(TODOS_LOS_PRODUCTOS),
        "n_users": len(USUARIOS_DB),
    }
 
 
@app.get("/health")
def health() -> dict:
    """Health check simple, para orquestadores/monitoreo."""
    return {"status": "healthy"}