"""Frontend Streamlit del sistema de recomendación de e-commerce.

Consume la API definida en `appi.py`. Ofrece dos flujos de solicitud:

- **Cliente con historial** (Warm Start): el usuario se elige de una
  lista poblada por `GET /users-list`; solo se envía `customer_id` y,
  opcionalmente, una categoría.
- **Cliente nuevo** (Cold Start): se piden edad, país y categoría.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

# ==========================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="Sistema de Recomendación E-commerce",
    page_icon="🛒",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000"
HTTP_TIMEOUT = 10  # segundos — evita que la app quede colgada si la API no responde

PAISES = [
    "Argentina",
    "Brazil",
    "Chile",
    "Portugal",
    "Spain",
    "USA",
]

CATEGORIAS = [
    "Todas las categorías",
    "Electronics",
    "Home & Kitchen",
    "Beauty",
    "Sports",
    "Fashion",
    "Books",
    "Toys",
]

# ==========================================================
# FUNCIONES AUXILIARES — LLAMADAS A LA API
# ==========================================================


def obtener_estado_api() -> Optional[dict]:
    """Consulta el estado de la API.

    Sin caché a propósito: es un chequeo de salud y debe reflejar el
    estado real en cada ejecución del script, no una copia de hasta
    60 segundos de antigüedad.
    """
    try:
        r = requests.get(f"{API_URL}/", timeout=HTTP_TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.RequestException:
        pass

    return None


@st.cache_data(ttl=60)
def obtener_metricas() -> Optional[dict]:
    """Obtiene las métricas de ambos modelos (cambian poco: sí cachea)."""
    try:
        r = requests.get(f"{API_URL}/model-metrics", timeout=HTTP_TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.RequestException:
        pass

    return None


@st.cache_data(ttl=60)
def obtener_usuarios() -> List[dict]:
    """Obtiene los usuarios elegibles para Warm Start (con historial)."""
    try:
        r = requests.get(f"{API_URL}/users-list", timeout=HTTP_TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.RequestException:
        pass

    return []


def solicitar_recomendaciones(payload: Dict[str, Any]) -> dict:
    """Envía la solicitud de recomendaciones a la API.

    Lanza una excepción con un mensaje legible si la API responde con
    un error, para que el llamador la muestre con `st.error`.
    """
    try:
        r = requests.post(f"{API_URL}/recommend", json=payload, timeout=HTTP_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        raise Exception(f"No se pudo contactar a la API: {exc}") from exc

    if r.status_code != 200:
        raise Exception(f"La API respondió con un error: {r.text}")

    return r.json()


def ejecutar_solicitud(payload: Dict[str, Any], mensaje_spinner: str) -> None:
    """Ejecuta una solicitud de recomendación y guarda el resultado.

    Centraliza el patrón (spinner + llamada + manejo de errores +
    guardado en `session_state`) que antes estaba duplicado entre el
    flujo Warm Start y el flujo Cold Start.
    """
    with st.spinner(mensaje_spinner):
        try:
            resultado = solicitar_recomendaciones(payload)
            st.session_state["resultado"] = resultado
        except Exception as exc:
            st.error(str(exc))


def categoria_o_vacio(category: str) -> str:
    """Traduce la opción 'Todas las categorías' al valor que espera la API."""
    return "" if category == "Todas las categorías" else category


# ==========================================================
# TÍTULO
# ==========================================================

st.title("🛒 Sistema Inteligente de Recomendación")

st.markdown(
    """
Este sistema recomienda productos utilizando dos estrategias:

- **Warm Start (LightGBM)** para usuarios con historial.
- **Cold Start (Content-Based Filtering)** para usuarios nuevos.

El modelo se selecciona automáticamente según el usuario consultado.
"""
)

# ==========================================================
# SIDEBAR — ESTADO DEL SISTEMA
# ==========================================================

st.sidebar.title("⚙️ Estado del Sistema")

estado = obtener_estado_api()

if estado is None:
    st.sidebar.error("🔴 API desconectada")
    st.stop()

st.sidebar.success("🟢 API conectada")
st.sidebar.markdown("---")
st.sidebar.subheader("Información del sistema")

col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Usuarios", estado["n_users"])
with col2:
    st.metric("Productos", estado["n_products"])

st.sidebar.markdown("### Modelos")
st.sidebar.write(f"**Warm Start:** {estado['warm_model']}")
st.sidebar.write(f"**Cold Start:** {estado['cold_model']}")
st.sidebar.markdown("---")

if st.sidebar.button("🔄 Actualizar"):
    st.cache_data.clear()
    st.rerun()

# ==========================================================
# SIDEBAR — MÉTRICAS DE LOS MODELOS
# ==========================================================

metricas = obtener_metricas()

if metricas is not None:
    with st.sidebar.expander("📈 Métricas del modelo"):
        st.markdown("### Warm Start")
        warm = metricas["warm_start"]
        st.write(f"Accuracy: **{warm['accuracy']:.4f}**")
        st.write(f"Precision: **{warm['precision']:.4f}**")
        st.write(f"Recall: **{warm['recall']:.4f}**")
        st.write(f"F1: **{warm['f1']:.4f}**")
        st.write(f"MAP@K: **{warm['map@k']:.4f}**")
        st.write(f"NDCG@K: **{warm['ndcg@k']:.4f}**")

        st.markdown("---")

        st.markdown("### Cold Start")
        cold = metricas["cold_start"]
        st.write(f"Accuracy: **{cold['accuracy']:.4f}**")
        st.write(f"Precision: **{cold['precision']:.4f}**")
        st.write(f"Recall: **{cold['recall']:.4f}**")
        st.write(f"F1: **{cold['f1']:.4f}**")
        st.write(f"MAP@K: **{cold['map@k']:.4f}**")
        st.write(f"NDCG@K: **{cold['ndcg@k']:.4f}**")

# ==========================================================
# FORMULARIO PRINCIPAL
# ==========================================================

st.markdown("---")
st.header("🎯 Solicitar recomendaciones")

tipo_usuario = st.radio(
    "Seleccione el tipo de usuario",
    ["👤 Usuario con historial", "🆕 Usuario nuevo"],
    horizontal=True,
)

st.markdown("")

# ==========================================================
# USUARIO CON HISTORIAL — WARM START
# ==========================================================

if tipo_usuario == "👤 Usuario con historial":

    st.subheader("Warm Start (LightGBM)")

    usuarios = obtener_usuarios()

    if not usuarios:
        st.warning(
            "No hay usuarios con historial disponibles en este momento "
            "(la API no devolvió resultados en /users-list)."
        )
    else:
        col1, col2 = st.columns(2)

        with col1:
            opciones = {
                f'{u["name"]} - {u["email"]}': u["customer_id"] for u in usuarios
            }
            seleccion = st.selectbox("Seleccione un cliente", list(opciones.keys()))
            customer_id = int(opciones[seleccion])

        with col2:
            category = st.selectbox("Categoría favorita", CATEGORIAS)

        if st.button("🚀 Obtener recomendaciones", use_container_width=True):
            payload = {
                "customer_id": customer_id,
                "context": {"category": categoria_o_vacio(category)},
            }
            ejecutar_solicitud(payload, "Generando recomendaciones...")

# ==========================================================
# USUARIO NUEVO — COLD START
# ==========================================================

else:

    st.subheader("Cold Start (Content-Based Filtering)")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Edad", 18, 80, 30)
        country = st.selectbox("País", PAISES)

    with col2:
        category = st.selectbox("Categoría de interés", CATEGORIAS)

    if st.button("🚀 Obtener recomendaciones", use_container_width=True):
        payload = {
            "customer_id": -1,
            "context": {
                "age": age,
                "country": country,
                "category": categoria_o_vacio(category),
            },
        }
        ejecutar_solicitud(payload, "Buscando productos similares...")

# ==========================================================
# MOSTRAR RECOMENDACIONES
# ==========================================================

if "resultado" in st.session_state:

    resultado = st.session_state["resultado"]

    st.markdown("---")
    st.header("📦 Recomendaciones obtenidas")

    # ------------------------------------------------------
    # Información general
    # ------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Modelo utilizado", resultado["modelo"])

    with col2:
        st.metric(
            "Usuario con historial",
            "Sí" if resultado["usuario_con_historial"] else "No",
        )

    with col3:
        st.metric("Productos recomendados", len(resultado["recommendations"]))

    st.markdown("")

    # ------------------------------------------------------
    # Historial del usuario (solo Warm Start)
    # ------------------------------------------------------

    if resultado["usuario_con_historial"]:
        with st.expander("👤 Información del usuario"):
            historial = resultado.get("historial", {})
            st.json(historial)

    # ------------------------------------------------------
    # Tabla de recomendaciones
    # ------------------------------------------------------

    recomendaciones = pd.DataFrame(resultado["recommendations"])

    if not recomendaciones.empty:

        columnas = {
            "product_id": "Product ID",
            "product_name": "Producto",
            "category": "Categoría",
            "price": "Precio (USD)",
            "score": "Score",
            "reason": "Motivo",
        }

        # Conservar únicamente las columnas que llegaron desde la API
        columnas_presentes = [
            col for col in columnas.keys() if col in recomendaciones.columns
        ]
        recomendaciones = recomendaciones[columnas_presentes]

        recomendaciones.rename(
            columns={col: columnas[col] for col in columnas_presentes},
            inplace=True,
        )

        if "Score" in recomendaciones.columns:
            recomendaciones["Score"] = recomendaciones["Score"].round(4)

        if "Precio (USD)" in recomendaciones.columns:
            recomendaciones["Precio (USD)"] = recomendaciones["Precio (USD)"].round(2)

        st.dataframe(recomendaciones, use_container_width=True, hide_index=True)

    else:
        st.warning("No se encontraron recomendaciones.")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown(
        """
        ### 📚 Proyecto Final - Sistema Inteligente de Recomendación

        **Tecnologías utilizadas**

        - 🐍 Python
        - ⚡ FastAPI
        - 🌐 Streamlit
        - 💡 LightGBM
        - 📊 Content-Based Filtering
        - 🐼 Pandas
        - 🔢 Scikit-Learn

        El sistema selecciona automáticamente el modelo de recomendación:

        - **Warm Start** para usuarios con historial.
        - **Cold Start** para usuarios nuevos.
        """
    )

with col2:
    st.success("🟢 API Online")
    st.caption("Versión 1.0")

    if st.button("🗑 Limpiar resultados"):
        if "resultado" in st.session_state:
            del st.session_state["resultado"]
        st.rerun()

st.markdown("---")

st.caption(
    "Desarrollado como Proyecto Final de Machine Learning · "
    "Sistema de Recomendación para E-commerce"
)