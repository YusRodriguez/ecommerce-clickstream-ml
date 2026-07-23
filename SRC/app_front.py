import streamlit as st
import requests
import pandas as pd
import os

# ==========================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="Sistema de Recomendación E-commerce",
    page_icon="🛒",
    layout="wide"
)

API_URL = "https://ecommerce-clickstream-ml.onrender.com"

# ==========================================================
# LISTAS COMPARTIDAS DEL FORMULARIO
# ==========================================================

# Código de país (tal como está en los datos) -> nombre completo
PAISES = {
    "JP": "Japón",
    "IN": "India",
    "BR": "Brasil",
    "FR": "Francia",
    "US": "Estados Unidos",
    "GB": "Reino Unido",
    "MX": "México",
    "AU": "Australia",
    "SG": "Singapur",
    "AE": "Emiratos Árabes Unidos",
    "PL": "Polonia",
    "CA": "Canadá",
    "DE": "Alemania",
    "NL": "Países Bajos",
    "ES": "España",
    "SE": "Suecia",
    "ZA": "Sudáfrica"
}

CATEGORIAS = [
    "Todas las categorías",
    "Electronics",
    "Home & Kitchen",
    "Beauty",
    "Sports",
    "Fashion",
    "Books",
    "Toys"
]

# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================

@st.cache_data(ttl=60)
def obtener_estado_api():
    """
    Consulta el estado de la API.
    """
    try:
        r = requests.get(f"{API_URL}/")
        if r.status_code == 200:
            return r.json()
    except:
        pass

    return None


@st.cache_data(ttl=60)
def obtener_metricas():
    """
    Obtiene las métricas de ambos modelos.
    """
    try:
        r = requests.get(f"{API_URL}/model-metrics")
        if r.status_code == 200:
            return r.json()
    except:
        pass

    return None


@st.cache_data(ttl=60)
def obtener_usuario(customer_id):
    """
    Consulta los datos reales de un usuario existente.
    """
    try:
        r = requests.get(f"{API_URL}/users/{customer_id}")
        if r.status_code == 200:
            return r.json()
    except:
        pass

    return None


@st.cache_data(ttl=60)
def obtener_lista_usuarios():
    """
    Obtiene la lista de usuarios con historial de compras
    (para el selector de usuarios del Warm Start).
    """
    try:
        r = requests.get(f"{API_URL}/users-list")
        if r.status_code == 200:
            return r.json()
    except:
        pass

    return None


def solicitar_recomendaciones(payload):
    """
    Envía la solicitud de recomendaciones a la API.
    """
    r = requests.post(
        f"{API_URL}/recommend",
        json=payload
    )

    if r.status_code != 200:
        raise Exception(r.text)

    return r.json()

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
# SIDEBAR
# ==========================================================

st.sidebar.title("⚙️ Estado del Sistema")

estado = obtener_estado_api()

if estado is None:

    st.sidebar.error("🔴 API desconectada")

    st.stop()

else:

    st.sidebar.success("🟢 API conectada")


st.sidebar.markdown("---")

st.sidebar.subheader("Información del sistema")

col1, col2 = st.sidebar.columns(2)

with col1:

    st.metric(
        "Usuarios",
        estado["n_users"]
    )

with col2:

    st.metric(
        "Productos",
        estado["n_products"]
    )


st.sidebar.markdown("### Modelos")

st.sidebar.write(f"**Warm Start:** {estado['warm_model']}")
st.sidebar.write(f"**Cold Start:** {estado['cold_model']}")

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Actualizar"):

    st.cache_data.clear()

    st.rerun()

# ==========================================================
# MÉTRICAS DE LOS MODELOS
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
    [
        "👤 Usuario con historial",
        "🆕 Usuario nuevo"
    ],
    horizontal=True
)

st.markdown("")

# ==========================================================
# USUARIO CON HISTORIAL
# ==========================================================

if tipo_usuario == "👤 Usuario con historial":

    st.subheader("Warm Start (LightGBM)")

    usuarios_lista = obtener_lista_usuarios()

    if not usuarios_lista:

        st.warning("No se pudo obtener la lista de usuarios con historial.")
        st.stop()

    opciones_usuario = {
        f"{u['name']} ({u['email']})": u["customer_id"]
        for u in usuarios_lista
    }

    seleccion = st.selectbox(
        "Seleccione un usuario para ver sus recomendaciones personalizadas",
        list(opciones_usuario.keys())
    )

    customer_id = opciones_usuario[seleccion]

    if st.button("🚀 Obtener recomendaciones", use_container_width=True):

        payload = {

            "customer_id": int(customer_id),

            "context": {}

        }

        with st.spinner("Generando recomendaciones..."):

            try:

                resultado = solicitar_recomendaciones(payload)

                st.session_state["resultado"] = resultado

            except Exception as e:

                st.error(str(e))

# ==========================================================
# USUARIO NUEVO
# ==========================================================

else:

    st.subheader("Cold Start (Content-Based Filtering)")

    col1, col2, col3 = st.columns(3)

    with col1:

        opciones_pais_cold = {"Todos los países": ""}
        opciones_pais_cold.update(
            {nombre: codigo for codigo, nombre in PAISES.items()}
        )

        pais_seleccionado = st.selectbox(
            "País",
            list(opciones_pais_cold.keys())
        )

        country = opciones_pais_cold[pais_seleccionado]

    with col2:

        category = st.selectbox(
            "Categoría de interés",
            CATEGORIAS
        )

    with col3:

        age = st.number_input(
            "Edad del usuario",
            min_value=10,
            max_value=100,
            value=30,
            step=1
        )

    if st.button("🚀 Obtener recomendaciones", use_container_width=True):
        if category == "Todas las categorías":
            category = ""

        payload = {

            "customer_id": -1,

            "age": age,

            "country": country,

            "context": {

                "country": country,

                "category": category

            }

        }

        with st.spinner("Buscando productos similares..."):

            try:

                resultado = solicitar_recomendaciones(payload)

                st.session_state["resultado"] = resultado

            except Exception as e:

                st.error(str(e))


# ==========================================================
# MOSTRAR RESULTADOS
# ==========================================================

if "resultado" in st.session_state:

    resultado = st.session_state["resultado"]

    st.markdown("---")

    # ------------------------------------------------------
    # Fila superior: info del modelo
    # ------------------------------------------------------

    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.metric("Modelo utilizado", resultado["modelo"])

    with col_m2:
        st.metric(
            "Tipo de usuario",
            "Con historial" if resultado["usuario_con_historial"] else "Nuevo"
        )

    with col_m3:
        st.metric("Productos recomendados", len(resultado["recommendations"]))

    # ------------------------------------------------------
    # Perfil del usuario
    # ------------------------------------------------------

    usuario = resultado.get("usuario", {})

    if usuario:
        st.markdown("")
        st.subheader("👤 Perfil del usuario")

        info_cols = st.columns(4)

        with info_cols[0]:
            st.markdown(f"**Nombre:** {usuario.get('name', '-')}")
            st.markdown(f"**País:** {usuario.get('country', '-')}")

        with info_cols[1]:
            st.markdown(f"**Edad:** {usuario.get('age', '-')} años")
            st.markdown(f"**Email:** {usuario.get('email', '-')}")

        with info_cols[2]:
            if resultado["usuario_con_historial"]:
                st.markdown(f"**Sesiones:** {usuario.get('n_sessions', 0)}")
                st.markdown(f"**Compras:** {usuario.get('n_purchases', 0)}")

        with info_cols[3]:
            if resultado["usuario_con_historial"]:
                st.markdown(f"**Ticket promedio:** ${usuario.get('ticket_promedio', 0):.2f}")
                st.markdown(f"**Rating promedio:** {usuario.get('rating_promedio_usr', 0):.2f}")

    # ------------------------------------------------------
    # Preferencias de categoría (solo Cold Start)
    # ------------------------------------------------------

    categorias_pref = resultado.get("categorias_preferidas", [])

    if categorias_pref and not resultado["usuario_con_historial"]:
        st.markdown("")
        st.subheader("🏷️ Preferencias de categoría")

        df_cats = pd.DataFrame(categorias_pref)

        cat_col, chart_col = st.columns([1, 2])

        with cat_col:
            df_display = df_cats.copy()
            df_display.columns = ["Categoría", "Peso"]
            df_display["Peso"] = df_display["Peso"].round(4)
            st.dataframe(df_display, use_container_width=True, hide_index=True)

        with chart_col:
            st.bar_chart(
                df_cats.set_index("category")["peso"],
                horizontal=True
            )

    # ------------------------------------------------------
    # Recomendaciones personalizadas
    # ------------------------------------------------------

    st.markdown("---")
    st.subheader("📦 Productos recomendados para este usuario")

    recomendaciones = pd.DataFrame(resultado["recommendations"])

    if not recomendaciones.empty:

        columnas = {
            "product_id": "ID",
            "product_name": "Producto",
            "category": "Categoría",
            "price": "Precio (USD)",
            "score": "Score",
            "reason": "Motivo"
        }

        columnas_presentes = [
            col for col in columnas.keys()
            if col in recomendaciones.columns
        ]

        recomendaciones = recomendaciones[columnas_presentes]

        recomendaciones.rename(
            columns={col: columnas[col] for col in columnas_presentes},
            inplace=True
        )

        if "Score" in recomendaciones.columns:
            recomendaciones["Score"] = recomendaciones["Score"].round(4)

        if "Precio (USD)" in recomendaciones.columns:
            recomendaciones["Precio (USD)"] = recomendaciones["Precio (USD)"].round(2)

        # Top 3 destacados
        if len(recomendaciones) >= 3:
            st.markdown("#### 🏆 Top 3")
            top3 = recomendaciones.head(3)
            top_cols = st.columns(3)
            for idx, (_, row) in enumerate(top3.iterrows()):
                with top_cols[idx]:
                    st.markdown(f"**{idx+1}. {row.get('Producto', '')}**")
                    st.caption(f"{row.get('Categoría', '')} - ${row.get('Precio (USD)', 0):.2f}")
                    if "Score" in row:
                        st.caption(f"Score: {row['Score']:.4f}")

            st.markdown("")

        # Tabla completa
        st.markdown("#### 📋 Top completo")
        st.dataframe(
            recomendaciones,
            use_container_width=True,
            hide_index=True
        )

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
    "Desarrollado como Proyecto Final de Machine Learning · Sistema de Recomendación para E-commerce"
)

