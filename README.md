# E-commerce Clickstream ML — Sistema de Recomendación de Productos

Proyecto final del Bootcamp de Data Science de **Henry**. Sistema inteligente de recomendación para e-commerce basado en filtrado colaborivo y content-based, desarrollado siguiendo la metodología **CRISP-DM**.

---

## Equipo

| Nombre | GitHub |
|--------|--------|
| Yustin Rodríguez | |
| Elías | |
| Carina | |
| Rocío | |

---

## Objetivo

Construir un motor de recomendación de productos que combine **filtrado colaborivo** (SVD/ALS) con **filtrado por contenido** (content-based) para resolver tanto el caso de usuarios recurrentes como el cold start de usuarios y productos nuevos.

El dataset es sintético y contiene ~1 millón de registros de una plataforma de e-commerce: clientes, productos, pedidos, items de pedido, sesiones de navegación, eventos de clickstream y reseñas. Las fechas fueron generadas aleatoriamente, lo que introdujo violaciones de coherencia temporal (~50%) que se corrigieron en la etapa de limpieza.

---

## Estructura del repositorio

```
ecommerce-clickstream-ml/
│
├── Data/
│   ├── Raw/                              # CSVs originales (7 tablas)
│   │   ├── customers.csv
│   │   ├── products.csv
│   │   ├── orders.csv
│   │   ├── order_items.csv
│   │   ├── sessions.csv
│   │   ├── events.csv
│   │   └── reviews.csv
│   └── Processed/                        # Archivos generados por feature engineering
│       ├── interaction_matrix.csv
│       ├── product_features.csv
│       ├── user_features.csv
│       └── user_item_df.csv
│
├── Notebooks/
│   ├── 01_business_understanding.ipynb   # Contexto del negocio y preguntas clave
│   ├── 02_eda_general.ipynb              # Análisis exploratorio de todas las tablas
│   ├── 03_data_preparation.ipynb         # Limpieza y validación de los 7 datasets
│   └── 04_feature_engineering.ipynb      # Construcción de features y matrices de entrada
│
├── SRC/
│   ├── utils.py                          # Funciones utilitarias reutilizables
│   ├── data_clean.py                     # Pipeline de limpieza completo
│   └── feature_engineering.py            # Pipeline de feature engineering completo
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

Dataset sintético de e-commerce con ~1 millón de registros en 7 tablas relacionadas.

### Tablas

| Tabla | Registros | Columnas | Descripción |
|-------|-----------|----------|-------------|
| `customers` | 20,000 | 7 | Clientes registrados: ID, nombre, email, edad (18-75), país (17 países), fecha de registro, aceptación de marketing |
| `products` | 1,197 | 6 | Catálogo de productos: ID, categoría, precio (USD), costo (USD), margen (USD), descripción |
| `orders` | 33,580 | 10 | Pedidos realizados: ID, cliente, fecha del pedido, subtotal, descuento (%), total, estado, dispositivo, fuente de tráfico |
| `order_items` | 59,090 | 5 | Items dentro de cada pedido: ID, pedido, producto, precio unitario, cantidad, total de línea |
| `sessions` | 120,000 | 6 | Sesiones de navegación: ID, cliente, fecha de inicio, dispositivo, fuente de tráfico, canal de marketing |
| `events` | 760,958 | 10 | Eventos dentro de cada sesión: ID, sesión, tipo de evento (page_view, add_to_cart, purchase), producto, timestamp, monto (USD) |
| `reviews` | 10,765 | 5 | Reseñas de productos por pedido: ID, pedido, producto, rating (1-5), fecha de reseña |

### Estadísticas clave

- **Rango temporal:** 2020-01-01 a 2025-11-01
- **Países representados:** 17 (América, Europa, Asia)
- **Tipos de eventos:** page_view, add_to_cart, purchase
- **Rating promedio de reseñas:** 3.74 / 5

### Problemas de calidad detectados

| Problema | Alcance | Solución aplicada |
|----------|---------|-------------------|
| Fechas anteriores al registro del cliente | ~50% de orders, sessions y events | Corrección automática: fecha de referencia + offset aleatorio (semilla 42) |
| Duplicados en order_items | 73 filas | Eliminación de duplicados exactos |
| Duplicados en reviews | 4 exactas + 26 contradictorias | Eliminación de exactas; conservar el más reciente en contradictorias |
| review_text no utilizable | 100% de las reseñas | Eliminación de la columna (5 frases fijas 1:1 con rating, NLP imposible) |
| Clientes sin sesiones | 55 clientes (0.27%) | Reporte sin modificación (cold start) |

---

## Pipeline

El proyecto sigue las 5 etapas de CRISP-DM:

### 1. Business Understanding

**Archivo:** `Notebooks/01_business_understanding.ipynb`

Definición del problema de negocio: desarrollar un sistema de recomendación de productos para una plataforma de e-commerce utilizando información histórica de clientes, productos, órdenes y eventos de navegación. Se establece el enfoque del proyecto y las preguntas clave a responder.

### 2. Exploratory Data Analysis (EDA)

**Archivo:** `Notebooks/02_eda_general.ipynb`

Análisis exploratorio completo de las 7 tablas del dataset:

- **Distribuciones** de variables numéricas (edad, precios, montos, ratings)
- **Relaciones** entre tablas (clientes-pedidos-productos-sesiones-eventos-reseñas)
- **Calidad de datos** (nulos, duplicados, tipos de dato)
- **Hallazgo clave:** ~50% de violaciones de coherencia temporal en orders, sessions y events (fechas anteriores al registro del cliente)
- **Patrones de navegación:** distribución de eventos por tipo, tasa de conversión de carrito a compra

### 3. Data Preparation

**Archivos:** `Notebooks/03_data_preparation.ipynb` + `SRC/data_clean.py`

Limpieza y validación de los 7 datasets. El notebook documenta cada paso con explicaciones; el módulo `data_clean.py` contiene la implementación reutilizable.

#### Pipeline de limpieza (`limpiar_tablas()`)

| Paso | Función | Qué hace |
|------|---------|----------|
| 1 | `convertir_fechas()` | Convierte todas las columnas de fechas a datetime |
| 2 | `clean_customers()` | Valida nulos, duplicados, rango de edad (18-100), formato de email |
| 3 | `clean_products()` | Valida nulos, duplicados, precios > 0, costo ≤ precio, margen consistente |
| 4 | `corregir_coherencia_temporal_orders()` | Corrige order_time para que sea ≥ signup_date del cliente |
| 5 | `clean_orders()` | Valida nulos, duplicados, montos > 0, descuento 0-100%, total consistente, FK a customers |
| 6 | `clean_order_items()` | Elimina 73 duplicados, valida montos, FK a products y orders |
| 7 | `eliminar_duplicados_events()` | Verifica que no haya duplicados (OK) |
| 8 | `eliminar_duplicados_sessions()` | Verifica que no haya duplicados (OK) |
| 9 | `eliminar_duplicados_reviews()` | Elimina 4 exactas + 26 contradictorias |
| 10 | `eliminar_review_text()` | Elimina columna review_text (5 frases fijas, NLP imposible) |
| 11 | `reportar_clientes_sesiones()` | Reporta 55 clientes sin sesiones (sin modificar) |
| 12 | `corregir_coherencia_temporal()` | Corrige fechas de events, sessions y reviews para respetar signup ≤ order ≤ review |
| 13 | `validar_integridad()` | Valida 7 relaciones FK → PK (0 registros huérfanos) |

#### Resultados de la limpieza

| Tabla | Filas antes | Filas después | Acciones |
|-------|-------------|---------------|----------|
| customers | 20,000 | 20,000 | Sin modificaciones (validación OK) |
| products | 1,197 | 1,197 | Sin modificaciones (validación OK) |
| orders | 33,580 | 33,580 | 16,923 fechas corregidas (50.4%) |
| order_items | 59,163 | 59,090 | 73 duplicados eliminados |
| sessions | 120,000 | 120,000 | 60,411 fechas corregidas |
| events | 760,958 | 760,958 | 382,899 fechas corregidas |
| reviews | 10,765 | 10,735 | 30 duplicados eliminados, review_text eliminada, fechas corregidas |

### 4. Feature Engineering

**Archivos:** `Notebooks/04_feature_engineering.ipynb` + `SRC/feature_engineering.py`

Construcción de las 4 estructuras de entrada para los modelos de recomendación.

#### 4.1 Matriz usuario-item (para filtrado colaborivo)

Une events con sessions para obtener customer_id, asigna pesos por tipo de evento:
- **page_view:** 1 punto
- **add_to_cart:** 3 puntos
- **purchase:** 5 puntos

Agrupa por (customer_id, product_id) y suma los scores.

| Métrica | Valor |
|---------|-------|
| Usuarios únicos | 19,945 |
| Productos únicos | 1,197 |
| Interacciones | 529,593 |
| Pares posibles | 23,874,165 |
| **Sparsity** | **97.78%** |

La alta sparsez es normal en e-commerce y es exactamente lo que los algoritmos SVD/ALS resuelven: predecir los ceros.

#### 4.2 Features de producto (para content-based y cold start)

Un DataFrame con una fila por producto y 11 features derivados:

| Feature | Fuente | Descripción |
|---------|--------|-------------|
| category | products | Categoría del producto |
| price_usd | products | Precio de venta |
| cost_usd | products | Costo |
| margin_usd | products | Margen de ganancia |
| n_views | events | Cantidad de vistas (page_view) |
| n_cart | events | Cantidad de agregados al carrito |
| n_purchases | events | Cantidad de compras |
| popularidad | order_items | Compradores únicos |
| rating_promedio | reviews | Rating promedio recibido |
| n_ratings | reviews | Cantidad de reseñas |

Sirve para: recomendaciones por similitud de contenido (content-based) y cold start de productos nuevos sin interacciones.

#### 4.3 Features de usuario (para cold start)

Un DataFrame con una fila por usuario y 10 features derivados:

| Feature | Fuente | Descripción |
|---------|--------|-------------|
| age | customers | Edad del usuario |
| country | customers | País de registro |
| marketing_opt_in | customers | Aceptación de marketing |
| n_sessions | sessions | Cantidad de sesiones |
| n_purchases | orders | Cantidad de compras únicas |
| ticket_promedio | orders | Gasto promedio por compra |
| n_products_viewed | events | Productos distintos vistos |
| n_products_carted | events | Productos distintos agregados al carrito |
| rating_promedio_usr | reviews | Rating promedio dado por el usuario |

Sirve para: cold start de clientes nuevos (recomendar basándose en perfil demográfico y comportamiento de usuarios similares).

#### 4.4 Preprocesamiento para modelado

| Transformación | Propósito |
|----------------|-----------|
| Pivot table | Convierte la lista de scores en matriz usuario × producto |
| LabelEncoder | Convierte texto a números (category, country) |
| StandardScaler | Centra variables numéricas alrededor de 0 |

#### Archivos de salida (`Data/Processed/`)

| Archivo | Dimensiones | Destino |
|---------|-------------|---------|
| `interaction_matrix.csv` | 19,945 × 1,197 | Filtrado colaborivo (SVD/ALS) |
| `product_features.csv` | 1,197 × 12 | Content-based y cold start de productos |
| `user_features.csv` | 20,000 × 11 | Cold start de clientes nuevos |
| `user_item_df.csv` | 529,593 × 3 | Análisis y modelos alternativos |

### 5. Modelado *(próximamente)*

Entrenamiento y evaluación de modelos de recomendación:
- **Filtrado colaborivo:** SVD y ALS sobre la matriz de interacción
- **Content-based:** Similitud coseno sobre features de producto
- **Híbrido:** Combinación de ambos para cubrir cold start y usuarios recurrentes
- **Métricas:** Precision@K, Recall@K

---

## Archivos SRC — Documentación completa

### `SRC/utils.py`

Funciones utilitarias reutilizables en todo el proyecto:

| Función | Parámetros | Retorna | Descripción |
|---------|------------|---------|-------------|
| `cargarDatos(nombre_archivo)` | `nombre_archivo: str` (sin extensión .csv) | `pd.DataFrame` | Carga un CSV desde `Data/Raw/`. Resuelve rutas de forma absoluta para que funcione desde cualquier directorio. |
| `verificar_integridad_referencial(child_df, child_col, parent_df, parent_col)` | DataFrames hijo/padre y columnas FK/PK | `int` | Cuenta cuántos valores del hijo no existen en el padre (registros huérfanos). |
| `calcular_sets_eventos_sesion(events)` | `events: DataFrame` | `pd.Series` | Agrupa eventos por sesión y retorna un set de tipos de evento por sesión. |
| `convertir_columnas_fecha(df, date_columns)` | DataFrame y lista de columnas | None | Convierte columnas a datetime usando `pd.to_datetime`. |

### `SRC/data_clean.py`

Módulo de limpieza de datos. Contiene todo el pipeline de limpieza del proyecto.

**Constantes:**
- `EMAIL_PATTERN`: Regex para validación de emails
- `ROUNDING_TOLERANCE_USD = 0.011`: Tolerancia para comparaciones de montos

**Funciones de limpieza por tabla:**

| Función | Tabla | Qué hace |
|---------|-------|----------|
| `clean_customers(customers)` | customers | Valida nulos, duplicados, rango de edad (18-100), formato de email |
| `clean_products(products)` | products | Valida nulos, duplicados, precios > 0, costo ≤ precio, margen consistente con tolerancia |
| `clean_orders(orders, customers)` | orders | Valida nulos, duplicados, montos > 0, descuento 0-100%, total consistente, FK a customers |
| `clean_order_items(order_items, products, orders)` | order_items | Elimina duplicados, valida montos (unit_price × quantity = line_total), FK a products y orders |
| `corregir_coherencia_temporal_orders(orders, customers)` | orders | Corrige order_time < signup_date usando signup_date + offset aleatorio (0-365 días, semilla 42) |

**Funciones de limpieza de tablas de eventos:**

| Función | Tabla | Qué hace |
|---------|-------|----------|
| `eliminar_duplicados_events(events)` | events | Verifica duplicados exactos (ninguno encontrado) |
| `eliminar_duplicados_sessions(sessions)` | sessions | Verifica duplicados exactos y session_id duplicados (ninguno encontrado) |
| `eliminar_duplicados_reviews(reviews)` | reviews | Elimina 4 duplicados exactos + resuelve 26 contradictorios (conserva el más reciente) |
| `eliminar_review_text(reviews)` | reviews | Elimina columna review_text (5 frases fijas 1:1 con rating, NLP imposible) |
| `reportar_clientes_sesiones(sessions, customers)` | customers | Reporta 55 clientes sin sesiones (0.27%, sin modificar) |
| `winsorizar_amount_usd(events, percentil)` | events | (Opcional) Recorta amount_usd en el percentil indicado para reducir outliers |

**Funciones de corrección temporal:**

| Función | Tablas afectadas | Qué hace |
|---------|------------------|----------|
| `corregir_coherencia_temporal(events, sessions, reviews, orders, customers)` | events, sessions, reviews | Corrige fechas para respetar: signup ≤ order ≤ review, signup ≤ start_time, signup ≤ timestamp. Solo modifica events/sessions/reviews; orders y customers son referencia. |

**Función de validación:**

| Función | Qué valida |
|---------|------------|
| `validar_integridad(events, sessions, reviews, order_items, orders, customers, products)` | 7 relaciones FK → PK: events→sessions, sessions→customers, reviews→orders, reviews→products, order_items→orders, order_items→products, events→products (solo page_view) |

**Función principal:**

```python
from data_clean import limpiar_tablas
events, sessions, reviews, orders, order_items, customers, products = limpiar_tablas()
```

Ejecuta todo el pipeline en orden: conversión de tipos → customers → products → orders → order_items → duplicados → review_text → clientes sin sesiones → coherencia temporal → integridad referencial.

### `SRC/feature_engineering.py`

Módulo de feature engineering para el sistema de recomendación.

**Funciones de construcción de estructuras:**

| Función | Qué produce |
|---------|-------------|
| `crear_matriz_usuario_item(events, sessions)` | Matriz de interacciones usuario-producto con scores implícitos (page_view=1, add_to_cart=3, purchase=5). Retorna DataFrame con [customer_id, product_id, score] y events con customer_id unido. |
| `crear_features_producto(products, events_con_user, reviews, order_items, orders)` | DataFrame con una fila por producto y 11 features: category, price_usd, cost_usd, margin_usd, n_views, n_cart, n_purchases, popularidad, rating_promedio, n_ratings |
| `crear_features_usuario(customers, sessions, events_con_user, orders, order_items, reviews)` | DataFrame con una fila por usuario y 10 features: age, country, marketing_opt_in, n_sessions, n_purchases, ticket_promedio, n_products_viewed, n_products_carted, rating_promedio_usr |
| `preprocesar_para_modelado(matriz_usuario_item, features_producto, features_usuario)` | Aplica pivot table, LabelEncoder (category, country) y StandardScaler a las variables numéricas. Retorna 4 estructuras listas para modelado. |

**Función principal:**

```python
from feature_engineering import generar_features
interaction_matrix, product_features, user_features, user_item_df, events_con_user = generar_features()
```

Ejecuta: limpieza → matriz usuario-item → features de producto → features de usuario → preprocesamiento.

**Uso directo (sin notebook):**

```python
# Guardar resultados
import os
os.makedirs("Data/Processed", exist_ok=True)

interaction_matrix.to_csv("Data/Processed/interaction_matrix.csv")
product_features.to_csv("Data/Processed/product_features.csv", index=False)
user_features.to_csv("Data/Processed/user_features.csv", index=False)
user_item_df.to_csv("Data/Processed/user_item_df.csv", index=False)
```

---

## Cómo ejecutar

### Requisitos

- Python 3.10+
- ~500 MB de espacio en disco (dataset completo)

### Instalación

```bash
# 1. Clonar el repositorio
git clone <url>
cd ecommerce-clickstream-ml

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Ejecución

**Opción A — Notebooks (recomendado para exploración):**

```bash
# Abrir Jupyter y ejecutar en orden: 01 → 02 → 03 → 04
jupyter notebook Notebooks/
```

**Opción B — Scripts directos (para pipelines automatizados):**

```bash
# Solo limpieza de datos
cd SRC
python data_clean.py

# Limpieza + feature engineering + guardado en Data/Processed/
cd SRC
python feature_engineering.py
```

**Opción C — Desde otro script o notebook:**

```python
import sys
sys.path.append("SRC")

from data_clean import limpiar_tablas
events, sessions, reviews, orders, order_items, customers, products = limpiar_tablas()

from feature_engineering import generar_features
interaction_matrix, product_features, user_features, user_item_df, _ = generar_features()
```

---

## Dependencias (`requirements.txt`)

| Paquete | Versión | Uso |
|---------|---------|-----|
| pandas | 3.0.3 | Manipulación de datos |
| numpy | 2.4.6 | Operaciones numéricas |
| scikit-learn | 1.9.0 | Preprocesamiento (StandardScaler, LabelEncoder) |
| seaborn | — | Visualizaciones |
| matplotlib | — | Visualizaciones (dependencia de seaborn) |
| jupyter / notebook / ipykernel | — | Ejecución de notebooks |
| ipywidgets | — | Widgets en notebooks |
| xgboost | 3.2.0 | Modelos (futuro) |
| fastapi | — | API (futuro) |
| streamlit | — | Dashboard (futuro) |
| uvicorn | — | Servidor ASGI (futuro) |
| joblib | 1.5.3 | Serialización de modelos |
| scipy | — | Cálculos científicos |
| pydantic | — | Validación de datos (futuro) |
| openpyxl | — | Lectura de Excel |
| python-dotenv | — | Variables de entorno |
| db-dtypes | — | Tipos de datos para bases de datos |

---

## Notas y limitaciones

- **Dataset sintético:** Las distribuciones de conversión son uniformes por dispositivo/fuente/país (~28% en todos los segmentos), lo que reduce el poder predictivo de los features demográficos. En un dataset real serían mucho más útiles.
- **Corrección temporal:** Se aplicó corrección automática con offset aleatorio (semilla 42 para reproducibilidad). Las fechas corregidas son aproximadas y no representan el comportamiento real de un usuario.
- **review_text:** Eliminada del dataset. Solo contenía 5 frases fijas repetidas que correspondían 1 a 1 con el rating, por lo que no es utilizable para NLP.
- **Sparsity alta (97.78%):** La matriz usuario-item es extremadamente dispersa. Los modelos de filtrado colaborivo (SVD/ALS) están diseñados para manejar esto.
- **Modelado pendiente:** Falta el entrenamiento y evaluación de los modelos de recomendación.
- **API/Deploy pendiente:** No hay servicio de inferencia ni dashboard implementado aún. Las dependencias (FastAPI, Streamlit, uvicorn) ya están en requirements.txt.
- **55 clientes inactivos:** Registrados pero sin sesiones. Requieren cold start con features demográficos.

---

## Próximos pasos

1. **Entrenamiento de modelos:** SVD, ALS, content-based con similitud coseno
2. **Evaluación:** Precision@K, Recall@K, comparación de modelos
3. **Enfoque híbrido:** Combinar filtrado colaborivo + content-based para cubrir cold start
4. **API de inferencia:** FastAPI con modelo serializado (joblib)
5. **Dashboard:** Streamlit para visualización de recomendaciones
