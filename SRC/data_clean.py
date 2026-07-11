"""
Módulo de limpieza de datos — events, sessions, reviews.

Dataset sintético con fechas aleatorias y ~50% de violaciones de
coherencia temporal (review antes de order, order antes de signup).

Este módulo:
  1. Convierte columnas de fechas a datetime
  2. Elimina duplicados (4 reviews exactas, 26 contradictorias)
  3. Corrige coherencia temporal: signup ≤ order ≤ review
  4. Elimina review_text (5 frases fijas, NLP no utilizable)
  5. Marca clientes sin sesiones (55 clientes inactivos)
  6. Valida integridad referencial entre tablas

Función principal: limpiar_tablas() ejecuta todo el pipeline.
"""

import pandas as pd
import numpy as np
from utils import cargarDatos, verificar_integridad_referencial


# =============================================================================
# 1. CONVERSIÓN DE TIPOS
# =============================================================================

def convertir_fechas(events, sessions, reviews):
    """
    Convierte las columnas de fechas a datetime64 en events, sessions y reviews.
    """
    print("=" * 60)
    print("PASO 1: Conversión de columnas de fechas a datetime")
    print("=" * 60)
    
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    print(f"  ✓ events['timestamp'] → {events['timestamp'].dtype}")
    
    sessions["start_time"] = pd.to_datetime(sessions["start_time"])
    print(f"  ✓ sessions['start_time'] → {sessions['start_time'].dtype}")
    
    reviews["review_time"] = pd.to_datetime(reviews["review_time"])
    print(f"  ✓ reviews['review_time'] → {reviews['review_time'].dtype}")
    
    print(f"\n  Rango de fechas:")
    print(f"    events   : {events['timestamp'].min().date()} → {events['timestamp'].max().date()}")
    print(f"    sessions : {sessions['start_time'].min().date()} → {sessions['start_time'].max().date()}")
    print(f"    reviews  : {reviews['review_time'].min().date()} → {reviews['review_time'].max().date()}")
    
    return events, sessions, reviews


# =============================================================================
# 2. DUPLICADOS
# =============================================================================

def eliminar_duplicados_events(events):
    """
    Events: no tiene duplicados. Función de validación.
    """
    print("\n" + "=" * 60)
    print("PASO 2a: Verificación de duplicados en events")
    print("=" * 60)
    
    n_antes = len(events)
    duplicados = events.duplicated().sum()
    
    print(f"  Filas totales: {n_antes:,}")
    print(f"  Filas duplicadas exactas: {duplicados:,}")
    
    if duplicados > 0:
        events = events.drop_duplicates()
        print(f"  → Se eliminaron {duplicados:,} filas duplicadas")
        print(f"  Filas restantes: {len(events):,}")
    else:
        print("  ✓ No hay duplicados. No se requiere acción.")
    
    return events


def eliminar_duplicados_sessions(sessions):
    """
    Sessions: no tiene duplicados. Función de validación.
    """
    print("\n" + "=" * 60)
    print("PASO 2b: Verificación de duplicados en sessions")
    print("=" * 60)
    
    n_antes = len(sessions)
    duplicados = sessions.duplicated().sum()
    duplicados_id = sessions["session_id"].duplicated().sum()
    
    print(f"  Filas totales: {n_antes:,}")
    print(f"  Filas duplicadas exactas: {duplicados:,}")
    print(f"  session_id duplicados: {duplicados_id:,}")
    
    if duplicados > 0:
        sessions = sessions.drop_duplicates()
        print(f"  → Se eliminaron {duplicados:,} filas duplicadas")
        print(f"  Filas restantes: {len(sessions):,}")
    else:
        print("  ✓ No hay duplicados. No se requiere acción.")
    
    return sessions


def eliminar_duplicados_reviews(reviews):
    """
    Reviews: eliminar duplicados exactos y resolver contradictorios.
    
    Duplicados exactos: misma order_id + product_id + rating → eliminar.
    Contradictorios: misma order_id + product_id, distinto rating → conservar el más reciente.
    """
    print("\n" + "=" * 60)
    print("PASO 2c: Limpieza de duplicados en reviews")
    print("=" * 60)
    
    n_antes = len(reviews)
    
    # 2c-i: Duplicados exactos (mismo order_id + product_id + rating)
    duplicados_exactos = reviews.duplicated(
        subset=["order_id", "product_id", "rating"], keep="first"
    ).sum()
    print(f"  Filas totales: {n_antes:,}")
    print(f"  Duplicados exactos (order_id + product_id + rating): {duplicados_exactos:,}")
    
    reviews = reviews.drop_duplicates(
        subset=["order_id", "product_id", "rating"], keep="first"
    )
    
    # 2c-ii: Contradictorios (mismo order_id + product_id, distinto rating)
    contradicciones = reviews.duplicated(
        subset=["order_id", "product_id"], keep=False
    ).sum()
    print(f"  Reviews contradictorios (mismo order+product, distinto rating): {contradicciones:,}")
    
    if contradicciones > 0:
        # Conservar solo el más reciente por order_id + product_id
        reviews = reviews.drop_duplicates(
            subset=["order_id", "product_id"], keep="last"
        )
        print(f"  → Se conservó el review más reciente por order_id + product_id")
    
    n_despues = len(reviews)
    print(f"  Filas eliminadas totales: {n_antes - n_despues:,}")
    print(f"  Filas restantes: {n_despues:,}")
    
    return reviews


def eliminar_review_text(reviews):
    """
    Elimina la columna review_text.
    Solo tiene 5 frases fijas 1:1 con el rating. No es posible aplicar NLP.
    """
    print("\n" + "=" * 60)
    print("PASO 2d: Eliminación de review_text")
    print("=" * 60)
    
    if "review_text" in reviews.columns:
        reviews = reviews.drop(columns=["review_text"])
        print("  ✓ Columna 'review_text' eliminada (5 frases fijas, 1:1 con rating)")
        print("  → No es posible aplicar NLP. Se usa solo el rating numérico.")
    else:
        print("  ✓ Columna 'review_text' no existe. No se requiere acción.")
    
    return reviews


def reportar_clientes_sesiones(sessions, customers):
    """
    Reporta cuántos clientes no tienen sesiones (sin modificar customers).
    55 clientes (0.28%) están registrados pero nunca generaron sesión.
    El flag 'tiene_sesion' lo agrega Elías en su módulo.
    """
    print("\n" + "=" * 60)
    print("PASO 2e: Reporte de clientes sin sesiones")
    print("=" * 60)
    
    clientes_con_sesion = sessions["customer_id"].unique()
    total_clientes = customers["customer_id"].nunique()
    sin_sesion = total_clientes - len(clientes_con_sesion)
    
    print(f"  Clientes con al menos 1 sesión: {len(clientes_con_sesion):,}")
    print(f"  Clientes SIN ninguna sesión: {sin_sesion:,} ({sin_sesion/total_clientes*100:.2f}%)")
    print("  → No se eliminan ni modifican. El flag 'tiene_sesion' lo agrega Elías.")


def winsorizar_amount_usd(events, percentil=99):
    """
    Opcional: recorta amount_usd en el percentil indicado.
    Máximo actual: $2,984 vs mediana $86.
    
    Solo aplica a eventos de tipo 'purchase'.
    """
    print("\n" + "=" * 60)
    print(f"PASO 2f: Winsorización de amount_usd (percentil {percentil})")
    print("=" * 60)
    
    if "amount_usd" not in events.columns:
        print("  ✓ Columna 'amount_usd' no existe. No se requiere acción.")
        return events
    
    # Solo purchase tiene amount_usd
    mascara = events["event_type"] == "purchase"
    n_antes = events.loc[mascara, "amount_usd"].notna().sum()
    max_antes = events.loc[mascara, "amount_usd"].max()
    
    umbral = events.loc[mascara, "amount_usd"].quantile(percentil / 100)
    n_recortados = (events.loc[mascara, "amount_usd"] > umbral).sum()
    
    events.loc[mascara, "amount_usd"] = events.loc[mascara, "amount_usd"].clip(upper=umbral)
    max_despues = events.loc[mascara, "amount_usd"].max()
    
    print(f"  Compras con amount_usd: {n_antes:,}")
    print(f"  Max antes: ${max_antes:,.2f}")
    print(f"  Umbral P{percentil}: ${umbral:,.2f}")
    print(f"  Filas recortadas: {n_recortados:,}")
    print(f"  Max después: ${max_despues:,.2f}")
    print(f"  → Probar si mejora el modelo. Si no, revertir.")
    
    return events


# =============================================================================
# 3. COHERENCIA TEMPORAL
# =============================================================================

def corregir_coherencia_temporal(events, sessions, reviews, orders, customers):
    """
    Corrige las fechas para respetar la cadena temporal:
      signup_date ≤ order_time ≤ review_time
      signup_date ≤ start_time (sessions)
      signup_date ≤ timestamp (events)
    
    Solo limpia events, sessions y reviews.
    orders y customers se usan como referencia (no se modifican).
    """
    print("\n" + "=" * 60)
    print("PASO 3: Corrección de coherencia temporal")
    print("=" * 60)
    
    rng = np.random.default_rng(42)
    
    # Convertir fechas de referencia inline (sin modificar los DataFrames originales)
    signup_map = pd.Series(
        pd.to_datetime(customers["signup_date"]).values,
        index=customers["customer_id"]
    )
    
    order_time_map = pd.Series(
        pd.to_datetime(orders["order_time"]).values,
        index=orders["order_id"]
    )
    
    # --- reviews: review_time ≥ order_time ---
    reviews["order_time_ref"] = reviews["order_id"].map(order_time_map)
    violaciones_review = (reviews["review_time"] < reviews["order_time_ref"]).sum()
    print(f"\n  Reviews: {violaciones_review:,} de {len(reviews):,} con review_time < order_time")
    
    if violaciones_review > 0:
        mascara = reviews["review_time"] < reviews["order_time_ref"]
        delta_dias = rng.integers(1, 31, size=mascara.sum())
        reviews.loc[mascara, "review_time"] = (
            reviews.loc[mascara, "order_time_ref"] + pd.to_timedelta(delta_dias, unit="D")
        )
        print(f"  → Corregidas {mascara.sum():,} fechas de review")
    
    reviews = reviews.drop(columns=["order_time_ref"])
    
    # --- sessions: start_time ≥ signup_date ---
    # Mapa session_id → signup_date (para sessions)
    session_signup_map = sessions.set_index("session_id")["customer_id"].map(signup_map)
    
    sessions["signup_date"] = sessions["customer_id"].map(signup_map)
    violaciones_session = (sessions["start_time"] < sessions["signup_date"]).sum()
    print(f"\n  Sessions: {violaciones_session:,} de {len(sessions):,} con start_time < signup_date")
    
    if violaciones_session > 0:
        mascara = sessions["start_time"] < sessions["signup_date"]
        delta_dias = rng.integers(0, 366, size=mascara.sum())
        sessions.loc[mascara, "start_time"] = (
            sessions.loc[mascara, "signup_date"] + pd.to_timedelta(delta_dias, unit="D")
        )
        print(f"  → Corregidas {mascara.sum():,} fechas de session")
    
    sessions = sessions.drop(columns=["signup_date"])
    
    # --- events: timestamp ≥ signup_date ---
    # events tiene session_id (no customer_id), uso session_signup_map
    events["signup_date"] = events["session_id"].map(session_signup_map)
    violaciones_event = (events["timestamp"] < events["signup_date"]).sum()
    print(f"\n  Events: {violaciones_event:,} de {len(events):,} con timestamp < signup_date")
    
    if violaciones_event > 0:
        mascara = events["timestamp"] < events["signup_date"]
        delta_dias = rng.integers(0, 366, size=mascara.sum())
        events.loc[mascara, "timestamp"] = (
            events.loc[mascara, "signup_date"] + pd.to_timedelta(delta_dias, unit="D")
        )
        print(f"  → Corregidas {mascara.sum():,} fechas de event")
    
    events = events.drop(columns=["signup_date"])
    
    print("\n  ✓ Coherencia temporal corregida en events, sessions y reviews.")
    print("  → orders y customers se limpian en el módulo de Elías.")
    
    return events, sessions, reviews


# =============================================================================
# 4. VALIDACIONES DE INTEGRIDAD
# =============================================================================

def validar_integridad(events, sessions, reviews, orders, customers, products):
    """
    Valida integridad referencial de events, sessions y reviews.
    orders, customers y products se usan como referencia (no se modifican acá).
    """
    print("\n" + "=" * 60)
    print("PASO 4: Validación de integridad referencial")
    print("=" * 60)
    
    checks = [
        ("events.session_id → sessions", events, "session_id", sessions, "session_id"),
        ("sessions.customer_id → customers", sessions, "customer_id", customers, "customer_id"),
        ("reviews.order_id → orders", reviews, "order_id", orders, "order_id"),
        ("reviews.product_id → products", reviews, "product_id", products, "product_id"),
    ]
    
    todo_ok = True
    for nombre, child_df, child_col, parent_df, parent_col in checks:
        n_huerfanos = verificar_integridad_referencial(child_df, child_col, parent_df, parent_col)
        estado = "✓" if n_huerfanos == 0 else "✗"
        print(f"  {estado} {nombre}: {n_huerfanos:,} huérfanos")
        if n_huerfanos > 0:
            todo_ok = False
    
    # Validación adicional: events.product_id solo para page_view (los demás tienen NaN estructural)
    events_con_producto = events[events["product_id"].notna()]
    huerfanos_prod = verificar_integridad_referencial(events_con_producto, "product_id", products, "product_id")
    estado = "✓" if huerfanos_prod == 0 else "✗"
    print(f"  {estado} events.product_id → products (solo page_view, {len(events_con_producto):,} filas): {huerfanos_prod:,} huérfanos")
    if huerfanos_prod > 0:
        todo_ok = False
    
    if todo_ok:
        print("\n  ✓ TODAS las integridades referenciales están OK.")
    else:
        print("\n  ⚠ Hay huérfanos detectados. Revisar.")
    
    # Resumen de tamaños (solo mis tablas)
    print("\n  Resumen de tamaños (tablas limpiadas):")
    print(f"    events     : {len(events):>10,} filas")
    print(f"    sessions   : {len(sessions):>10,} filas")
    print(f"    reviews    : {len(reviews):>10,} filas")


# =============================================================================
# 5. PIPELINE PRINCIPAL
# =============================================================================

def limpiar_tablas(ruta_data="Data/Raw"):
    """
    Pipeline completo de limpieza para events, sessions y reviews.
    
    orders, customers y products se cargan como referencia para
    integridad referencial y coherencia temporal.
    
    Retorna: events, sessions, reviews (limpios)
             orders, customers, products (sin modificar, como referencia)
    """
    print("\n" + "#" * 60)
    print("# PIPELINE DE LIMPIEZA DE DATOS")
    print("#" * 60)
    
    # Carga
    print("\nCargando datos...")
    events = cargarDatos("events")
    sessions = cargarDatos("sessions")
    reviews = cargarDatos("reviews")
    orders = cargarDatos("orders")       # referencia
    customers = cargarDatos("customers") # referencia
    products = cargarDatos("products")   # referencia
    
    # Paso 1: Conversión de fechas (solo mis tablas)
    events, sessions, reviews = convertir_fechas(events, sessions, reviews)
    
    # Paso 2: Duplicados
    events = eliminar_duplicados_events(events)
    sessions = eliminar_duplicados_sessions(sessions)
    reviews = eliminar_duplicados_reviews(reviews)
    
    # Paso 2 extra: Limpieza adicional de reviews
    reviews = eliminar_review_text(reviews)
    
    # Paso 2 extra: Reportar clientes sin sesiones (sin modificar customers)
    reportar_clientes_sesiones(sessions, customers)
    
    # Paso 2 extra: Winsorizar amount_usd (opcional, comentar si no se desea)
    # events = winsorizar_amount_usd(events, percentil=99)
    
    # Paso 3: Coherencia temporal (solo mis tablas, usa orders/customers como referencia)
    events, sessions, reviews = corregir_coherencia_temporal(
        events, sessions, reviews, orders, customers
    )
    
    # Paso 4: Validación de integridad (solo mis tablas)
    validar_integridad(events, sessions, reviews, orders, customers, products)
    
    print("\n" + "#" * 60)
    print("# LIMPIEZA COMPLETADA")
    print("#" * 60)
    return events, sessions, reviews, orders, customers, products


if __name__ == "__main__":
    events, sessions, reviews, orders, customers, products = limpiar_tablas()
