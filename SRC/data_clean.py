"""
Módulo de limpieza de datos — order_items, orders, products, customers.
"""

import numpy as np
import pandas as pd

EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
ROUNDING_TOLERANCE_USD = 0.011


def _reportar(descripcion, n_fallos, total):
    """Imprime el resultado de una validación sin interrumpir la ejecución."""
    if n_fallos > 0:
        print(f"  ✗ {descripcion}: {n_fallos:,}/{total:,} registros no cumplen")
    else:
        print(f"  ✓ {descripcion}: OK ({total:,} registros)")


# =============================================================================
# 1. CONVERSIÓN DE FECHAS
# =============================================================================

def convertir_fechas(orders, customers):
    print("=" * 60)
    print("PASO 1: Conversión de columnas de fechas a datetime")
    print("=" * 60)

    orders["order_time"] = pd.to_datetime(orders["order_time"])
    print(f"  ✓ orders['order_time'] → {orders['order_time'].dtype}")

    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    print(f"  ✓ customers['signup_date'] → {customers['signup_date'].dtype}")

    return orders, customers


# =============================================================================
# 2. LIMPIEZA POR TABLA
# =============================================================================

def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Limpiando customers ---")
    df = customers.copy()
    total = len(df)

    _reportar("Sin valores nulos", df.isnull().any(axis=1).sum(), total)
    _reportar("Sin filas duplicadas", df.duplicated().sum(), total)
    _reportar("age entre 18 y 100", (~df["age"].between(18, 100)).sum(), total)
    _reportar("email con formato válido", (~df["email"].str.match(EMAIL_PATTERN)).sum(), total)

    return df


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Limpiando products ---")
    df = products.copy()
    total = len(df)

    _reportar("Sin valores nulos", df.isnull().any(axis=1).sum(), total)
    _reportar("Sin filas duplicadas", df.duplicated().sum(), total)
    _reportar("price_usd > 0", (~(df["price_usd"] > 0)).sum(), total)
    _reportar("cost_usd > 0", (~(df["cost_usd"] > 0)).sum(), total)
    _reportar("cost_usd <= price_usd", (~(df["cost_usd"] <= df["price_usd"])).sum(), total)

    margen_calculado = (df["price_usd"] - df["cost_usd"]).round(2)
    diferencia = (margen_calculado - df["margin_usd"]).abs()
    _reportar(
        f"margin_usd coincide con price_usd - cost_usd (tolerancia {ROUNDING_TOLERANCE_USD})",
        (diferencia > ROUNDING_TOLERANCE_USD).sum(),
        total,
    )

    return df


def corregir_coherencia_temporal(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige order_time para respetar signup_date <= order_time.

    Mismo criterio y método que el módulo de events/sessions/reviews (Rocío,
    SRC/data_clean.py en feature/rocio-data-cleaning): las filas violatorias
    se reasignan a signup_date + offset aleatorio de 0 a 365 días
    (semilla fija 42, para reproducibilidad).
    """
    print("\n--- Corrigiendo coherencia temporal (orders vs customers) ---")
    df = orders.copy()
    total = len(df)

    signup_map = pd.Series(customers["signup_date"].values, index=customers["customer_id"])
    df["signup_date_ref"] = df["customer_id"].map(signup_map)

    mascara = df["order_time"] < df["signup_date_ref"]
    n_viol = mascara.sum()
    print(f"  order_time < signup_date: {n_viol:,}/{total:,} registros")

    if n_viol > 0:
        rng = np.random.default_rng(42)
        delta_dias = rng.integers(0, 366, size=n_viol)
        df.loc[mascara, "order_time"] = (
            df.loc[mascara, "signup_date_ref"] + pd.to_timedelta(delta_dias, unit="D")
        )
        print(f"  → Corregidas {n_viol:,} fechas de order_time (signup_date + offset aleatorio 0-365 días)")
    else:
        print("  ✓ No hay violaciones. No se requiere corrección.")

    return df.drop(columns=["signup_date_ref"])


def clean_orders(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Limpiando orders ---")
    df = orders.copy()
    total = len(df)

    _reportar("Sin valores nulos", df.isnull().any(axis=1).sum(), total)
    _reportar("Sin filas duplicadas", df.duplicated().sum(), total)
    _reportar("subtotal_usd > 0", (~(df["subtotal_usd"] > 0)).sum(), total)
    _reportar("total_usd > 0", (~(df["total_usd"] > 0)).sum(), total)
    _reportar("discount_pct entre 0 y 100", (~df["discount_pct"].between(0, 100)).sum(), total)

    total_calculado = (df["subtotal_usd"] * (1 - df["discount_pct"] / 100)).round(2)
    diferencia = (total_calculado - df["total_usd"]).abs()
    _reportar(
        f"total_usd coincide con subtotal_usd tras descuento (tolerancia {ROUNDING_TOLERANCE_USD})",
        (diferencia > ROUNDING_TOLERANCE_USD).sum(),
        total,
    )

    _reportar("customer_id existe en customers", (~df["customer_id"].isin(customers["customer_id"])).sum(), total)

    print("  Verificando coherencia temporal (order_time vs signup_date)...")
    con_signup = df.merge(customers[["customer_id", "signup_date"]], on="customer_id", how="left")
    anteriores_a_signup = (con_signup["order_time"] < con_signup["signup_date"]).sum()
    _reportar("order_time es posterior o igual a signup_date del cliente", anteriores_a_signup, total)

    return df


def clean_order_items(order_items: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Limpiando order_items ---")
    n_antes = len(order_items)
    df = order_items.drop_duplicates().reset_index(drop=True)
    print(f"  Duplicados eliminados: {n_antes - len(df):,}")
    total = len(df)

    _reportar("Sin valores nulos", df.isnull().any(axis=1).sum(), total)
    _reportar("unit_price_usd > 0", (~(df["unit_price_usd"] > 0)).sum(), total)
    _reportar("quantity > 0", (~(df["quantity"] > 0)).sum(), total)
    _reportar("line_total_usd > 0", (~(df["line_total_usd"] > 0)).sum(), total)

    total_calculado = (df["unit_price_usd"] * df["quantity"]).round(2)
    diferencia = (total_calculado - df["line_total_usd"]).abs()
    _reportar(
        f"line_total_usd coincide con unit_price_usd * quantity (tolerancia {ROUNDING_TOLERANCE_USD})",
        (diferencia > ROUNDING_TOLERANCE_USD).sum(),
        total,
    )

    _reportar("product_id existe en products", (~df["product_id"].isin(products["product_id"])).sum(), total)
    _reportar("order_id existe en orders", (~df["order_id"].isin(orders["order_id"])).sum(), total)

    return df


# =============================================================================
# 3. PIPELINE
# =============================================================================

def limpiar_tablas(order_items: pd.DataFrame, orders: pd.DataFrame, products: pd.DataFrame, customers: pd.DataFrame):
    """Ejecuta el pipeline completo de limpieza y devuelve los DataFrames limpios."""
    orders, customers = convertir_fechas(orders, customers)

    customers_limpio = clean_customers(customers)
    products_limpio = clean_products(products)
    orders_corregido = corregir_coherencia_temporal(orders, customers_limpio)
    orders_limpio = clean_orders(orders_corregido, customers_limpio)
    order_items_limpio = clean_order_items(order_items, products_limpio, orders_limpio)

    print("\n" + "=" * 60)
    print("PIPELINE DE LIMPIEZA FINALIZADO")
    print("=" * 60)

    return {
        "order_items": order_items_limpio,
        "orders": orders_limpio,
        "products": products_limpio,
        "customers": customers_limpio,
    }
