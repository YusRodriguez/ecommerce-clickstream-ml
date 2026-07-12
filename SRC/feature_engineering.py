"""
Módulo de feature engineering — popularidad y margen de producto (Elías).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

TOP_SELLER_PERCENTILE = 0.9
DATA_DIR = Path(__file__).resolve().parent.parent / "Data" / "Raw"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"

COLOR_BLUE = "#2a78d6"
COLOR_AQUA = "#1baf7a"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"


# =============================================================================
# 1. POPULARIDAD DEL PRODUCTO
# =============================================================================

def calcular_popularidad_producto(products: pd.DataFrame, order_items: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega a `products` las variables de popularidad derivadas de `order_items`:
    unidades vendidas, ingreso generado, cantidad de órdenes distintas y el
    percentil de ventas dentro de la propia categoría.
    """
    print("\n--- Calculando popularidad de producto ---")
    df = products.copy()

    ventas = order_items.groupby("product_id").agg(
        qty_sold=("quantity", "sum"),
        revenue_total=("line_total_usd", "sum"),
        n_orders=("order_id", "nunique"),
    )

    df = df.merge(ventas, on="product_id", how="left")
    df[["qty_sold", "revenue_total", "n_orders"]] = df[["qty_sold", "revenue_total", "n_orders"]].fillna(0)

    df["popularity_pct"] = df.groupby("category")["qty_sold"].rank(pct=True)
    df["is_top_seller"] = df["popularity_pct"] >= TOP_SELLER_PERCENTILE

    print(f"  ✓ qty_sold, revenue_total, n_orders calculados ({len(df):,} productos)")
    print(f"  ✓ popularity_pct (percentil dentro de categoría) y is_top_seller (top {round((1 - TOP_SELLER_PERCENTILE) * 100)}%)")
    print(f"  Productos sin ventas registradas: {(df['qty_sold'] == 0).sum():,}")

    return df


# =============================================================================
# 2. MARGEN PROMEDIO
# =============================================================================

def calcular_margen_producto(products: pd.DataFrame, order_items: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega a `products` el margen porcentual por producto, el margen en USD
    realmente generado según las unidades vendidas, y el margen promedio de
    la categoría a la que pertenece cada producto.
    """
    print("\n--- Calculando margen de producto ---")
    df = products.copy()

    df["margin_pct"] = (df["margin_usd"] / df["price_usd"] * 100).round(2)

    ventas_margen = order_items[["product_id", "quantity"]].merge(
        products[["product_id", "margin_usd"]], on="product_id", how="left"
    )
    ventas_margen["margin_total_line"] = ventas_margen["margin_usd"] * ventas_margen["quantity"]

    margen_realizado = ventas_margen.groupby("product_id").agg(margin_usd_total=("margin_total_line", "sum"))
    df = df.merge(margen_realizado, on="product_id", how="left")
    df["margin_usd_total"] = df["margin_usd_total"].fillna(0)

    resumen_categoria = df.groupby("category").agg(
        category_avg_margin_pct=("margin_pct", "mean"),
        category_avg_margin_usd=("margin_usd", "mean"),
    ).round(2)
    df = df.merge(resumen_categoria, on="category", how="left")

    print(f"  ✓ margin_pct, margin_usd_total calculados ({len(df):,} productos)")
    print(f"  ✓ category_avg_margin_pct, category_avg_margin_usd calculados ({df['category'].nunique()} categorías)")

    return df


# =============================================================================
# 3. GRÁFICOS
# =============================================================================

def _limpiar_ejes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", colors=COLOR_MUTED, labelsize=9)
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def graficar_top_productos(df: pd.DataFrame, top_n: int = 10, output_dir: Path = FIGURES_DIR) -> Path:
    """Guarda un bar chart horizontal con los `top_n` productos más vendidos (qty_sold)."""
    top = df.sort_values("qty_sold", ascending=False).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.barh(top["name"], top["qty_sold"], color=COLOR_BLUE, height=0.6)
    _limpiar_ejes(ax)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("Unidades vendidas", color=COLOR_MUTED, fontsize=9)
    ax.set_title(f"Top {top_n} productos más vendidos", color=COLOR_INK, fontsize=12, loc="left", pad=12)

    for barra, valor in zip(barras, top["qty_sold"]):
        ax.text(
            barra.get_width() + top["qty_sold"].max() * 0.01,
            barra.get_y() + barra.get_height() / 2,
            f"{valor:,.0f}",
            va="center", ha="left", color=COLOR_INK, fontsize=9,
        )

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    ruta = output_dir / "top_productos_populares.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def graficar_margen_categoria(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Guarda dos bar charts lado a lado: margen promedio en USD y en % por categoría."""
    resumen = (
        df.drop_duplicates("category")
        .sort_values("category_avg_margin_usd", ascending=False)
        [["category", "category_avg_margin_usd", "category_avg_margin_pct"]]
    )

    fig, (ax_usd, ax_pct) = plt.subplots(1, 2, figsize=(11, 5))

    ax_usd.bar(resumen["category"], resumen["category_avg_margin_usd"], color=COLOR_BLUE, width=0.6)
    _limpiar_ejes(ax_usd)
    ax_usd.set_title("Margen promedio por categoría (USD)", color=COLOR_INK, fontsize=11, loc="left", pad=10)
    ax_usd.tick_params(axis="x", rotation=30)
    for x, valor in enumerate(resumen["category_avg_margin_usd"]):
        ax_usd.text(x, valor + resumen["category_avg_margin_usd"].max() * 0.01, f"${valor:,.0f}",
                    ha="center", va="bottom", color=COLOR_INK, fontsize=8)

    ax_pct.bar(resumen["category"], resumen["category_avg_margin_pct"], color=COLOR_AQUA, width=0.6)
    _limpiar_ejes(ax_pct)
    ax_pct.set_title("Margen promedio por categoría (%)", color=COLOR_INK, fontsize=11, loc="left", pad=10)
    ax_pct.tick_params(axis="x", rotation=30)
    for x, valor in enumerate(resumen["category_avg_margin_pct"]):
        ax_pct.text(x, valor + resumen["category_avg_margin_pct"].max() * 0.01, f"{valor:.1f}%",
                    ha="center", va="bottom", color=COLOR_INK, fontsize=8)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    ruta = output_dir / "margen_por_categoria.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


# =============================================================================
# 4. PIPELINE
# =============================================================================

def generar_features_producto(products: pd.DataFrame, order_items: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta el pipeline de features de producto (popularidad + margen) y devuelve products enriquecido."""
    df = calcular_popularidad_producto(products, order_items)
    df = calcular_margen_producto(df, order_items)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING DE PRODUCTO FINALIZADO")
    print("=" * 60)

    return df


# =============================================================================
# 5. EJECUCIÓN DIRECTA (smoke test desde terminal)
# =============================================================================

if __name__ == "__main__":
    products = pd.read_csv(DATA_DIR / "products.csv")
    order_items = pd.read_csv(DATA_DIR / "order_items.csv")

    resultado = generar_features_producto(products, order_items)

    print("\nColumnas nuevas:", [c for c in resultado.columns if c not in products.columns])

    print("\nTop 5 productos más populares (qty_sold):")
    print(
        resultado.sort_values("qty_sold", ascending=False)
        .head(5)[["product_id", "name", "category", "qty_sold", "is_top_seller"]]
        .to_string(index=False)
    )

    print("\nMargen promedio por categoría (USD y %):")
    print(
        resultado.drop_duplicates("category")
        .sort_values("category_avg_margin_usd", ascending=False)[
            ["category", "category_avg_margin_usd", "category_avg_margin_pct"]
        ]
        .to_string(index=False)
    )

    ruta_top_productos = graficar_top_productos(resultado)
    ruta_margen = graficar_margen_categoria(resultado)
    print(f"\nGráficos guardados en:\n  {ruta_top_productos}\n  {ruta_margen}")