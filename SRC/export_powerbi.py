"""
Exporta las 7 tablas limpias a Data/PowerBI/ para consumo de Power BI.

Reutiliza limpiar_tablas() de data_clean.py para no duplicar lógica.
Genera CSVs sin escalado ni encoding, tal como necesita Power BI para visualización.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_clean import limpiar_tablas


def exportar_para_powerbi(ruta_salida="Data/PowerBI"):
    os.makedirs(ruta_salida, exist_ok=True)

    print("\nEjecutando pipeline de limpieza...")
    events, sessions, reviews, orders, order_items, customers, products = limpiar_tablas()

    tablas = {
        "customers_clean.csv": customers,
        "products_clean.csv": products,
        "orders_clean.csv": orders,
        "order_items_clean.csv": order_items,
        "sessions_clean.csv": sessions,
        "events_clean.csv": events,
        "reviews_clean.csv": reviews,
    }

    print(f"\nExportando a {ruta_salida}/")
    for nombre, df in tablas.items():
        ruta = os.path.join(ruta_salida, nombre)
        df.to_csv(ruta, index=False)
        print(f"  OK {nombre}: {len(df):,} filas -> {ruta}")

    print(f"\nExportación de tablas completada. {len(tablas)} archivos en {ruta_salida}/")

    # Exportar metricas de modelos si existen los .joblib
    from export_model_metrics import exportar_metricas
    exportar_metricas(os.path.join(ruta_salida, "Models"))


if __name__ == "__main__":
    exportar_para_powerbi()
