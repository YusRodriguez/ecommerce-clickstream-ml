"""
Exporta métricas de modelos y feature importance a CSV para Power BI.

Lee los artefactos entrenados de Models/ (generados por pipeline_modelos.py)
y genera:
  - Data/PowerBI/model_results.csv
  - Data/PowerBI/feature_importance.csv
"""
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
MODELS_DIR = PROJECT_DIR / "Models"
RAW_DIR = PROJECT_DIR / "Data" / "Raw"


def exportar_metricas(ruta_salida="Data/PowerBI/Models"):
    os.makedirs(ruta_salida, exist_ok=True)

    # --- 1. Leer modelos ---
    warm_path = MODELS_DIR / "warm_start_lightgbm.joblib"
    cold_path = MODELS_DIR / "cold_start_content_based.joblib"

    if not warm_path.exists() or not cold_path.exists():
        print("ERROR: Faltan archivos .joblib en Models/. Ejecutá primero:")
        print("  python SRC/pipeline_modelos.py")
        return

    warm = joblib.load(warm_path)
    cold = joblib.load(cold_path)

    # --- 2. Construir tabla de métricas ---
    rows = []
    for nombre, modelo, metrics in [
        ("LightGBM", "Warm-start", warm["metrics"]),
        ("Content-Based", "Cold-start", cold["metrics"]),
    ]:
        rows.append({
            "Modelo": nombre,
            "Tipo": modelo,
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1": round(metrics["f1"], 4),
            "MAP_10": round(metrics["map@k"], 4),
            "NDCG_10": round(metrics["ndcg@k"], 4),
            "Usuarios_Eval": metrics["n_usuarios_evaluados"],
        })

    df_results = pd.DataFrame(rows)
    results_path = os.path.join(ruta_salida, "model_results.csv")
    df_results.to_csv(results_path, index=False)
    print(f"  OK model_results.csv: {len(df_results)} modelos -> {results_path}")

    # --- 3. Feature importance (solo LightGBM) ---
    modelo_lgb = warm["modelo"]
    feature_cols = warm["feature_cols"]
    importancias = modelo_lgb.feature_importances_

    df_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importancia": importancias,
    }).sort_values("Importancia", ascending=False).reset_index(drop=True)

    importance_path = os.path.join(ruta_salida, "feature_importance.csv")
    df_importance.to_csv(importance_path, index=False)
    print(f"  OK feature_importance.csv: {len(df_importance)} features -> {importance_path}")

    print(f"\nExportación de métricas completada.")


if __name__ == "__main__":
    exportar_metricas()
