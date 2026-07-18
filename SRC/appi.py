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

# 1. Cargar los modelos y artefactos- variables
try:
    warm_start_dict = joblib.load("Models/warm_start_lightgbm.joblib")
    cold_start_dict = joblib.load("Models/cold_start_content_based.joblib")
    
    # Extraer componentes (Ahora estas variables son globales para el archivo)
    model = warm_start_dict["modelo"]
    feature_cols = warm_start_dict["feature_cols"]
    imputer = warm_start_dict["imputer"]
    cat_mappings = warm_start_dict["category_mappings"]
    
    scaler = cold_start_dict["scaler"]
    product_vectors = cold_start_dict["product_vectors"]
    product_ids = cold_start_dict["product_ids"]
    numeric_cols_content = cold_start_dict["numeric_cols"] # <-- EXTRAER ESTO
    
except Exception as e:
    print(f"Error cargando modelos: {e}")
    raise RuntimeError(f"No se pudieron cargar los archivos: {e}")

class RecommendationRequest(BaseModel):
    user_id: str
    is_new_user: bool
    context: dict # Aquí se reciben datos como 'categoria', 'dispositivo', etc.

@app.post("/recommend")
def get_recommendation(request: RecommendationRequest):
    try:
        # --- ENFOQUE 1: Cold Start ---
        if request.is_new_user:
            # 1. Recuperar info del entrenamiento
            # Necesitamos el scaler y los nombres de columnas categóricas (o el número total de dimensiones)
            n_numeric = len(numeric_cols_content)
            
            # 2. Construir vector base (numéricos)
            numeric_vals = [request.context.get(col, 0) for col in numeric_cols_content]
            
            # 3. Construir vector de categorías (One-Hot)
            # Debemos saber qué categoría envió el usuario y marcarla en el vector
            cat_val = request.context.get("category", "missing")
            cat_vector = [1 if f"cat_{cat_val}" == c else 0 for c in cold_start_dict["category_columns"]]
            
            # 4. Unir todo
            full_vector = np.array(numeric_vals + cat_vector).reshape(1, -1)
            input_array = np.array(numeric_vals + cat_vector, dtype=float).reshape(1, -1)
            
            # 5. Escalar (El scaler solo debe ver los numéricos según tu pipeline)
            # Tu pipeline entrena el scaler solo con NUMERIC_COLS_CONTENT
            scaled_num = scaler.transform(full_vector[:, :n_numeric])
            
            # Unir escalados con las categorías (que no se escalan)
            final_input = np.hstack([scaled_num.astype(float), input_array[:, n_numeric:].astype(float)])
            
            # 6. Calcular similitud
            similitudes = cosine_similarity(final_input, product_vectors)
            
            top_indices = similitudes[0].argsort()[-3:][::-1]
            recos = [int(product_ids[i]) for i in top_indices]
            
            return {"user_id": request.user_id, "type": "Cold Start", "recommendations": recos}
        # --- ENFOQUE 2: Warm Start ---
        else:
            candidatos = [101, 102, 103, 104, 105] 
            datos_pred = []
            for pid in candidatos:
                row = request.context.copy()
                row['product_id'] = pid
                datos_pred.append(row)
            
            input_df = pd.DataFrame(datos_pred)
            
            for col, meta in cat_mappings.items():
                if col in input_df.columns:
                    input_df[col] = input_df[col].map(meta["mapping"]).fillna(meta["desconocida"])
            
            input_df = input_df.reindex(columns=feature_cols, fill_value=0)
            input_df = pd.DataFrame(imputer.transform(input_df), columns=feature_cols)
            
            # Cambiado model_warm por model (la variable global cargada)
            scores = model.predict_proba(input_df)[:, 1]
            
            resultados = pd.DataFrame({'pid': candidatos, 'score': scores})
            top = resultados.sort_values(by='score', ascending=False).head(3)['pid'].tolist()
            
            return {"user_id": request.user_id, "type": "Personalized", "recommendations": top}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/model-metrics")
def get_metrics():
    return {
        "warm_start_metrics": warm_start_dict.get("metrics", {}),
        "cold_start_metrics": cold_start_dict.get("metrics", {})
    }
