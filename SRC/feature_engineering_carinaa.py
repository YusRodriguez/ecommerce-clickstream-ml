import pandas as pd
from data_clean import limpiar_tablas
from sklearn.preprocessing import StandardScaler
import os


def preparar_datos():
    # 1. Pipeline de limpieza de tus compañeros
    events, sessions, reviews, orders, order_items, customers, products = limpiar_tablas()
    print("✓ Datos limpios y listos.")

    return (
        customers,
        orders,
        products,
        order_items,
        events,
        sessions,
        reviews,
    )


def generar_matriz_preferencias(events):
    # Asignamos pesos a las acciones (puedes ajustarlos)
    event_weights = {'page_view': 1, 'add_to_cart': 3, 'purchase': 5}
    events['score'] = events['event_type'].map(event_weights).fillna(1)
    
    # Agrupamos por usuario y producto para obtener el interés acumulado
    matriz_user_item = events.groupby(['customer_id', 'product_id'])['score'].sum().reset_index()
    
    return matriz_user_item

def generar_tabla_maestra_recomendacion(events, sessions, products, customers):
    print("\n============================================================")
    print("CONSTRUYENDO TABLA MAESTRA PARA RECOMENDACIÓN")
    print("============================================================")
    
    # 1. Unir events con sessions para obtener el customer_id
    events_con_user = events.merge(sessions[['session_id', 'customer_id']], on='session_id', how='left')
    
    # 2. Matriz de Afinidad (User-Item)
    event_weights = {'page_view': 1, 'add_to_cart': 3, 'purchase': 5}
    events_con_user['score'] = events_con_user['event_type'].map(event_weights).fillna(1)
    
    # Agrupamos para obtener la preferencia total por cliente/producto
    df_maestra = events_con_user.groupby(['customer_id', 'product_id'])['score'].sum().reset_index()
    
    # 3. Integrar características del producto
    df_maestra = df_maestra.merge(products, on='product_id', how='left')
    
    # 4. Integrar perfil del usuario
    df_maestra = df_maestra.merge(customers, on='customer_id', how='left')
    
    # 5. ELIMINACIÓN DE DATA LEAKAGE (Columnas que no debe ver el modelo)
    # Estas columnas pueden sesgar el modelo o no aportan valor predictivo
    columnas_a_eliminar = [
        'signup_date', 'customer_name', 'email', 'birthdate', 'address', 
        'session_id', 'timestamp' # Eliminamos estas por seguridad
    ]
    df_maestra = df_maestra.drop(columns=[c for c in columnas_a_eliminar if c in df_maestra.columns])
    
    print(f" ✓ Tabla Maestra creada: {df_maestra.shape[0]} filas.")
    print("--- COLUMNAS FINALES PARA EL MODELO ---")
    print(df_maestra.columns.tolist())

    # Ajuste de nombres para evitar confusiones
    df_maestra = df_maestra.rename(columns={'name_x': 'product_name', 'name_y': 'customer_name'})
    
    # Eliminación final para un modelo limpio
    cols_a_borrar = ['customer_name', 'marketing_opt_in', 'margin_usd', 'cost_usd']
    df_maestra = df_maestra.drop(columns=[c for c in cols_a_borrar if c in df_maestra.columns])
    
    # Lista final para verificar
    print(f"Columnas definitivas: {df_maestra.columns.tolist()}")
    return df_maestra
    
    return df_maestra

def finalizar_preparacion_modelo(df):
    print("\n--- INICIANDO PREPARACIÓN FINAL PARA MODELADO ---")
    
    # 1. Imputación básica: llenar nulos en numéricos
    # Evita que el modelo falle por valores vacíos
    df['age'] = df['age'].fillna(df['age'].median())
    df['price_usd'] = df['price_usd'].fillna(df['price_usd'].mean())
    
    # 2. Encoding: Convertir variables categóricas en numéricas (dummies)
    # Crea columnas binarias para cada categoría
    df_final = pd.get_dummies(df, columns=['category', 'country'], drop_first=True)
    
    # 3. Escalado: Estandarizar valores numéricos
    # Esto evita que 'price_usd' domine sobre 'age' por tener números más grandes
    scaler = StandardScaler()
    cols_a_escalar = ['price_usd', 'age', 'score']
    df_final[cols_a_escalar] = scaler.fit_transform(df_final[cols_a_escalar])
    
    print(f" ✓ Preparación finalizada. Columnas resultantes: {df_final.shape[1]}")
    return df_final

if __name__ == "__main__":

    events, sessions, reviews, orders, order_items, customers, products = limpiar_tablas()

    tabla_maestra = generar_tabla_maestra_recomendacion(
        events,
        sessions,
        products,
        customers
    )

    tabla_final = finalizar_preparacion_modelo(tabla_maestra)

    os.makedirs("Data/Processed", exist_ok=True)
    tabla_final.to_csv(
        "Data/Processed/tabla_maestra_final.csv",
        index=False
    )

    print("\n✓ Proceso finalizado. Archivo 'Data/Processed/tabla_maestra_final.csv' guardado y listo.")
