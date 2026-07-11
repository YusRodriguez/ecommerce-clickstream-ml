import os
import pandas as pd


def cargarDatos(nombre_archivo):
    """
    Carga un archivo CSV de la carpeta data.

    Parámetros
    ----------
    nombre_archivo : str
        Nombre del archivo sin la extensión .csv.
        Ejemplo:
            "customers"
            "orders"
            "products"

    Retorna
    -------
    pandas.DataFrame
        DataFrame con los datos cargados.
    """

    # 1. Ruta absoluta del directorio donde está este archivo (src)
    ruta_actual = os.path.dirname(os.path.abspath(__file__))

    # 2. Subir un nivel hasta la carpeta raíz del proyecto
    ruta_proyecto = os.path.dirname(ruta_actual)

    # 3. Construir la ruta completa al archivo CSV
    ruta_csv = os.path.join(
        ruta_proyecto,
        "data",
        "raw",
        f"{nombre_archivo}.csv"
    )

    # 4. Leer el archivo CSV
    df = pd.read_csv(ruta_csv)

    print(f"\nArchivo cargado correctamente: {nombre_archivo}.csv")
    print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")

    return df


if __name__ == "__main__":

    # Ejemplo de uso
    datos = cargarDatos("customers")

    print("\nPrimeras filas:")
    print(datos.head())

    print("\nColumnas:")
    print(datos.columns)


def verificar_integridad_referencial(child_df, child_col, parent_df, parent_col):
    """
    Verifica integridad referencial: cuántos valores del hijo no existen en el padre.

    Parámetros
    ----------
    child_df : DataFrame
        DataFrame hijo (el que tiene la foreign key).
    child_col : str
        Nombre de la columna foreign key en el hijo.
    parent_df : DataFrame
        DataFrame padre (el que tiene la primary key).
    parent_col : str
        Nombre de la columna primary key en el padre.

    Retorna
    -------
    int
        Cantidad de filas huérfanas (valores del hijo que no existen en el padre).
    """
    return (~child_df[child_col].isin(parent_df[parent_col])).sum()


def calcular_sets_eventos_sesion(events):
    """
    Agrupa eventos por sesión y retorna un set de tipos de evento por sesión.

    Parámetros
    ----------
    events : DataFrame
        DataFrame de eventos con columnas 'session_id' y 'event_type'.

    Retorna
    -------
    pandas.Series
        Serie con session_id como índice y un set de event_type como valor.
    """
    return events.groupby("session_id")["event_type"].apply(set)


def convertir_columnas_fecha(df, date_columns):
    """
    Convierte columnas a datetime usando pd.to_datetime.

    Parámetros
    ----------
    df : DataFrame
        DataFrame a modificar.
    date_columns : list
        Lista de nombres de columnas a convertir.
    """
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])