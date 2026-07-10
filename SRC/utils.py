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