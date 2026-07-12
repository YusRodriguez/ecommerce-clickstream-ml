"""
Funciones auxiliares reutilizables del proyecto.

Este módulo reúne utilidades comunes utilizadas en diferentes etapas del
desarrollo, con el objetivo de reducir la duplicación de código y mantener
una implementación consistente entre los notebooks y scripts del proyecto.
"""

from typing import Iterable

import pandas as pd


def verificar_integridad_referencial(
    child_df: pd.DataFrame,
    child_col: str,
    parent_df: pd.DataFrame,
    parent_col: str,
) -> int:
    """
    Verifica la integridad referencial entre dos tablas.

    Retorna la cantidad de registros de la tabla hija cuyo identificador
    no existe en la tabla padre.
    """
    huerfanos = ~child_df[child_col].isin(parent_df[parent_col])
    return int(huerfanos.sum())


def convertir_columnas_fecha(
    df: pd.DataFrame,
    date_columns: Iterable[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Convierte una o varias columnas al tipo datetime.

    La función devuelve una copia del DataFrame original y convierte los
    valores no válidos a NaT utilizando ``errors="coerce"``.
    """
    if isinstance(date_columns, str):
        date_columns = [date_columns]

    df = df.copy()

    for col in date_columns:
        nulos_antes = df[col].isna().sum()
        df[col] = pd.to_datetime(df[col], errors="coerce")
        nuevos_nat = df[col].isna().sum() - nulos_antes

        if verbose and nuevos_nat > 0:
            print(
                f"'{col}': {nuevos_nat} valores no se pudieron convertir "
                "a fecha (quedaron como NaT)."
            )

    return df


def calcular_sets_eventos_sesion(
    events: pd.DataFrame,
    group_col: str = "session_id",
    value_col: str = "event_type",
) -> pd.Series:
    """
    Agrupa los eventos por sesión y genera un conjunto con los distintos
    tipos de eventos registrados en cada una.

    Este resultado facilita el análisis del comportamiento de navegación y
    la construcción del embudo de conversión.
    """
    return events.groupby(group_col)[value_col].apply(set)