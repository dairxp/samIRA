import functools
from pathlib import Path
import pandas as pd

from samira_app.constants import (
    PATH_CALIBRACION_PROV,
    PATH_COBERTURA,
    PATH_METRICAS_TEST,
    PATH_METRICAS_VAL,
    PATH_PANEL_PROVINCIAL,
    PATH_PANEL_REGIONAL,
    PATH_PREDICCIONES,
    PATH_ROLLING_ORIGIN,
)


def _verificar_archivo(ruta: Path) -> None:
    """Verifica que el archivo exista en disco."""
    if not ruta.is_file():
        raise RuntimeError(
            f"Archivo de datos no encontrado: {ruta}. "
            "Asegúrese de ejecutar desde la raíz del proyecto samIRA."
        )


def _validar_columnas(df: pd.DataFrame, requeridas: set[str], origen: str) -> None:
    """Valida la presencia de columnas requeridas tras el proceso de renombrado."""
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise RuntimeError(
            f"Falta columna obligatoria en {origen}: {sorted(list(faltantes))}. "
            f"Columnas disponibles tras renombrado: {list(df.columns)}"
        )


@functools.lru_cache(maxsize=1)
def load_panel_provincial() -> pd.DataFrame:
    """
    Carga el panel epidemiológico provincial estandarizado.

    Rename:
      ano -> anio
      ira_no_neumonia -> episodios
    """
    _verificar_archivo(PATH_PANEL_PROVINCIAL)
    df = pd.read_parquet(PATH_PANEL_PROVINCIAL)

    rename_map = {
        "ano": "anio",
        "ira_no_neumonia": "episodios",
    }
    df = df.rename(columns=rename_map)

    columnas_requeridas = {"provincia", "anio", "semana", "episodios", "es_evaluable"}
    _validar_columnas(df, columnas_requeridas, "panel_semanal_provincial_puno")

    # Asegurar tipos estándar
    df["provincia"] = df["provincia"].astype(str)
    df["anio"] = df["anio"].astype(int)
    df["semana"] = df["semana"].astype(int)
    df["episodios"] = df["episodios"].astype(float)
    return df


@functools.lru_cache(maxsize=1)
def load_panel_regional() -> pd.DataFrame:
    """
    Carga el panel epidemiológico regional estandarizado.

    Rename:
      ano -> anio
      ira_no_neumonia -> episodios
    """
    _verificar_archivo(PATH_PANEL_REGIONAL)
    df = pd.read_parquet(PATH_PANEL_REGIONAL)

    rename_map = {
        "ano": "anio",
        "ira_no_neumonia": "episodios",
    }
    df = df.rename(columns=rename_map)

    columnas_requeridas = {"anio", "semana", "episodios", "es_evaluable"}
    _validar_columnas(df, columnas_requeridas, "panel_semanal_regional_puno")

    df["anio"] = df["anio"].astype(int)
    df["semana"] = df["semana"].astype(int)
    df["episodios"] = df["episodios"].astype(float)
    return df


@functools.lru_cache(maxsize=1)
def load_predicciones() -> pd.DataFrame:
    """
    Carga el dataset de predicciones completas estandarizado.

    Rename:
      entidad -> provincia
      target_ano -> anio
      target_semana -> semana
      horizonte_h -> h
      y_true -> observado
      y_pred -> pronostico
      y_pred_lower_95 -> lim_inf
      y_pred_upper_95 -> lim_sup
    Filtro interno:
      ambito == 'PROVINCIAL'
    """
    _verificar_archivo(PATH_PREDICCIONES)
    df = pd.read_parquet(PATH_PREDICCIONES)

    # Filtrar ámbito provincial
    if "ambito" in df.columns:
        df = df[df["ambito"] == "PROVINCIAL"].copy()

    rename_map = {
        "entidad": "provincia",
        "target_ano": "anio",
        "target_semana": "semana",
        "horizonte_h": "h",
        "y_true": "observado",
        "y_pred": "pronostico",
        "y_pred_lower_95": "lim_inf",
        "y_pred_upper_95": "lim_sup",
    }
    df = df.rename(columns=rename_map)

    columnas_requeridas = {
        "provincia",
        "anio",
        "semana",
        "h",
        "observado",
        "pronostico",
        "lim_inf",
        "lim_sup",
        "modelo",
        "particion",
    }
    _validar_columnas(df, columnas_requeridas, "predicciones_completas")

    df["provincia"] = df["provincia"].astype(str)
    df["anio"] = df["anio"].astype(int)
    df["semana"] = df["semana"].astype(int)
    df["h"] = df["h"].astype(int)
    df["modelo"] = df["modelo"].astype(str)
    df["particion"] = df["particion"].astype(str)
    return df


@functools.lru_cache(maxsize=2)
def load_metricas(particion: str = "test") -> pd.DataFrame:
    """
    Carga las métricas predictivas globales estandarizadas para 'val' o 'test'.

    Rename:
      Particion -> particion
      Horizonte_h -> h
      Modelo -> modelo
      MAE -> mae
      RMSE -> rmse
      "WAPE (%)" -> wape
      Bias -> bias
      N -> n
    """
    ruta = PATH_METRICAS_TEST if particion.lower() == "test" else PATH_METRICAS_VAL
    _verificar_archivo(ruta)
    df = pd.read_csv(ruta)

    rename_map = {
        "Particion": "particion",
        "Horizonte_h": "h",
        "Modelo": "modelo",
        "MAE": "mae",
        "RMSE": "rmse",
        "WAPE (%)": "wape",
        "Bias": "bias",
        "N": "n",
    }
    df = df.rename(columns=rename_map)

    columnas_requeridas = {"particion", "h", "modelo", "mae", "rmse", "wape", "bias", "n"}
    _validar_columnas(df, columnas_requeridas, f"metricas_{particion}")

    df["h"] = df["h"].astype(int)
    df["modelo"] = df["modelo"].astype(str)
    return df


@functools.lru_cache(maxsize=1)
def load_cobertura() -> pd.DataFrame:
    """
    Carga la cobertura de intervalos al 95% estandarizada.

    Rename:
      Nivel -> nivel
      Provincia -> provincia
      Particion -> particion
      Horizonte_h -> h
      Modelo -> modelo
      Cobertura_pct -> cobertura
      Ancho_Promedio -> ancho
      N -> n
    """
    _verificar_archivo(PATH_COBERTURA)
    df = pd.read_csv(PATH_COBERTURA)

    rename_map = {
        "Nivel": "nivel",
        "Provincia": "provincia",
        "Particion": "particion",
        "Horizonte_h": "h",
        "Modelo": "modelo",
        "Cobertura_pct": "cobertura",
        "Ancho_Promedio": "ancho",
        "N": "n",
    }
    df = df.rename(columns=rename_map)

    columnas_requeridas = {"nivel", "provincia", "particion", "h", "modelo", "cobertura", "ancho", "n"}
    _validar_columnas(df, columnas_requeridas, "cobertura_intervalos_95")

    df["h"] = df["h"].astype(int)
    df["provincia"] = df["provincia"].astype(str)
    df["modelo"] = df["modelo"].astype(str)
    return df


@functools.lru_cache(maxsize=1)
def load_calibracion() -> pd.DataFrame:
    """
    Carga la comparación de calibración provincial (conformal prediction) estandarizada.

    Rename:
      Modelo -> modelo
      h -> h
      Provincia -> provincia
      Media_2025 -> media_2025
      "Cob_Orig (%)" -> cob_orig
      "Cob_Calib (%)" -> cob_calib
      Ancho_Orig -> ancho_orig
      Ancho_Calib -> ancho_calib
      Winkler_Orig -> winkler_orig
      Winkler_Calib -> winkler_calib
      "Mejora_Winkler (%)" -> mejora_winkler
    """
    _verificar_archivo(PATH_CALIBRACION_PROV)
    df = pd.read_csv(PATH_CALIBRACION_PROV)

    rename_map = {
        "Modelo": "modelo",
        "h": "h",
        "Provincia": "provincia",
        "Media_2025": "media_2025",
        "Cob_Orig (%)": "cob_orig",
        "Cob_Calib (%)": "cob_calib",
        "Ancho_Orig": "ancho_orig",
        "Ancho_Calib": "ancho_calib",
        "Winkler_Orig": "winkler_orig",
        "Winkler_Calib": "winkler_calib",
        "Mejora_Winkler (%)": "mejora_winkler",
    }
    df = df.rename(columns=rename_map)

    columnas_requeridas = {
        "modelo",
        "h",
        "provincia",
        "cob_orig",
        "cob_calib",
        "ancho_orig",
        "ancho_calib",
        "winkler_orig",
        "winkler_calib",
    }
    _validar_columnas(df, columnas_requeridas, "comparacion_provincial_calibracion")

    df["h"] = df["h"].astype(int)
    df["provincia"] = df["provincia"].astype(str)
    df["modelo"] = df["modelo"].astype(str)
    return df


@functools.lru_cache(maxsize=1)
def load_rolling() -> pd.DataFrame:
    """
    Carga las métricas de validación Rolling-Origin (2020-2024) estandarizadas.

    Rename:
      Modelo -> modelo
      Horizonte_h -> h
      Ventana -> ventana
      Train_Fin -> train_fin
      Test_Ano -> anio_test
      N -> n
      MAE -> mae
      RMSE -> rmse
      "WAPE (%)" -> wape
      Bias -> bias
    """
    _verificar_archivo(PATH_ROLLING_ORIGIN)
    df = pd.read_csv(PATH_ROLLING_ORIGIN)

    rename_map = {
        "Modelo": "modelo",
        "Horizonte_h": "h",
        "Ventana": "ventana",
        "Train_Fin": "train_fin",
        "Test_Ano": "anio_test",
        "N": "n",
        "MAE": "mae",
        "RMSE": "rmse",
        "WAPE (%)": "wape",
        "Bias": "bias",
    }
    df = df.rename(columns=rename_map)

    columnas_requeridas = {
        "modelo",
        "h",
        "ventana",
        "train_fin",
        "anio_test",
        "n",
        "mae",
        "rmse",
        "wape",
        "bias",
    }
    _validar_columnas(df, columnas_requeridas, "metricas_rolling_origin")

    df["h"] = df["h"].astype(int)
    df["modelo"] = df["modelo"].astype(str)
    return df
