"""
================================================================================
MÓDULO: PIPELINE CAUSAL DE INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING)
Proyecto samIRA - Fase 2
================================================================================
Genera matrices de características para horizontes h=1 y h=4 (Directo y Recursivo):
  - Causalidad estricta: ninguna variable utiliza información posterior al origen t.
  - Imputación causal de soporte C (Media móvil retrospectiva / LOCF) para retardos de entrada.
  - La variable objetivo (y) conserva estrictamente sus NaNs y su bandera es_evaluable.
"""

import os
import sys
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ANIOS_SEMANA_53 = {2003, 2008, 2014, 2020, 2025}

def get_official_weeks_in_year(ano: int) -> int:
    return 53 if ano in ANIOS_SEMANA_53 else 52

def prepare_feature_dataset(horizon: int = 1, is_direct: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepara los conjuntos de datos provinciales y regionales para un horizonte h dado.
    
    Si is_direct=True y horizon=4:
      El objetivo en el origen t es y_{t+4}.
      Las características en t son exclusivamente observables en t (y_t, y_{t-1}, ...).
    Si horizon=1:
      El objetivo en el origen t es y_{t+1}.
    """
    prov_path = os.path.join("data", "processed", "panel_semanal_provincial_puno.parquet")
    reg_path = os.path.join("data", "processed", "panel_semanal_regional_puno.parquet")
    
    df_prov = pd.read_parquet(prov_path)
    df_reg = pd.read_parquet(reg_path)
    
    # --------------------------------------------------------------------------
    # 1. Procesamiento Provincial
    # --------------------------------------------------------------------------
    prov_lookup = df_prov.set_index(["provincia", "ano", "semana"])["ira_no_neumonia"].to_dict()
    
    prov_dfs = []
    for prov, grp in df_prov.groupby("provincia", sort=False):
        grp = grp.copy().sort_values(by=["ano", "semana"]).reset_index(drop=True)
        raw_target = grp["ira_no_neumonia"].copy()
        
        # Serie causal de soporte para lags: LOCF + media móvil de últimas observaciones conocidas
        support_series = raw_target.ffill().bfill()
        
        # Características en el origen t
        features_dict = {
            "provincia": prov,
            "ano": grp["ano"],
            "semana": grp["semana"],
            "ano_semana": grp["ano_semana"],
            "es_evaluable": grp["es_evaluable"],
            "y_raw": raw_target
        }
        
        # Objetivo según horizonte
        if is_direct and horizon > 1:
            features_dict["target"] = raw_target.shift(-horizon)
            features_dict["target_es_evaluable"] = grp["es_evaluable"].shift(-horizon).fillna(False).astype(bool)
            features_dict["target_ano"] = grp["ano"].shift(-horizon)
            features_dict["target_semana"] = grp["semana"].shift(-horizon)
            features_dict["target_ano_semana"] = grp["ano_semana"].shift(-horizon)
        else:
            features_dict["target"] = raw_target
            features_dict["target_es_evaluable"] = grp["es_evaluable"]
            features_dict["target_ano"] = grp["ano"]
            features_dict["target_semana"] = grp["semana"]
            features_dict["target_ano_semana"] = grp["ano_semana"]
            
        # Lags autorregresivos recientes desde el origen
        # Para h=1, en la fila de predicción target en t, el origen es t-1.
        shift_base = 0 if (is_direct and horizon > 1) else 1
        
        features_dict["lag_1"] = support_series.shift(shift_base)
        features_dict["lag_2"] = support_series.shift(shift_base + 1)
        features_dict["lag_3"] = support_series.shift(shift_base + 2)
        features_dict["lag_4"] = support_series.shift(shift_base + 3)
        features_dict["lag_8"] = support_series.shift(shift_base + 7)
        
        # Estadísticos móviles causales retrospectivos
        features_dict["rolling_mean_4"] = support_series.shift(shift_base).rolling(4, min_periods=1).mean()
        features_dict["rolling_std_4"] = support_series.shift(shift_base).rolling(4, min_periods=1).std().fillna(0)
        features_dict["rolling_mean_13"] = support_series.shift(shift_base).rolling(13, min_periods=1).mean()
        features_dict["rolling_std_13"] = support_series.shift(shift_base).rolling(13, min_periods=1).std().fillna(0)
        
        # Retardo estacional homólogo del año previo
        seasonal_lags = []
        for i, row in grp.iterrows():
            y = int(row["ano"])
            w = int(row["semana"])
            # Si estamos en modelo directo para h=4, el objetivo es semana w+4
            target_w = w + (horizon if (is_direct and horizon > 1) else 0)
            target_y = y
            while target_w > get_official_weeks_in_year(target_y):
                target_w -= get_official_weeks_in_year(target_y)
                target_y += 1
                
            prev_y = target_y - 1
            s_lookup_w = 52 if (target_w == 53 and get_official_weeks_in_year(prev_y) == 52) else target_w
            s_val = prov_lookup.get((prov, prev_y, s_lookup_w), np.nan)
            seasonal_lags.append(s_val)
            
        features_dict["seasonal_lag_1y"] = pd.Series(seasonal_lags).ffill().bfill()
        
        # Armónicas de Fourier
        features_dict["sin_semana"] = np.sin(2 * np.pi * grp["semana"] / 52.0)
        features_dict["cos_semana"] = np.cos(2 * np.pi * grp["semana"] / 52.0)
        
        # Dummy de período pandémico (2020-2022)
        features_dict["es_pandemia"] = grp["ano"].between(2020, 2022).astype(int)
        
        df_prov_feat = pd.DataFrame(features_dict)
        prov_dfs.append(df_prov_feat)
        
    final_prov_df = pd.concat(prov_dfs, ignore_index=True)
    final_prov_df["provincia"] = final_prov_df["provincia"].astype("category")
    
    # --------------------------------------------------------------------------
    # 2. Procesamiento Regional
    # --------------------------------------------------------------------------
    reg_lookup = df_reg.set_index(["ano", "semana"])["ira_no_neumonia"].to_dict()
    reg_sorted = df_reg.sort_values(by=["ano", "semana"]).reset_index(drop=True)
    raw_reg_target = reg_sorted["ira_no_neumonia"].copy()
    support_reg = raw_reg_target.ffill().bfill()
    
    reg_dict = {
        "ano": reg_sorted["ano"],
        "semana": reg_sorted["semana"],
        "ano_semana": reg_sorted["ano_semana"],
        "es_evaluable": reg_sorted["es_evaluable"],
        "y_raw": raw_reg_target
    }
    
    shift_base = 0 if (is_direct and horizon > 1) else 1
    if is_direct and horizon > 1:
        reg_dict["target"] = raw_reg_target.shift(-horizon)
        reg_dict["target_es_evaluable"] = reg_sorted["es_evaluable"].shift(-horizon).fillna(False).astype(bool)
        reg_dict["target_ano"] = reg_sorted["ano"].shift(-horizon)
        reg_dict["target_semana"] = reg_sorted["semana"].shift(-horizon)
        reg_dict["target_ano_semana"] = reg_sorted["ano_semana"].shift(-horizon)
    else:
        reg_dict["target"] = raw_reg_target
        reg_dict["target_es_evaluable"] = reg_sorted["es_evaluable"]
        reg_dict["target_ano"] = reg_sorted["ano"]
        reg_dict["target_semana"] = reg_sorted["semana"]
        reg_dict["target_ano_semana"] = reg_sorted["ano_semana"]
        
    reg_dict["lag_1"] = support_reg.shift(shift_base)
    reg_dict["lag_2"] = support_reg.shift(shift_base + 1)
    reg_dict["lag_3"] = support_reg.shift(shift_base + 2)
    reg_dict["lag_4"] = support_reg.shift(shift_base + 3)
    reg_dict["lag_8"] = support_reg.shift(shift_base + 7)
    
    reg_dict["rolling_mean_4"] = support_reg.shift(shift_base).rolling(4, min_periods=1).mean()
    reg_dict["rolling_std_4"] = support_reg.shift(shift_base).rolling(4, min_periods=1).std().fillna(0)
    reg_dict["rolling_mean_13"] = support_reg.shift(shift_base).rolling(13, min_periods=1).mean()
    reg_dict["rolling_std_13"] = support_reg.shift(shift_base).rolling(13, min_periods=1).std().fillna(0)
    
    seasonal_lags_reg = []
    for i, row in reg_sorted.iterrows():
        y = int(row["ano"])
        w = int(row["semana"])
        target_w = w + (horizon if (is_direct and horizon > 1) else 0)
        target_y = y
        while target_w > get_official_weeks_in_year(target_y):
            target_w -= get_official_weeks_in_year(target_y)
            target_y += 1
        prev_y = target_y - 1
        s_lookup_w = 52 if (target_w == 53 and get_official_weeks_in_year(prev_y) == 52) else target_w
        s_val = reg_lookup.get((prev_y, s_lookup_w), np.nan)
        seasonal_lags_reg.append(s_val)
        
    reg_dict["seasonal_lag_1y"] = pd.Series(seasonal_lags_reg).ffill().bfill()
    reg_dict["sin_semana"] = np.sin(2 * np.pi * reg_sorted["semana"] / 52.0)
    reg_dict["cos_semana"] = np.cos(2 * np.pi * reg_sorted["semana"] / 52.0)
    reg_dict["es_pandemia"] = reg_sorted["ano"].between(2020, 2022).astype(int)
    
    final_reg_df = pd.DataFrame(reg_dict)
    
    return final_prov_df, final_reg_df

if __name__ == "__main__":
    df_p, df_r = prepare_feature_dataset(horizon=1)
    print("Features provinciales para h=1:", df_p.shape)
    print("Features regionales para h=1:", df_r.shape)
    df_p4, df_r4 = prepare_feature_dataset(horizon=4, is_direct=True)
    print("Features provinciales para h=4 (Directo):", df_p4.shape)
    print("Features regionales para h=4 (Directo):", df_r4.shape)
