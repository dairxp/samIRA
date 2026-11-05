"""
================================================================================
MÓDULO: ENTRENAMIENTO Y EVALUACIÓN FINAL DE MODELOS ML Y BASELINES
Proyecto samIRA - Fase 2
================================================================================
Entrena y evalúa todos los modelos para h=1 y h=4 (Directo y Recursivo):
  - Modelos Baselines: Naive, Seasonal Naive, ETS
  - Modelos ML Globales: LightGBM y XGBoost
  - Modelos ML Independientes por Provincia: LightGBM por provincia
  - Modelos ML Regionales
  - Generación de Intervalos de Predicción al 95% empíricamente calibrados en Validación
  - Exportación de predicciones y tablas completas de resultados
"""

import os
import sys
import time
import joblib
import importlib.util
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Cargar dinámicamente feature pipeline
spec = importlib.util.spec_from_file_location("feature_pipeline", os.path.join("src", "fase2", "05_feature_pipeline.py"))
feat_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat_mod)
prepare_feature_dataset = feat_mod.prepare_feature_dataset

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

RANDOM_SEED = 42

def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return {"mae": np.nan, "rmse": np.nan, "wape_pct": np.nan, "bias": np.nan, "n": 0}
    mae = mean_absolute_error(yt, yp)
    rmse = root_mean_squared_error(yt, yp)
    wape = np.sum(np.abs(yt - yp)) / np.sum(yt) * 100
    bias = np.mean(yp - yt)
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "wape_pct": round(wape, 3),
        "bias": round(bias, 3),
        "n": len(yt)
    }

def main():
    print("=" * 80)
    print("ENTRENAMIENTO Y EVALUACIÓN CIENTÍFICA INTEGRAL - FASE 2")
    print("=" * 80)
    
    # 1. Cargar parámetros óptimos
    opt_df = pd.read_csv(os.path.join("reports", "fase2", "hiperparametros_optimos.csv"))
    lgb_p1 = eval(opt_df.loc[(opt_df["algoritmo"] == "LightGBM") & (opt_df["horizonte_h"] == 1), "parametros_optimos"].values[0])
    xgb_p1 = eval(opt_df.loc[(opt_df["algoritmo"] == "XGBoost") & (opt_df["horizonte_h"] == 1), "parametros_optimos"].values[0])
    lgb_p4 = eval(opt_df.loc[(opt_df["algoritmo"] == "LightGBM") & (opt_df["horizonte_h"] == 4), "parametros_optimos"].values[0])
    xgb_p4 = eval(opt_df.loc[(opt_df["algoritmo"] == "XGBoost") & (opt_df["horizonte_h"] == 4), "parametros_optimos"].values[0])
    
    # 2. Cargar predicciones baselines precalculadas
    base_preds_df = pd.read_parquet(os.path.join("data", "processed", "predicciones_baselines.parquet"))
    
    all_predictions = []
    
    # Copiar predicciones de baselines en el formato unificado
    for _, row in base_preds_df.iterrows():
        # Naive
        all_predictions.append({
            "ambito": row["ambito"],
            "entidad": row["entidad"],
            "ano": row["ano"],
            "semana": row["semana"],
            "ano_semana": row["ano_semana"],
            "horizonte_h": row["horizonte_h"],
            "estrategia_horizonte": "N/A",
            "particion": row["particion"],
            "es_evaluable": row["es_evaluable"],
            "y_true": row["y_true"],
            "modelo": "Naive",
            "y_pred": row["pred_naive"],
            "y_pred_lower_95": np.nan,
            "y_pred_upper_95": np.nan
        })
        # Seasonal Naive
        all_predictions.append({
            "ambito": row["ambito"],
            "entidad": row["entidad"],
            "ano": row["ano"],
            "semana": row["semana"],
            "ano_semana": row["ano_semana"],
            "horizonte_h": row["horizonte_h"],
            "estrategia_horizonte": "N/A",
            "particion": row["particion"],
            "es_evaluable": row["es_evaluable"],
            "y_true": row["y_true"],
            "modelo": "Seasonal Naive",
            "y_pred": row["pred_snaive"],
            "y_pred_lower_95": np.nan,
            "y_pred_upper_95": np.nan
        })
        # ETS
        all_predictions.append({
            "ambito": row["ambito"],
            "entidad": row["entidad"],
            "ano": row["ano"],
            "semana": row["semana"],
            "ano_semana": row["ano_semana"],
            "horizonte_h": row["horizonte_h"],
            "estrategia_horizonte": "N/A",
            "particion": row["particion"],
            "es_evaluable": row["es_evaluable"],
            "y_true": row["y_true"],
            "modelo": "ETS / Holt-Winters",
            "y_pred": row["pred_ets"],
            "y_pred_lower_95": np.nan,
            "y_pred_upper_95": np.nan
        })
        
    feature_cols = [
        "lag_1", "lag_2", "lag_3", "lag_4", "lag_8",
        "rolling_mean_4", "rolling_std_4", "rolling_mean_13", "rolling_std_13",
        "seasonal_lag_1y", "sin_semana", "cos_semana", "es_pandemia"
    ]
    
    # --------------------------------------------------------------------------
    # MODELO GLOBAL PROVINCIAL (h=1)
    # --------------------------------------------------------------------------
    print("\n[1/5] Entrenando Modelos Globales Provinciales para h=1...")
    df_p1, df_r1 = prepare_feature_dataset(horizon=1, is_direct=True)
    
    # Train: 2001-2022
    p_train = df_p1[(df_p1["ano"].between(2001, 2022)) & (df_p1["target_es_evaluable"] == True)].copy()
    p_val = df_p1[(df_p1["target_ano"].between(2023, 2024)) & (df_p1["target_es_evaluable"] == True)].copy()
    p_test = df_p1[(df_p1["target_ano"] == 2025) & (df_p1["target_es_evaluable"] == True)].copy()
    
    # Features LightGBM
    X_train_lgb1 = p_train[feature_cols + ["provincia"]]
    y_train1 = p_train["target"].values
    
    X_val_lgb1 = p_val[feature_cols + ["provincia"]]
    y_val1 = p_val["target"].values
    
    X_test_lgb1 = p_test[feature_cols + ["provincia"]]
    y_test1 = p_test["target"].values
    
    # Entrenar LightGBM h=1
    lgb_model1 = lgb.LGBMRegressor(**lgb_p1, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)
    lgb_model1.fit(X_train_lgb1, y_train1)
    joblib.dump(lgb_model1, os.path.join("models", "fase2", "lgbm_global_h1.joblib"))
    
    val_preds_lgb1 = lgb_model1.predict(X_val_lgb1)
    test_preds_lgb1 = lgb_model1.predict(X_test_lgb1)
    
    # Calibración de intervalo de predicción al 95% usando cuantiles de residuos empíricos en validación
    res_val_lgb1 = y_val1 - val_preds_lgb1
    q025_lgb1 = np.percentile(res_val_lgb1, 2.5)
    q975_lgb1 = np.percentile(res_val_lgb1, 97.5)
    
    for i, row in p_val.reset_index(drop=True).iterrows():
        all_predictions.append({
            "ambito": "PROVINCIAL",
            "entidad": row["provincia"],
            "ano": int(row["target_ano"]),
            "semana": int(row["target_semana"]),
            "ano_semana": row["target_ano_semana"],
            "horizonte_h": 1,
            "estrategia_horizonte": "Directa",
            "particion": "val",
            "es_evaluable": row["target_es_evaluable"],
            "y_true": row["target"],
            "modelo": "LightGBM Global",
            "y_pred": max(0.0, float(val_preds_lgb1[i])),
            "y_pred_lower_95": max(0.0, float(val_preds_lgb1[i] + q025_lgb1)),
            "y_pred_upper_95": max(0.0, float(val_preds_lgb1[i] + q975_lgb1))
        })
        
    for i, row in p_test.reset_index(drop=True).iterrows():
        all_predictions.append({
            "ambito": "PROVINCIAL",
            "entidad": row["provincia"],
            "ano": int(row["target_ano"]),
            "semana": int(row["target_semana"]),
            "ano_semana": row["target_ano_semana"],
            "horizonte_h": 1,
            "estrategia_horizonte": "Directa",
            "particion": "test",
            "es_evaluable": row["target_es_evaluable"],
            "y_true": row["target"],
            "modelo": "LightGBM Global",
            "y_pred": max(0.0, float(test_preds_lgb1[i])),
            "y_pred_lower_95": max(0.0, float(test_preds_lgb1[i] + q025_lgb1)),
            "y_pred_upper_95": max(0.0, float(test_preds_lgb1[i] + q975_lgb1))
        })
        
    # Features XGBoost (One-hot provincia)
    X_train_xgb1 = pd.get_dummies(p_train[feature_cols + ["provincia"]], columns=["provincia"], drop_first=True)
    X_val_xgb1 = pd.get_dummies(p_val[feature_cols + ["provincia"]], columns=["provincia"], drop_first=True)
    X_test_xgb1 = pd.get_dummies(p_test[feature_cols + ["provincia"]], columns=["provincia"], drop_first=True)
    X_train_xgb1, X_val_xgb1 = X_train_xgb1.align(X_val_xgb1, join="left", axis=1, fill_value=0)
    X_train_xgb1, X_test_xgb1 = X_train_xgb1.align(X_test_xgb1, join="left", axis=1, fill_value=0)
    
    xgb_model1 = xgb.XGBRegressor(**xgb_p1, random_state=RANDOM_SEED, verbosity=0, n_jobs=-1)
    xgb_model1.fit(X_train_xgb1, y_train1)
    joblib.dump(xgb_model1, os.path.join("models", "fase2", "xgboost_global_h1.joblib"))
    
    val_preds_xgb1 = xgb_model1.predict(X_val_xgb1)
    test_preds_xgb1 = xgb_model1.predict(X_test_xgb1)
    
    res_val_xgb1 = y_val1 - val_preds_xgb1
    q025_xgb1 = np.percentile(res_val_xgb1, 2.5)
    q975_xgb1 = np.percentile(res_val_xgb1, 97.5)
    
    for i, row in p_val.reset_index(drop=True).iterrows():
        all_predictions.append({
            "ambito": "PROVINCIAL",
            "entidad": row["provincia"],
            "ano": int(row["target_ano"]),
            "semana": int(row["target_semana"]),
            "ano_semana": row["target_ano_semana"],
            "horizonte_h": 1,
            "estrategia_horizonte": "Directa",
            "particion": "val",
            "es_evaluable": row["target_es_evaluable"],
            "y_true": row["target"],
            "modelo": "XGBoost Global",
            "y_pred": max(0.0, float(val_preds_xgb1[i])),
            "y_pred_lower_95": max(0.0, float(val_preds_xgb1[i] + q025_xgb1)),
            "y_pred_upper_95": max(0.0, float(val_preds_xgb1[i] + q975_xgb1))
        })
        
    for i, row in p_test.reset_index(drop=True).iterrows():
        all_predictions.append({
            "ambito": "PROVINCIAL",
            "entidad": row["provincia"],
            "ano": int(row["target_ano"]),
            "semana": int(row["target_semana"]),
            "ano_semana": row["target_ano_semana"],
            "horizonte_h": 1,
            "estrategia_horizonte": "Directa",
            "particion": "test",
            "es_evaluable": row["target_es_evaluable"],
            "y_true": row["target"],
            "modelo": "XGBoost Global",
            "y_pred": max(0.0, float(test_preds_xgb1[i])),
            "y_pred_lower_95": max(0.0, float(test_preds_xgb1[i] + q025_xgb1)),
            "y_pred_upper_95": max(0.0, float(test_preds_xgb1[i] + q975_xgb1))
        })
        
    # --------------------------------------------------------------------------
    # MODELOS INDEPENDIENTES POR PROVINCIA (h=1)
    # --------------------------------------------------------------------------
    print("\n[2/5] Entrenando Modelos Independientes por Provincia (LightGBM h=1)...")
    indep_val_preds = []
    indep_test_preds = []
    
    for prov in sorted(df_p1["provincia"].unique()):
        p_sub_train = p_train[p_train["provincia"] == prov]
        p_sub_val = p_val[p_val["provincia"] == prov]
        p_sub_test = p_test[p_test["provincia"] == prov]
        
        m_indep = lgb.LGBMRegressor(**lgb_p1, random_state=RANDOM_SEED, verbose=-1, n_jobs=1)
        m_indep.fit(p_sub_train[feature_cols], p_sub_train["target"])
        
        pv_preds = m_indep.predict(p_sub_val[feature_cols])
        pt_preds = m_indep.predict(p_sub_test[feature_cols])
        
        res_indep = p_sub_val["target"].values - pv_preds
        q025_indep = np.percentile(res_indep, 2.5) if len(res_indep) > 0 else 0
        q975_indep = np.percentile(res_indep, 97.5) if len(res_indep) > 0 else 0
        
        for idx, row in p_sub_val.reset_index(drop=True).iterrows():
            all_predictions.append({
                "ambito": "PROVINCIAL",
                "entidad": prov,
                "ano": int(row["target_ano"]),
                "semana": int(row["target_semana"]),
                "ano_semana": row["target_ano_semana"],
                "horizonte_h": 1,
                "estrategia_horizonte": "Directa",
                "particion": "val",
                "es_evaluable": row["target_es_evaluable"],
                "y_true": row["target"],
                "modelo": "LightGBM Independiente",
                "y_pred": max(0.0, float(pv_preds[idx])),
                "y_pred_lower_95": max(0.0, float(pv_preds[idx] + q025_indep)),
                "y_pred_upper_95": max(0.0, float(pv_preds[idx] + q975_indep))
            })
            
        for idx, row in p_sub_test.reset_index(drop=True).iterrows():
            all_predictions.append({
                "ambito": "PROVINCIAL",
                "entidad": prov,
                "ano": int(row["target_ano"]),
                "semana": int(row["target_semana"]),
                "ano_semana": row["target_ano_semana"],
                "horizonte_h": 1,
                "estrategia_horizonte": "Directa",
                "particion": "test",
                "es_evaluable": row["target_es_evaluable"],
                "y_true": row["target"],
                "modelo": "LightGBM Independiente",
                "y_pred": max(0.0, float(pt_preds[idx])),
                "y_pred_lower_95": max(0.0, float(pt_preds[idx] + q025_indep)),
                "y_pred_upper_95": max(0.0, float(pt_preds[idx] + q975_indep))
            })
            
    # --------------------------------------------------------------------------
    # MODELO GLOBAL PROVINCIAL (h=4 DIRECTO)
    # --------------------------------------------------------------------------
    print("\n[3/5] Entrenando Modelos Globales Provinciales para h=4 (Directo)...")
    df_p4, df_r4 = prepare_feature_dataset(horizon=4, is_direct=True)
    
    p_train4 = df_p4[(df_p4["ano"].between(2001, 2022)) & (df_p4["target_es_evaluable"] == True)].copy()
    p_val4 = df_p4[(df_p4["target_ano"].between(2023, 2024)) & (df_p4["target_es_evaluable"] == True)].copy()
    p_test4 = df_p4[(df_p4["target_ano"] == 2025) & (df_p4["target_es_evaluable"] == True)].copy()
    
    X_train_lgb4 = p_train4[feature_cols + ["provincia"]]
    y_train4 = p_train4["target"].values
    X_val_lgb4 = p_val4[feature_cols + ["provincia"]]
    y_val4 = p_val4["target"].values
    X_test_lgb4 = p_test4[feature_cols + ["provincia"]]
    y_test4 = p_test4["target"].values
    
    lgb_model4 = lgb.LGBMRegressor(**lgb_p4, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)
    lgb_model4.fit(X_train_lgb4, y_train4)
    joblib.dump(lgb_model4, os.path.join("models", "fase2", "lgbm_global_h4_direct.joblib"))
    
    val_preds_lgb4 = lgb_model4.predict(X_val_lgb4)
    test_preds_lgb4 = lgb_model4.predict(X_test_lgb4)
    
    res_val_lgb4 = y_val4 - val_preds_lgb4
    q025_lgb4 = np.percentile(res_val_lgb4, 2.5)
    q975_lgb4 = np.percentile(res_val_lgb4, 97.5)
    
    for i, row in p_val4.reset_index(drop=True).iterrows():
        all_predictions.append({
            "ambito": "PROVINCIAL",
            "entidad": row["provincia"],
            "ano": int(row["target_ano"]),
            "semana": int(row["target_semana"]),
            "ano_semana": row["target_ano_semana"],
            "horizonte_h": 4,
            "estrategia_horizonte": "Directa",
            "particion": "val",
            "es_evaluable": row["target_es_evaluable"],
            "y_true": row["target"],
            "modelo": "LightGBM Global Directo",
            "y_pred": max(0.0, float(val_preds_lgb4[i])),
            "y_pred_lower_95": max(0.0, float(val_preds_lgb4[i] + q025_lgb4)),
            "y_pred_upper_95": max(0.0, float(val_preds_lgb4[i] + q975_lgb4))
        })
        
    for i, row in p_test4.reset_index(drop=True).iterrows():
        all_predictions.append({
            "ambito": "PROVINCIAL",
            "entidad": row["provincia"],
            "ano": int(row["target_ano"]),
            "semana": int(row["target_semana"]),
            "ano_semana": row["target_ano_semana"],
            "horizonte_h": 4,
            "estrategia_horizonte": "Directa",
            "particion": "test",
            "es_evaluable": row["target_es_evaluable"],
            "y_true": row["target"],
            "modelo": "LightGBM Global Directo",
            "y_pred": max(0.0, float(test_preds_lgb4[i])),
            "y_pred_lower_95": max(0.0, float(test_preds_lgb4[i] + q025_lgb4)),
            "y_pred_upper_95": max(0.0, float(test_preds_lgb4[i] + q975_lgb4))
        })
        
    # XGBoost Global Directo h=4
    X_train_xgb4 = pd.get_dummies(p_train4[feature_cols + ["provincia"]], columns=["provincia"], drop_first=True)
    X_val_xgb4 = pd.get_dummies(p_val4[feature_cols + ["provincia"]], columns=["provincia"], drop_first=True)
    X_test_xgb4 = pd.get_dummies(p_test4[feature_cols + ["provincia"]], columns=["provincia"], drop_first=True)
    X_train_xgb4, X_val_xgb4 = X_train_xgb4.align(X_val_xgb4, join="left", axis=1, fill_value=0)
    X_train_xgb4, X_test_xgb4 = X_train_xgb4.align(X_test_xgb4, join="left", axis=1, fill_value=0)
    
    xgb_model4 = xgb.XGBRegressor(**xgb_p4, random_state=RANDOM_SEED, verbosity=0, n_jobs=-1)
    xgb_model4.fit(X_train_xgb4, y_train4)
    joblib.dump(xgb_model4, os.path.join("models", "fase2", "xgboost_global_h4_direct.joblib"))
    
    val_preds_xgb4 = xgb_model4.predict(X_val_xgb4)
    test_preds_xgb4 = xgb_model4.predict(X_test_xgb4)
    
    res_val_xgb4 = y_val4 - val_preds_xgb4
    q025_xgb4 = np.percentile(res_val_xgb4, 2.5)
    q975_xgb4 = np.percentile(res_val_xgb4, 97.5)
    
    for i, row in p_val4.reset_index(drop=True).iterrows():
        all_predictions.append({
            "ambito": "PROVINCIAL",
            "entidad": row["provincia"],
            "ano": int(row["target_ano"]),
            "semana": int(row["target_semana"]),
            "ano_semana": row["target_ano_semana"],
            "horizonte_h": 4,
            "estrategia_horizonte": "Directa",
            "particion": "val",
            "es_evaluable": row["target_es_evaluable"],
            "y_true": row["target"],
            "modelo": "XGBoost Global Directo",
            "y_pred": max(0.0, float(val_preds_xgb4[i])),
            "y_pred_lower_95": max(0.0, float(val_preds_xgb4[i] + q025_xgb4)),
            "y_pred_upper_95": max(0.0, float(val_preds_xgb4[i] + q975_xgb4))
        })
        
    for i, row in p_test4.reset_index(drop=True).iterrows():
        all_predictions.append({
            "ambito": "PROVINCIAL",
            "entidad": row["provincia"],
            "ano": int(row["target_ano"]),
            "semana": int(row["target_semana"]),
            "ano_semana": row["target_ano_semana"],
            "horizonte_h": 4,
            "estrategia_horizonte": "Directa",
            "particion": "test",
            "es_evaluable": row["target_es_evaluable"],
            "y_true": row["target"],
            "modelo": "XGBoost Global Directo",
            "y_pred": max(0.0, float(test_preds_xgb4[i])),
            "y_pred_lower_95": max(0.0, float(test_preds_xgb4[i] + q025_xgb4)),
            "y_pred_upper_95": max(0.0, float(test_preds_xgb4[i] + q975_xgb4))
        })

    # --------------------------------------------------------------------------
    # MODELO GLOBAL PROVINCIAL (h=4 RECURSIVO con LightGBM)
    # --------------------------------------------------------------------------
    print("\n[4/5] Evaluando Estrategia Recursiva para h=4 con LightGBM...")
    # Para recursivo: desde cada origen t, predecimos t+1, actualizamos lag_1, lag_2, ..., predecimos t+2, t+3, t+4.
    # Evaluamos sobre las semanas de test y val
    for part, part_anos in [("val", [2023, 2024]), ("test", [2025])]:
        for prov in sorted(df_p1["provincia"].unique()):
            prov_series = df_p1[df_p1["provincia"] == prov].sort_values(by=["ano", "semana"]).reset_index(drop=True)
            
            # Índices donde el target_ano cae en part_anos
            eval_indices = prov_series[prov_series["ano"].isin(part_anos)].index
            
            for idx in eval_indices:
                origin_idx = idx - 4
                if origin_idx < 0:
                    continue
                    
                target_row = prov_series.iloc[idx]
                target_val = target_row["y_raw"]
                is_eval = target_row["es_evaluable"]
                
                # Simular avance paso a paso desde origin_idx
                # Historial disponible hasta origin_idx
                history = prov_series.loc[:origin_idx, "y_raw"].ffill().bfill().tolist()
                
                # Predecir paso 1, 2, 3, 4
                curr_pred = 0.0
                for step in range(1, 5):
                    # Armar fila de features sintética
                    s_lag1 = history[-1]
                    s_lag2 = history[-2] if len(history) >= 2 else s_lag1
                    s_lag3 = history[-3] if len(history) >= 3 else s_lag2
                    s_lag4 = history[-4] if len(history) >= 4 else s_lag3
                    s_lag8 = history[-8] if len(history) >= 8 else s_lag4
                    
                    r_mean4 = np.mean(history[-4:])
                    r_std4 = np.std(history[-4:])
                    r_mean13 = np.mean(history[-13:])
                    r_std13 = np.std(history[-13:])
                    
                    step_idx = origin_idx + step
                    step_row = prov_series.iloc[min(step_idx, len(prov_series)-1)]
                    
                    feat_dict = {
                        "lag_1": [s_lag1], "lag_2": [s_lag2], "lag_3": [s_lag3], "lag_4": [s_lag4], "lag_8": [s_lag8],
                        "rolling_mean_4": [r_mean4], "rolling_std_4": [r_std4], "rolling_mean_13": [r_mean13], "rolling_std_13": [r_std13],
                        "seasonal_lag_1y": [step_row["seasonal_lag_1y"]],
                        "sin_semana": [step_row["sin_semana"]],
                        "cos_semana": [step_row["cos_semana"]],
                        "es_pandemia": [step_row["es_pandemia"]],
                        "provincia": pd.Categorical([prov], categories=p_train["provincia"].cat.categories)
                    }
                    X_step = pd.DataFrame(feat_dict)[X_train_lgb1.columns]
                    curr_pred = max(0.0, float(lgb_model1.predict(X_step)[0]))
                    history.append(curr_pred)
                    
                all_predictions.append({
                    "ambito": "PROVINCIAL",
                    "entidad": prov,
                    "ano": int(target_row["ano"]),
                    "semana": int(target_row["semana"]),
                    "ano_semana": target_row["ano_semana"],
                    "horizonte_h": 4,
                    "estrategia_horizonte": "Recursiva",
                    "particion": part,
                    "es_evaluable": is_eval,
                    "y_true": target_val,
                    "modelo": "LightGBM Global Recursivo",
                    "y_pred": curr_pred,
                    "y_pred_lower_95": max(0.0, curr_pred + q025_lgb4),
                    "y_pred_upper_95": max(0.0, curr_pred + q975_lgb4)
                })

    # --------------------------------------------------------------------------
    # MODELOS REGIONALES AGREGADOS (h=1 y h=4)
    # --------------------------------------------------------------------------
    print("\n[5/5] Entrenando Modelos Regionales Agregados (LightGBM h=1 y h=4)...")
    reg_train1 = df_r1[(df_r1["ano"].between(2001, 2022)) & (df_r1["target_es_evaluable"] == True)]
    reg_val1 = df_r1[(df_r1["target_ano"].between(2023, 2024)) & (df_r1["target_es_evaluable"] == True)]
    reg_test1 = df_r1[(df_r1["target_ano"] == 2025) & (df_r1["target_es_evaluable"] == True)]
    
    m_reg1 = lgb.LGBMRegressor(**lgb_p1, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)
    m_reg1.fit(reg_train1[feature_cols], reg_train1["target"])
    
    v_preds_r1 = m_reg1.predict(reg_val1[feature_cols])
    t_preds_r1 = m_reg1.predict(reg_test1[feature_cols])
    
    res_reg1 = reg_val1["target"].values - v_preds_r1
    q025_reg1 = np.percentile(res_reg1, 2.5)
    q975_reg1 = np.percentile(res_reg1, 97.5)
    
    for i, row in reg_val1.reset_index(drop=True).iterrows():
        all_predictions.append({
            "ambito": "REGIONAL",
            "entidad": "PUNO_REGION",
            "ano": int(row["target_ano"]),
            "semana": int(row["target_semana"]),
            "ano_semana": row["target_ano_semana"],
            "horizonte_h": 1,
            "estrategia_horizonte": "Directa",
            "particion": "val",
            "es_evaluable": row["target_es_evaluable"],
            "y_true": row["target"],
            "modelo": "LightGBM Regional",
            "y_pred": max(0.0, float(v_preds_r1[i])),
            "y_pred_lower_95": max(0.0, float(v_preds_r1[i] + q025_reg1)),
            "y_pred_upper_95": max(0.0, float(v_preds_r1[i] + q975_reg1))
        })
        
    for i, row in reg_test1.reset_index(drop=True).iterrows():
        all_predictions.append({
            "ambito": "REGIONAL",
            "entidad": "PUNO_REGION",
            "ano": int(row["target_ano"]),
            "semana": int(row["target_semana"]),
            "ano_semana": row["target_ano_semana"],
            "horizonte_h": 1,
            "estrategia_horizonte": "Directa",
            "particion": "test",
            "es_evaluable": row["target_es_evaluable"],
            "y_true": row["target"],
            "modelo": "LightGBM Regional",
            "y_pred": max(0.0, float(t_preds_r1[i])),
            "y_pred_lower_95": max(0.0, float(t_preds_r1[i] + q025_reg1)),
            "y_pred_upper_95": max(0.0, float(t_preds_r1[i] + q975_reg1))
        })
        
    # Regional h=4
    reg_train4 = df_r4[(df_r4["ano"].between(2001, 2022)) & (df_r4["target_es_evaluable"] == True)]
    reg_val4 = df_r4[(df_r4["target_ano"].between(2023, 2024)) & (df_r4["target_es_evaluable"] == True)]
    reg_test4 = df_r4[(df_r4["target_ano"] == 2025) & (df_r4["target_es_evaluable"] == True)]
    
    m_reg4 = lgb.LGBMRegressor(**lgb_p4, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)
    m_reg4.fit(reg_train4[feature_cols], reg_train4["target"])
    
    v_preds_r4 = m_reg4.predict(reg_val4[feature_cols])
    t_preds_r4 = m_reg4.predict(reg_test4[feature_cols])
    
    res_reg4 = reg_val4["target"].values - v_preds_r4
    q025_reg4 = np.percentile(res_reg4, 2.5)
    q975_reg4 = np.percentile(res_reg4, 97.5)
    
    for i, row in reg_val4.reset_index(drop=True).iterrows():
        all_predictions.append({
            "ambito": "REGIONAL",
            "entidad": "PUNO_REGION",
            "ano": int(row["target_ano"]),
            "semana": int(row["target_semana"]),
            "ano_semana": row["target_ano_semana"],
            "horizonte_h": 4,
            "estrategia_horizonte": "Directa",
            "particion": "val",
            "es_evaluable": row["target_es_evaluable"],
            "y_true": row["target"],
            "modelo": "LightGBM Regional Directo",
            "y_pred": max(0.0, float(v_preds_r4[i])),
            "y_pred_lower_95": max(0.0, float(v_preds_r4[i] + q025_reg4)),
            "y_pred_upper_95": max(0.0, float(v_preds_r4[i] + q975_reg4))
        })
        
    for i, row in reg_test4.reset_index(drop=True).iterrows():
        all_predictions.append({
            "ambito": "REGIONAL",
            "entidad": "PUNO_REGION",
            "ano": int(row["target_ano"]),
            "semana": int(row["target_semana"]),
            "ano_semana": row["target_ano_semana"],
            "horizonte_h": 4,
            "estrategia_horizonte": "Directa",
            "particion": "test",
            "es_evaluable": row["target_es_evaluable"],
            "y_true": row["target"],
            "modelo": "LightGBM Regional Directo",
            "y_pred": max(0.0, float(t_preds_r4[i])),
            "y_pred_lower_95": max(0.0, float(t_preds_r4[i] + q025_reg4)),
            "y_pred_upper_95": max(0.0, float(t_preds_r4[i] + q975_reg4))
        })

    # ==========================================================================
    # CONSOLIDACIÓN Y EXPORTACIÓN DE TABLAS DE RESULTADOS
    # ==========================================================================
    df_pred_final = pd.DataFrame(all_predictions)
    
    # Exportar predicciones detalladas
    pred_file = os.path.join("reports", "fase2", "predicciones_2025.csv")
    test_preds_export = df_pred_final[df_pred_final["particion"] == "test"].copy()
    test_preds_export.to_csv(pred_file, index=False, encoding="utf-8-sig")
    print(f"\nPredicciones del conjunto de prueba 2025 exportadas: {pred_file} ({len(test_preds_export):,} filas)")
    
    # Calcular métricas para todas las combinaciones
    provincias_list = sorted(df_p1["provincia"].unique())
    summary_metrics = []
    prov_metrics = []
    
    eval_df = df_pred_final[df_pred_final["es_evaluable"] == True].copy()
    
    for (ambito, part, h, modelo), grp in eval_df.groupby(["ambito", "particion", "horizonte_h", "modelo"]):
        # 1. Global
        m_glob = compute_metrics(grp["y_true"].values, grp["y_pred"].values)
        
        # Cobertura empírica de intervalo si existe
        if grp["y_pred_lower_95"].notna().any():
            inside = (grp["y_true"] >= grp["y_pred_lower_95"]) & (grp["y_true"] <= grp["y_pred_upper_95"])
            cov_95 = round(inside.mean() * 100, 2)
        else:
            cov_95 = np.nan
            
        # 2. Macro (para provincial)
        if ambito == "PROVINCIAL":
            macro_maes = []
            macro_wapes = []
            for p in provincias_list:
                p_grp = grp[grp["entidad"] == p]
                if len(p_grp) > 0:
                    mp = compute_metrics(p_grp["y_true"].values, p_grp["y_pred"].values)
                    macro_maes.append(mp["mae"])
                    macro_wapes.append(mp["wape_pct"])
                    
                    prov_metrics.append({
                        "provincia": p,
                        "particion": part,
                        "horizonte_h": h,
                        "modelo": modelo,
                        "mae": mp["mae"],
                        "rmse": mp["rmse"],
                        "wape_pct": mp["wape_pct"],
                        "bias": mp["bias"],
                        "n_semanas": mp["n"]
                    })
            macro_mae = round(np.mean(macro_maes), 3)
            macro_wape = round(np.mean(macro_wapes), 3)
        else:
            macro_mae = m_glob["mae"]
            macro_wape = m_glob["wape_pct"]
            
        summary_metrics.append({
            "ambito": ambito,
            "particion": part,
            "horizonte_h": h,
            "modelo": modelo,
            "mae_global": m_glob["mae"],
            "rmse_global": m_glob["rmse"],
            "wape_global_pct": m_glob["wape_pct"],
            "bias_global": m_glob["bias"],
            "mae_macro": macro_mae,
            "wape_macro_pct": macro_wape,
            "cobertura_ic_95_pct": cov_95,
            "n_evaluable": m_glob["n"]
        })
        
    df_metrics = pd.DataFrame(summary_metrics)
    
    # Calcular mejora respecto a Seasonal Naive dentro de cada ámbito, partición y horizonte
    enriched_rows = []
    for (amb, part, h), grp in df_metrics.groupby(["ambito", "particion", "horizonte_h"]):
        snaive_row = grp[grp["modelo"] == "Seasonal Naive"]
        if len(snaive_row) > 0:
            s_wape = snaive_row["wape_global_pct"].values[0]
            s_mae = snaive_row["mae_global"].values[0]
        else:
            s_wape = np.nan
            s_mae = np.nan
            
        for _, r in grp.iterrows():
            row_dict = r.to_dict()
            if not np.isnan(s_wape) and s_wape > 0:
                row_dict["mejora_wape_vs_snaive_pct"] = round((s_wape - r["wape_global_pct"]) / s_wape * 100, 2)
                row_dict["mejora_mae_vs_snaive_pct"] = round((s_mae - r["mae_global"]) / s_mae * 100, 2)
            else:
                row_dict["mejora_wape_vs_snaive_pct"] = 0.0
                row_dict["mejora_mae_vs_snaive_pct"] = 0.0
            enriched_rows.append(row_dict)
            
    df_all_metrics = pd.DataFrame(enriched_rows)
    
    # Separar tablas de validación y prueba
    val_table = df_all_metrics[df_all_metrics["particion"] == "val"].sort_values(by=["ambito", "horizonte_h", "wape_global_pct"])
    test_table = df_all_metrics[df_all_metrics["particion"] == "test"].sort_values(by=["ambito", "horizonte_h", "wape_global_pct"])
    prov_table = pd.DataFrame(prov_metrics)
    
    val_file = os.path.join("reports", "fase2", "resultados_validacion.csv")
    test_file = os.path.join("reports", "fase2", "resultados_prueba_2025.csv")
    prov_file = os.path.join("reports", "fase2", "resultados_por_provincia.csv")
    
    val_table.to_csv(val_file, index=False, encoding="utf-8-sig")
    test_table.to_csv(test_file, index=False, encoding="utf-8-sig")
    prov_table.to_csv(prov_file, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 80)
    print("RESULTADOS EN CONJUNTO DE PRUEBA INDEPENDIENTE 2025 (OUT-OF-SAMPLE):")
    print("=" * 80)
    print(test_table[["ambito", "horizonte_h", "modelo", "mae_global", "rmse_global", "wape_global_pct", "bias_global", "mejora_wape_vs_snaive_pct", "cobertura_ic_95_pct"]].to_string(index=False))
    
    print(f"\nTablas exportadas en reports/fase2/:")
    print(f"  - {val_file}")
    print(f"  - {test_file}")
    print(f"  - {prov_file}")
    
    # Guardar dataframe completo de predicciones evaluables para módulo de alertas
    df_pred_final.to_parquet(os.path.join("data", "processed", "predicciones_completas_fase2.parquet"), index=False)
    
    return val_table, test_table, prov_table, df_pred_final

if __name__ == "__main__":
    main()
