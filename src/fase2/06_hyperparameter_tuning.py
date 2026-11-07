"""
MÓDULO: OPTIMIZACIÓN DE HIPERPARÁMETROS MEDIANTE VALIDACIÓN TEMPORAL (2023-2024)
Proyecto samIRA - Fase 2
"""

import os
import sys
import time
import importlib.util
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Cargar dinámicamente el pipeline de características
spec = importlib.util.spec_from_file_location("feature_pipeline", os.path.join("src", "fase2", "05_feature_pipeline.py"))
feat_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat_mod)
prepare_feature_dataset = feat_mod.prepare_feature_dataset

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

RANDOM_SEED = 42

def compute_wape(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    return np.sum(np.abs(yt - yp)) / np.sum(yt) * 100

def run_hyperparameter_optimization():
    print("=" * 80)
    print("INICIANDO OPTIMIZACIÓN DE HIPERPARÁMETROS EN VALIDACIÓN (2023-2024)")
    print("=" * 80)
    
    start_global_time = time.time()
    
    # Grid de búsqueda razonable y exhaustivo
    lgb_param_grid = [
        {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8},
        {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 31, "subsample": 0.8},
        {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "subsample": 1.0},
        {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "subsample": 0.8},
        {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 63, "subsample": 0.8},
    ]
    
    xgb_param_grid = [
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8},
        {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 4, "subsample": 0.8},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "subsample": 1.0},
        {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 4, "subsample": 0.8},
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8},
    ]
    
    optimal_records = []
    
    for h in [1, 4]:
        print(f"\n--- OPTIMIZACIÓN PARA HORIZONTE h={h} SEMANAS ---")
        df_prov_feat, df_reg_feat = prepare_feature_dataset(horizon=h, is_direct=True)
        
        feature_cols = [
            "lag_1", "lag_2", "lag_3", "lag_4", "lag_8",
            "rolling_mean_4", "rolling_std_4", "rolling_mean_13", "rolling_std_13",
            "seasonal_lag_1y", "sin_semana", "cos_semana", "es_pandemia"
        ]
        
        train_mask = (df_prov_feat["ano"].between(2001, 2022)) & (df_prov_feat["target_es_evaluable"] == True)
        val_mask = (df_prov_feat["target_ano"].between(2023, 2024)) & (df_prov_feat["target_es_evaluable"] == True)
        
        prov_train_df = df_prov_feat[train_mask].copy()
        prov_val_df = df_prov_feat[val_mask].copy()
        
        X_train_lgb = prov_train_df[feature_cols + ["provincia"]]
        X_val_lgb = prov_val_df[feature_cols + ["provincia"]]
        
        X_train_xgb = pd.get_dummies(prov_train_df[feature_cols + ["provincia"]], columns=["provincia"], drop_first=True)
        X_val_xgb = pd.get_dummies(prov_val_df[feature_cols + ["provincia"]], columns=["provincia"], drop_first=True)
        X_train_xgb, X_val_xgb = X_train_xgb.align(X_val_xgb, join="left", axis=1, fill_value=0)
        
        y_train = prov_train_df["target"].values
        y_val = prov_val_df["target"].values
        
        # 1. Búsqueda LightGBM
        print(f"Evaluando {len(lgb_param_grid)} configuraciones para LightGBM (h={h})...")
        best_lgb_wape = 999.0
        best_lgb_params = None
        t0 = time.time()
        for p in lgb_param_grid:
            m = lgb.LGBMRegressor(**p, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)
            m.fit(X_train_lgb, y_train)
            preds = m.predict(X_val_lgb)
            wape = compute_wape(y_val, preds)
            if wape < best_lgb_wape:
                best_lgb_wape = wape
                best_lgb_params = p
                
        lgb_time = time.time() - t0
        print(f"  > Mejor LightGBM h={h}: WAPE={best_lgb_wape:.2f}% | Parámetros: {best_lgb_params} (Tiempo: {lgb_time:.1f}s)")
        
        optimal_records.append({
            "algoritmo": "LightGBM",
            "horizonte_h": h,
            "estrategia": "Directa",
            "configs_evaluadas": len(lgb_param_grid),
            "criterio_seleccion": "Min WAPE (%) en Validación (2023-2024)",
            "wape_validacion_pct": round(best_lgb_wape, 3),
            "parametros_optimos": str(best_lgb_params),
            "tiempo_segundos": round(lgb_time, 2),
            "random_seed": RANDOM_SEED,
            "version_libreria": lgb.__version__
        })
        
        # 2. Búsqueda XGBoost
        print(f"Evaluando {len(xgb_param_grid)} configuraciones para XGBoost (h={h})...")
        best_xgb_wape = 999.0
        best_xgb_params = None
        t0 = time.time()
        for p in xgb_param_grid:
            m = xgb.XGBRegressor(**p, random_state=RANDOM_SEED, verbosity=0, n_jobs=-1)
            m.fit(X_train_xgb, y_train)
            preds = m.predict(X_val_xgb)
            wape = compute_wape(y_val, preds)
            if wape < best_xgb_wape:
                best_xgb_wape = wape
                best_xgb_params = p
                
        xgb_time = time.time() - t0
        print(f"  > Mejor XGBoost h={h}: WAPE={best_xgb_wape:.2f}% | Parámetros: {best_xgb_params} (Tiempo: {xgb_time:.1f}s)")
        
        optimal_records.append({
            "algoritmo": "XGBoost",
            "horizonte_h": h,
            "estrategia": "Directa",
            "configs_evaluadas": len(xgb_param_grid),
            "criterio_seleccion": "Min WAPE (%) en Validación (2023-2024)",
            "wape_validacion_pct": round(best_xgb_wape, 3),
            "parametros_optimos": str(best_xgb_params),
            "tiempo_segundos": round(xgb_time, 2),
            "random_seed": RANDOM_SEED,
            "version_libreria": xgb.__version__
        })
        
    df_opt = pd.DataFrame(optimal_records)
    out_path = os.path.join("reports", "fase2", "hiperparametros_optimos.csv")
    df_opt.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 80)
    print("TABLA DE HIPERPARÁMETROS ÓPTIMOS EXPORTADA:")
    print("=" * 80)
    print(df_opt[["algoritmo", "horizonte_h", "wape_validacion_pct", "parametros_optimos", "tiempo_segundos"]].to_string(index=False))
    print(f"\nArchivo guardado en: {out_path}")
    print(f"Tiempo total de búsqueda: {time.time() - start_global_time:.1f} segundos.")
    return df_opt

if __name__ == "__main__":
    run_hyperparameter_optimization()
