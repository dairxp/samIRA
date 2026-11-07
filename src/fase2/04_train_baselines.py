"""
MÓDULO: IMPLEMENTACIÓN Y EVALUACIÓN DE MODELOS BASELINE DE REFERENCIA
Proyecto samIRA - Fase 2
Implementa los tres modelos de referencia obligatorios:
  1. Naive (Último valor observado en el origen del pronóstico).
  2. Seasonal Naive (Alineación por calendario epidemiológico y-1, w+h).
  3. ETS / Holt-Winters (Suavizamiento exponencial con estacionalidad anual).

Se evalúan para horizontes h=1 y h=4 semanas:
  - En Validación (2023–2024)
  - En Prueba Ciega (2025)
Métricas calculadas EXCLUSIVAMENTE donde es_evaluable == True.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ANIOS_SEMANA_53 = {2003, 2008, 2014, 2020, 2025}

def get_official_weeks_in_year(ano: int) -> int:
    return 53 if ano in ANIOS_SEMANA_53 else 52

def compute_metrics(y_true, y_pred):
    """Calcula MAE, RMSE, WAPE (%) y Bias."""
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

def run_baselines_evaluation():
    print("=" * 80)
    print("EVALUACIÓN DE MODELOS BASELINE: NAIVE, SEASONAL NAIVE Y ETS (h=1 y h=4)")
    print("=" * 80)
    
    # 1. Cargar datos provinciales y regionales
    df_prov = pd.read_parquet(os.path.join("data", "processed", "panel_semanal_provincial_puno.parquet"))
    df_reg = pd.read_parquet(os.path.join("data", "processed", "panel_semanal_regional_puno.parquet"))
    
    # Tabla hash global para lookup de Seasonal Naive: (prov/reg, ano, semana) -> valor
    prov_lookup = df_prov.set_index(["provincia", "ano", "semana"])["ira_no_neumonia"].to_dict()
    reg_lookup = df_reg.set_index(["ano", "semana"])["ira_no_neumonia"].to_dict()
    
    baseline_records = []
    detailed_preds = []
    
    # A. MODELOS BASELINE A NIVEL PROVINCIAL (13 Provincias)
    provincias = sorted(df_prov["provincia"].unique())
    horizons = [1, 4]
    partitions = ["val", "test"] # val: 2023-2024, test: 2025
    
    for prov in provincias:
        p_df = df_prov[df_prov["provincia"] == prov].sort_values(by=["ano", "semana"]).reset_index(drop=True)
        # Serie causal con ffill para Naive en caso de que origen t sea NaN
        causal_p_series = p_df["ira_no_neumonia"].ffill()
        
        # Ajustar modelo ETS en partición de entrenamiento (2000-2022)
        # Para ETS necesitamos serie continua sin NaNs: usamos ffill hacia adelante y bfill si inicia con NaN
        train_mask = p_df["ano"] <= 2022
        train_series = p_df.loc[train_mask, "ira_no_neumonia"].ffill().bfill()
        
        try:
            # Modelo Holt-Winters aditivo con amortiguamiento
            ets_model = ExponentialSmoothing(
                train_series.values,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=52,
                initialization_method="estimated"
            ).fit()
            alpha = ets_model.params.get("smoothing_level")
            beta = ets_model.params.get("smoothing_trend")
            gamma = ets_model.params.get("smoothing_seasonal")
            phi = ets_model.params.get("damping_trend", 0.95)
            ets_fitted = True
        except Exception as e:
            print(f"Error ajustando ETS base para provincia {prov}: {e}")
            ets_fitted = False
            
        for h in horizons:
            # Construir predicciones para cada semana objetivo t+h
            for idx in range(len(p_df)):
                origin_idx = idx - h
                if origin_idx < 0:
                    continue
                    
                target_row = p_df.iloc[idx]
                target_year = int(target_row["ano"])
                target_week = int(target_row["semana"])
                target_val = target_row["ira_no_neumonia"]
                is_eval = target_row["es_evaluable"]
                
                # Partición
                if target_year in [2023, 2024]:
                    part = "val"
                elif target_year == 2025:
                    part = "test"
                else:
                    continue
                    
                # 1. Naive (valor en origin_idx)
                pred_naive = causal_p_series.iloc[origin_idx]
                
                # 2. Seasonal Naive: semana equivalente del año anterior (target_year - 1, target_week)
                prev_year = target_year - 1
                lookup_week = 52 if (target_week == 53 and get_official_weeks_in_year(prev_year) == 52) else target_week
                pred_snaive = prov_lookup.get((prov, prev_year, lookup_week), np.nan)
                if np.isnan(pred_snaive):
                    pred_snaive = pred_naive
                    
                # 3. ETS Forecast: predicción propia rolling desde el origen t (sin fuga temporal)
                if ets_fitted:
                    try:
                        s_up_to_origin = p_df.loc[:origin_idx, "ira_no_neumonia"].ffill().bfill().values
                        m_step = ExponentialSmoothing(
                            s_up_to_origin,
                            trend="add",
                            damped_trend=True,
                            seasonal="add",
                            seasonal_periods=52,
                            initialization_method="estimated"
                        ).fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, damping_trend=phi, use_brute=False)
                        fc = m_step.forecast(h)
                        pred_ets = max(0.0, float(np.asarray(fc)[-1]))
                    except Exception as e:
                        # Registro explícito de error sin fallback encubierto
                        pred_ets = np.nan
                else:
                    pred_ets = np.nan
                    
                detailed_preds.append({
                    "ambito": "PROVINCIAL",
                    "entidad": prov,
                    "ano": target_year,
                    "semana": target_week,
                    "ano_semana": target_row["ano_semana"],
                    "horizonte_h": h,
                    "particion": part,
                    "es_evaluable": is_eval,
                    "y_true": target_val,
                    "pred_naive": pred_naive,
                    "pred_snaive": pred_snaive,
                    "pred_ets": pred_ets
                })
                
    # B. MODELOS BASELINE A NIVEL REGIONAL
    reg_df_sorted = df_reg.sort_values(by=["ano", "semana"]).reset_index(drop=True)
    causal_reg_series = reg_df_sorted["ira_no_neumonia"].ffill()
    
    # ETS regional
    train_reg_series = reg_df_sorted.loc[reg_df_sorted["ano"] <= 2022, "ira_no_neumonia"]
    try:
        ets_reg_model = ExponentialSmoothing(
            train_reg_series.values,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=52,
            initialization_method="estimated"
        ).fit()
        alpha_reg = ets_reg_model.params.get("smoothing_level")
        beta_reg = ets_reg_model.params.get("smoothing_trend")
        gamma_reg = ets_reg_model.params.get("smoothing_seasonal")
        phi_reg = ets_reg_model.params.get("damping_trend", 0.95)
        ets_reg_fitted = True
    except Exception as e:
        print(f"Error ajustando ETS regional base: {e}")
        ets_reg_fitted = False
        
    for h in horizons:
        for idx in range(len(reg_df_sorted)):
            origin_idx = idx - h
            if origin_idx < 0:
                continue
            target_row = reg_df_sorted.iloc[idx]
            target_year = int(target_row["ano"])
            target_week = int(target_row["semana"])
            target_val = target_row["ira_no_neumonia"]
            is_eval = target_row["es_evaluable"]
            
            if target_year in [2023, 2024]:
                part = "val"
            elif target_year == 2025:
                part = "test"
            else:
                continue
                
            pred_naive = causal_reg_series.iloc[origin_idx]
            
            prev_year = target_year - 1
            lookup_week = 52 if (target_week == 53 and get_official_weeks_in_year(prev_year) == 52) else target_week
            pred_snaive = reg_lookup.get((prev_year, lookup_week), np.nan)
            if np.isnan(pred_snaive):
                pred_snaive = pred_naive
                
            # ETS Regional Forecast: predicción rolling desde origen t
            if ets_reg_fitted:
                try:
                    s_up_to_origin = reg_df_sorted.loc[:origin_idx, "ira_no_neumonia"].ffill().bfill().values
                    m_step = ExponentialSmoothing(
                        s_up_to_origin,
                        trend="add",
                        damped_trend=True,
                        seasonal="add",
                        seasonal_periods=52,
                        initialization_method="estimated"
                    ).fit(smoothing_level=alpha_reg, smoothing_trend=beta_reg, smoothing_seasonal=gamma_reg, damping_trend=phi_reg, use_brute=False)
                    fc = m_step.forecast(h)
                    pred_ets = max(0.0, float(np.asarray(fc)[-1]))
                except Exception as e:
                    pred_ets = np.nan
            else:
                pred_ets = np.nan
                
            detailed_preds.append({
                "ambito": "REGIONAL",
                "entidad": "PUNO_REGION",
                "ano": target_year,
                "semana": target_week,
                "ano_semana": target_row["ano_semana"],
                "horizonte_h": h,
                "particion": part,
                "es_evaluable": is_eval,
                "y_true": target_val,
                "pred_naive": pred_naive,
                "pred_snaive": pred_snaive,
                "pred_ets": pred_ets
            })
            
    df_all_preds = pd.DataFrame(detailed_preds)
    
    # C. CÁLCULO DE MÉTRICAS CONSOLIDADAS
    # Métricas calculadas ÚNICAMENTE donde es_evaluable == True
    eval_preds = df_all_preds[df_all_preds["es_evaluable"] == True].copy()
    
    results = []
    for ambito in ["PROVINCIAL", "REGIONAL"]:
        for part in ["val", "test"]:
            for h in [1, 4]:
                sub = eval_preds[(eval_preds["ambito"] == ambito) & (eval_preds["particion"] == part) & (eval_preds["horizonte_h"] == h)]
                
                # Evaluación por modelo
                for model_key, model_name in [("pred_naive", "Naive"), ("pred_snaive", "Seasonal Naive"), ("pred_ets", "ETS / Holt-Winters")]:
                    if ambito == "PROVINCIAL":
                        # 1. Métrica global ponderada por volumen
                        m_global = compute_metrics(sub["y_true"].values, sub[model_key].values)
                        
                        # 2. Métrica macro (promedio de las 13 provincias)
                        macro_maes = []
                        macro_wapes = []
                        for prov in provincias:
                            p_sub = sub[sub["entidad"] == prov]
                            m_p = compute_metrics(p_sub["y_true"].values, p_sub[model_key].values)
                            if m_p["n"] > 0:
                                macro_maes.append(m_p["mae"])
                                macro_wapes.append(m_p["wape_pct"])
                        macro_mae = np.mean(macro_maes)
                        macro_wape = np.mean(macro_wapes)
                    else:
                        m_global = compute_metrics(sub["y_true"].values, sub[model_key].values)
                        macro_mae = m_global["mae"]
                        macro_wape = m_global["wape_pct"]
                        
                    results.append({
                        "ambito": ambito,
                        "particion": part,
                        "horizonte_h": h,
                        "modelo": model_name,
                        "mae_global": m_global["mae"],
                        "rmse_global": m_global["rmse"],
                        "wape_global_pct": m_global["wape_pct"],
                        "bias_global": m_global["bias"],
                        "mae_macro": round(macro_mae, 3),
                        "wape_macro_pct": round(macro_wape, 3),
                        "n_evaluable": m_global["n"]
                    })
                    
    df_metrics = pd.DataFrame(results)
    
    # Calcular mejora porcentual respecto a Seasonal Naive dentro de cada grupo
    final_rows = []
    for (amb, part, h), grp in df_metrics.groupby(["ambito", "particion", "horizonte_h"]):
        snaive_wape = grp.loc[grp["modelo"] == "Seasonal Naive", "wape_global_pct"].values[0]
        snaive_mae = grp.loc[grp["modelo"] == "Seasonal Naive", "mae_global"].values[0]
        for _, row in grp.iterrows():
            r = row.to_dict()
            if row["modelo"] == "Seasonal Naive":
                r["mejora_mae_vs_snaive_pct"] = 0.0
                r["mejora_wape_vs_snaive_pct"] = 0.0
            else:
                # Mejora positiva si el error es menor
                r["mejora_mae_vs_snaive_pct"] = round((snaive_mae - row["mae_global"]) / snaive_mae * 100, 2)
                r["mejora_wape_vs_snaive_pct"] = round((snaive_wape - row["wape_global_pct"]) / snaive_wape * 100, 2)
            final_rows.append(r)
            
    df_final = pd.DataFrame(final_rows)
    
    print("\nRESUMEN DE DESEMPEÑO DE MODELOS BASELINE:")
    print(df_final[["ambito", "particion", "horizonte_h", "modelo", "mae_global", "rmse_global", "wape_global_pct", "bias_global", "mejora_wape_vs_snaive_pct"]].to_string(index=False))
    
    # Guardar predicciones base en parquet para comparativa
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    df_all_preds.to_parquet(os.path.join("data", "processed", "predicciones_baselines.parquet"), index=False)
    print("\nPredicciones baseline exportadas a: data/processed/predicciones_baselines.parquet")
    return df_final, df_all_preds

if __name__ == "__main__":
    run_baselines_evaluation()
