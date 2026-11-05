"""
================================================================================
MÓDULO: COMPARACIÓN EMPÍRICA DE ESTRATEGIAS DE IMPUTACIÓN CAUSAL DE RETARDOS
Proyecto samIRA - Fase 2
================================================================================
Compara en la partición de Validación (2023–2024) cuatro estrategias para el
tratamiento de faltantes en las variables explicativas (retardos):
  A. Retardo estacional (valor de la misma semana del año anterior).
  B. LOCF (último valor observado conocido previo).
  C. Media móvil retrospectiva causal (promedio de las 3 semanas pasadas conocidas).
  D. Tratamiento nativo con NaN y bandera indicadora de ausencia (LightGBM nativo).

Reglas estrictas:
- NINGUNA imputación se realiza sobre la variable objetivo.
- Las métricas se calculan exclusivamente donde es_evaluable == True.
- Se utiliza exclusivamente información previa (causalidad temporal estricta).
- El año 2025 NO interviene en esta comparación.
"""

import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def build_features_with_imputation_strategy(df_input: pd.DataFrame, strategy: str = "A") -> pd.DataFrame:
    """
    Construye variables de retardo (t-1, t-2, t-3, t-4) para cada provincia aplicando
    la estrategia de imputación seleccionada de forma estrictamente causal.
    """
    df = df_input.copy().sort_values(by=["provincia", "ano", "semana"]).reset_index(drop=True)
    
    # Tabla hash para lookup estacional (y-1, w)
    lookup_cal = df.set_index(["provincia", "ano", "semana"])["ira_no_neumonia"].to_dict()
    
    # Construcción por provincia
    processed_dfs = []
    for prov, grp in df.groupby("provincia", sort=False):
        grp = grp.copy().sort_values(by=["ano", "semana"]).reset_index(drop=True)
        raw_series = grp["ira_no_neumonia"].copy()
        
        # 1. Aplicar estrategia de soporte causal si un valor pasado fue NaN
        if strategy == "A":
            # Estrategia A: Retardo estacional histórico (y-1, w)
            clean_series = raw_series.copy()
            for i, val in enumerate(raw_series):
                if np.isnan(val):
                    y = int(grp.loc[i, "ano"])
                    w = int(grp.loc[i, "semana"])
                    target_w = 52 if (w == 53 and y - 1 not in {2003, 2008, 2014, 2020}) else w
                    alt_val = lookup_cal.get((prov, y - 1, target_w), np.nan)
                    if not np.isnan(alt_val):
                        clean_series.iloc[i] = alt_val
                    else:
                        clean_series.iloc[i] = raw_series.iloc[:i].dropna().iloc[-1] if len(raw_series.iloc[:i].dropna()) > 0 else 0
                        
        elif strategy == "B":
            # Estrategia B: LOCF (Last Observation Carried Forward estricto hacia adelante)
            clean_series = raw_series.ffill().fillna(0)
            
        elif strategy == "C":
            # Estrategia C: Media móvil retrospectiva de las últimas 3 observaciones conocidas
            clean_series = raw_series.copy()
            for i, val in enumerate(raw_series):
                if np.isnan(val):
                    past_vals = clean_series.iloc[:i].dropna()
                    if len(past_vals) >= 1:
                        clean_series.iloc[i] = past_vals.tail(3).mean()
                    else:
                        clean_series.iloc[i] = 0
                        
        elif strategy == "D":
            # Estrategia D: Conservar NaN nativo y generar bandera indicadora
            clean_series = raw_series.copy()
            
        else:
            raise ValueError(f"Estrategia desconocida: {strategy}")
            
        # Generar retardos t-1, t-2, t-3, t-4
        for lag in [1, 2, 3, 4]:
            grp[f"lag_{lag}"] = clean_series.shift(lag)
            if strategy == "D":
                grp[f"lag_{lag}_missing"] = grp[f"lag_{lag}"].isna().astype(int)
                
        # Retardo estacional (1 año antes)
        seasonal_lags = []
        for i, row in grp.iterrows():
            y = int(row["ano"])
            w = int(row["semana"])
            target_w = 52 if (w == 53 and y - 1 not in {2003, 2008, 2014, 2020}) else w
            s_val = lookup_cal.get((prov, y - 1, target_w), np.nan)
            seasonal_lags.append(s_val)
        grp["seasonal_lag_1y"] = seasonal_lags
        if strategy != "D":
            grp["seasonal_lag_1y"] = grp["seasonal_lag_1y"].ffill().fillna(0)
        else:
            grp["seasonal_lag_1y_missing"] = grp["seasonal_lag_1y"].isna().astype(int)
            
        # Variables armónicas de calendario
        grp["sin_semana"] = np.sin(2 * np.pi * grp["semana"] / 52.0)
        grp["cos_semana"] = np.cos(2 * np.pi * grp["semana"] / 52.0)
        grp["semana_num"] = grp["semana"].astype(int)
        
        # Variable indicadora pandémica (2020-2022)
        grp["es_pandemia"] = grp["ano"].between(2020, 2022).astype(int)
        
        processed_dfs.append(grp)
        
    final_df = pd.concat(processed_dfs, ignore_index=True)
    # Codificación de provincia
    final_df["provincia_cat"] = final_df["provincia"].astype("category")
    return final_df


def compare_imputation_strategies():
    print("=" * 80)
    print("EVALUACIÓN COMPARATIVA DE ESTRATEGIAS DE IMPUTACIÓN EN VALIDACIÓN (2023-2024)")
    print("=" * 80)
    
    prov_path = os.path.join("data", "processed", "panel_semanal_provincial_puno.parquet")
    df_base = pd.read_parquet(prov_path)
    
    strategies = ["A", "B", "C", "D"]
    strategy_names = {
        "A": "A. Retardo Estacional Homólogo (y-1, w)",
        "B": "B. LOCF (Último Valor Observado Pasado)",
        "C": "C. Media Móvil Retrospectiva Causal",
        "D": "D. Tratamiento Nativo con NaN + Bandera"
    }
    
    results = []
    
    for strat in strategies:
        print(f"\nEntrenando modelo con Estrategia {strategy_names[strat]}...")
        df_strat = build_features_with_imputation_strategy(df_base, strategy=strat)
        
        # Filtro de entrenamiento (2000-2022) y validación (2023-2024)
        # Excluimos año 2000 del entrenamiento para que los retardos estacionales de 1 año estén completos
        train_df = df_strat[(df_strat["ano"].between(2001, 2022)) & (df_strat["es_evaluable"] == True)].copy()
        val_df = df_strat[(df_strat["ano"].between(2023, 2024)) & (df_strat["es_evaluable"] == True)].copy()
        
        feature_cols = [c for c in df_strat.columns if c.startswith("lag_") or c in ["seasonal_lag_1y", "seasonal_lag_1y_missing", "sin_semana", "cos_semana", "semana_num", "es_pandemia", "provincia_cat"]]
        
        X_train = train_df[feature_cols]
        y_train = train_df["ira_no_neumonia"]
        
        X_val = val_df[feature_cols]
        y_val = val_df["ira_no_neumonia"]
        
        model = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        preds_val = model.predict(X_val)
        
        # Métricas calculadas estrictamente sobre semanas evaluables
        mae = mean_absolute_error(y_val, preds_val)
        rmse = root_mean_squared_error(y_val, preds_val)
        wape = np.sum(np.abs(y_val - preds_val)) / np.sum(y_val) * 100
        bias = np.mean(preds_val - y_val)
        
        print(f"  Resultados Validación 2023-2024: MAE = {mae:.2f} | RMSE = {rmse:.2f} | WAPE = {wape:.2f}% | Bias = {bias:+.2f}")
        
        results.append({
            "codigo_estrategia": strat,
            "descripcion_estrategia": strategy_names[strat],
            "filas_train": len(train_df),
            "filas_val_evaluables": len(val_df),
            "mae_val": round(mae, 3),
            "rmse_val": round(rmse, 3),
            "wape_val_pct": round(wape, 3),
            "bias_val": round(bias, 3)
        })
        
    df_res = pd.DataFrame(results)
    
    # Exportar tabla
    out_dir = os.path.join("reports", "fase2")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "comparacion_imputaciones.csv")
    df_res.to_csv(out_file, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 80)
    print("TABLA COMPARATIVA FINAL DE ESTRATEGIAS DE IMPUTACIÓN:")
    print("=" * 80)
    print(df_res[["codigo_estrategia", "descripcion_estrategia", "mae_val", "rmse_val", "wape_val_pct", "bias_val"]].to_string(index=False))
    print(f"\nTabla exportada en: {out_file}")
    
    best_strat = df_res.loc[df_res["wape_val_pct"].idxmin()]
    print(f"\n>>> ESTRATEGIA SELECCIONADA POR VALIDACIÓN: {best_strat['codigo_estrategia']} ({best_strat['descripcion_estrategia']}) con WAPE = {best_strat['wape_val_pct']:.2f}%")
    return df_res

if __name__ == "__main__":
    compare_imputation_strategies()
