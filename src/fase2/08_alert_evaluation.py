"""
FASE 2 - EVALUACIÓN DE UTILIDAD EN PLANIFICACIÓN SANITARIA Y ALERTAS EPIDEMIOLÓGICAS
Evalúa la capacidad de los modelos predictivos para anticipar semanas de alta notificación
o brotes según la metodología oficial del Canal Endémico (Bortman / CDC-MINSA).

Umbral epidemiológico:
- Zona de Alerta / Epidemia: Percentil 75 histórico de la semana epidemiológica homóloga
  durante el período de referencia basal (2000-2022).
  Se calcula por provincia y a nivel regional sobre semanas históricas evaluables.

Métricas evaluadas sobre el conjunto de PRUEBA (2025 holdout):
- Verdaderos Positivos (TP): Brote ocurrido y anticipado correctamente.
- Falsos Positivos (FP): Falsa alarma (se predijo alerta pero la observación no superó el umbral).
- Falsos Negativos (FN): Alerta perdida (ocurrió alza/brote y no se anticipó).
- Verdaderos Negativos (TN): Semana normal predicha como normal.
- Precisión: TP / (TP + FP)
- Recall (Sensibilidad): TP / (TP + FN)
- F1-Score: Media armónica entre Precisión y Recall.
- Tasa de Falsas Alarmas: FP / (FP + TN) [o porcentaje de falsas alertas sobre predicciones]
- Tasa de Alertas Perdidas: FN / (TP + FN)

Genera:
- reports/fase2/metricas_alertas.csv
"""

import os
import sys
import numpy as np
import pandas as pd

def compute_endemic_thresholds(df_prov, df_reg):
    """
    Calcula el percentil 75 histórico por provincia y semana (2000-2022),
    y a nivel regional.
    """
    # Filtrar período basal 2000-2022 con es_evaluable == True
    prov_baseline = df_prov[(df_prov["ano"].between(2000, 2022)) & (df_prov["es_evaluable"] == True)]
    reg_baseline = df_reg[(df_reg["ano"].between(2000, 2022)) & (df_reg["es_evaluable"] == True)]
    
    # P75 Provincial por provincia y semana
    prov_p75 = prov_baseline.groupby(["provincia", "semana"])["ira_no_neumonia"].quantile(0.75).reset_index()
    prov_p75.rename(columns={"ira_no_neumonia": "umbral_p75"}, inplace=True)
    
    # En caso de semana 53 (que tiene pocas observaciones históricas), imputar con semana 52
    for prov in df_prov["provincia"].unique():
        s52 = prov_p75[(prov_p75["provincia"] == prov) & (prov_p75["semana"] == 52)]["umbral_p75"].values
        val_s52 = s52[0] if len(s52) > 0 else 10.0
        s53_idx = prov_p75[(prov_p75["provincia"] == prov) & (prov_p75["semana"] == 53)].index
        if len(s53_idx) == 0:
            prov_p75 = pd.concat([prov_p75, pd.DataFrame([{
                "provincia": prov, "semana": 53, "umbral_p75": val_s52
            }])], ignore_index=True)
        else:
            if pd.isna(prov_p75.loc[s53_idx[0], "umbral_p75"]):
                prov_p75.loc[s53_idx[0], "umbral_p75"] = val_s52

    # P75 Regional por semana
    reg_p75 = reg_baseline.groupby("semana")["ira_no_neumonia"].quantile(0.75).reset_index()
    reg_p75.rename(columns={"ira_no_neumonia": "umbral_p75"}, inplace=True)
    reg_p75["entidad"] = "PUNO_REGION"
    
    reg_s52 = reg_p75[reg_p75["semana"] == 52]["umbral_p75"].values[0]
    reg_s53_idx = reg_p75[reg_p75["semana"] == 53].index
    if len(reg_s53_idx) == 0:
        reg_p75 = pd.concat([reg_p75, pd.DataFrame([{
            "entidad": "PUNO_REGION", "semana": 53, "umbral_p75": reg_s52
        }])], ignore_index=True)
    else:
        if pd.isna(reg_p75.loc[reg_s53_idx[0], "umbral_p75"]):
            reg_p75.loc[reg_s53_idx[0], "umbral_p75"] = reg_s52
            
    prov_p75["entidad"] = prov_p75["provincia"]
    
    thresholds = pd.concat([
        prov_p75[["entidad", "semana", "umbral_p75"]],
        reg_p75[["entidad", "semana", "umbral_p75"]]
    ], ignore_index=True)
    
    return thresholds

def evaluate_alert_classification(df_preds, df_thresholds, partition="test"):
    """
    Evalúa la matriz de confusión y métricas de clasificación de alertas
    para cada modelo y horizonte en la partición especificada (por defecto 'test' 2025).
    """
    # Filtrar solo observaciones evaluables
    df_eval = df_preds[(df_preds["particion"] == partition) & (df_preds["es_evaluable"] == True)].copy()
    
    # Merge con umbrales por entidad y semana
    df_eval = df_eval.merge(df_thresholds, on=["entidad", "semana"], how="left")
    
    # Banderas binarias de alerta
    df_eval["alerta_real"] = (df_eval["y_true"] >= df_eval["umbral_p75"]).astype(int)
    df_eval["alerta_pred"] = (df_eval["y_pred"] >= df_eval["umbral_p75"]).astype(int)
    
    results = []
    
    group_cols = ["ambito", "horizonte_h", "estrategia_horizonte", "modelo"]
    for keys, grp in df_eval.groupby(group_cols):
        ambito, h, strat, model = keys
        
        tp = int(((grp["alerta_real"] == 1) & (grp["alerta_pred"] == 1)).sum())
        fp = int(((grp["alerta_real"] == 0) & (grp["alerta_pred"] == 1)).sum())
        fn = int(((grp["alerta_real"] == 1) & (grp["alerta_pred"] == 0)).sum())
        tn = int(((grp["alerta_real"] == 0) & (grp["alerta_pred"] == 0)).sum())
        
        total_alertas_reales = tp + fn
        total_alertas_pred = tp + fp
        total_obs = len(grp)
        
        precision = tp / total_alertas_pred if total_alertas_pred > 0 else 0.0
        recall = tp / total_alertas_reales if total_alertas_reales > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        tasa_falsa_alarma = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tasa_alerta_perdida = fn / total_alertas_reales if total_alertas_reales > 0 else 0.0
        exactitud = (tp + tn) / total_obs if total_obs > 0 else 0.0
        
        results.append({
            "ambito": ambito,
            "horizonte_h": h,
            "estrategia_horizonte": strat,
            "modelo": model,
            "n_total": total_obs,
            "brotes_reales": total_alertas_reales,
            "brotes_predichos": total_alertas_pred,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": round(precision, 4),
            "recall_sensibilidad": round(recall, 4),
            "f1_score": round(f1, 4),
            "exactitud": round(exactitud, 4),
            "tasa_falsa_alarma": round(tasa_falsa_alarma, 4),
            "tasa_alerta_perdida": round(tasa_alerta_perdida, 4)
        })
        
    df_metrics = pd.DataFrame(results).sort_values(by=["ambito", "horizonte_h", "f1_score"], ascending=[True, True, False])
    return df_metrics, df_eval

def main():
    print("=" * 80)
    print("EVALUACIÓN DE UTILIDAD EN ALERTAS SANITARIAS - FASE 2")
    print("=" * 80)
    
    prov_path = os.path.join("data", "processed", "panel_semanal_provincial_puno.parquet")
    reg_path = os.path.join("data", "processed", "panel_semanal_regional_puno.parquet")
    preds_path = os.path.join("data", "processed", "predicciones_completas_fase2.parquet")
    
    if not os.path.exists(preds_path):
        print(f"ERROR: No se encontró el archivo de predicciones: {preds_path}")
        sys.exit(1)
        
    df_prov = pd.read_parquet(prov_path)
    df_reg = pd.read_parquet(reg_path)
    df_preds = pd.read_parquet(preds_path)
    
    print(f"Cargadas {len(df_preds):,} predicciones completas.")
    
    # Calcular umbrales P75 históricos (2000-2022)
    thresholds = compute_endemic_thresholds(df_prov, df_reg)
    print(f"Umbrales calculados para {thresholds['entidad'].nunique()} entidades.")
    
    # Evaluar alertas en conjunto de prueba 2025
    df_alert_metrics, df_eval_detail = evaluate_alert_classification(df_preds, thresholds, partition="test")
    
    os.makedirs(os.path.join("reports", "fase2"), exist_ok=True)
    out_csv = os.path.join("reports", "fase2", "metricas_alertas.csv")
    df_alert_metrics.to_csv(out_csv, index=False)
    print(f"\n[OK] Métricas de alerta guardadas exitosamente en: {out_csv}")
    
    # Mostrar tabla resumen para horizonte h=1 y h=4
    print("\n--- RESUMEN DE ALERTAS SANITARIAS EN TEST (2025) ---")
    print(df_alert_metrics[["ambito", "horizonte_h", "modelo", "brotes_reales", "TP", "FP", "FN", "precision", "recall_sensibilidad", "f1_score", "tasa_falsa_alarma"]].to_string(index=False))

if __name__ == "__main__":
    main()
