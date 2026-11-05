"""
Evaluación rigurosa de alternativas de detección de semanas de alta notificación
Calibración y sintonización estricta sobre Validación 2023-2024.
Evaluación out-of-sample en Prueba 2025.
"""

import pandas as pd
import numpy as np

# Cargar datos
df_prov = pd.read_parquet("data/processed/panel_semanal_provincial_puno.parquet")
df_preds = pd.read_parquet("data/processed/predicciones_completas_fase2.parquet")
lgb1 = df_preds[
    (df_preds["ambito"] == "PROVINCIAL") &
    (df_preds["horizonte_h"] == 1) &
    (df_preds["modelo"] == "LightGBM Global") &
    (df_preds["es_evaluable"] == True)
].copy()

# 1. Umbral P75 Histórico Tradicional (2000-2022)
prov_baseline = df_prov[(df_prov["ano"].between(2000, 2022)) & (df_prov["es_evaluable"] == True)]
p75_hist = prov_baseline.groupby(["provincia", "semana"])["ira_no_neumonia"].quantile(0.75).reset_index().rename(columns={"ira_no_neumonia": "umbral_p75_hist"})

# Imputación semana 53
for p in df_prov["provincia"].unique():
    s52_val = p75_hist[(p75_hist["provincia"] == p) & (p75_hist["semana"] == 52)]["umbral_p75_hist"].values[0]
    s53_idx = p75_hist[(p75_hist["provincia"] == p) & (p75_hist["semana"] == 53)].index
    if len(s53_idx) == 0:
        p75_hist = pd.concat([p75_hist, pd.DataFrame([{"provincia": p, "semana": 53, "umbral_p75_hist": s52_val}])], ignore_index=True)
    elif pd.isna(p75_hist.loc[s53_idx[0], "umbral_p75_hist"]):
        p75_hist.loc[s53_idx[0], "umbral_p75_hist"] = s52_val

# Merge con predicciones
lgb1["provincia"] = lgb1["entidad"]
df_merged = lgb1.merge(p75_hist, on=["provincia", "semana"], how="left")

# Calcular residuos en validación para calibrar cuantiles probabilísticos alternativos
val_df = df_merged[df_merged["particion"] == "val"].copy()
test_df = df_merged[df_merged["particion"] == "test"].copy()

val_df["res"] = val_df["y_true"] - val_df["y_pred"]
test_df["res"] = test_df["y_true"] - test_df["y_pred"]

# Encontrar en validación el cuantil óptimo q que maximiza F1 para la alerta probabilística: (y_pred + q >= umbral)
# Probamos cuantiles de residuos desde 50% (mediana) hasta 97.5%
best_q = None
best_f1_val = -1
records_val = []

y_val_real_alert = (val_df["y_true"] >= val_df["umbral_p75_hist"]).astype(int)

for pct in [50, 60, 70, 75, 80, 85, 90, 95, 97.5]:
    q_val = np.percentile(val_df["res"], pct)
    pred_alert = (val_df["y_pred"] + q_val >= val_df["umbral_p75_hist"]).astype(int)
    
    tp = int(((y_val_real_alert == 1) & (pred_alert == 1)).sum())
    fp = int(((y_val_real_alert == 0) & (pred_alert == 1)).sum())
    fn = int(((y_val_real_alert == 1) & (pred_alert == 0)).sum())
    tn = int(((y_val_real_alert == 0) & (pred_alert == 0)).sum())
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    tasa_fa = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    records_val.append({
        "percentil_residuo": pct, "q_valor": round(q_val, 2),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "tasa_fa": round(tasa_fa, 4)
    })
    if f1 > best_f1_val:
        best_f1_val = f1
        best_q = q_val
        best_pct = pct

print("=== SINTONIZACIÓN DE UMBRAL PROBABILÍSTICO EN VALIDACIÓN (2023-2024) ===")
print(pd.DataFrame(records_val).to_string(index=False))
print(f"\nMejor percentil de residuo en validación: P{best_pct} (q = {best_q:.2f}) con F1 = {best_f1_val:.4f}")

# Ahora evaluar en Test 2025 usando las 3 reglas comparadas:
# Regla 1: Determinista Puntual (y_pred >= P75)
# Regla 2: Probabilística Extrema (y_pred_upper_95 >= P75)
# Regla 3: Probabilística Sintonizada en Validación (y_pred + best_q >= P75)
y_test_real_alert = (test_df["y_true"] >= test_df["umbral_p75_hist"]).astype(int)

test_rules = []
rules_dict = {
    "1. Pronóstico Puntual (Conservador)": test_df["y_pred"] >= test_df["umbral_p75_hist"],
    "2. Límite Superior IC 95% (Extremo)": test_df["y_pred"] + np.percentile(val_df["res"], 97.5) >= test_df["umbral_p75_hist"],
    f"3. Probabilístico Sintonizado en Val (P{best_pct})": test_df["y_pred"] + best_q >= test_df["umbral_p75_hist"]
}

for r_name, p_alert in rules_dict.items():
    pred_bin = p_alert.astype(int)
    tp = int(((y_test_real_alert == 1) & (pred_bin == 1)).sum())
    fp = int(((y_test_real_alert == 0) & (pred_bin == 1)).sum())
    fn = int(((y_test_real_alert == 1) & (pred_bin == 0)).sum())
    tn = int(((y_test_real_alert == 0) & (pred_bin == 0)).sum())
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    tasa_fa = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    test_rules.append({
        "regla": r_name,
        "brotes_reales": tp + fn, "alertas_emitidas": tp + fp,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(prec, 4), "recall_sensibilidad": round(rec, 4),
        "f1_score": round(f1, 4), "tasa_falsa_alarma": round(tasa_fa, 4)
    })

print("\n=== EVALUACIÓN COMPARATIVA EN TEST 2025 (OUT-OF-SAMPLE) ===")
print(pd.DataFrame(test_rules).to_string(index=False))
