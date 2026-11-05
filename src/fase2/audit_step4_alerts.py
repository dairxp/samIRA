"""
Auditoría Paso 4: Auditoría profunda de Alertas Epidemiológicas, Canal Endémico P75 y Matrices de Confusión
"""

import pandas as pd
import numpy as np

df_prov = pd.read_parquet("data/processed/panel_semanal_provincial_puno.parquet")
df_reg = pd.read_parquet("data/processed/panel_semanal_regional_puno.parquet")
df_preds = pd.read_parquet("data/processed/predicciones_completas_fase2.parquet")

# 1. Período histórico 2000-2022
baseline_prov = df_prov[(df_prov["ano"].between(2000, 2022)) & (df_prov["es_evaluable"] == True)].copy()
baseline_reg = df_reg[(df_reg["ano"].between(2000, 2022)) & (df_reg["es_evaluable"] == True)].copy()

# Cálculo del P75 histórico por provincia y semana
p75_prov = baseline_prov.groupby(["provincia", "semana"])["ira_no_neumonia"].quantile(0.75).reset_index()
p75_prov.rename(columns={"ira_no_neumonia": "umbral_p75"}, inplace=True)

# Manejo de semana 53
for p in df_prov["provincia"].unique():
    s52_val = p75_prov[(p75_prov["provincia"] == p) & (p75_prov["semana"] == 52)]["umbral_p75"].values[0]
    s53_idx = p75_prov[(p75_prov["provincia"] == p) & (p75_prov["semana"] == 53)].index
    if len(s53_idx) == 0:
        p75_prov = pd.concat([p75_prov, pd.DataFrame([{"provincia": p, "semana": 53, "umbral_p75": s52_val}])], ignore_index=True)
    elif pd.isna(p75_prov.loc[s53_idx[0], "umbral_p75"]):
        p75_prov.loc[s53_idx[0], "umbral_p75"] = s52_val

p75_prov["entidad"] = p75_prov["provincia"]

# P75 Regional
p75_reg = baseline_reg.groupby("semana")["ira_no_neumonia"].quantile(0.75).reset_index()
p75_reg.rename(columns={"ira_no_neumonia": "umbral_p75"}, inplace=True)
p75_reg["entidad"] = "PUNO_REGION"
s52_reg = p75_reg[p75_reg["semana"] == 52]["umbral_p75"].values[0]
s53_reg_idx = p75_reg[p75_reg["semana"] == 53].index
if len(s53_reg_idx) == 0:
    p75_reg = pd.concat([p75_reg, pd.DataFrame([{"entidad": "PUNO_REGION", "semana": 53, "umbral_p75": s52_reg}])], ignore_index=True)
elif pd.isna(p75_reg.loc[s53_reg_idx[0], "umbral_p75"]):
    p75_reg.loc[s53_reg_idx[0], "umbral_p75"] = s52_reg

thresholds = pd.concat([p75_prov[["entidad", "semana", "umbral_p75"]], p75_reg[["entidad", "semana", "umbral_p75"]]], ignore_index=True)

# 2. Evaluar sobre conjunto de prueba 2025
df_test = df_preds[(df_preds["particion"] == "test") & (df_preds["es_evaluable"] == True)].copy()
df_test = df_test.merge(thresholds, on=["entidad", "semana"], how="left")

# Verificar número real de semanas que superan el umbral en 2025
real_surges_prov = df_test[df_test["ambito"] == "PROVINCIAL"].drop_duplicates(subset=["entidad", "semana"])
n_surges_prov = (real_surges_prov["y_true"] >= real_surges_prov["umbral_p75"]).sum()
total_prov_weeks = len(real_surges_prov)

real_surges_reg = df_test[df_test["ambito"] == "REGIONAL"].drop_duplicates(subset=["entidad", "semana"])
n_surges_reg = (real_surges_reg["y_true"] >= real_surges_reg["umbral_p75"]).sum()

print("=== VERIFICACIÓN DE SEMANAS DE ALTA NOTIFICACIÓN (2025) ===")
print(f"Total semanas-provincia en 2025: {total_prov_weeks}")
print(f"Semanas-provincia reales con y_true >= P75 histórico: {n_surges_prov} ({n_surges_prov/total_prov_weeks*100:.2f}%)")
print(f"Total semanas regionales en 2025: {len(real_surges_reg)}")
print(f"Semanas regionales reales con y_true >= P75 histórico: {n_surges_reg}")

print("\nDetalle de las 20 semanas-provincia que superaron el umbral P75:")
surges_detail = real_surges_prov[real_surges_prov["y_true"] >= real_surges_prov["umbral_p75"]][["entidad", "semana", "y_true", "umbral_p75"]]
print(surges_detail.to_string(index=False))

# 3. Matriz de confusión para cada modelo
print("\n=== MATRIZ DE CONFUSIÓN DETALLADA POR MODELO (TEST 2025) ===")
results = []
for (amb, h, mod, strat), grp in df_test.groupby(["ambito", "horizonte_h", "modelo", "estrategia_horizonte"]):
    y_real_alert = (grp["y_true"] >= grp["umbral_p75"]).astype(int)
    y_pred_alert = (grp["y_pred"] >= grp["umbral_p75"]).astype(int)
    
    tp = int(((y_real_alert == 1) & (y_pred_alert == 1)).sum())
    fp = int(((y_real_alert == 0) & (y_pred_alert == 1)).sum())
    fn = int(((y_real_alert == 1) & (y_pred_alert == 0)).sum())
    tn = int(((y_real_alert == 0) & (y_pred_alert == 0)).sum())
    
    n_tot = len(grp)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    tasa_fa = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    prop_alertas_falsas = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Evaluar también si tiene intervalo superior al 95%
    if grp["y_pred_upper_95"].notna().any():
        y_upper_alert = (grp["y_pred_upper_95"] >= grp["umbral_p75"]).astype(int)
        tp_up = int(((y_real_alert == 1) & (y_upper_alert == 1)).sum())
        fp_up = int(((y_real_alert == 0) & (y_upper_alert == 1)).sum())
        fn_up = int(((y_real_alert == 1) & (y_upper_alert == 0)).sum())
        tn_up = int(((y_real_alert == 0) & (y_upper_alert == 0)).sum())
        rec_up = tp_up / (tp_up + fn_up) if (tp_up + fn_up) > 0 else 0.0
        prec_up = tp_up / (tp_up + fp_up) if (tp_up + fp_up) > 0 else 0.0
    else:
        tp_up, fp_up, fn_up, tn_up, rec_up, prec_up = None, None, None, None, None, None
        
    results.append({
        "ambito": amb, "horizonte_h": h, "modelo": mod, "estrategia": strat,
        "n_total": n_tot, "brotes_reales": tp + fn, "alertas_predichas": tp + fp,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(prec, 4), "recall": round(rec, 4), "especificidad": round(spec, 4),
        "f1": round(f1, 4), "tasa_falsa_alarma": round(tasa_fa, 4),
        "prop_alertas_falsas": round(prop_alertas_falsas, 4),
        "TP_upper95": tp_up, "FP_upper95": fp_up, "recall_upper95": round(rec_up, 4) if rec_up is not None else np.nan
    })

res_alerts_df = pd.DataFrame(results)
print(res_alerts_df.to_string(index=False))
