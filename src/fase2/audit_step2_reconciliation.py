"""
Auditoría Paso 2: Reconciliación de MAE, RMSE, WAPE y Totales Observados
"""

import os
import numpy as np
import pandas as pd

# Cargar archivos relevantes
pred_path = os.path.join("reports", "fase2", "predicciones_2025.csv")
res_test_path = os.path.join("reports", "fase2", "resultados_prueba_2025.csv")
prov_res_path = os.path.join("reports", "fase2", "resultados_por_provincia.csv")
panel_prov_path = os.path.join("data", "processed", "panel_semanal_provincial_puno.parquet")
panel_reg_path = os.path.join("data", "processed", "panel_semanal_regional_puno.parquet")

df_preds = pd.read_csv(pred_path)
df_res_test = pd.read_csv(res_test_path)
df_prov_res = pd.read_csv(prov_res_path)
df_panel_prov = pd.read_parquet(panel_prov_path)
df_panel_reg = pd.read_parquet(panel_reg_path)

print("=== 1. AUDITORÍA ESPECÍFICA LIGHTGBM GLOBAL (PROVINCIAL, h=1, TEST 2025) ===")
sub = df_preds[
    (df_preds["ambito"] == "PROVINCIAL") &
    (df_preds["horizonte_h"] == 1) &
    (df_preds["modelo"] == "LightGBM Global") &
    (df_preds["particion"] == "test")
].copy()

print(f"Total filas en predicciones_2025.csv para LightGBM Global h=1: {len(sub)}")
print(f"Distribución de es_evaluable: {sub['es_evaluable'].value_counts().to_dict()}")
print(f"Valores nulos en y_true: {sub['y_true'].isna().sum()}")
print(f"Valores nulos en y_pred: {sub['y_pred'].isna().sum()}")

# Filtrar evaluables
sub_eval = sub[(sub["es_evaluable"] == True) & (sub["y_true"].notna()) & (sub["y_pred"].notna())].copy()
N = len(sub_eval)
sum_y_true = sub_eval["y_true"].sum()
sum_y_pred = sub_eval["y_pred"].sum()
abs_errors = np.abs(sub_eval["y_true"] - sub_eval["y_pred"])
sum_abs_error = abs_errors.sum()
sq_errors = (sub_eval["y_true"] - sub_eval["y_pred"]) ** 2
sum_sq_error = sq_errors.sum()

mae_calc = sum_abs_error / N
rmse_calc = np.sqrt(sum_sq_error / N)
wape_calc = (sum_abs_error / sum_y_true) * 100
bias_calc = (sum_y_pred - sum_y_true) / N # o y_pred - y_true medio

print(f"\nResultados del recálculo directo:")
print(f"  N observaciones evaluadas: {N}")
print(f"  Suma real de episodios observados (sum_y_true): {sum_y_true:,.2f}")
print(f"  Suma de predicciones (sum_y_pred): {sum_y_pred:,.2f}")
print(f"  Suma de errores absolutos: {sum_abs_error:,.2f}")
print(f"  MAE recalculado: {mae_calc:.4f}")
print(f"  RMSE recalculado: {rmse_calc:.4f}")
print(f"  WAPE recalculado (%): {wape_calc:.4f}%")
print(f"  Sesgo recalculado (y_pred - y_true): {bias_calc:.4f}")

# Comprobación de identidad matemática:
# WAPE (%) = 100 * (MAE * N) / sum_y_true
wape_identity = 100 * (mae_calc * N) / sum_y_true
print(f"\nComprobación de la identidad matemática WAPE = 100 * MAE * N / sum_y_true:")
print(f"  100 * ({mae_calc:.4f} * {N}) / {sum_y_true:.2f} = {wape_identity:.4f}%")
print(f"  Diferencia absoluta con wape_calc: {abs(wape_calc - wape_identity):.8f}")

# Comparar con lo publicado en resultados_prueba_2025.csv
row_pub = df_res_test[
    (df_res_test["ambito"] == "PROVINCIAL") &
    (df_res_test["horizonte_h"] == 1) &
    (df_res_test["modelo"] == "LightGBM Global")
].iloc[0]

print(f"\nComparación con resultados_prueba_2025.csv publicado:")
print(f"  MAE publicado: {row_pub['mae_global']} vs Recalculado: {mae_calc:.3f}")
print(f"  RMSE publicado: {row_pub['rmse_global']} vs Recalculado: {rmse_calc:.3f}")
print(f"  WAPE (%) publicado: {row_pub['wape_global_pct']} vs Recalculado: {wape_calc:.3f}%")
print(f"  Sesgo publicado: {row_pub['bias_global']} vs Recalculado: {bias_calc:.3f}")

print("\n=== 2. TOTALES REALES DEL PANEL LIMPIO EN 2025 ===")
panel_2025_prov = df_panel_prov[df_panel_prov["ano"] == 2025].copy()
panel_2025_reg = df_panel_reg[df_panel_reg["ano"] == 2025].copy()

total_prov_2025 = panel_2025_prov["ira_no_neumonia"].sum()
total_reg_2025 = panel_2025_reg["ira_no_neumonia"].sum()
print(f"Suma de episodios en panel provincial 2025: {total_prov_2025:,.2f}")
print(f"Suma de episodios en panel regional 2025: {total_reg_2025:,.2f}")
print(f"Diferencia suma provincias vs regional 2025: {total_prov_2025 - total_reg_2025:.2f}")

print(f"\nTotal observaciones panel provincial 2025: {len(panel_2025_prov)} (13 prov * 53 semanas = 689)")
print(f"Observaciones evaluables en panel provincial 2025: {panel_2025_prov['es_evaluable'].sum()}")
print(f"Observaciones NaN en panel provincial 2025: {panel_2025_prov['ira_no_neumonia'].isna().sum()}")

# Ahora desglosamos por provincia en 2025
prov_totals = panel_2025_prov.groupby("provincia", observed=False).agg(
    total_episodios=("ira_no_neumonia", "sum"),
    n_semanas=("semana", "count"),
    n_evaluable=("es_evaluable", "sum"),
    n_nan=("ira_no_neumonia", lambda x: x.isna().sum())
).reset_index()

print("\nTotales por provincia en 2025:")
print(prov_totals.to_string(index=False))
