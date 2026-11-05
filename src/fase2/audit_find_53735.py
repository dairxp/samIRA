import pandas as pd

df_prov = pd.read_parquet("data/processed/panel_semanal_provincial_puno.parquet")
df_reg = pd.read_parquet("data/processed/panel_semanal_regional_puno.parquet")
df_prov_res = pd.read_csv("reports/fase2/resultados_por_provincia.csv")
df_preds = pd.read_csv("reports/fase2/predicciones_2025.csv")

print("=== TOTALES ANUALES HISTÓRICOS (PROVINCIAL) ===")
for y in range(2018, 2026):
    sub = df_prov[df_prov["ano"] == y]
    tot = sub["ira_no_neumonia"].sum()
    sr = sub[sub["provincia"] == "SAN ROMAN"]["ira_no_neumonia"].sum()
    pu = sub[sub["provincia"] == "PUNO"]["ira_no_neumonia"].sum()
    print(f"Año {y}: Total = {tot:,.0f} | San Román = {sr:,.0f} | Puno = {pu:,.0f}")

print("\n=== VERIFICAR TOTALES EN RESULTADOS_POR_PROVINCIA.CSV ===")
# En resultados_por_provincia.csv:
# Ver qué métricas están reportadas para cada provincia en test
test_prov = df_prov_res[(df_prov_res["particion"] == "test") & (df_prov_res["horizonte_h"] == 1)].copy()
print("Modelos en test_prov:", test_prov["modelo"].unique())

lgb_prov = test_prov[test_prov["modelo"] == "LightGBM Global"]
print("\nLightGBM Global por provincia en resultados_por_provincia.csv:")
print(lgb_prov[["provincia", "mae", "rmse", "wape_pct", "bias", "n_semanas"]])

# Qué pasa si calculamos la suma observada de 2025 dividiendo mae * n / (wape / 100)?
print("\nSuma implícita de y_true por provincia a partir de MAE y WAPE en resultados_por_provincia.csv:")
lgb_prov["implied_y_true"] = (lgb_prov["mae"] * lgb_prov["n_semanas"]) / (lgb_prov["wape_pct"] / 100.0)
for _, r in lgb_prov.iterrows():
    real_y = df_prov[(df_prov["ano"] == 2025) & (df_prov["provincia"] == r["provincia"])]["ira_no_neumonia"].sum()
    print(f"{r['provincia']:22s} | Implied: {r['implied_y_true']:8.1f} | Real 2025: {real_y:8.1f} | Diff: {r['implied_y_true'] - real_y:8.1f}")

print(f"Total Implied y_true: {lgb_prov['implied_y_true'].sum():,.1f}")
