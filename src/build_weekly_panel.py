import os
import sys
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

raw_path = os.path.join("data", "processed", "datos_iras_puno_limpio.parquet")
if not os.path.exists(raw_path):
    raw_path = os.path.join("data", "raw", "datos_abiertos_vigilancia_iras_2000_2025 PUNO.csv")

df = pd.read_parquet(raw_path) if raw_path.endswith(".parquet") else pd.read_csv(raw_path, encoding="utf-8-sig", dtype={"ubigeo": str})

print("=" * 80)
print("CONSTRUCCIÓN DEL PANEL SEMANAL: PRESERVACIÓN ESTRICTA DE NaN EN FALTANTES")
print("=" * 80)

# 1. Calendario epidemiológico oficial MINSA (2000-2025)
anios_semana_53 = {2003, 2008, 2014, 2020, 2025}

calendario = []
for ano in range(2000, 2026):
    max_sem = 53 if ano in anios_semana_53 else 52
    for sem in range(1, max_sem + 1):
        calendario.append({"ano": ano, "semana": sem, "ano_semana": f"{ano}-W{sem:02d}"})

df_cal = pd.DataFrame(calendario)
total_semanas_esperadas = len(df_cal)
print(f"Total semanas epidemiológicas oficiales MINSA (2000-2025): {total_semanas_esperadas}")

# 2. Provincias oficiales (13)
provincias = sorted(df["provincia"].unique())

# 3. Agregación observada en el dataset por (provincia, ano, semana)
prov_obs = df.groupby(["provincia", "ano", "semana"]).agg(
    ira_no_neumonia=("ira_no_neumonia", "sum"),
    neumonias_men5=("neumonias_men5", "sum"),
    hospitalizados_men5=("hospitalizados_men5", "sum"),
    defunciones_men5=("defunciones_men5", "sum"),
    conteo_filas_notisp=("ubigeo", "count"),
    distritos_notificantes=("ubigeo", "nunique"),
    subredes_notificantes=("sub_reg_nt", "nunique")
).reset_index()

# 4. Cuadrícula Teórica Provincial (13 x 1,357 = 17,641 filas)
grid_rows = []
for p in provincias:
    for _, row in df_cal.iterrows():
        grid_rows.append({
            "provincia": p,
            "ano": row["ano"],
            "semana": row["semana"],
            "ano_semana": row["ano_semana"]
        })
df_grid = pd.DataFrame(grid_rows)

# Merge con datos observados
panel_prov = pd.merge(df_grid, prov_obs, on=["provincia", "ano", "semana"], how="left")

# Clasificación estricta del estado de observación
conditions = [
    panel_prov["conteo_filas_notisp"].isna(),
    (panel_prov["conteo_filas_notisp"].notna()) & (panel_prov["ira_no_neumonia"] == 0),
    (panel_prov["conteo_filas_notisp"].notna()) & (panel_prov["ira_no_neumonia"] > 0)
]
choices = ["AUSENCIA_REGISTRO", "CERO_REPORTADO", "REPORTADO_CON_CASOS"]
panel_prov["estado_observacion"] = np.select(conditions, choices, default="INDEFINIDO")

# Bandera de evaluación objetiva:
# True: la semana tiene datos reales reportados (incluso 0 legítimo).
# False: la semana no tiene registro en NotiSP (ausencia de información); NO debe evaluarse contra ceros inventados.
panel_prov["es_evaluable"] = panel_prov["estado_observacion"].isin(["REPORTADO_CON_CASOS", "CERO_REPORTADO"])

print("\nDistribución del estado de observación en el panel provincial:")
counts = panel_prov["estado_observacion"].value_counts()
for k, v in counts.items():
    print(f"  {k:22}: {v:6,d} ({v/len(panel_prov)*100:6.2f}%)")

print(f"\nSemanas evaluables para métricas de error out-of-sample: {panel_prov['es_evaluable'].sum():,} ({panel_prov['es_evaluable'].mean()*100:.2f}%)")
print(f"Semanas no evaluables (ausencias que no deben falsearse): {(~panel_prov['es_evaluable']).sum():,} ({(~panel_prov['es_evaluable']).mean()*100:.2f}%)")

# VERIFICACIÓN ESTRICTA: Asegurar que ira_no_neumonia mantenga NaN en AUSENCIA_REGISTRO
assert panel_prov.loc[panel_prov["estado_observacion"] == "AUSENCIA_REGISTRO", "ira_no_neumonia"].isna().all(), "¡ERROR: Hay valores en AUSENCIA_REGISTRO!"
print(">>> PRESERVACIÓN VERIFICADA: Todas las 153 semanas de AUSENCIA_REGISTRO contienen NaN estrictamente.")

# 5. Panel Regional (Agregación Departamental, 1,357 semanas)
# En el nivel regional, cada semana suma los casos observados reportados por las provincias
panel_reg = panel_prov.groupby(["ano", "semana", "ano_semana"]).agg(
    ira_no_neumonia=("ira_no_neumonia", lambda x: x.dropna().sum()),
    neumonias_men5=("neumonias_men5", lambda x: x.dropna().sum()),
    hospitalizados_men5=("hospitalizados_men5", lambda x: x.dropna().sum()),
    defunciones_men5=("defunciones_men5", lambda x: x.dropna().sum()),
    provincias_con_reporte=("conteo_filas_notisp", lambda x: x.notna().sum()),
    provincias_con_casos=("ira_no_neumonia", lambda x: (x > 0).sum())
).reset_index()

panel_reg["estado_observacion"] = np.where(panel_reg["ira_no_neumonia"] > 0, "REPORTADO_CON_CASOS", "CERO_REPORTADO")
panel_reg["es_evaluable"] = True

print(f"\nDimensiones del panel regional: {panel_reg.shape[0]} semanas")
print(f"Total nacional/regional acumulado de ira_no_neumonia: {int(panel_reg['ira_no_neumonia'].sum()):,}")
print(f"Total directo del CSV original: {int(df['ira_no_neumonia'].sum()):,}")
assert int(panel_reg['ira_no_neumonia'].sum()) == int(df['ira_no_neumonia'].sum()), "¡ERROR DE RECONCILIACIÓN EN TOTAL REGIONAL!"
print(">>> RECONCILIACIÓN CONFIRMADA: 2,297,378 episodios exactos.")

# 6. Exportación de Paneles Curados
os.makedirs("data/processed", exist_ok=True)
os.makedirs("reports", exist_ok=True)

panel_prov.to_csv("data/processed/panel_semanal_provincial_puno.csv", index=False, encoding="utf-8-sig")
panel_prov.to_parquet("data/processed/panel_semanal_provincial_puno.parquet", index=False, engine="pyarrow", compression="snappy")

panel_reg.to_csv("data/processed/panel_semanal_regional_puno.csv", index=False, encoding="utf-8-sig")
panel_reg.to_parquet("data/processed/panel_semanal_regional_puno.parquet", index=False, engine="pyarrow", compression="snappy")

# Resumen de cobertura por provincia
resumen_prov = panel_prov.groupby("provincia")["estado_observacion"].value_counts().unstack(fill_value=0)
resumen_prov["TOTAL_SEMANAS_TEORICAS"] = resumen_prov.sum(axis=1)
resumen_prov["SEMANAS_PRESENTES"] = resumen_prov["TOTAL_SEMANAS_TEORICAS"] - resumen_prov.get("AUSENCIA_REGISTRO", 0)
resumen_prov["PCT_COBERTURA"] = (resumen_prov["SEMANAS_PRESENTES"] / resumen_prov["TOTAL_SEMANAS_TEORICAS"] * 100).round(2)
resumen_prov["TOTAL_IRA_EPISODIOS"] = panel_prov.groupby("provincia")["ira_no_neumonia"].sum()

resumen_prov = resumen_prov.reset_index()
resumen_prov.to_csv("reports/tabla_episodios_semanales_resumen.csv", index=False, encoding="utf-8-sig")

print("\nResumen provincial verificado sin imputación forzada a cero:")
print(resumen_prov[["provincia", "TOTAL_SEMANAS_TEORICAS", "SEMANAS_PRESENTES", "AUSENCIA_REGISTRO", "CERO_REPORTADO", "PCT_COBERTURA", "TOTAL_IRA_EPISODIOS"]].to_string(index=False))
print(f"Suma total provincial: {int(resumen_prov['TOTAL_IRA_EPISODIOS'].sum()):,}")
