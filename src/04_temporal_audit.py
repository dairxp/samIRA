import os
import sys
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

raw_path = os.path.join("data", "raw", "datos_abiertos_vigilancia_iras_2000_2025 PUNO.csv")
df = pd.read_csv(raw_path, encoding="utf-8-sig", dtype={"ubigeo": str})

# Total expected weeks by year in epidemiology
# In epidemiological calendar: 2003, 2008, 2014, 2020, 2025 have 53 weeks; others 52.
expected_weeks = {
    ano: 53 if ano in [2003, 2008, 2014, 2020, 2025] else 52
    for ano in range(2000, 2026)
}

# Coverage by Province and Year
coverage = df.groupby(["provincia", "ano"]).agg(
    semanas_observadas=("semana", "nunique"),
    filas_reportadas=("semana", "count"),
    distritos_reportando=("ubigeo", "nunique"),
    total_ira_no_neumonia=("ira_no_neumonia", "sum"),
    total_neumonias_men5=("neumonias_men5", "sum"),
).reset_index()

coverage["semanas_esperadas"] = coverage["ano"].map(expected_weeks)
coverage["pct_cobertura_semanal"] = (coverage["semanas_observadas"] / coverage["semanas_esperadas"] * 100).round(2)

print("=" * 80)
print("RESUMEN DE COBERTURA SEMANAL POR PROVINCIA (MÍNIMA Y MÁXIMA)")
print("=" * 80)
prov_summary = coverage.groupby("provincia").agg(
    min_cobertura_pct=("pct_cobertura_semanal", "min"),
    media_cobertura_pct=("pct_cobertura_semanal", "mean"),
    max_cobertura_pct=("pct_cobertura_semanal", "max"),
    total_ira=("total_ira_no_neumonia", "sum"),
    total_neum=("total_neumonias_men5", "sum"),
).reset_index()

print(prov_summary.to_string(index=False))

# Check any year-province with less than 100% weekly coverage
incomplete_cov = coverage[coverage["pct_cobertura_semanal"] < 100]
print(f"\nTotal combinaciones provincia-año con menos del 100% de semanas: {len(incomplete_cov)}")
if len(incomplete_cov) > 0:
    print(incomplete_cov[["provincia", "ano", "semanas_observadas", "semanas_esperadas", "pct_cobertura_semanal"]].to_string(index=False))

# Pivot table: Semanas observadas por provincia y año
pivot_coverage = coverage.pivot(index="provincia", columns="ano", values="semanas_observadas")
print("\nPivot: Semanas observadas por provincia y año:")
print(pivot_coverage)

# Export coverage table
os.makedirs("reports", exist_ok=True)
coverage.to_csv("reports/tabla_cobertura_semanal_provincia_ano.csv", index=False, encoding="utf-8-sig")
pivot_coverage.to_csv("reports/tabla_pivote_semanas_provincia_ano.csv", encoding="utf-8-sig")
print("\nExportadas tablas de cobertura a reports/")
