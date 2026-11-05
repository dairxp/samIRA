import sys
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

raw_path = "data/raw/datos_abiertos_vigilancia_iras_2000_2025 PUNO.csv"
df = pd.read_csv(raw_path, encoding="utf-8-sig", dtype={"ubigeo": str})

# 1. Check weeks present in dataset per year
yearly_weeks = df.groupby("ano")["semana"].agg(["min", "max", "nunique"]).reset_index()
print("Semanas presentes en el dataset por año:")
print(yearly_weeks.to_string(index=False))

# Check which years have week 53 in MINSA official epidemiological calendars
# In epidemiology (CDC Peru / DGE), week 53 occurs in years where the 53rd week satisfies the CDC/WHO rule.
# In our dataset:
# 2003: 53 weeks
# 2008: 53 weeks
# 2014: 53 weeks
# 2020: 53 weeks
# 2025: 53 weeks
# Total years = 26 (2000 to 2025).
# 21 years with 52 weeks = 1,092 weeks.
# 5 years with 53 weeks = 265 weeks.
# Total expected weeks across 2000-2025 = 1,092 + 265 = 1,357 weeks.

# Let's verify whether the regional series has all 1,357 weeks present:
regional_weeks = df.groupby(["ano", "semana"]).size().reset_index()
print(f"\nTotal combinaciones (ano, semana) presentes a nivel REGIONAL: {len(regional_weeks)}")
print(f"Total esperado a nivel regional: 1,357")
print(f"Semanas faltantes a nivel regional: {1357 - len(regional_weeks)}")

# 2. Check provincial level
provinces = sorted(df["provincia"].unique())
print(f"\nNúmero de provincias: {len(provinces)}")

# Theoretical maximum
total_teorico = len(regional_weeks) * len(provinces)
print(f"Total teórico de pares (provincia, ano, semana) [1,357 semanas x 13 provincias]: {total_teorico:,}")

# Actual observed combinations of (provincia, ano, semana)
prov_weekly = df.groupby(["provincia", "ano", "semana"]).size().reset_index()
total_observado = len(prov_weekly)
total_faltante = total_teorico - total_observado
pct_cobertura_global = (total_observado / total_teorico) * 100

print(f"Total pares observados (presentes): {total_observado:,}")
print(f"Total pares faltantes (sin registro): {total_faltante:,}")
print(f"Porcentaje de cobertura global provincial: {pct_cobertura_global:.4f}% ({pct_cobertura_global:.2f}%)")

# Detailed breakdown per province
prov_breakdown = []
for p in provinces:
    sub = df[df["provincia"] == p]
    obs = sub.groupby(["ano", "semana"]).size().shape[0]
    faltantes = 1357 - obs
    pct = (obs / 1357) * 100
    prov_breakdown.append({
        "provincia": p,
        "semanas_esperadas": 1357,
        "semanas_observadas": obs,
        "semanas_faltantes": faltantes,
        "pct_cobertura": round(pct, 2)
    })

df_prov_cov = pd.DataFrame(prov_breakdown)
print("\nDesglose exacto de cobertura por provincia (2000-2025):")
print(df_prov_cov.to_string(index=False))
