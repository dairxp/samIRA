import os
import sys
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

raw_path = os.path.join("data", "raw", "datos_abiertos_vigilancia_iras_2000_2025 PUNO.csv")
df = pd.read_csv(raw_path, encoding="utf-8-sig", dtype={"ubigeo": str})

print("=" * 70)
print("COBERTURA TEMPORAL POR AÑO")
print("=" * 70)

yearly = df.groupby("ano").agg(
    total_filas=("semana", "count"),
    semanas_unicas=("semana", "nunique"),
    semana_min=("semana", "min"),
    semana_max=("semana", "max"),
    ubigeos_unicos=("ubigeo", "nunique"),
    total_ira_no_neumonia=("ira_no_neumonia", "sum"),
    total_neumonias_men5=("neumonias_men5", "sum"),
    total_hosp_men5=("hospitalizados_men5", "sum"),
    total_def_men5=("defunciones_men5", "sum"),
).reset_index()

print(yearly.to_string(index=False))

print("\n" + "=" * 70)
print("AÑO 2025: DETALLE DE SEMANAS Y FECHA DE CORTE")
print("=" * 70)
df_2025 = df[df["ano"] == 2025]
print(f"Filas en 2025: {len(df_2025)}")
print(f"Semanas presentes en 2025: {sorted(df_2025['semana'].unique())}")
print(f"Provincias presentes en 2025: {df_2025['provincia'].nunique()}")
print(f"Distritos presentes en 2025: {df_2025['ubigeo'].nunique()}")

print("\n" + "=" * 70)
print("DETALLE DUPLICADOS Y SUB_REG_NT")
print("=" * 70)
key_cols = ["ano", "semana", "ubigeo"]
key_dup_mask = df.duplicated(subset=key_cols, keep=False)
dup_records = df[key_dup_mask].sort_values(by=key_cols + ["sub_reg_nt"])

print(f"Total registros con clave repetida [ano, semana, ubigeo]: {len(dup_records)}")
num_cols = ["ira_no_neumonia", "neumonias_men5", "neumonias_60mas", "hospitalizados_men5", "hospitalizados_60mas", "defunciones_men5", "defunciones_60mas"]

dup_types = []
for (ano, sem, ubi), group in dup_records.groupby(key_cols):
    is_identical_data = (group[num_cols].nunique() == 1).all()
    subregs = group["sub_reg_nt"].tolist()
    dup_types.append({
        "ano": ano,
        "semana": sem,
        "ubigeo": ubi,
        "provincia": group["provincia"].iloc[0],
        "distrito": group["distrito"].iloc[0],
        "num_filas": len(group),
        "sub_regs": str(subregs),
        "datos_iguales": is_identical_data,
        "ira_valores": group["ira_no_neumonia"].tolist(),
        "neum_men5_valores": group["neumonias_men5"].tolist(),
    })

dup_summary_df = pd.DataFrame(dup_types)
print(f"Clasificación de las {len(dup_summary_df)} claves repetidas:")
print(f"  Claves con datos numéricos idénticos: {(dup_summary_df['datos_iguales']).sum()}")
print(f"  Claves con datos numéricos diferentes: {(~dup_summary_df['datos_iguales']).sum()}")

# Export suspicious duplicates table to reports
os.makedirs("reports", exist_ok=True)
dup_records.to_csv("reports/tabla_duplicados_sospechosos.csv", index=False, encoding="utf-8-sig")
print("Exportado reports/tabla_duplicados_sospechosos.csv")
