import os
import sys
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

raw_path = os.path.join("data", "raw", "datos_abiertos_vigilancia_iras_2000_2025 PUNO.csv")
df = pd.read_csv(raw_path, encoding="utf-8-sig", dtype={"ubigeo": str})

print("=" * 60)
print("ESTRUCTURA GENERAL DEL DATASET")
print("=" * 60)
print(f"Filas: {df.shape[0]:,}")
print(f"Columnas: {df.shape[1]}")
print(f"Memoria consumida: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

print("\n" + "=" * 60)
print("COLUMNAS Y TIPOS DE DATOS")
print("=" * 60)
for col in df.columns:
    print(f"  {col:25} | Dtype: {str(df[col].dtype):10} | Nulos: {df[col].isnull().sum():6} ({df[col].isnull().mean()*100:.2f}%)")

print("\n" + "=" * 60)
print("VALIDACIÓN DE VALORES NULOS Y CADENAS VACÍAS")
print("=" * 60)
str_cols = ["departamento", "provincia", "distrito", "ubigeo"]
for col in str_cols:
    blanks = df[col].astype(str).str.strip().eq("").sum()
    print(f"  {col:25} | Cadenas vacías o espacios: {blanks}")

print("\n" + "=" * 60)
print("VALIDACIÓN DE UBIGEO")
print("=" * 60)
df["ubigeo_len"] = df["ubigeo"].astype(str).str.len()
print("Distribución de longitudes de UBIGEO:")
print(df["ubigeo_len"].value_counts())
non_digits = (~df["ubigeo"].astype(str).str.isdigit()).sum()
print(f"UBIGEOs no numéricos: {non_digits}")
not_puno_prefix = (~df["ubigeo"].astype(str).str.startswith("21")).sum()
print(f"UBIGEOs que no inician con '21' (código Puno): {not_puno_prefix}")
unique_ubigeos = df["ubigeo"].nunique()
print(f"Total UBIGEOs únicos en el dataset: {unique_ubigeos}")

print("\n" + "=" * 60)
print("VALIDACIÓN TEMPORAL")
print("=" * 60)
print(f"Rango de años: {df['ano'].min()} a {df['ano'].max()}")
out_years = ((df["ano"] < 2000) | (df["ano"] > 2025)).sum()
print(f"Años fuera del rango [2000, 2025]: {out_years}")
print(f"Rango de semanas epidemiológicas: {df['semana'].min()} a {df['semana'].max()}")
out_weeks = ((df["semana"] < 1) | (df["semana"] > 53)).sum()
print(f"Semanas fuera del rango [1, 53]: {out_weeks}")

print("\n" + "=" * 60)
print("VALIDACIÓN DE VARIABLES NUMÉRICAS (EPIDEMIOLÓGICAS)")
print("=" * 60)
num_cols = [
    "ira_no_neumonia",
    "neumonias_men5",
    "neumonias_60mas",
    "hospitalizados_men5",
    "hospitalizados_60mas",
    "defunciones_men5",
    "defunciones_60mas",
]

stats_list = []
for col in num_cols:
    s = df[col]
    neg_cnt = (s < 0).sum()
    zero_cnt = (s == 0).sum()
    stats_list.append({
        "Variable": col,
        "Min": int(s.min()),
        "Q25": float(s.quantile(0.25)),
        "Mediana": float(s.median()),
        "Media": float(s.mean()),
        "Q75": float(s.quantile(0.75)),
        "Max": int(s.max()),
        "Negativos": int(neg_cnt),
        "Ceros": int(zero_cnt),
        "Pct_Ceros": float(zero_cnt / len(df) * 100),
    })

stats_df = pd.DataFrame(stats_list)
print(stats_df.to_string(index=False))

print("\n" + "=" * 60)
print("CONSISTENCIA LÓGICA ENTRE INDICADORES")
print("=" * 60)
hosp_gt_pneum_men5 = (df["hospitalizados_men5"] > df["neumonias_men5"]).sum()
def_gt_pneum_men5 = (df["defunciones_men5"] > df["neumonias_men5"]).sum()
hosp_gt_pneum_60mas = (df["hospitalizados_60mas"] > df["neumonias_60mas"]).sum()
def_gt_pneum_60mas = (df["defunciones_60mas"] > df["neumonias_60mas"]).sum()

print(f"Casos donde hospitalizados_men5 > neumonias_men5: {hosp_gt_pneum_men5:,} ({hosp_gt_pneum_men5/len(df)*100:.2f}%)")
print(f"Casos donde defunciones_men5 > neumonias_men5: {def_gt_pneum_men5:,} ({def_gt_pneum_men5/len(df)*100:.2f}%)")
print(f"Casos donde hospitalizados_60mas > neumonias_60mas: {hosp_gt_pneum_60mas:,} ({hosp_gt_pneum_60mas/len(df)*100:.2f}%)")
print(f"Casos donde defunciones_60mas > neumonias_60mas: {def_gt_pneum_60mas:,} ({def_gt_pneum_60mas/len(df)*100:.2f}%)")
