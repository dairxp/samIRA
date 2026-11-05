import os
import sys
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

raw_path = os.path.join("data", "raw", "datos_abiertos_vigilancia_iras_2000_2025 PUNO.csv")
df = pd.read_csv(raw_path, encoding="utf-8-sig", dtype={"ubigeo": str})

print("=" * 70)
print("INICIANDO LIMPIEZA JUSTIFICADA Y EXPORTACIÓN")
print("=" * 70)

# 1. Audit missing / invalid values
quality_rows = []
for col in df.columns:
    n_null = int(df[col].isnull().sum())
    pct_null = float(n_null / len(df) * 100)
    
    # Blanks / empty strings
    if df[col].dtype == object or str(df[col].dtype) == "str":
        n_blank = int(df[col].astype(str).str.strip().eq("").sum())
        n_invalid = int(df[col].astype(str).str.contains("\ufffd").sum())
    else:
        n_blank = 0
        n_invalid = 0
    
    # Range / negative validation
    if col in ["ira_no_neumonia", "neumonias_men5", "neumonias_60mas", "hospitalizados_men5", "hospitalizados_60mas", "defunciones_men5", "defunciones_60mas"]:
        n_neg = int((df[col] < 0).sum())
        n_zeros = int((df[col] == 0).sum())
        pct_zeros = float(n_zeros / len(df) * 100)
    elif col == "ano":
        n_neg = int(((df[col] < 2000) | (df[col] > 2025)).sum())
        n_zeros = 0
        pct_zeros = 0.0
    elif col == "semana":
        n_neg = int(((df[col] < 1) | (df[col] > 53)).sum())
        n_zeros = 0
        pct_zeros = 0.0
    else:
        n_neg = 0
        n_zeros = 0
        pct_zeros = 0.0

    quality_rows.append({
        "variable": col,
        "tipo_dato": str(df[col].dtype),
        "total_registros": len(df),
        "nulos": n_null,
        "pct_nulos": round(pct_null, 4),
        "cadenas_vacias": n_blank,
        "caracteres_invalidos": n_invalid,
        "valores_fuera_rango_o_negativos": n_neg,
        "conteo_ceros": n_zeros,
        "pct_ceros": round(pct_zeros, 2)
    })

quality_df = pd.DataFrame(quality_rows)
os.makedirs("reports", exist_ok=True)
quality_df.to_csv("reports/tabla_valores_faltantes_invalidos.csv", index=False, encoding="utf-8-sig")
print("Guardada tabla_valores_faltantes_invalidos.csv en reports/")

# 2. Document transformations Before/After
transform_log = []

# Replacement of damaged district names
districts_corrupted_before = df["distrito"].str.contains("\ufffd").sum()
corrections_map = {
    "210208": "MUÑANI",
    "210806": "NUÑOA",
    "210109": "MAÑAZO"
}

df_clean = df.copy()

for ubi, correct_name in corrections_map.items():
    old_name = df_clean.loc[df_clean["ubigeo"] == ubi, "distrito"].iloc[0]
    count = (df_clean["ubigeo"] == ubi).sum()
    df_clean.loc[df_clean["ubigeo"] == ubi, "distrito"] = correct_name
    transform_log.append({
        "campo": "distrito",
        "ubigeo": ubi,
        "valor_anterior": old_name,
        "valor_nuevo": correct_name,
        "filas_afectadas": count,
        "justificacion": f"Reparación de caracter de sustitución (U+FFFD) basado en catálogo oficial UBIGEO INEI ({correct_name})"
    })

districts_corrupted_after = df_clean["distrito"].str.contains("\ufffd").sum()
print(f"Distritos con caracteres inválidos: Antes={districts_corrupted_before}, Después={districts_corrupted_after}")

# Format UBIGEO as 6-character string with leading zero if needed
df_clean["ubigeo"] = df_clean["ubigeo"].astype(str).str.strip().str.zfill(6)
transform_log.append({
    "campo": "ubigeo",
    "ubigeo": "TODOS",
    "valor_anterior": "object / int inferido",
    "valor_nuevo": "string (6 caracteres, zfill)",
    "filas_afectadas": len(df_clean),
    "justificacion": "Preservar formato estándar UBIGEO de 6 dígitos requerido para uniones cartográficas INEI"
})

# Add temporal identifier ano_semana
df_clean["ano_semana"] = df_clean["ano"].astype(str) + "-W" + df_clean["semana"].astype(str).str.zfill(2)
transform_log.append({
    "campo": "ano_semana",
    "ubigeo": "TODOS",
    "valor_anterior": "No existía",
    "valor_nuevo": "YYYY-Www (string)",
    "filas_afectadas": len(df_clean),
    "justificacion": "Identificador temporal estandarizado para modelado de series de tiempo epidemiológicas"
})

transform_df = pd.DataFrame(transform_log)
transform_df.to_csv("reports/registro_transformaciones.csv", index=False, encoding="utf-8-sig")
print("Guardado registro_transformaciones.csv en reports/")

# 3. Export Clean Dataset to CSV UTF-8-SIG and Parquet
os.makedirs("data/processed", exist_ok=True)
csv_out = os.path.join("data", "processed", "datos_iras_puno_limpio.csv")
parquet_out = os.path.join("data", "processed", "datos_iras_puno_limpio.parquet")

df_clean.to_csv(csv_out, index=False, encoding="utf-8-sig")
df_clean.to_parquet(parquet_out, index=False, engine="pyarrow", compression="snappy")

print(f"Exportado CSV: {csv_out} ({os.path.getsize(csv_out):,} bytes)")
print(f"Exportado Parquet: {parquet_out} ({os.path.getsize(parquet_out):,} bytes)")

print("\n" + "=" * 70)
print("RESUMEN DE TRANSFORMACIONES")
print("=" * 70)
print(transform_df.to_string(index=False))
