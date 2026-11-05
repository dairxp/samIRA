"""
Auditoría Paso 3: Claves primarias, duplicados e integridad de predicciones_2025.csv
"""

import pandas as pd
import numpy as np

df_preds = pd.read_csv("reports/fase2/predicciones_2025.csv")
print("=== ESTRUCTURA DE PREDICCIONES_2025.CSV ===")
print(f"Total filas: {len(df_preds):,}")
print("Columnas:", list(df_preds.columns))

print("\nConteo por ámbito:")
print(df_preds["ambito"].value_counts())

print("\nConteo por horizonte (h):")
print(df_preds["horizonte_h"].value_counts())

print("\nConteo por modelo:")
print(df_preds["modelo"].value_counts())

print("\nModelos cruzados por ámbito y horizonte:")
cross = df_preds.groupby(["ambito", "horizonte_h", "modelo", "estrategia_horizonte"]).size().reset_index(name="filas")
print(cross.to_string(index=False))

# Verificar claves primarias
# Clave teórica: (ambito, entidad, ano, semana, horizonte_h, modelo, estrategia_horizonte)
key_cols = ["ambito", "entidad", "ano", "semana", "horizonte_h", "modelo", "estrategia_horizonte"]
dups = df_preds[df_preds.duplicated(subset=key_cols, keep=False)]
print(f"\nDuplicados exactos con clave completa: {len(dups)}")

# Qué pasa si no incluimos estrategia_horizonte?
key_cols_no_strat = ["ambito", "entidad", "ano", "semana", "horizonte_h", "modelo"]
dups_no_strat = df_preds[df_preds.duplicated(subset=key_cols_no_strat, keep=False)]
print(f"Duplicados con clave sin estrategia_horizonte: {len(dups_no_strat)}")

# Verificar semanas por entidad y modelo
print("\nSemanas por entidad y modelo en provincial:")
p_sub = df_preds[df_preds["ambito"] == "PROVINCIAL"]
prov_summary = p_sub.groupby(["horizonte_h", "modelo", "estrategia_horizonte"]).agg(
    n_filas=("semana", "count"),
    n_provincias=("entidad", "nunique"),
    semanas_min=("semana", "min"),
    semanas_max=("semana", "max"),
    n_evaluable=("es_evaluable", "sum"),
    n_nan_y_true=("y_true", lambda x: x.isna().sum()),
    n_nan_y_pred=("y_pred", lambda x: x.isna().sum())
).reset_index()
print(prov_summary.to_string(index=False))

# Verificar regional
print("\nSemanas en regional:")
r_sub = df_preds[df_preds["ambito"] == "REGIONAL"]
reg_summary = r_sub.groupby(["horizonte_h", "modelo", "estrategia_horizonte"]).agg(
    n_filas=("semana", "count"),
    semanas_min=("semana", "min"),
    semanas_max=("semana", "max"),
    n_evaluable=("es_evaluable", "sum"),
    n_nan_y_true=("y_true", lambda x: x.isna().sum()),
    n_nan_y_pred=("y_pred", lambda x: x.isna().sum())
).reset_index()
print(reg_summary.to_string(index=False))
