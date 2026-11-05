"""
Prueba y comparación rigurosa de métodos de calibración de intervalos al 95%
Utilizando exclusivamente la partición de validación (2023-2024) para calibrar.
"""

import pandas as pd
import numpy as np

df_preds = pd.read_parquet("data/processed/predicciones_completas_fase2.parquet")
lgb1 = df_preds[
    (df_preds["ambito"] == "PROVINCIAL") &
    (df_preds["horizonte_h"] == 1) &
    (df_preds["modelo"] == "LightGBM Global") &
    (df_preds["es_evaluable"] == True)
].copy()

val = lgb1[lgb1["particion"] == "val"].copy()
test = lgb1[lgb1["particion"] == "test"].copy()

val["res"] = val["y_true"] - val["y_pred"]
test["res"] = test["y_true"] - test["y_pred"]

# Método 1: Cuantiles globales aditivos (Método original)
q025_glob = np.percentile(val["res"], 2.5)
q975_glob = np.percentile(val["res"], 97.5)

# Método 2: Cuantiles empíricos provinciales (N=104 semanas de validación por provincia)
prov_q = val.groupby("entidad")["res"].agg(
    q025_prov=lambda x: np.percentile(x, 2.5),
    q975_prov=lambda x: np.percentile(x, 97.5)
).reset_index()

# Método 3: Estratificado por volumen demográfico/epidemiológico
vol_groups = {
    "SAN ROMAN": "ALTO", "PUNO": "ALTO",
    "MELGAR": "MEDIO", "AZANGARO": "MEDIO", "CHUCUITO": "MEDIO", "CARABAYA": "MEDIO", "LAMPA": "MEDIO", "SANDIA": "MEDIO", "HUANCANE": "MEDIO",
    "EL COLLAO": "BAJO", "SAN ANTONIO DE PUTINA": "BAJO", "YUNGUYO": "BAJO", "MOHO": "BAJO"
}
val["grupo"] = val["entidad"].map(vol_groups)
strat_q = val.groupby("grupo")["res"].agg(
    q025_strat=lambda x: np.percentile(x, 2.5),
    q975_strat=lambda x: np.percentile(x, 97.5)
).reset_index()

# Método 4: Escalado heterocedástico por desviación estándar local (Conformal / z-scaling)
val_std = val.groupby("entidad")["res"].std().reset_index().rename(columns={"res": "std_prov"})
val = val.merge(val_std, on="entidad")
val["res_scaled"] = val["res"] / val["std_prov"]
q025_scaled = np.percentile(val["res_scaled"], 2.5)
q975_scaled = np.percentile(val["res_scaled"], 97.5)

# Aplicar a test 2025
test["grupo"] = test["entidad"].map(vol_groups)
test = test.merge(prov_q, on="entidad", how="left")
test = test.merge(strat_q, on="grupo", how="left")
test = test.merge(val_std, on="entidad", how="left")

# Predicciones de límites inferiores y superiores
test["low_glob"] = np.maximum(0, test["y_pred"] + q025_glob)
test["up_glob"] = np.maximum(0, test["y_pred"] + q975_glob)
test["cov_glob"] = (test["y_true"] >= test["low_glob"]) & (test["y_true"] <= test["up_glob"])
test["w_glob"] = test["up_glob"] - test["low_glob"]

test["low_prov"] = np.maximum(0, test["y_pred"] + test["q025_prov"])
test["up_prov"] = np.maximum(0, test["y_pred"] + test["q975_prov"])
test["cov_prov"] = (test["y_true"] >= test["low_prov"]) & (test["y_true"] <= test["up_prov"])
test["w_prov"] = test["up_prov"] - test["low_prov"]

test["low_strat"] = np.maximum(0, test["y_pred"] + test["q025_strat"])
test["up_strat"] = np.maximum(0, test["y_pred"] + test["q975_strat"])
test["cov_strat"] = (test["y_true"] >= test["low_strat"]) & (test["y_true"] <= test["up_strat"])
test["w_strat"] = test["up_strat"] - test["low_strat"]

test["low_scaled"] = np.maximum(0, test["y_pred"] + q025_scaled * test["std_prov"])
test["up_scaled"] = np.maximum(0, test["y_pred"] + q975_scaled * test["std_prov"])
test["cov_scaled"] = (test["y_true"] >= test["low_scaled"]) & (test["y_true"] <= test["up_scaled"])
test["w_scaled"] = test["up_scaled"] - test["low_scaled"]

comp_table = test.groupby("entidad").agg(
    n=("semana", "count"),
    y_mean=("y_true", "mean"),
    cov_glob=("cov_glob", lambda x: round(x.mean() * 100, 2)),
    cov_prov=("cov_prov", lambda x: round(x.mean() * 100, 2)),
    cov_strat=("cov_strat", lambda x: round(x.mean() * 100, 2)),
    cov_scaled=("cov_scaled", lambda x: round(x.mean() * 100, 2)),
    w_glob=("w_glob", lambda x: round(x.mean(), 1)),
    w_prov=("w_prov", lambda x: round(x.mean(), 1)),
    w_strat=("w_strat", lambda x: round(x.mean(), 1)),
    w_scaled=("w_scaled", lambda x: round(x.mean(), 1))
).reset_index()

print("=== COMPARACIÓN DE COBERTURA Y ANCHO DE INTERVALOS EN TEST 2025 ===")
print(comp_table.to_string(index=False))

print("\n=== RESUMEN GLOBAL (N=689) ===")
print(f"Método Global: Cobertura = {test['cov_glob'].mean()*100:.2f}%, Ancho Promedio = {test['w_glob'].mean():.2f}")
print(f"Método Provincial: Cobertura = {test['cov_prov'].mean()*100:.2f}%, Ancho Promedio = {test['w_prov'].mean():.2f}")
print(f"Método Estratificado: Cobertura = {test['cov_strat'].mean()*100:.2f}%, Ancho Promedio = {test['w_strat'].mean():.2f}")
print(f"Método Heterocedástico Escalado: Cobertura = {test['cov_scaled'].mean()*100:.2f}%, Ancho Promedio = {test['w_scaled'].mean():.2f}")
