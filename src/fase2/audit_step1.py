"""
Auditoría Paso 1: Verificación de existencia, metadatos, columnas y SHA-256
"""

import os
import hashlib
import pandas as pd

files_to_check = [
    "reports/fase2/informe_fase2_modelado.md",
    "reports/fase2/resultados_prueba_2025.csv",
    "reports/fase2/resultados_validacion.csv",
    "reports/fase2/resultados_por_provincia.csv",
    "reports/fase2/predicciones_2025.csv",
    "reports/fase2/metricas_alertas.csv",
    "reports/fase2/comparacion_imputaciones.csv",
    "data/processed/panel_semanal_provincial_puno.parquet",
    "data/processed/panel_semanal_regional_puno.parquet"
]

scripts_fase2 = [
    "src/fase2/01_verify_inputs.py",
    "src/fase2/02_calendar_and_lags.py",
    "src/fase2/03_imputation_comparison.py",
    "src/fase2/04_train_baselines.py",
    "src/fase2/05_feature_pipeline.py",
    "src/fase2/06_hyperparameter_tuning.py",
    "src/fase2/07_train_and_evaluate_all_models.py",
    "src/fase2/08_alert_evaluation.py",
    "src/fase2/09_export_figures.py",
    "src/fase2/10_build_notebook.py"
]

all_files = files_to_check + scripts_fase2

results = []
for fpath in all_files:
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        mtime = os.path.getmtime(fpath)
        dt = pd.to_datetime(mtime, unit="s")
        with open(fpath, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        
        r, c = None, None
        cols = []
        if fpath.endswith(".csv"):
            df = pd.read_csv(fpath)
            r, c = df.shape
            cols = list(df.columns)
        elif fpath.endswith(".parquet"):
            df = pd.read_parquet(fpath)
            r, c = df.shape
            cols = list(df.columns)
        elif fpath.endswith(".md") or fpath.endswith(".py"):
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            r = len(lines)
            c = 1
            cols = ["line_count"]
            
        results.append({
            "archivo": fpath,
            "existe": True,
            "filas": r,
            "columnas_count": c,
            "tamano_bytes": size,
            "fecha_modificacion": str(dt),
            "sha256": h,
            "columnas": ", ".join(cols) if cols != ["line_count"] else f"{r} líneas"
        })
    else:
        results.append({
            "archivo": fpath,
            "existe": False,
            "filas": 0,
            "columnas_count": 0,
            "tamano_bytes": 0,
            "fecha_modificacion": "N/A",
            "sha256": "N/A",
            "columnas": "N/A"
        })

df_res = pd.DataFrame(results)
print("=== REVISIÓN DE ARCHIVOS Y METADATOS ===")
for idx, row in df_res.iterrows():
    print(f"Archivo: {row['archivo']}")
    print(f"  Existe: {row['existe']} | Filas: {row['filas']} | Cols: {row['columnas_count']} | Bytes: {row['tamano_bytes']}")
    print(f"  Mtime: {row['fecha_modificacion']} | SHA256: {row['sha256']}")
    print(f"  Columnas: {row['columnas'][:120]}...")
    print("-" * 60)
