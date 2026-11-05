import os
import sys
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def verify_all_preconditions():
    print("=" * 80)
    print("VERIFICACIÓN PROGRAMÁTICA OBLIGATORIA DE CONDICIONES DE FASE 1")
    print("=" * 80)
    
    # 1. Raw CSV check
    raw_path = os.path.join("data", "raw", "datos_abiertos_vigilancia_iras_2000_2025 PUNO.csv")
    assert os.path.exists(raw_path), f"Archivo no encontrado: {raw_path}"
    raw_df = pd.read_csv(raw_path, encoding="utf-8-sig", dtype={"ubigeo": str})
    
    n_raw_rows = len(raw_df)
    sum_raw_episodes = int(raw_df["ira_no_neumonia"].sum())
    print(f"1. Registros originales en CSV:        {n_raw_rows:,} (Esperado: 126,118)")
    print(f"2. Episodios históricos notificados:   {sum_raw_episodes:,} (Esperado: 2,297,378)")
    assert n_raw_rows == 126118, f"Error: filas raw = {n_raw_rows}"
    assert sum_raw_episodes == 2297378, f"Error: suma raw = {sum_raw_episodes}"
    
    # 2. Regional panel check
    reg_path = os.path.join("data", "processed", "panel_semanal_regional_puno.parquet")
    assert os.path.exists(reg_path), f"Archivo no encontrado: {reg_path}"
    reg_df = pd.read_parquet(reg_path)
    n_reg_weeks = len(reg_df)
    sum_reg_episodes = int(reg_df["ira_no_neumonia"].sum())
    print(f"3. Semanas regionales continuas:       {n_reg_weeks:,} (Esperado: 1,357)")
    print(f"4. Episodios panel regional:           {sum_reg_episodes:,} (Esperado: 2,297,378)")
    assert n_reg_weeks == 1357, f"Error: semanas regionales = {n_reg_weeks}"
    assert sum_reg_episodes == 2297378, f"Error: suma regional = {sum_reg_episodes}"
    
    # 3. Provincial panel check
    prov_path = os.path.join("data", "processed", "panel_semanal_provincial_puno.parquet")
    assert os.path.exists(prov_path), f"Archivo no encontrado: {prov_path}"
    prov_df = pd.read_parquet(prov_path)
    
    n_prov_rows = len(prov_df)
    n_provincias = prov_df["provincia"].nunique()
    n_evaluable = int(prov_df["es_evaluable"].sum())
    n_missing = int((~prov_df["es_evaluable"]).sum())
    n_nan = int(prov_df["ira_no_neumonia"].isna().sum())
    sum_prov_episodes = int(prov_df["ira_no_neumonia"].sum())
    
    print(f"5. Provincias oficiales de Puno:       {n_provincias} (Esperado: 13)")
    print(f"6. Combinaciones provincia-semana:     {n_prov_rows:,} (Esperado: 17,641)")
    print(f"7. Observaciones con información:      {n_evaluable:,} (Esperado: 17,488)")
    print(f"8. Observaciones sin registro (NaN):   {n_missing:,} (Esperado: 153)")
    print(f"9. Preservación estricta de NaN:       {n_nan:,} (Esperado: 153)")
    print(f"10. Suma agregada provincial:          {sum_prov_episodes:,} (Esperado: 2,297,378)")
    
    assert n_provincias == 13, f"Error: provincias = {n_provincias}"
    assert n_prov_rows == 17641, f"Error: combinaciones = {n_prov_rows}"
    assert n_evaluable == 17488, f"Error: evaluables = {n_evaluable}"
    assert n_missing == 153, f"Error: faltantes = {n_missing}"
    assert n_nan == 153, f"Error: NaNs = {n_nan}"
    assert sum_prov_episodes == 2297378, f"Error: suma provincial = {sum_prov_episodes}"
    
    # 4. Temporal splits check
    train_prov = prov_df[prov_df["ano"] <= 2022]
    val_prov = prov_df[prov_df["ano"].between(2023, 2024)]
    test_prov = prov_df[prov_df["ano"] == 2025]
    print("\nPartición Temporal Walk-Forward:")
    print(f"  - Entrenamiento (2000-2022): {len(train_prov):,} filas ({train_prov['es_evaluable'].sum():,} evaluables)")
    print(f"  - Validación (2023-2024):    {len(val_prov):,} filas ({val_prov['es_evaluable'].sum():,} evaluables)")
    print(f"  - Prueba (2025):             {len(test_prov):,} filas ({test_prov['es_evaluable'].sum():,} evaluables)")
    
    print("\n" + "=" * 80)
    print(">>> VERIFICACIÓN PROGRAMÁTICA CONCLUIDA EXITOSAMENTE: 100% CUMPLIDO <<<")
    print("=" * 80)
    return True

if __name__ == "__main__":
    verify_all_preconditions()
