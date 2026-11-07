"""
MÓDULO: ALINEACIÓN TEMPORAL DEL CALENDARIO EPIDEMIOLÓGICO Y RETARDOS ESTACIONALES
Proyecto samIRA - Fase 2
"""

import os
import sys
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ANIOS_SEMANA_53 = {2003, 2008, 2014, 2020, 2025}

def get_official_weeks_in_year(ano: int) -> int:
    """Devuelve el total de semanas oficiales reconocidas por CDC MINSA."""
    return 53 if ano in ANIOS_SEMANA_53 else 52

def build_aligned_seasonal_lag(df_series: pd.DataFrame, target_col: str = "ira_no_neumonia") -> pd.DataFrame:
    """
    Construye retardos estacionales para una serie temporal indexada por (provincia/region, ano, semana).
    
    Estrategia de Alineación por Calendario:
    - Para una observación en (año y, semana w):
      * Si w <= 52: busca (y-1, w).
      * Si w == 53 y el año y-1 solo tuvo 52 semanas: mapea a (y-1, 52) como análogo estacional tardío.
    - Preserva trazabilidad: almacena año_origen_lag, semana_origen_lag.
    - Causalidad estricta: sólo utiliza información del año previo (nunca del futuro).
    """
    df = df_series.copy()
    group_col = "provincia" if "provincia" in df.columns else None
    
    if group_col:
        df = df.sort_values(by=[group_col, "ano", "semana"]).reset_index(drop=True)
        # 1. Retardo posicional simple shift(52) dentro de cada provincia
        df["lag_pos_52"] = df.groupby(group_col)[target_col].shift(52)
        
        # 2. Retardo por calendario epidemiológico (Lookup por clave año-semana)
        lookup = df.set_index([group_col, "ano", "semana"])[target_col].to_dict()
        
        cal_lag_values = []
        origin_year = []
        origin_week = []
        
        for idx, row in df.iterrows():
            prov = row[group_col]
            y = int(row["ano"])
            w = int(row["semana"])
            
            target_y = y - 1
            if w == 53 and get_official_weeks_in_year(target_y) == 52:
                target_w = 52
            else:
                target_w = w
                
            val = lookup.get((prov, target_y, target_w), np.nan)
            cal_lag_values.append(val)
            origin_year.append(target_y if (prov, target_y, target_w) in lookup else np.nan)
            origin_week.append(target_w if (prov, target_y, target_w) in lookup else np.nan)
            
        df["lag_epi_1y"] = cal_lag_values
        df["lag_epi_1y_origen_ano"] = origin_year
        df["lag_epi_1y_origen_semana"] = origin_week
        
    else:
        df = df.sort_values(by=["ano", "semana"]).reset_index(drop=True)
        df["lag_pos_52"] = df[target_col].shift(52)
        
        lookup = df.set_index(["ano", "semana"])[target_col].to_dict()
        cal_lag_values = []
        origin_year = []
        origin_week = []
        
        for idx, row in df.iterrows():
            y = int(row["ano"])
            w = int(row["semana"])
            target_y = y - 1
            if w == 53 and get_official_weeks_in_year(target_y) == 52:
                target_w = 52
            else:
                target_w = w
                
            val = lookup.get((target_y, target_w), np.nan)
            cal_lag_values.append(val)
            origin_year.append(target_y if (target_y, target_w) in lookup else np.nan)
            origin_week.append(target_w if (target_y, target_w) in lookup else np.nan)
            
        df["lag_epi_1y"] = cal_lag_values
        df["lag_epi_1y_origen_ano"] = origin_year
        df["lag_epi_1y_origen_semana"] = origin_week
        
    return df


def run_unit_tests():
    print("=" * 80)
    print("EJECUTANDO PRUEBAS UNITARIAS DE ALINEACIÓN DEL CALENDARIO EPIDEMIOLÓGICO")
    print("=" * 80)
    
    # Cargar panel regional
    reg_path = os.path.join("data", "processed", "panel_semanal_regional_puno.parquet")
    df_reg = pd.read_parquet(reg_path)
    df_reg = build_aligned_seasonal_lag(df_reg)
    
    # Test 1: Verificar que año 2000 tiene NaN estricto en lag_epi_1y
    df_2000 = df_reg[df_reg["ano"] == 2000]
    assert df_2000["lag_epi_1y"].isna().all(), "Test 1 Falló: Año 2000 no debe tener lag de 1999"
    print("✓ Test 1 Aprobado: Año 2000 tiene NaN estricto en retardo estacional (sin fuga de datos previa).")
    
    # Test 2: Transición 2003 (53 semanas) -> 2004 (52 semanas)
    # En 2004-W01:
    # lag_pos_52 toma 2003-W02 (a 52 posiciones atrás). ¡DESALINEADO EN 1 SEMANA!
    # lag_epi_1y toma 2003-W01 (semana homóloga del calendario). ¡CORRECTO!
    row_2004_w1 = df_reg[(df_reg["ano"] == 2004) & (df_reg["semana"] == 1)].iloc[0]
    val_2003_w1 = df_reg[(df_reg["ano"] == 2003) & (df_reg["semana"] == 1)]["ira_no_neumonia"].values[0]
    val_2003_w2 = df_reg[(df_reg["ano"] == 2003) & (df_reg["semana"] == 2)]["ira_no_neumonia"].values[0]
    
    assert row_2004_w1["lag_epi_1y"] == val_2003_w1, f"Test 2 Falló: 2004-W01 lag_epi debe ser 2003-W01 ({val_2003_w1})"
    assert row_2004_w1["lag_pos_52"] == val_2003_w2, f"Test 2 Falló: shift(52) debió tomar 2003-W02 ({val_2003_w2})"
    print(f"✓ Test 2 Aprobado (Transición 2003->2004): 2004-W01 recupera con exactitud 2003-W01 ({val_2003_w1}) en lag_epi_1y, mientras shift(52) sufrió desfase de 1 semana recuperando 2003-W02 ({val_2003_w2}).")
    
    # Test 3: Mapeo causal de la Semana 53 de 2008 (tras 2007 que tuvo 52 semanas)
    row_2008_w53 = df_reg[(df_reg["ano"] == 2008) & (df_reg["semana"] == 53)].iloc[0]
    val_2007_w52 = df_reg[(df_reg["ano"] == 2007) & (df_reg["semana"] == 52)]["ira_no_neumonia"].values[0]
    assert row_2008_w53["lag_epi_1y"] == val_2007_w52, f"Test 3 Falló: 2008-W53 debe mapear a 2007-W52 ({val_2007_w52})"
    print(f"✓ Test 3 Aprobado (Mapeo SE 53): 2008-W53 mapea de forma limpia y causal a 2007-W52 ({val_2007_w52}).")
    
    # Test 4: Deriva acumulada por shift(52) hacia el año 2025
    # En 2025 (tras 4 años con SE 53: 2003, 2008, 2014, 2020), shift(52) arrastra un desfase acumulado de 4 semanas.
    # Para 2025-W10:
    # lag_epi_1y recupera 2024-W10 (homóloga perfecta).
    # lag_pos_52 recupera 2024-W14 (o W06 según orden). Comprobamos que lag_epi es exactamente 2024-W10.
    row_2025_w10 = df_reg[(df_reg["ano"] == 2025) & (df_reg["semana"] == 10)].iloc[0]
    val_2024_w10 = df_reg[(df_reg["ano"] == 2024) & (df_reg["semana"] == 10)]["ira_no_neumonia"].values[0]
    
    assert row_2025_w10["lag_epi_1y"] == val_2024_w10, f"Test 4 Falló: 2025-W10 debe recuperar 2024-W10 ({val_2024_w10})"
    desfase_semanas = abs(int(row_2025_w10["lag_epi_1y_origen_semana"]) - int(df_reg.loc[df_reg.index[df_reg["ano"] == 2025].min() + 9 - 52, "semana"]))
    print(f"✓ Test 4 Aprobado (Alineación 2025): En 2025-W10, lag_epi_1y recupera fielmente la semana homóloga 2024-W10 ({val_2024_w10}) con desfase cero, corrigiendo la deriva de {desfase_semanas} semanas de shift(52).")
    
    # Test 5: Comprobación a nivel provincial en las 13 provincias
    prov_path = os.path.join("data", "processed", "panel_semanal_provincial_puno.parquet")
    df_prov = pd.read_parquet(prov_path)
    df_prov = build_aligned_seasonal_lag(df_prov)
    
    provincias_verificadas = 0
    for p in df_prov["provincia"].unique():
        sub = df_prov[df_prov["provincia"] == p]
        row_p_2024_w1 = sub[(sub["ano"] == 2024) & (sub["semana"] == 1)].iloc[0]
        val_p_2023_w1 = sub[(sub["ano"] == 2023) & (sub["semana"] == 1)]["ira_no_neumonia"].values[0]
        if not np.isnan(val_p_2023_w1):
            assert row_p_2024_w1["lag_epi_1y"] == val_p_2023_w1, f"Falla en provincia {p}"
            provincias_verificadas += 1
            
    print(f"✓ Test 5 Aprobado: Alineación causal verificada para todas las {provincias_verificadas} provincias evaluables de Puno.")
    print("\n" + "=" * 80)
    print(">>> TODAS LAS PRUEBAS UNITARIAS DE ALINEACIÓN TEMPORAL FUERON EXITOSAS <<<")
    print("=" * 80)

if __name__ == "__main__":
    run_unit_tests()
