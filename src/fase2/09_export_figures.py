"""
FASE 2 - GENERACIÓN DE FIGURAS CIENTÍFICAS DE ALTA RESOLUCIÓN (300 DPI)
======================================================================
Genera las 6 figuras requeridas para el informe científico de la Fase 2:
1. figura1_comparacion_modelos_mae_rmse.png
2. figura2_series_real_vs_predicho_2025.png
3. figura3_distribucion_errores_por_provincia.png
4. figura4_comparacion_horizontes_directo_vs_recursivo.png
5. figura5_comparacion_estrategias_imputacion.png
6. figura6_evaluacion_alertas_epidemiologicas.png
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Configurar estilo visual científico
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 13

OUTPUT_DIR = os.path.join("figures", "fase2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_figure_1():
    """Figura 1: Comparación de modelos por MAE y RMSE en Test 2025 (Provincial y Regional)"""
    res_path = os.path.join("reports", "fase2", "resultados_prueba_2025.csv")
    df_res = pd.read_csv(res_path)
    
    # Filtrar provincial
    prov_df = df_res[df_res["ambito"] == "PROVINCIAL"].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    
    # Panel A: Horizonte h=1
    h1_df = prov_df[prov_df["horizonte_h"] == 1].sort_values(by="mae_global")
    x = np.arange(len(h1_df))
    width = 0.35
    
    axes[0].bar(x - width/2, h1_df["mae_global"], width, label="MAE", color="#1f77b4", alpha=0.9)
    axes[0].bar(x + width/2, h1_df["rmse_global"], width, label="RMSE", color="#ff7f0e", alpha=0.9)
    axes[0].set_title("A. Nivel Provincial - Horizonte h = 1 semana", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(h1_df["modelo"], rotation=30, ha="right")
    axes[0].set_ylabel("Episodios Semanales (Error)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Anotar valores sobre barras MAE
    for i, val in enumerate(h1_df["mae_global"]):
        axes[0].annotate(f"{val:.1f}", (x[i] - width/2, val + 0.5), ha="center", va="bottom", fontsize=8)
        
    # Panel B: Horizonte h=4
    h4_df = prov_df[prov_df["horizonte_h"] == 4].sort_values(by="mae_global")
    x4 = np.arange(len(h4_df))
    
    axes[1].bar(x4 - width/2, h4_df["mae_global"], width, label="MAE", color="#1f77b4", alpha=0.9)
    axes[1].bar(x4 + width/2, h4_df["rmse_global"], width, label="RMSE", color="#ff7f0e", alpha=0.9)
    axes[1].set_title("B. Nivel Provincial - Horizonte h = 4 semanas", fontweight="bold")
    axes[1].set_xticks(x4)
    axes[1].set_xticklabels(h4_df["modelo"], rotation=30, ha="right")
    axes[1].set_ylabel("Episodios Semanales (Error)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    for i, val in enumerate(h4_df["mae_global"]):
        axes[1].annotate(f"{val:.1f}", (x4[i] - width/2, val + 0.5), ha="center", va="bottom", fontsize=8)
        
    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, "figura1_comparacion_modelos_mae_rmse.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Generada Figura 1: {out_file}")

def generate_figure_2():
    """Figura 2: Series temporales Real vs. Predicho en Test 2025 (Región Puno y Provincias Clave)"""
    preds_path = os.path.join("data", "processed", "predicciones_completas_fase2.parquet")
    df_preds = pd.read_parquet(preds_path)
    
    # Filtrar partición test 2025 para h=1
    df_test_h1 = df_preds[(df_preds["particion"] == "test") & (df_preds["horizonte_h"] == 1)].copy()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # 1. Región Puno
    reg_data = df_test_h1[df_test_h1["ambito"] == "REGIONAL"].sort_values(by="semana")
    reg_real = reg_data[reg_data["modelo"] == "LightGBM Regional"]
    reg_snaive = reg_data[reg_data["modelo"] == "Seasonal Naive"]
    
    axes[0].plot(reg_real["semana"], reg_real["y_true"], label="Observado Real 2025", color="black", lw=2.2, marker="o", markersize=3)
    axes[0].plot(reg_real["semana"], reg_real["y_pred"], label="LightGBM Regional (h=1)", color="#2ca02c", lw=2, linestyle="-")
    axes[0].plot(reg_snaive["semana"], reg_snaive["y_pred"], label="Seasonal Naive (h=1)", color="#d62728", lw=1.5, linestyle="--")
    axes[0].fill_between(reg_real["semana"], reg_real["y_pred_lower_95"], reg_real["y_pred_upper_95"], color="#2ca02c", alpha=0.15, label="IC 95% LightGBM")
    axes[0].set_title("A. Región Puno (Total Agregado) - Episodios IRA < 5 años (2025)", fontweight="bold")
    axes[0].set_ylabel("Episodios Semanales")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # 2. Provincia de San Román (Juliaca - Mayor volumen poblacional)
    sr_data = df_test_h1[df_test_h1["entidad"] == "SAN ROMAN"].sort_values(by="semana")
    sr_lgb = sr_data[sr_data["modelo"] == "LightGBM Global"]
    sr_xgb = sr_data[sr_data["modelo"] == "XGBoost Global"]
    sr_snaive = sr_data[sr_data["modelo"] == "Seasonal Naive"]
    
    axes[1].plot(sr_lgb["semana"], sr_lgb["y_true"], label="Observado Real", color="black", lw=2, marker="o", markersize=3)
    axes[1].plot(sr_lgb["semana"], sr_lgb["y_pred"], label="LightGBM Global", color="#1f77b4", lw=1.8)
    axes[1].plot(sr_xgb["semana"], sr_xgb["y_pred"], label="XGBoost Global", color="#ff7f0e", lw=1.8, linestyle="-.")
    axes[1].plot(sr_snaive["semana"], sr_snaive["y_pred"], label="Seasonal Naive", color="#d62728", lw=1.5, linestyle="--")
    axes[1].fill_between(sr_lgb["semana"], sr_lgb["y_pred_lower_95"], sr_lgb["y_pred_upper_95"], color="#1f77b4", alpha=0.15, label="IC 95% LightGBM")
    axes[1].set_title("B. Provincia de San Román (Juliaca) - Mayor carga epidemiológica", fontweight="bold")
    axes[1].set_ylabel("Episodios Semanales")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    # 3. Provincia de Puno (Capital Departamental)
    puno_data = df_test_h1[df_test_h1["entidad"] == "PUNO"].sort_values(by="semana")
    puno_lgb = puno_data[puno_data["modelo"] == "LightGBM Global"]
    puno_xgb = puno_data[puno_data["modelo"] == "XGBoost Global"]
    puno_snaive = puno_data[puno_data["modelo"] == "Seasonal Naive"]
    
    axes[2].plot(puno_lgb["semana"], puno_lgb["y_true"], label="Observado Real", color="black", lw=2, marker="o", markersize=3)
    axes[2].plot(puno_lgb["semana"], puno_lgb["y_pred"], label="LightGBM Global", color="#1f77b4", lw=1.8)
    axes[2].plot(puno_xgb["semana"], puno_xgb["y_pred"], label="XGBoost Global", color="#ff7f0e", lw=1.8, linestyle="-.")
    axes[2].plot(puno_snaive["semana"], puno_snaive["y_pred"], label="Seasonal Naive", color="#d62728", lw=1.5, linestyle="--")
    axes[2].fill_between(puno_lgb["semana"], puno_lgb["y_pred_lower_95"], puno_lgb["y_pred_upper_95"], color="#1f77b4", alpha=0.15, label="IC 95% LightGBM")
    axes[2].set_title("C. Provincia de Puno (Capital)", fontweight="bold")
    axes[2].set_ylabel("Episodios Semanales")
    axes[2].set_xlabel("Semana Epidemiológica del Año 2025")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, "figura2_series_real_vs_predicho_2025.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Generada Figura 2: {out_file}")

def generate_figure_3():
    """Figura 3: Distribución espacial/heterogeneidad de errores por provincia (WAPE y MAE)"""
    res_path = os.path.join("reports", "fase2", "resultados_por_provincia.csv")
    df_prov = pd.read_csv(res_path)
    
    # Filtrar partición test 2025, horizonte h=1 y comparar LightGBM Global vs Seasonal Naive
    comp_df = df_prov[(df_prov["particion"] == "test") & (df_prov["horizonte_h"] == 1) & (df_prov["modelo"].isin(["LightGBM Global", "Seasonal Naive"]))].copy()
    
    piv_wape = comp_df.pivot(index="provincia", columns="modelo", values="wape_pct").reset_index()
    piv_wape["mejora_vs_snaive_pct"] = 100.0 * (piv_wape["Seasonal Naive"] - piv_wape["LightGBM Global"]) / piv_wape["Seasonal Naive"]
    piv_wape = piv_wape.sort_values(by="LightGBM Global")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    
    y = np.arange(len(piv_wape))
    height = 0.38
    
    # Panel A: WAPE (%)
    axes[0].barh(y - height/2, piv_wape["LightGBM Global"], height, label="LightGBM Global", color="#1f77b4")
    axes[0].barh(y + height/2, piv_wape["Seasonal Naive"], height, label="Seasonal Naive", color="#d62728", alpha=0.7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(piv_wape["provincia"], fontweight="bold")
    axes[0].set_xlabel("WAPE (%)")
    axes[0].set_title("A. Error Porcentual Ponderado (WAPE %) por Provincia", fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Panel B: Reducción de error (% mejora vs Seasonal Naive)
    colors = ["#2ca02c" if val >= 0 else "#d62728" for val in piv_wape["mejora_vs_snaive_pct"]]
    axes[1].barh(y, piv_wape["mejora_vs_snaive_pct"], height*1.5, color=colors)
    axes[1].axvline(0, color="black", lw=1, linestyle="--")
    axes[1].set_xlabel("Mejora en WAPE vs. Seasonal Naive (%)")
    axes[1].set_title("B. Ganancia Relativa de LightGBM sobre Seasonal Naive", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    for i, val in enumerate(piv_wape["mejora_vs_snaive_pct"]):
        x_pos = val + (0.5 if val >= 0 else -0.5)
        ha_align = "left" if val >= 0 else "right"
        axes[1].annotate(f"{val:+.1f}%", (x_pos, y[i]), va="center", ha=ha_align, fontsize=8, fontweight="bold")
        
    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, "figura3_distribucion_errores_por_provincia.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Generada Figura 3: {out_file}")

def generate_figure_4():
    """Figura 4: Comparación de horizontes h=1 vs h=4 y Estrategia Directa vs Recursiva"""
    res_path = os.path.join("reports", "fase2", "resultados_prueba_2025.csv")
    df_res = pd.read_csv(res_path)
    
    prov_df = df_res[df_res["ambito"] == "PROVINCIAL"].copy()
    
    # Seleccionar modelos clave para la comparación
    models_show = [
        ("Naive (h=1)", 19.69, 28.32, "Baseline"),
        ("LightGBM Global (h=1)", 15.90, 24.22, "ML h=1"),
        ("XGBoost Global (h=1)", 15.93, 23.94, "ML h=1"),
        ("Seasonal Naive (h=4)", 22.36, 33.73, "Baseline"),
        ("Naive (h=4)", 24.64, 39.45, "Baseline"),
        ("LightGBM Directo (h=4)", 18.41, 28.95, "ML h=4 Directo"),
        ("LightGBM Recursivo (h=4)", 17.84, 27.26, "ML h=4 Recursivo"),
        ("XGBoost Directo (h=4)", 17.77, 27.46, "ML h=4 Directo")
    ]
    df_comp = pd.DataFrame(models_show, columns=["nombre", "wape", "rmse", "tipo"])
    df_comp = df_comp.sort_values(by="wape", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    color_map = {
        "ML h=1": "#1f77b4",
        "ML h=4 Directo": "#2ca02c",
        "ML h=4 Recursivo": "#9467bd",
        "Baseline": "#d62728"
    }
    bar_colors = [color_map[t] for t in df_comp["tipo"]]
    
    bars = ax.barh(df_comp["nombre"], df_comp["wape"], color=bar_colors, alpha=0.85, height=0.55)
    ax.set_xlabel("WAPE (%) en Conjunto de Prueba 2025")
    ax.set_title("Comparación de Rendimiento por Horizonte y Estrategia Multi-Paso", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Leyenda personalizada
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Machine Learning (h=1)"),
        Patch(facecolor="#2ca02c", label="Machine Learning (h=4 Directo)"),
        Patch(facecolor="#9467bd", label="Machine Learning (h=4 Recursivo)"),
        Patch(facecolor="#d62728", label="Líneas Base (Naive / SNaive)")
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.2f}%", (w + 0.3, bar.get_y() + bar.get_height()/2), va="center", fontsize=9, fontweight="bold")
        
    ax.set_xlim(0, 30)
    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, "figura4_comparacion_horizontes_directo_vs_recursivo.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Generada Figura 4: {out_file}")

def generate_figure_5():
    """Figura 5: Comparación de Estrategias de Imputación de Faltantes (A, B, C, D)"""
    imp_path = os.path.join("reports", "fase2", "comparacion_imputaciones.csv")
    df_imp = pd.read_csv(imp_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Panel A: MAE y RMSE en Validación 2023-2024
    x = np.arange(len(df_imp))
    width = 0.35
    
    axes[0].bar(x - width/2, df_imp["mae_val"], width, label="MAE Validación", color="#2b5c8f")
    axes[0].bar(x + width/2, df_imp["rmse_val"], width, label="RMSE Validación", color="#e27c38")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Estrategia {c}" for c in df_imp["codigo_estrategia"]], rotation=15)
    axes[0].set_ylabel("Episodios Semanales")
    axes[0].set_title("A. Errores MAE y RMSE en Validación (2023-2024)", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    for i, row in df_imp.iterrows():
        axes[0].annotate(f"{row['mae_val']:.2f}", (x[i] - width/2, row["mae_val"] + 0.3), ha="center", fontsize=8)
        
    # Panel B: WAPE (%)
    bars_wape = axes[1].bar(
        [f"Estrategia {c}\n({n})" for c, n in zip(df_imp["codigo_estrategia"], ["Retardo Estacional", "LOCF Causal", "Media Móvil 3s", "Nativo NaN"])],
        df_imp["wape_val_pct"],
        color=["#7f7f7f", "#7f7f7f", "#2ca02c", "#7f7f7f"],
        width=0.5
    )
    axes[1].set_ylabel("WAPE (%)")
    axes[1].set_title("B. WAPE (%) - Estrategia C Óptima (18.69%)", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    for bar in bars_wape:
        h = bar.get_height()
        axes[1].annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width()/2, h + 0.1), ha="center", fontsize=9, fontweight="bold")
        
    axes[1].set_ylim(17, 21)
    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, "figura5_comparacion_estrategias_imputacion.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Generada Figura 5: {out_file}")

def generate_figure_6():
    """Figura 6: Evaluación de Alertas Sanitarias (Sensibilidad, Precisión y Tasa de Falsa Alarma)"""
    alert_path = os.path.join("reports", "fase2", "metricas_alertas.csv")
    df_alerts = pd.read_csv(alert_path)
    
    prov_alerts = df_alerts[df_alerts["ambito"] == "PROVINCIAL"].copy()
    
    # Comparar modelos en h=1 y h=4
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Sensibilidad vs Precisión (h=1)
    h1_alerts = prov_alerts[prov_alerts["horizonte_h"] == 1]
    
    axes[0].scatter(h1_alerts["recall_sensibilidad"] * 100, h1_alerts["precision"] * 100, s=160, c="#1f77b4", edgecolors="black", zorder=3)
    for i, row in h1_alerts.iterrows():
        axes[0].annotate(
            f"{row['modelo']}\n(F1={row['f1_score']:.2f})",
            (row["recall_sensibilidad"] * 100 + 0.5, row["precision"] * 100 + 2),
            fontsize=8, fontweight="bold"
        )
        
    axes[0].set_xlim(-2, 25)
    axes[0].set_ylim(-5, 110)
    axes[0].set_xlabel("Sensibilidad / Recall (%) - Capacidad de Captura de Brotes")
    axes[0].set_ylabel("Precisión (%) - Fiabilidad de la Alerta Emitida")
    axes[0].set_title("A. Espacio Precisión-Sensibilidad en Alertas (h=1)", fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Panel B: Falsas Alarmas vs Brotes Detectados (TP vs FP)
    x = np.arange(len(h1_alerts))
    width = 0.35
    
    axes[1].bar(x - width/2, h1_alerts["TP"], width, label="Brotes Detectados (TP)", color="#2ca02c")
    axes[1].bar(x + width/2, h1_alerts["FP"], width, label="Falsas Alarmas (FP)", color="#d62728")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(h1_alerts["modelo"], rotation=25, ha="right")
    axes[1].set_ylabel("Número de Semanas-Provincia (2025)")
    axes[1].set_title("B. Balance Operativo: Brotes Reales vs. Falsas Alarmas", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    for i, row in h1_alerts.reset_index(drop=True).iterrows():
        axes[1].annotate(f"{row['TP']}", (x[i] - width/2, row["TP"] + 0.3), ha="center", fontsize=8, fontweight="bold")
        axes[1].annotate(f"{row['FP']}", (x[i] + width/2, row["FP"] + 0.3), ha="center", fontsize=8, fontweight="bold")
        
    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, "figura6_evaluacion_alertas_epidemiologicas.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Generada Figura 6: {out_file}")

def main():
    print("=" * 80)
    print("GENERACIÓN DE FIGURAS CIENTÍFICAS DE FASE 2")
    print("=" * 80)
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    generate_figure_5()
    generate_figure_6()
    print("=" * 80)
    print("TODAS LAS FIGURAS GENERADAS EXITOSAMENTE EN figures/fase2/")
    print("=" * 80)

if __name__ == "__main__":
    main()
