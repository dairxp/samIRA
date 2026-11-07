import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Configure plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

os.makedirs("figures", exist_ok=True)
data_path = os.path.join("data", "processed", "datos_iras_puno_limpio.parquet")
df = pd.read_parquet(data_path)

print(f"Dataset cargado con éxito: {df.shape[0]:,} filas, {df.shape[1]} columnas.")

# Color palette
PALETTE = sns.color_palette("tab10")

# FIGURA 1: Evolución Anual y Serie Temporal Semanal Regional de IRA no neumonía
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# 1A. Casos anuales totales
annual_ira = df.groupby("ano")["ira_no_neumonia"].sum().reset_index()
bars = ax1.bar(annual_ira["ano"], annual_ira["ira_no_neumonia"], color="#1f77b4", alpha=0.85, edgecolor="#0d47a1")
# Highlight pandemic years
for i, bar in enumerate(bars):
    year = annual_ira["ano"].iloc[i]
    if year in [2020, 2021, 2022]:
        bar.set_color("#d95f02")
ax1.set_title("A. Volumen Anual de Episodios Notificados de IRA no Neumonía en Menores de 5 Años (Región Puno, 2000-2025)", fontsize=13, fontweight="bold", pad=10)
ax1.set_xlabel("Año", fontsize=11)
ax1.set_ylabel("Episodios Notificados (Casos Brutos)", fontsize=11)
ax1.set_xticks(annual_ira["ano"])
ax1.set_xticklabels(annual_ira["ano"], rotation=45, ha="right")
ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
# Add footnote
ax1.text(0.01, 0.90, "Nota: Barras naranjas indican el trienio pandémico COVID-19 (2020-2022)", transform=ax1.transAxes, fontsize=10, style="italic", bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#ffeeba"))

# 1B. Serie temporal continua semanal agregada a nivel regional
weekly_reg = df.groupby(["ano", "semana"])["ira_no_neumonia"].sum().reset_index()
weekly_reg["time_idx"] = np.arange(len(weekly_reg))

ax2.plot(weekly_reg["time_idx"], weekly_reg["ira_no_neumonia"], color="#02818a", linewidth=1.2, label="Episodios semanales")
# 13-week rolling average (~ quarterly trend)
weekly_reg["rolling_13"] = weekly_reg["ira_no_neumonia"].rolling(13, center=True).mean()
ax2.plot(weekly_reg["time_idx"], weekly_reg["rolling_13"], color="#e7298a", linewidth=2.0, label="Media móvil (13 semanas)")

# Year tick labels
year_starts = weekly_reg.drop_duplicates(subset=["ano"], keep="first")
ax2.set_xticks(year_starts["time_idx"])
ax2.set_xticklabels(year_starts["ano"], rotation=45, ha="right", fontsize=9)
ax2.set_title("B. Serie de Tiempo Semanal Continua de IRA no Neumonía (Nivel Departamental Puno)", fontsize=13, fontweight="bold", pad=10)
ax2.set_xlabel("Año Epidemiológico", fontsize=11)
ax2.set_ylabel("Episodios Semanales", fontsize=11)
ax2.legend(loc="upper right", frameon=True)
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
fig.savefig("figures/fig01_evolucion_anual_semanal_ira.png", dpi=300)
plt.close()
print("Generada: figures/fig01_evolucion_anual_semanal_ira.png")


# FIGURA 2: Estacionalidad por Semana Epidemiológica (Ciclo Anual Típico)
fig, ax = plt.subplots(figsize=(13, 6))

# Aggregate weekly regional cases per year-week
reg_by_week = df.groupby(["ano", "semana"])["ira_no_neumonia"].sum().reset_index()

# Exclude 2020-2021 pandemic anomaly for the baseline seasonal curve, or compare both
baseline_pre = reg_by_week[reg_by_week["ano"] < 2020].groupby("semana")["ira_no_neumonia"]
median_pre = baseline_pre.median()
q25_pre = baseline_pre.quantile(0.25)
q75_pre = baseline_pre.quantile(0.75)

weeks_axis = median_pre.index

ax.fill_between(weeks_axis, q25_pre, q75_pre, color="#67a9cf", alpha=0.35, label="Rango Intercuartil Histórico (P25-P75, 2000-2019)")
ax.plot(weeks_axis, median_pre, color="#02818a", linewidth=2.5, marker="o", markersize=4, label="Mediana Histórica Pre-COVID (2000-2019)")

# Pandemic years
for yr, col, ls in [(2020, "#d95f02", "--"), (2021, "#e7298a", "-."), (2024, "#7570b3", "-")]:
    df_yr = reg_by_week[reg_by_week["ano"] == yr].sort_values("semana")
    ax.plot(df_yr["semana"], df_yr["ira_no_neumonia"], color=col, linestyle=ls, linewidth=1.8, label=f"Año {yr}")

# Highlight winter / friaje season in Puno (weeks 20 to 32: mid-May to mid-August)
ax.axvspan(20, 32, color="#fee08b", alpha=0.3, label="Temporada de Bajas Temperaturas / Heladas (SE 20-32)")

ax.set_title("Patrón Estacional de IRA no Neumonía en Menores de 5 Años por Semana Epidemiológica", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Semana Epidemiológica (SE 1 - 53)", fontsize=11)
ax.set_ylabel("Episodios Semanales (Casos Notificados)", fontsize=11)
ax.set_xticks(range(1, 54, 2))
ax.set_xlim(1, 53)
ax.legend(loc="upper right", frameon=True, fontsize=9)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
fig.savefig("figures/fig02_estacionalidad_semanal_ira.png", dpi=300)
plt.close()
print("Generada: figures/fig02_estacionalidad_semanal_ira.png")


# FIGURA 3: Comparación entre las 13 Provincias de Puno
fig, ax = plt.subplots(figsize=(12, 7))

prov_ira = df.groupby("provincia")["ira_no_neumonia"].sum().sort_values(ascending=True)

y_pos = np.arange(len(prov_ira))
bars = ax.barh(y_pos, prov_ira.values, color="#3690c0", edgecolor="#02818a", height=0.65)

# Annotate values
for i, v in enumerate(prov_ira.values):
    ax.text(v + 5000, i, f"{v:,} ({v/prov_ira.sum()*100:.1f}%)", va="center", fontsize=9.5, fontweight="semibold")

ax.set_yticks(y_pos)
ax.set_yticklabels(prov_ira.index, fontsize=10.5)
ax.set_title("Volumen Acumulado de IRA no Neumonía por Provincia (Puno, 2000-2025)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Total Episodios Notificados (Casos Brutos)", fontsize=11)
ax.set_xlim(0, prov_ira.max() * 1.18)
ax.grid(True, linestyle="--", alpha=0.5, axis="x")

# Disclaimer footnote
plt.figtext(0.12, -0.01, "ADVERTENCIA METODOLÓGICA: Cifras corresponden a frecuencias absolutas de notificación, no a tasas de incidencia.\nLas provincias de San Román (Juliaca) y Puno concentran los mayores volúmenes por su alta densidad demográfica y concentración hospitalaria.", wrap=True, fontsize=9, style="italic", color="#555555")

plt.tight_layout()
fig.savefig("figures/fig03_comparacion_13_provincias_ira.png", dpi=300, bbox_inches="tight")
plt.close()
print("Generada: figures/fig03_comparacion_13_provincias_ira.png")


# FIGURA 4: Distribución y Valores Extremos Semanales por Provincia (Boxplots)
fig, ax = plt.subplots(figsize=(14, 7))

# Aggregate by provincia, ano, semana
prov_weekly = df.groupby(["provincia", "ano", "semana"])["ira_no_neumonia"].sum().reset_index()

# Sort provinces by median
prov_order = prov_weekly.groupby("provincia")["ira_no_neumonia"].median().sort_values(ascending=False).index

sns.boxplot(
    data=prov_weekly,
    x="provincia",
    y="ira_no_neumonia",
    order=prov_order,
    ax=ax,
    palette="Blues_r",
    showmeans=True,
    meanprops={"marker": "^", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 7},
    flierprops={"marker": "o", "markersize": 3, "alpha": 0.4, "markerfacecolor": "grey"}
)

ax.set_title("Distribución de Casos Semanales de IRA no Neumonía por Provincia (2000-2025)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Provincia", fontsize=11)
ax.set_ylabel("Episodios Notificados por Semana (Casos Brutos)", fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9.5)
ax.grid(True, linestyle="--", alpha=0.5, axis="y")

# Legend for mean
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='^', color='w', label='Media provincial', markerfacecolor='red', markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Semana atípica (outlier)', markerfacecolor='grey', markersize=5)
]
ax.legend(handles=legend_elements, loc="upper right", frameon=True)

plt.tight_layout()
fig.savefig("figures/fig04_distribucion_episodios_boxplot.png", dpi=300)
plt.close()
print("Generada: figures/fig04_distribucion_episodios_boxplot.png")


# FIGURA 5: Comparación Histórica: Pre-Pandemia, Pandemia y Post-Pandemia
fig, ax = plt.subplots(figsize=(13, 6))

df_periods = df.copy()
conditions = [
    (df_periods["ano"] < 2020),
    (df_periods["ano"].between(2020, 2022)),
    (df_periods["ano"] >= 2023)
]
period_labels = ["Pre-pandemia (2000-2019)", "Pandemia COVID-19 (2020-2022)", "Post-pandemia (2023-2025)"]
df_periods["periodo"] = np.select(conditions, period_labels, default="Otro")

period_weekly = df_periods.groupby(["periodo", "semana"])["ira_no_neumonia"].mean().reset_index()

colors = {"Pre-pandemia (2000-2019)": "#2b83ba", "Pandemia COVID-19 (2020-2022)": "#d7191c", "Post-pandemia (2023-2025)": "#fdae61"}

for per in period_labels:
    sub = period_weekly[period_weekly["periodo"] == per].sort_values("semana")
    ax.plot(sub["semana"], sub["ira_no_neumonia"], label=per, color=colors[per], linewidth=2.5, marker="o" if "Post" in per else None, markersize=4)

ax.set_title("Impacto del Período Pandémico en la Notificación Semanal Promedio de IRA no Neumonía", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Semana Epidemiológica (SE 1 - 53)", fontsize=11)
ax.set_ylabel("Promedio de Episodios Notificados por Semana (Región Puno)", fontsize=11)
ax.set_xticks(range(1, 54, 2))
ax.set_xlim(1, 53)
ax.legend(loc="upper right", frameon=True, fontsize=10)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
fig.savefig("figures/fig05_impacto_covid_periodos.png", dpi=300)
plt.close()
print("Generada: figures/fig05_impacto_covid_periodos.png")


# FIGURA 6: Evolución Separada de Neumonías, Hospitalizados y Defunciones
fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

annual_severe = df.groupby("ano").agg(
    neum_men5=("neumonias_men5", "sum"),
    neum_60mas=("neumonias_60mas", "sum"),
    hosp_men5=("hospitalizados_men5", "sum"),
    hosp_60mas=("hospitalizados_60mas", "sum"),
    def_men5=("defunciones_men5", "sum"),
    def_60mas=("defunciones_60mas", "sum"),
).reset_index()

# 6A. Neumonías
axes[0].plot(annual_severe["ano"], annual_severe["neum_men5"], marker="o", color="#d95f02", linewidth=2, label="Menores de 5 años")
axes[0].plot(annual_severe["ano"], annual_severe["neum_60mas"], marker="s", color="#7570b3", linewidth=2, linestyle="--", label="Adultos mayores (60+ años)")
axes[0].set_ylabel("Neumonías Anuales", fontsize=10.5)
axes[0].set_title("A. Neumonías Notificadas (2000-2025)", fontsize=12, fontweight="bold")
axes[0].legend(loc="upper right")
axes[0].grid(True, linestyle="--", alpha=0.5)

# 6B. Hospitalizados
axes[1].plot(annual_severe["ano"], annual_severe["hosp_men5"], marker="o", color="#e7298a", linewidth=2, label="Menores de 5 años")
axes[1].plot(annual_severe["ano"], annual_severe["hosp_60mas"], marker="s", color="#1b9e77", linewidth=2, linestyle="--", label="Adultos mayores (60+ años)")
axes[1].set_ylabel("Hospitalizados Anuales", fontsize=10.5)
axes[1].set_title("B. Hospitalizaciones por Neumonía (2000-2025)", fontsize=12, fontweight="bold")
axes[1].legend(loc="upper right")
axes[1].grid(True, linestyle="--", alpha=0.5)

# 6C. Defunciones
axes[2].plot(annual_severe["ano"], annual_severe["def_men5"], marker="o", color="#e41a1c", linewidth=2, label="Menores de 5 años")
axes[2].plot(annual_severe["ano"], annual_severe["def_60mas"], marker="s", color="#377eb8", linewidth=2, linestyle="--", label="Adultos mayores (60+ años)")
axes[2].set_ylabel("Defunciones Anuales", fontsize=10.5)
axes[2].set_title("C. Defunciones Notificadas por Neumonía (2000-2025)", fontsize=12, fontweight="bold")
axes[2].legend(loc="upper right")
axes[2].grid(True, linestyle="--", alpha=0.5)
axes[2].set_xlabel("Año", fontsize=11)
axes[2].set_xticks(annual_severe["ano"])
axes[2].set_xticklabels(annual_severe["ano"], rotation=45, ha="right")

plt.tight_layout()
fig.savefig("figures/fig06_evolucion_neumonias_hosp_def.png", dpi=300)
plt.close()
print("Generada: figures/fig06_evolucion_neumonias_hosp_def.png")


# FIGURA 7: Proporción de Ceros por Variable y Período
fig, ax = plt.subplots(figsize=(13, 6))

indicators = [
    "ira_no_neumonia",
    "neumonias_men5",
    "neumonias_60mas",
    "hospitalizados_men5",
    "hospitalizados_60mas",
    "defunciones_men5",
    "defunciones_60mas",
]

labels_clean = [
    "IRA no neumonía (<5a)",
    "Neumonías (<5a)",
    "Neumonías (60+a)",
    "Hospitalizados (<5a)",
    "Hospitalizados (60+a)",
    "Defunciones (<5a)",
    "Defunciones (60+a)"
]

zero_pct_global = [(df[col] == 0).mean() * 100 for col in indicators]
zero_pct_pre = [(df[df["ano"] < 2020][col] == 0).mean() * 100 for col in indicators]
zero_pct_pan = [(df[df["ano"].between(2020, 2022)][col] == 0).mean() * 100 for col in indicators]
zero_pct_post = [(df[df["ano"] >= 2023][col] == 0).mean() * 100 for col in indicators]

x = np.arange(len(indicators))
width = 0.22

rects1 = ax.bar(x - width*1.5, zero_pct_pre, width, label="Pre-COVID (2000-2019)", color="#2b83ba")
rects2 = ax.bar(x - width*0.5, zero_pct_pan, width, label="Pandemia (2020-2022)", color="#d7191c")
rects3 = ax.bar(x + width*0.5, zero_pct_post, width, label="Post-COVID (2023-2025)", color="#fdae61")
rects4 = ax.bar(x + width*1.5, zero_pct_global, width, label="Global (2000-2025)", color="#2ca25f", alpha=0.7)

ax.set_ylabel("Proporción de Ceros (%) en Registros Semanales", fontsize=11)
ax.set_title("Prevalencia de Ceros por Indicador Epidemiológico y Período Histórico", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(labels_clean, rotation=25, ha="right", fontsize=9.5)
ax.set_ylim(0, 105)
ax.legend(loc="lower right", frameon=True)
ax.grid(True, linestyle="--", alpha=0.5, axis="y")

# Add text above IRA no neumonía
ax.text(x[0], 6, f"{zero_pct_global[0]:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#005a32")

plt.tight_layout()
fig.savefig("figures/fig07_proporcion_ceros_periodo.png", dpi=300)
plt.close()
print("Generada: figures/fig07_proporcion_ceros_periodo.png")


# FIGURA 8: Matriz de Correlación entre Indicadores Numéricos
fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(15, 6))

corr_cols = [
    "ira_no_neumonia",
    "neumonias_men5",
    "hospitalizados_men5",
    "defunciones_men5",
    "neumonias_60mas",
    "hospitalizados_60mas",
    "defunciones_60mas",
]

corr_labels = [
    "IRA no neum (<5a)",
    "Neum (<5a)",
    "Hosp (<5a)",
    "Def (<5a)",
    "Neum (60+a)",
    "Hosp (60+a)",
    "Def (60+a)",
]

pearson_corr = df[corr_cols].corr(method="pearson")
spearman_corr = df[corr_cols].corr(method="spearman")

mask = np.triu(np.ones_like(pearson_corr, dtype=bool))

sns.heatmap(pearson_corr, mask=mask, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax_p, cbar_kws={"shrink": 0.8}, xticklabels=corr_labels, yticklabels=corr_labels)
ax_p.set_title("A. Correlación Lineal de Pearson", fontsize=12, fontweight="bold")

sns.heatmap(spearman_corr, mask=mask, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax_s, cbar_kws={"shrink": 0.8}, xticklabels=corr_labels, yticklabels=corr_labels)
ax_s.set_title("B. Correlación por Rangos de Spearman", fontsize=12, fontweight="bold")

plt.figtext(0.1, -0.02, "LÍMITES DE INTERPRETACIÓN: La correlación a nivel de registro semanal/distrital refleja concurrencia de notificación y tamaño poblacional,\nno causalidad clínica directa sincrónica. Spearman es más adecuado dado el alto sesgo por exceso de ceros en formas graves.", wrap=True, fontsize=8.5, style="italic")

plt.tight_layout()
fig.savefig("figures/fig08_matriz_correlacion_indicadores.png", dpi=300, bbox_inches="tight")
plt.close()
print("Generada: figures/fig08_matriz_correlacion_indicadores.png")

print("Todas las 8 figuras han sido generadas y guardadas exitosamente en figures/.")
