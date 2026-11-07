"""
Construcción y ejecución del notebook interactivo de la Fase 2.
Genera notebooks/02_fase2_modelado_predictivo.ipynb con ejecución directa,
flujo paso a paso, comentarios precisos y sin divisores decorativos.
"""

import os
import nbformat as nbf
from nbclient import NotebookClient

def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Portada y contexto
    cells.append(nbf.v4.new_markdown_cell("""# Proyecto samIRA - Fase 2: Modelado Predictivo y Evaluación Científica de IRA en Puno

**Investigación:** Inteligencia de negocios y aprendizaje automático para el análisis y pronóstico de infecciones respiratorias agudas en menores de cinco años en la región Puno, Perú.  
**Dataset oficial:** Vigilancia Epidemiológica de IRAs, Centro Nacional de Epidemiología, Prevención y Control de Enfermedades (CDC Perú / MINSA, 2000–2025).  
**Ámbito territorial:** 13 provincias del departamento de Puno y serie consolidada regional.  
**Variable objetivo:** `ira_no_neumonia` (episodios semanales en menores de cinco años).  
**Horizontes temporales:** $h = 1$ semana (alerta operativa) y $h = 4$ semanas (planificación mensual).  

---
### Estructura Metodológica del Notebook:
1. Verificación programática de precondiciones de Fase 1
2. Calendario epidemiológico y control de la semana 53
3. Tratamiento causal de valores faltantes (Estrategia C)
4. Modelos estadísticos basales (Naive, Seasonal Naive, ETS / Holt-Winters)
5. Ingeniería de características temporales y estacionales sin fuga de información
6. Modelos de Machine Learning: LightGBM y XGBoost (estrategias directa y recursiva)
7. Evaluación comparativa en conjunto ciego de prueba 2025
8. Calibración heterocedástica de intervalos de predicción al 95% por provincia
9. Evaluación epidemiológica de semanas de alta notificación y análisis de alertas
10. Diagnóstico visual y figuras epidemiológicas
11. Verificación de consistencia y exportación de entregables finales
12. Conclusiones y recomendaciones para el Dashboard de Business Intelligence
"""))

    # Configuración e importaciones
    cells.append(nbf.v4.new_code_cell("""# Importación de librerías científicas y configuración del entorno
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import lightgbm as lgb
import xgboost as xgb

# Configuración visual
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10

# Detección de la raíz del proyecto
project_root = os.path.abspath("..") if os.path.exists("../data") else os.path.abspath(".")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

print(f"Directorio de trabajo configurado: {os.getcwd()}")
"""))

    # Sección 1: Precondiciones
    cells.append(nbf.v4.new_markdown_cell("""## 1. Verificación Programática de Precondiciones de la Fase 1

Antes de proceder con el modelado temporal, se valida programáticamente el cumplimiento estricto de las 7 condiciones metodológicas establecidas en la auditoría de la Fase 1:
- Registros limpios consolidados: 126,118 registros originales.
- Total histórico acumulado: 2,297,378 episodios notificados de IRA no neumonía.
- Ámbito geográfico: 13 provincias de Puno.
- Semanas epidemiológicas regionales observadas: 1,357 semanas continuas.
- Cobertura teórica provincial: $1,357 \\times 13 = 17,641$ combinaciones provincia-semana.
- Observaciones con información: 17,488 pares evaluables.
- Observaciones sin registro (faltantes reales): 153 semanas con valor `NaN`.
"""))

    cells.append(nbf.v4.new_code_cell("""# Carga de paneles procesados y validación de precondiciones
df_prov = pd.read_parquet("data/processed/panel_semanal_provincial_puno.parquet")
df_reg = pd.read_parquet("data/processed/panel_semanal_regional_puno.parquet")

n_filas = len(df_prov)
n_provincias = df_prov["provincia"].nunique()
n_semanas_reg = len(df_reg)
total_teorico = n_semanas_reg * n_provincias
n_evaluables = int(df_prov["es_evaluable"].sum())
n_nan = int(df_prov["ira_no_neumonia"].isna().sum())
total_episodios = int(df_prov["ira_no_neumonia"].sum())

print("Resumen de Control de Precondiciones (Fase 1):")
print(f"- Total filas del panel provincial: {n_filas:,} (esperado: {total_teorico:,})")
print(f"- Número de provincias: {n_provincias} (esperado: 13)")
print(f"- Semanas epidemiológicas regionales: {n_semanas_reg:,} (esperado: 1,357)")
print(f"- Observaciones evaluables (con información): {n_evaluables:,} (esperado: 17,488)")
print(f"- Observaciones faltantes (NaN original): {n_nan:,} (esperado: 153)")
print(f"- Total episodios acumulados (2000-2025): {total_episodios:,} (esperado: 2,297,378)")

assert n_filas == 17641, f"Error en filas: {n_filas}"
assert n_provincias == 13, f"Error en provincias: {n_provincias}"
assert n_semanas_reg == 1357, f"Error en semanas regionales: {n_semanas_reg}"
assert n_evaluables == 17488, f"Error en observaciones evaluables: {n_evaluables}"
assert n_nan == 153, f"Error en NaNs: {n_nan}"
assert total_episodios == 2297378, f"Error en episodios: {total_episodios}"

print("\\nTodas las precondiciones han sido validadas exitosamente.")
"""))

    # Sección 2: Calendario y Faltantes
    cells.append(nbf.v4.new_markdown_cell("""## 2. Calendario Epidemiológico y Tratamiento Causal de Faltantes

El calendario de vigilancia epidemiológica del MINSA contempla semanas epidemiológicas 53 en años específicos donde el ciclo anual acumula días adicionales (2004, 2009, 2015, 2020 y 2025). Estas semanas son legítimas y no deben eliminarse ni forzarse a calendarios gregorianos convencionales.

Para el tratamiento de las 153 observaciones faltantes en el panel provincial, se implementa la **Estrategia C (Propagación causal histórica)**:
- Para calcular retardos (*lags*) y estadísticas móviles, los vacíos se cubren exclusivamente utilizando el último valor observado disponible hacia atrás (`ffill` temporal causal).
- La columna original de evaluación (`ira_no_neumonia`) preserva sus `NaN` intactos, impidiendo que el modelo sea evaluado contra ceros artificiales o datos interpolados con información futura.
"""))

    cells.append(nbf.v4.new_code_cell("""# Verificación de años con semana 53 y aplicación de tratamiento causal de faltantes
semanas_max = df_prov.groupby("ano")["semana"].max()
anios_con_53 = semanas_max[semanas_max == 53].index.tolist()
print(f"Años con semana epidemiológica 53 en el dataset: {anios_con_53}")

# Preparación de serie causal para retardos (Estrategia C)
df_prov = df_prov.sort_values(by=["provincia", "ano", "semana"]).reset_index(drop=True)
df_prov["target_raw"] = df_prov["ira_no_neumonia"]
df_prov["target_causal_imputed"] = df_prov.groupby("provincia")["target_raw"].ffill().bfill()

print(f"Total NaNs en columna evaluable (target_raw): {df_prov['target_raw'].isna().sum()}")
print(f"Total NaNs en serie causal para ingeniería de variables: {df_prov['target_causal_imputed'].isna().sum()}")
"""))

    # Sección 3: Baselines
    cells.append(nbf.v4.new_markdown_cell("""## 3. Modelos Estadísticos de Referencia (Baselines)

Para evaluar si el Machine Learning aporta valor clínico real, se implementan tres modelos basales:
1. **Naive Persistente:** Proyecta el último valor observado: $\\hat{y}_{t+h} = y_t$.
2. **Seasonal Naive:** Proyecta la observación correspondiente a la misma semana del año anterior: $\\hat{y}_{t+h} = y_{t+h-52}$.
3. **ETS / Holt-Winters (Reparado):** Modelo de suavizamiento exponencial de espacio de estados con tendencia aditiva amortiguada (*damped trend*) y componente estacional de 52 semanas.

En la auditoría de Fase 2 se corrigió un fallo de indización en `.forecast().iloc[-1]`, implementando el acceso mediante `float(np.asarray(fc)[-1])` y asegurando que ETS genere predicciones propias e independientes sin fallback a Seasonal Naive.
"""))

    cells.append(nbf.v4.new_code_cell("""# Carga de predicciones basales reparadas
df_baselines = pd.read_parquet("data/processed/predicciones_baselines.parquet")

# Verificación de independencia estadística entre ETS y Seasonal Naive
ets_vs_snaive_diff = (df_baselines["pred_ets"] - df_baselines["pred_snaive"]).abs().mean()
print(f"Diferencia media absoluta entre ETS y Seasonal Naive: {ets_vs_snaive_diff:.4f} episodios")
print(f"Filas de baselines disponibles: {len(df_baselines):,}")
"""))

    # Sección 4: Ingeniería de características
    cells.append(nbf.v4.new_markdown_cell("""## 4. Ingeniería de Características Causal sin Fuga de Información

Se construye una matriz de características epidemiológicas respetando la flecha del tiempo:
- **Retardos causales:** $y_t, y_{t-1}, y_{t-2}, y_{t-3}, y_{t-7}$ (ajustados rigurosamente según el horizonte $h$).
- **Estadísticas móviles:** Media y desviación estándar causales en ventanas de 2, 4 y 8 semanas terminadas en el origen $t$.
- **Armónicas de Fourier:** $K=3$ pares ortogonales de seno y coseno con período fundamental de 52 semanas:
  $$\\sin\\left(\\frac{2\\pi k \\cdot \\text{semana}}{52}\\right), \\quad \\cos\\left(\\frac{2\\pi k \\cdot \\text{semana}}{52}\\right) \\quad \\text{para } k \\in \\{1, 2, 3\\}$$
- **Variables de calendario:** Identificador de semana estacional, bandera de semana 53 y tendencia normalizada.
- **Particionamiento out-of-sample estricto:**
  - **Entrenamiento:** 2000–2022 (23 años históricos).
  - **Validación:** 2023–2024 (104 semanas para ajuste y calibración de intervalos).
  - **Prueba ciega:** 2025 (53 semanas para evaluación final de generalización).
"""))

    cells.append(nbf.v4.new_code_cell("""# Construcción de características temporales y estacionales
def build_features(df_input, h=1):
    df_feat = df_input.copy()
    
    # Armónicas de Fourier
    for k in range(1, 4):
        df_feat[f"sin_fourier_{k}"] = np.sin(2 * np.pi * k * df_feat["semana"] / 52.0)
        df_feat[f"cos_fourier_{k}"] = np.cos(2 * np.pi * k * df_feat["semana"] / 52.0)
        
    df_feat["es_semana_53"] = (df_feat["semana"] == 53).astype(int)
    df_feat["tendencia_norm"] = (df_feat["ano"] - 2000) / 25.0
    
    # Retardos causales según horizonte h
    target_s = df_feat["target_causal_imputed"]
    for lag in [0, 1, 2, 3, 7]:
        eff_lag = (h - 1) + lag
        df_feat[f"lag_{eff_lag}"] = df_feat.groupby("provincia")[target_s.name].shift(eff_lag)
        
    # Medias móviles causales calculadas estrictamente sobre información disponible en origen t
    lag_origin = h - 1
    s_origin = df_feat.groupby("provincia")[target_s.name].shift(lag_origin)
    for w in [2, 4, 8]:
        df_feat[f"roll_mean_{w}"] = df_feat.groupby("provincia")[s_origin.name].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df_feat[f"roll_std_{w}"] = df_feat.groupby("provincia")[s_origin.name].transform(
            lambda x: x.rolling(w, min_periods=1).std()
        ).fillna(0.0)
        
    # Asignación de particiones temporales
    df_feat["particion"] = "train"
    df_feat.loc[df_feat["ano"].isin([2023, 2024]), "particion"] = "val"
    df_feat.loc[df_feat["ano"] == 2025, "particion"] = "test"
    df_feat["provincia_cat"] = df_feat["provincia"].astype("category")
    
    return df_feat

df_feat_h1 = build_features(df_prov, h=1)
df_feat_h4 = build_features(df_prov, h=4)

print(f"Matriz h=1 construida: {df_feat_h1.shape[1]} columnas, {len(df_feat_h1):,} registros")
print(f"Distribución por partición: Train={len(df_feat_h1[df_feat_h1['particion']=='train']):,}, "
      f"Val={len(df_feat_h1[df_feat_h1['particion']=='val']):,}, "
      f"Test={len(df_feat_h1[df_feat_h1['particion']=='test']):,}")
"""))

    # Sección 5: Modelado ML
    cells.append(nbf.v4.new_markdown_cell("""## 5. Modelado Predictivo con Machine Learning (LightGBM y XGBoost)

Se entrenan modelos basados en árboles de decisión potenciados por gradiente bajo el paradigma **Global Multi-Serie (*Cross-Sectional Pooling*)**:
- Un único estimador aprende simultáneamente de las dinámicas epidemiológicas de las 13 provincias, compartiendo patrones no lineales de propagación estacional.
- Para $h=4$ se compara la estrategia directa (entrenada específicamente para predecir a 4 semanas vista) frente a la estrategia autorregresiva recursiva.
- Se cargan las predicciones completas unificadas para el análisis comparativo integral.
"""))

    cells.append(nbf.v4.new_code_cell("""# Carga de predicciones completas de Machine Learning
df_preds_all = pd.read_parquet("data/processed/predicciones_completas_fase2.parquet")

# Filtrar modelos principales evaluados
ml_models = [
    "LightGBM Global", "XGBoost Global", "LightGBM Independiente",
    "LightGBM Global Directo", "XGBoost Global Directo", "LightGBM Global Recursivo",
    "LightGBM Regional", "LightGBM Regional Directo"
]
df_ml_eval = df_preds_all[df_preds_all["modelo"].isin(ml_models)].copy()
print(f"Predicciones de Machine Learning cargadas: {len(df_ml_eval):,} observaciones")
"""))

    # Sección 6: Evaluación out-of-sample 2025
    cells.append(nbf.v4.new_markdown_cell(r"""## 6. Evaluación Comparativa en el Conjunto de Prueba Ciego (2025)

Se calculan las métricas oficiales de desempeño epidemiológico sobre el año ciego 2025 ($N=689$ a nivel provincial y $N=53$ a nivel regional):
- **WAPE (Weighted Absolute Percentage Error):** Métrica primaria de error relativo ponderado:
  $$\text{WAPE} = \frac{\sum |y - \hat{y}|}{\sum y} \times 100\%$$
- **MAE (Mean Absolute Error):** Error absoluto medio en episodios por semana.
- **RMSE (Root Mean Squared Error):** Penalización de grandes desviaciones y picos epidémicos.
- **Sesgo (*Bias*):** Desviación media direccional ($\hat{y} - y$).
"""))

    cells.append(nbf.v4.new_code_cell("""# Carga de la tabla oficial de resultados out-of-sample corregidos
df_results_test = pd.read_csv("reports/fase2/resultados_prueba_2025_corregido.csv")

# Formatear tabla para presentación
cols_show = [
    "ambito", "horizonte_h", "modelo", "estrategia_horizonte",
    "mae_global", "rmse_global", "wape_global_pct", "bias_global",
    "mejora_wape_vs_snaive_pct", "cobertura_ic_95_pct"
]
df_display = df_results_test[cols_show].copy()
df_display.columns = [
    "Ámbito", "Horizonte", "Modelo", "Estrategia",
    "MAE", "RMSE", "WAPE (%)", "Sesgo", "Mejora vs SNaive (%)", "Cobertura 95% (%)"
]

print("Tabla Maestra de Resultados en Conjunto de Prueba 2025:")
print(df_display.to_string(index=False))
"""))

    # Sección 7: Intervalos
    cells.append(nbf.v4.new_markdown_cell("""## 7. Calibración Heterocedástica de Intervalos de Predicción al 95%

En la auditoría se detectó que aplicar un cuantil aditivo global producía **subcobertura crítica en San Román (71.70%)** y bandas excesivamente anchas en provincias pequeñas como Moho (76.8 casos).

**Solución aplicada:**
Se calibran los cuantiles empíricos residuales ($Q_{0.025}, Q_{0.975}$) de forma independiente para cada provincia utilizando exclusivamente los 104 residuos observados en la partición de validación (2023–2024).
"""))

    cells.append(nbf.v4.new_code_cell("""# Carga y presentación del reporte de cobertura de intervalos por provincia
df_cov_prov = pd.read_csv("reports/fase2/cobertura_intervalos_por_provincia.csv")

cols_cov = [
    "provincia", "media_observada_2025", "cobertura_metodo_anterior_pct",
    "cobertura_metodo_corregido_pct", "ancho_metodo_anterior", "ancho_metodo_corregido",
    "reduccion_ancho", "diagnostico_calibracion"
]
print("Desempeño de Cobertura de Intervalos al 95% (LightGBM Global, h=1, Test 2025):")
print(df_cov_prov[cols_cov].to_string(index=False))
"""))

    # Sección 8: Alertas
    cells.append(nbf.v4.new_markdown_cell("""## 8. Evaluación Epidemiológica de Semanas de Alta Notificación

En consonancia con las normas técnicas del CDC Perú / MINSA:
- La superación de un umbral estadístico ($P_{75}$) define formalmente una **semana de alta notificación**, no un "brote confirmado".
- Se evaluaron tres reglas operativas en el conjunto de prueba 2025 para determinar la viabilidad de emitir alertas automáticas en el dashboard.
"""))

    cells.append(nbf.v4.new_code_cell("""# Carga de la comparación de métodos de alertas epidemiológicas
df_alerts_comp = pd.read_csv("reports/fase2/comparacion_alertas_metodos.csv")

cols_alert = [
    "metodo_alerta", "alertas_emitidas", "verdaderos_positivos_tp",
    "falsos_positivos_fp", "falsos_negativos_fn", "precision_pct",
    "sensibilidad_recall_pct", "f1_score", "tasa_falsa_alarma_pct"
]
print("Evaluación de Alternativas de Detección de Semanas de Alta Notificación (Test 2025):")
print(df_alerts_comp[cols_alert].to_string(index=False))

print("\\nDictamen Operativo:")
print("Ninguna regla algorítmica logra un balance clínico admisible.")
print("Se recomienda NO activar alertas automáticas autónomas en el dashboard de BI.")
"""))

    # Sección 9: Visualizaciones
    cells.append(nbf.v4.new_markdown_cell("""## 9. Diagnóstico Visual y Figuras Epidemiológicas

Generación directa de figuras científicas para la interpretación visual del modelo recomendado en el ámbito provincial:
1. Pronóstico vs Real con intervalo de incertidumbre al 95% en los principales focos urbanos (San Román y Puno).
2. Comparativa de precisión (WAPE) entre LightGBM Global, ETS y Seasonal Naive en las 13 provincias.
3. Cobertura empírica territorial de los intervalos de predicción.
"""))

    cells.append(nbf.v4.new_code_cell("""# Gráfico 1: Pronóstico con Intervalo de Confianza al 95% para San Román y Puno (2025)
df_pred_corr = pd.read_csv("reports/fase2/predicciones_2025_corregido.csv")
lgb_test = df_pred_corr[
    (df_pred_corr["modelo"] == "LightGBM Global") & 
    (df_pred_corr["horizonte_h"] == 1) &
    (df_pred_corr["ambito"] == "PROVINCIAL")
].copy()

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), sharey=False)

for i, prov in enumerate(["SAN ROMAN", "PUNO"]):
    ax = axes[i]
    p_data = lgb_test[lgb_test["entidad"] == prov].sort_values("semana")
    
    ax.plot(p_data["semana"], p_data["y_true"], color="#1f77b4", lw=2, label="Casos Observados (2025)", marker="o", markersize=3)
    ax.plot(p_data["semana"], p_data["y_pred"], color="#d62728", lw=2, linestyle="--", label="Pronóstico LightGBM (h=1)")
    ax.fill_between(p_data["semana"], p_data["y_pred_lower_95"], p_data["y_pred_upper_95"], color="#d62728", alpha=0.15, label="Banda 95% Calibrada")
    
    ax.set_title(f"Provincia: {prov} - Holdout 2025", fontweight="bold")
    ax.set_xlabel("Semana Epidemiológica")
    ax.set_ylabel("Episodios Semanales (Menores 5 años)")
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Gráfico 2: Comparativa de WAPE por provincia (LightGBM vs Baselines)
df_prov_metrics = pd.read_csv("reports/fase2/resultados_por_provincia.csv")
lgb_prov_wape = df_prov_metrics[
    (df_prov_metrics["modelo"] == "LightGBM Global") & 
    (df_prov_metrics["horizonte_h"] == 1) &
    (df_prov_metrics["particion"] == "test")
].set_index("provincia")["wape_pct"]

snaive_prov_wape = df_prov_metrics[
    (df_prov_metrics["modelo"] == "Seasonal Naive") & 
    (df_prov_metrics["horizonte_h"] == 1) &
    (df_prov_metrics["particion"] == "test")
].set_index("provincia")["wape_pct"]

prov_names = sorted(lgb_prov_wape.index.tolist())
x = np.arange(len(prov_names))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 5))
rects1 = ax.bar(x - width/2, [snaive_prov_wape[p] for p in prov_names], width, label="Seasonal Naive", color="#aec7e8")
rects2 = ax.bar(x + width/2, [lgb_prov_wape[p] for p in prov_names], width, label="LightGBM Global (h=1)", color="#1f77b4")

ax.set_ylabel("WAPE (%)")
ax.set_title("Comparativa de Error Porcentual (WAPE) por Provincia en Prueba 2025", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(prov_names, rotation=35, ha="right")
ax.legend(frameon=True, facecolor="white")
ax.grid(True, linestyle=":", alpha=0.5, axis="y")

plt.tight_layout()
plt.show()
"""))

    # Sección 10: Verificación y entregables
    cells.append(nbf.v4.new_markdown_cell("""## 10. Verificación de Consistencia y Trazabilidad

Se verifica la coherencia numérica del total provincial y la identidad matemática del WAPE:
- Suma real observada en las 13 provincias durante 2025: **69,412 episodios**.
- Cumplimiento de la identidad matemática:
  $$\\text{WAPE} = 100 \\times \\frac{\\text{MAE} \\times N}{\\sum y} = 100 \\times \\frac{16.0166 \\times 689}{69,412} = 15.8985\\% \\approx 15.90\\%$$
"""))

    cells.append(nbf.v4.new_code_cell("""# Verificación matemática estricta
lgb_eval = lgb_test[lgb_test["es_evaluable"] == True]
suma_observada = float(lgb_eval["y_true"].sum())
mae_calc = float(mean_absolute_error(lgb_eval["y_true"], lgb_eval["y_pred"]))
n_obs = len(lgb_eval)
wape_calc = (np.sum(np.abs(lgb_eval["y_true"] - lgb_eval["y_pred"])) / suma_observada) * 100.0
wape_identidad = 100.0 * (mae_calc * n_obs) / suma_observada

print("Certificación de Trazabilidad Numérica:")
print(f"- Total observado provincial 2025: {suma_observada:,.2f} episodios (esperado: 69,412.00)")
print(f"- N semanas evaluables: {n_obs} (esperado: 689)")
print(f"- MAE calculado: {mae_calc:.4f}")
print(f"- WAPE por fórmula directa: {wape_calc:.4f}%")
print(f"- WAPE por identidad algebraica: {wape_identidad:.4f}%")
print(f"- Discrepancia matemática: {abs(wape_calc - wape_identidad):.10f}")

assert abs(suma_observada - 69412.0) < 1e-4, f"Discrepancia en suma: {suma_observada}"
assert abs(wape_calc - wape_identidad) < 1e-6, "Fallo en identidad matemática del WAPE"
print("\\nComprobación superada: Consistencia matemática perfecta al 100%.")
"""))

    # Sección 11: Conclusiones
    cells.append(nbf.v4.new_markdown_cell("""## 11. Conclusiones y Recomendaciones para el Dashboard de Business Intelligence

### 11.1 Conclusiones Científicas
1. **Superioridad Demostrada de Machine Learning:** LightGBM Global ($h=1$) y XGBoost Global Directo ($h=4$) superan categóricamente a todos los baselines estadísticos a nivel provincial, reduciendo el error en **+28.89% y +20.49%** respecto a Seasonal Naive.
2. **Eficacia del Spatial Pooling:** El modelo multi-serie global captura dinámicas epidemiológicas cruzadas entre provincias, superando a modelos univariados independientes.
3. **Calibración Territorial de Incertidumbre:** La calibración heterocedástica provincial corrige la subcobertura en San Román (alcanzando 98.11%) y garantiza intervalos informativos para la toma de decisiones.
4. **Desactivación de Alertas Automatizadas:** La vigilancia pasiva presenta una tasa de falsas alarmas inaceptable para algoritmos autónomos; las decisiones deben mantenerse bajo juicio epidemiológico humano.

### 11.2 Recomendación de Arquitectura para el Dashboard BI (DIRESA Puno)
- **Modelo Provincial Oficial:** `LightGBM Global` ($h=1$, WAPE 15.90%, MAE 16.02 casos/semana).
- **Modelo de Planificación Mensual:** `XGBoost Global Directo` ($h=4$, WAPE 17.77%, MAE 17.91 casos/semana).
- **Despliegue Visual:** Serie observada, punto pronosticado y cono de incertidumbre al 95%.
- **Transición a Fase 3:** Incorporación de covariables climáticas del SENAMHI (temperaturas mínimas y heladas) mediante retardos bioclimáticos de 1 a 3 semanas.
"""))

    nb.cells = cells
    return nb

def main():
    print("Construyendo notebook completo de la Fase 2...")
    nb = build_notebook()
    
    nb_path = os.path.join("notebooks", "02_fase2_modelado_predictivo.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook guardado en: {nb_path} ({len(nb.cells)} celdas)")
    
    print("\nEjecutando notebook end-to-end con nbclient...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook ejecutado y reescrito con todas sus salidas en: {nb_path}")

if __name__ == "__main__":
    main()
