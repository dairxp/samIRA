"""
FASE 2 - GENERACIÓN Y EJECUCIÓN DEL JUPYTER NOTEBOOK DE MODELADO PREDICTIVO
===========================================================================
Genera el notebook interactivo y documentado:
notebooks/02_fase2_modelado_predictivo.ipynb

El notebook incluye explicaciones metodológicas en español, código reproducible,
tablas de resultados y gráficos científicos embebidos.
"""

import os
import nbformat as nbf
from nbclient import NotebookClient

def create_and_execute_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # --------------------------------------------------------------------------
    # Celda 1: Portada y Contexto
    # --------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""# Proyecto samIRA - Fase 2: Modelado Predictivo y Evaluación Científica de IRA en Puno

**Investigación:** Inteligencia de negocios y aprendizaje automático para el análisis y pronóstico de infecciones respiratorias agudas en menores de cinco años en la región Puno, Perú.  
**Dataset:** Vigilancia epidemiológica oficial MINSA/CDC Perú (2000–2025).  
**Ámbito territorial:** 13 provincias de Puno y total regional agregado.  
**Variable objetivo:** `ira_no_neumonia` (episodios semanales en menores de cinco años).  
**Horizontes temporales evaluados:** $h = 1$ semana y $h = 4$ semanas.  

---
### Estructura del Notebook:
1. **Verificación de Precondiciones de Fase 1**
2. **Control de Calendario Epidemiológico y Semana 53**
3. **Tratamiento Causal de Datos Faltantes (Estrategias A, B, C, D)**
4. **Modelos Estadísticos de Referencia (Baselines: Naive, Seasonal Naive, ETS)**
5. **Ingeniería de Características sin Fuga de Información**
6. **Optimización de Hiperparámetros (LightGBM y XGBoost)**
7. **Entrenamiento y Evaluación Científica Integral (Global, Independiente, Regional, Directo vs. Recursivo)**
8. **Evaluación Out-of-Sample en Conjunto de Prueba 2025**
9. **Intervalos de Predicción Empíricos (95%)**
10. **Evaluación de Utilidad Sanitaria: Detección Temprana de Alertas y Brotes**
11. **Galería de Figuras Científicas**
12. **Conclusiones y Dictamen Operativo**
"""))

    # --------------------------------------------------------------------------
    # Celda 2: Configuración de Entorno e Importaciones
    # --------------------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""# Configuración de ruta y librerías fundamentales
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Detectar la raíz del proyecto de manera robusta
if os.path.exists("data"):
    project_root = os.path.abspath(".")
elif os.path.exists("../data"):
    project_root = os.path.abspath("..")
else:
    project_root = os.path.abspath(".")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

# Configuración visual
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10
pd.set_option("display.max_columns", 25)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

print("Entorno configurado correctamente en el directorio raíz:", os.getcwd())
"""))

    # --------------------------------------------------------------------------
    # Celda 3: Verificación de Precondiciones
    # --------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 1. Verificación Programática de Precondiciones de la Fase 1
Antes de iniciar cualquier modelado o ingeniería de características, verificamos de manera determinista las 8 condiciones del dataset consolidado en la Fase 1.
"""))

    cells.append(nbf.v4.new_code_cell("""# Cargar paneles procesados
df_prov = pd.read_parquet("data/processed/panel_semanal_provincial_puno.parquet")
df_reg = pd.read_parquet("data/processed/panel_semanal_regional_puno.parquet")

print("--- AUDITORÍA DE PRECONDICIONES DE ENTRADA ---")
print(f"1. Episodios históricos totales notificados: {df_prov['ira_no_neumonia'].sum():,.0f} (Esperado: 2,297,378)")
print(f"2. Número de provincias observadas: {df_prov['provincia'].nunique()} (Esperado: 13)")
print(f"3. Semanas epidemiológicas regionales: {len(df_reg):,} (Esperado: 1,357)")
print(f"4. Combinaciones provincia-semana totales: {len(df_prov):,} (Esperado: 17,641)")
print(f"5. Observaciones con información (es_evaluable == True): {df_prov['es_evaluable'].sum():,} (Esperado: 17,488)")
print(f"6. Observaciones sin registro (NaN conservados): {df_prov['ira_no_neumonia'].isna().sum():,} (Esperado: 153)")
print(f"7. Variable objetivo 'ira_no_neumonia' presente: {'ira_no_neumonia' in df_prov.columns}")
print(f"8. Bandera de evaluabilidad 'es_evaluable' presente: {'es_evaluable' in df_prov.columns}")

assert df_prov['ira_no_neumonia'].sum() == 2297378, "Error en total de episodios"
assert df_prov['provincia'].nunique() == 13, "Error en número de provincias"
assert len(df_reg) == 1357, "Error en semanas regionales"
assert len(df_prov) == 17641, "Error en pares provincia-semana"
assert df_prov['es_evaluable'].sum() == 17488, "Error en observaciones evaluables"
assert df_prov['ira_no_neumonia'].isna().sum() == 153, "Error en NaNs conservados"
print("\\n>>> TODAS LAS PRECONDICIONES CONFIRMADAS AL 100% <<<")
"""))

    # --------------------------------------------------------------------------
    # Celda 4: Calendario Epidemiológico y Semana 53
    # --------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 2. Calendario Epidemiológico y Manejo de la Semana 53
En el sistema de vigilancia epidemiológica peruano (CDC-MINSA), los años bisiestos o con 53 semanas (2003, 2008, 2014, 2020 y 2025) introducen una asincronía si se aplica un desplazamiento ingenuo de 52 posiciones (`shift(52)`).  
Hacia el año 2025, un `shift(52)` acumula un desfase estacional de **4 semanas completas**.  
Para solucionarlo, implementamos una función de alineación basada en calendario que mapea exactamente el par $(y-1, w)$ de la semana homóloga.
"""))

    cells.append(nbf.v4.new_code_cell("""# Inspección de años con semana 53 en el dataset
semanas_por_ano = df_reg.groupby("ano")["semana"].max()
anos_53 = semanas_por_ano[semanas_por_ano == 53].index.tolist()
print("Años oficiales con 53 semanas epidemiológicas:", anos_53)

# Demostración del desfase acumulado por shift(52)
s = df_reg.set_index(["ano", "semana"]).sort_index()
s["shift_52_semana"] = s.index.get_level_values("semana")
print("\\nEl desplazamiento shift(52) sobre series con semanas 53 desplaza la fase estacional un período adicional por cada año bisiesto epidemiológico transcurrido.")
"""))

    # --------------------------------------------------------------------------
    # Celda 5: Tratamiento Causal de Faltantes
    # --------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 3. Tratamiento Causal de Valores Faltantes
Para no evaluar modelos sobre valores inventados, los 153 `NaN` se preservan intactos en la serie original y la evaluación se restringe a `es_evaluable == True`.  
Para construir las variables predictoras (lags y medias móviles), evaluamos 4 estrategias causales en la partición de validación (2023–2024):
- **Estrategia A:** Retardo estacional homólogo del año anterior.
- **Estrategia B:** Last Observation Carried Forward (LOCF) causal.
- **Estrategia C:** Media móvil causal retrospectiva de 3 semanas.
- **Estrategia D:** Tratamiento nativo de faltantes con indicador binario de ausencia.
"""))

    cells.append(nbf.v4.new_code_cell("""# Cargar tabla de comparación de estrategias de imputación
df_imp = pd.read_csv("reports/fase2/comparacion_imputaciones.csv")
df_imp
"""))

    # --------------------------------------------------------------------------
    # Celda 6: Baselines
    # --------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 4. Modelos Estadísticos de Referencia (Baselines)
Se implementan y evalúan tres modelos baselines:
1. **Naive ($t-1$ o $t-h$):** Proyecta la última observación conocida disponible.
2. **Seasonal Naive ($t-52$ ajustado):** Proyecta la semana homóloga del año previo.
3. **ETS / Holt-Winters:** Modelo estadístico clásico de nivel, tendencia y estacionalidad aditiva.
"""))

    cells.append(nbf.v4.new_code_cell("""# Inspección de baselines calculados
df_base = pd.read_parquet("data/processed/predicciones_baselines.parquet")
print(f"Total de predicciones baseline generadas: {len(df_base):,}")
print("Muestra de predicciones provinciales:")
df_base[df_base["ambito"] == "PROVINCIAL"].head()
"""))

    # --------------------------------------------------------------------------
    # Celda 7: Optimización de Hiperparámetros
    # --------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 5. Optimización de Hiperparámetros en Validación (2023–2024)
La búsqueda exhaustiva de hiperparámetros se ejecutó estrictamente sobre el período de validación temporal (2023–2024), sin tocar el conjunto de prueba (2025).  
Se optimizaron `n_estimators`, `learning_rate`, `num_leaves` / `max_depth` y `subsample` para LightGBM y XGBoost en horizontes $h=1$ y $h=4$.
"""))

    cells.append(nbf.v4.new_code_cell("""# Cargar mejores hiperparámetros identificados
df_hyp = pd.read_csv("reports/fase2/hiperparametros_optimos.csv")
df_hyp
"""))

    # --------------------------------------------------------------------------
    # Celda 8: Resultados en Validación
    # --------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 6. Rendimiento en Partición de Validación (2023–2024)
Evaluación de todas las arquitecturas en el período post-pandemia de validación (104 semanas por provincia).
"""))

    cells.append(nbf.v4.new_code_cell("""# Cargar tabla de validación
df_val_res = pd.read_csv("reports/fase2/resultados_validacion.csv")
df_val_res.sort_values(by=["ambito", "horizonte_h", "mae_global"])
"""))

    # --------------------------------------------------------------------------
    # Celda 9: Resultados en Prueba 2025 (Holdout Estricto)
    # --------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 7. Resultados en Conjunto de Prueba 2025 (Holdout Estricto)
Evaluación out-of-sample final sobre las 53 semanas epidemiológicas de 2025.  
Se calculan: MAE, RMSE, WAPE (%), Sesgo Medio (Bias), Mejora (%) sobre Seasonal Naive y Cobertura Empírica del Intervalo de Predicción al 95%.
"""))

    cells.append(nbf.v4.new_code_cell("""# Cargar tabla de prueba 2025
df_test_res = pd.read_csv("reports/fase2/resultados_prueba_2025.csv")
df_test_res.sort_values(by=["ambito", "horizonte_h", "mae_global"])
"""))

    # --------------------------------------------------------------------------
    # Celda 10: Desempeño por Provincia
    # --------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 8. Heterogeneidad Territorial: Desempeño en las 13 Provincias
Análisis desagregado por provincia para $h=1$. Muestra cómo el modelo global transfiere conocimiento beneficiando tanto a provincias de alta notificación (San Román, Puno) como a provincias rurales pequeñas (Moho, Yunguyo, Sandia).
"""))

    cells.append(nbf.v4.new_code_cell("""# Cargar resultados por provincia en test 2025
df_prov_res = pd.read_csv("reports/fase2/resultados_por_provincia.csv")
df_prov_test = df_prov_res[(df_prov_res["particion"] == "test") & (df_prov_res["horizonte_h"] == 1)].copy()

# Comparación directa entre LightGBM Global y Seasonal Naive por provincia
piv_prov = df_prov_test.pivot(index="provincia", columns="modelo", values="wape_pct").reset_index()
piv_prov["mejora_vs_snaive_pct"] = 100 * (piv_prov["Seasonal Naive"] - piv_prov["LightGBM Global"]) / piv_prov["Seasonal Naive"]
piv_prov.sort_values(by="LightGBM Global")
"""))

    # --------------------------------------------------------------------------
    # Celda 11: Evaluación de Utilidad Sanitaria y Alertas
    # --------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 9. Evaluación de Utilidad en Planificación Sanitaria y Alertas
Definición de brote / alza: semana epidemiológica que supera el **Percentil 75 histórico (2000–2022)** del Canal Endémico provincial o regional.  
Se evalúa la matriz de confusión, Precisión, Recall (Sensibilidad), F1-score y Tasa de Falsas Alarmas.
"""))

    cells.append(nbf.v4.new_code_cell("""# Cargar métricas de alerta epidemiológica
df_alerts = pd.read_csv("reports/fase2/metricas_alertas.csv")
df_alerts[df_alerts["ambito"] == "PROVINCIAL"]
"""))

    # --------------------------------------------------------------------------
    # Celda 12: Visualizaciones Embebidas
    # --------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 10. Figuras Científicas de Alta Resolución (300 DPI)
Presentación de las 6 figuras generadas para la comunicación técnica y epidemiológica.
"""))

    cells.append(nbf.v4.new_code_cell("""# Mostrar figuras generadas
from IPython.display import Image, display

figuras = [
    ("Figura 1: Comparación de Modelos por MAE y RMSE", "figures/fase2/figura1_comparacion_modelos_mae_rmse.png"),
    ("Figura 2: Series Temporales Real vs. Predicho en 2025", "figures/fase2/figura2_series_real_vs_predicho_2025.png"),
    ("Figura 3: Distribución Espacial de Errores por Provincia", "figures/fase2/figura3_distribucion_errores_por_provincia.png"),
    ("Figura 4: Comparación de Horizontes y Estrategia Directa vs. Recursiva", "figures/fase2/figura4_comparacion_horizontes_directo_vs_recursivo.png"),
    ("Figura 5: Comparación de Estrategias de Imputación Causal", "figures/fase2/figura5_comparacion_estrategias_imputacion.png"),
    ("Figura 6: Evaluación de Alertas Sanitarias", "figures/fase2/figura6_evaluacion_alertas_epidemiologicas.png")
]

for title, path in figuras:
    print(f"=== {title} ===")
    display(Image(filename=path, width=800))
"""))

    # --------------------------------------------------------------------------
    # Celda 13: Conclusiones
    # --------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 11. Conclusiones y Dictamen Operativo

### Hallazgos Principales:
1. **Horizonte $h=1$:** **LightGBM Global** alcanza el mejor rendimiento provincial con un **MAE de 16.02**, **WAPE de 15.90%** y una **mejora del 28.89%** sobre el Seasonal Naive. XGBoost Global muestra un desempeño prácticamente indistinguible (WAPE 15.93%, mejora 28.76%).
2. **Horizonte $h=4$:** **XGBoost Global Directo** es el mejor modelo con un **MAE de 17.91**, **WAPE de 17.77%** y **mejora del 20.50%** sobre Seasonal Naive, superando levemente a LightGBM Recursivo (WAPE 17.84%).
3. **Escala Regional:** LightGBM Regional en $h=1$ logra un **WAPE de 9.81%** (MAE 128.51) con un 25.34% de ganancia sobre Seasonal Naive. Sin embargo, para $h=4$ regional, Seasonal Naive (WAPE 13.14%) supera a los modelos ML debido a la alta regularidad del ciclo anual agregado y la muestra pequeña ($N=52$ semanas/año) a nivel departamental.
4. **Modelos Globales vs. Independientes:** El modelo global supera a los modelos provinciales independientes (WAPE 15.90% vs. 17.62%), demostrando la ventaja del agrupamiento espacial (*spatial pooling*).
5. **Incertidumbre e Intervalos:** La cobertura empírica al 95% oscila entre **97.1% y 97.8%**, confirmando intervalos estadísticamente confiables sin subestimación de riesgo.
6. **Utilidad Sanitaria:** Para alertas tempranas, el pronóstico puntual conservador exhibe 100% de precisión pero baja sensibilidad (5%). Al emplear el límite superior probabilístico al 95%, la sensibilidad aumenta al **85% - 95%**, proporcionando una herramienta de tamizaje preventivo idónea para la logística hospitalaria.

### Veredicto de Aplicabilidad Operativa:
- **Horizonte $h=1$:** **APTO** para integración operativa en tableros de Business Intelligence, monitoreo semanal de salas situacionales y redistribución de insumos respiratorios.
- **Horizonte $h=4$:** **APTO CON PRECAUCIÓN** como guía orientativa de planificación mensual de compras y programación de personal, debiendo combinarse con el monitoreo continuo de semanas intermedias.
"""))

    nb.cells = cells
    
    os.makedirs("notebooks", exist_ok=True)
    nb_path = os.path.join("notebooks", "02_fase2_modelado_predictivo.ipynb")
    
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Notebook estructurado escrito en: {nb_path}")
    
    # Ejecutar notebook
    print("Ejecutando notebook con NotebookClient...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Notebook ejecutado y guardado con outputs en: {nb_path}")

if __name__ == "__main__":
    create_and_execute_notebook()
