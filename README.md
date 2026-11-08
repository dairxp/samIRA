# samIRA: Vigilancia Epidemiológica y Pronóstico de Infecciones Respiratorias Agudas en Puno

Proyecto de investigación científica aplicada para la auditoría, análisis exploratorio y pronóstico epidemiológico de episodios de Infecciones Respiratorias Agudas (IRA no neumonía) en las 13 provincias del departamento de Puno, Perú (2000–2025).

---

## 1. Estructura del Repositorio

El repositorio está organizado conforme a estándares de reproducibilidad académica:

```text
samIRA/
├── data/
│   ├── raw/
│   │   └── datos_abiertos_vigilancia_iras_2000_2025 PUNO.csv # Registro oficial MINSA NotiSP
│   └── processed/
│       ├── datos_iras_puno_limpio.parquet                # Dataset curado a nivel de registro
│       ├── panel_semanal_provincial_puno.parquet         # Panel balanceado (13 provincias x 1,357 semanas)
│       └── panel_semanal_regional_puno.parquet           # Panel agregado departamental (1,357 semanas)
├── notebooks/
│   ├── 01_limpieza_eda.ipynb                             # Auditoría, curación geográfica y EDA
│   ├── 02_modelado_evaluacion.ipynb                      # Modelos ML, baselines, optimización y evaluación
│   └── 03_reporting_analitico.ipynb                      # Reporting: analítica descriptiva, diagnóstica, predictiva y prescriptiva
├── models/
│   ├── lgbm_global_h1.joblib                             # LightGBM Global h=1
│   ├── lgbm_global_h4_direct.joblib                      # LightGBM Global h=4 Directo (corregido)
│   ├── xgboost_global_h1.joblib                          # XGBoost Global h=1
│   └── xgboost_global_h4_direct.joblib                   # XGBoost Global h=4 Directo (corregido)
├── results/
│   ├── predicciones_completas.parquet                    # Predicciones de todos los modelos
│   ├── predicciones_completas.csv                        # Predicciones unificadas en formato tabular
│   ├── metricas_validacion_modelos.csv                   # MAE, RMSE, WAPE y Bias en Validación (2023–2024)
│   ├── metricas_prueba_2025.csv                          # MAE, RMSE, WAPE y Bias en Prueba (2025)
│   ├── cobertura_intervalos_95.csv                       # Cobertura empírica de intervalos al 95%
│   └── evaluacion_alertas_epidemiologicas.csv            # Detección de brotes epidémicos (P75)
├── pyproject.toml                                        # Dependencias y configuración del entorno
├── uv.lock                                               # Resolución determinista de paquetes
└── README.md                                             # Documentación de reproducción
```

---

## 2. Requisitos e Instalación

El proyecto utiliza [`uv`](https://docs.astral.sh/uv/) como gestor de paquetes y entornos virtuales sobre **Python 3.12**.

### Clonar y preparar el entorno
```bash
# 1. Clonar el repositorio
git clone <URL_DEL_REPOSITORIO>
cd samIRA

# 2. Sincronizar el entorno virtual con uv
uv sync
```

---

## 3. Ejecución de los Cuadernos de Investigación

Todo el código analítico, transformaciones, verificaciones con `assert`, tablas y figuras reside directamente dentro de los cuadernos Jupyter en `notebooks/`.

### Opción A: Ejecución Interactiva en Jupyter
```bash
uv run jupyter lab
```
Abra y ejecute en orden secuencial:
1. `notebooks/01_limpieza_eda.ipynb`
2. `notebooks/02_modelado_evaluacion.ipynb`
3. `notebooks/03_reporting_analitico.ipynb`

### Opción B: Ejecución Completa por Línea de Comandos
```bash
# Ejecutar Cuaderno 01 (Auditoría, curación y exportación de paneles Parquet)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_limpieza_eda.ipynb

# Ejecutar Cuaderno 02 (Ajuste de modelos, validación causal, optimización y exportación de resultados)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_modelado_evaluacion.ipynb

# Ejecutar Cuaderno 03 (Reporting integral: analítica descriptiva, diagnóstica, predictiva y prescriptiva)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_reporting_analitico.ipynb
```

---

## 4. Metodología Científica y Correcciones Críticas

### A. Preservación Estricta de Faltantes
- Se identificaron exactamente **153 semanas sin notificación** en el sistema NotiSP MINSA.
- Estas observaciones se conservan estrictamente como `NaN` (`es_evaluable = False`). Ninguna ausencia se sustituye por cero para el cómputo de métricas out-of-sample.

### B. Corrección de Contaminación Temporal en $h=4$
- **Diagnóstico:** En versiones preliminares existía una superposición de 52 observaciones entre entrenamiento y validación al filtrar el entrenamiento por año de origen $t$ y la validación por año objetivo $t+4$.
- **Solución implementada:** Las particiones de corte se realizan de forma unívoca sobre la variable `target_ano`:
  - **Entrenamiento:** $target\_ano \le 2022$ ($2001 \le target\_ano \le 2022$).
  - **Validación:** $target\_ano \in [2023, 2024]$.
  - **Prueba:** $target\_ano == 2025$.
- Las armónicas de Fourier se computan sobre la **semana objetivo** $w_{t+h}$ para representar fielmente la fase estacional a predecir.
- Los modelos para $h=4$ fueron reentrenados y evaluados bajo esta partición estricta sin fugas.

### C. Reporting y las Cuatro Analíticas (Cuaderno 03)
- **Descriptiva:** Caracterización multianual y territorial (2000–2025, 2,297,378 casos) en 13 provincias.
- **Diagnóstica:** Patrones temporales e interanuales, shock COVID-19 (2020), y separación estricta entre hechos, asociaciones e hipótesis no comprobables.
- **Predictiva:** Proyecciones out-of-sample en $h=1$ y $h=4$ directo con intervalos de predicción al 95%.
- **Prescriptiva:** Simulación cuantitativa de capacidad sanitaria bajo supuestos académicos y escenarios de demanda (baja, central, alta) para priorización logística.

---

## 5. Resumen de Desempeño Predictivo

Métricas calculadas exclusivamente sobre observaciones evaluables (`es_evaluable == True`):

### Validación (2023–2024, $N = 1,347$)
| Horizonte | Modelo | MAE | RMSE | WAPE (%) | Bias |
|---|---|---|---|---|---|
| **$h=1$** | **LightGBM Global** | **19.33** | **30.32** | **18.25%** | +1.69 |
| $h=1$ | XGBoost Global | 19.71 | 31.65 | 18.61% | +1.24 |
| $h=1$ | LightGBM Independiente | 21.14 | 32.93 | 19.96% | +2.97 |
| $h=1$ | ETS / Holt-Winters | 21.18 | 33.73 | 20.00% | +0.21 |
| $h=1$ | Naive | 23.06 | 36.05 | 21.77% | -0.05 |
| $h=1$ | Seasonal Naive | 31.47 | 49.25 | 29.71% | -7.44 |
| **$h=4$** | **LightGBM Global Directo** | **24.51** | **39.32** | **23.14%** | +4.98 |
| $h=4$ | XGBoost Global Directo | 24.61 | 39.82 | 23.24% | +3.51 |
| $h=4$ | ETS / Holt-Winters | 25.65 | 41.82 | 24.21% | +0.56 |
| $h=4$ | Naive | 31.07 | 51.54 | 29.33% | +0.29 |
| $h=4$ | Seasonal Naive | 31.49 | 49.26 | 29.72% | -7.45 |

### Prueba Fuera de Muestra (2025, $N = 689$)
| Horizonte | Modelo | MAE | RMSE | WAPE (%) | Bias |
|---|---|---|---|---|---|
| **$h=1$** | **XGBoost Global** | **16.03** | **23.89** | **15.91%** | +2.69 |
| $h=1$ | LightGBM Global | 16.26 | 24.48 | 16.14% | +3.04 |
| $h=1$ | ETS / Holt-Winters | 17.62 | 25.91 | 17.49% | -0.02 |
| $h=1$ | Seasonal Naive | 22.52 | 33.73 | 22.36% | +6.38 |
| **$h=4$** | **XGBoost Global Directo** | **19.11** | **29.65** | **18.97%** | +6.47 |
| $h=4$ | LightGBM Global Directo | 19.35 | 30.18 | 19.21% | +7.47 |
| $h=4$ | ETS / Holt-Winters | 21.78 | 33.11 | 21.62% | -0.68 |
| $h=4$ | Seasonal Naive | 22.52 | 33.73 | 22.36% | +6.38 |
