# Resumen del Análisis Exploratorio de Datos (EDA – Versión Corregida y Reconciliada)
## Vigilancia de Infecciones Respiratorias Agudas (IRAs) en Menores de 5 Años – Región Puno (2000–2025)

**Autor:** Científico de Datos Senior en Epidemiología y Business Intelligence  
**Dataset Base:** `data/processed/datos_iras_puno_limpio.parquet` (126,118 registros procesados)  
**Paneles Modelables:** `data/processed/panel_semanal_provincial_puno.parquet` (17,641 pares) y `panel_semanal_regional_puno.parquet` (1,357 semanas)  
**Trazabilidad:** Versión previa archivada en [`reports/resumen_eda_v1.md`](file:///F:/dxp-projects/DevBasics/R%20y%20statistics/samIRA/reports/resumen_eda_v1.md)  

---

## 1. Síntesis Epidemiológica y Hallazgos Descriptivos

El análisis exploratorio sobre los 26 años completos de vigilancia NotiSP en Puno revela las siguientes características observacionales fundamentales:

1. **Volumen Acumulado Reconciliado:**  
   Entre 2000 y 2025 se notificaron **2,297,378 episodios de `ira_no_neumonia`** en menores de 5 años. Se reconcilió de forma definitiva la discrepancia de 732 episodios detectada en la versión preliminar, confirmando que la suma directa de la columna en el archivo fuente coincide exactamente con la suma agregada de las 13 provincias ($2,297,378$).
2. **Ciclicidad Anual Observada (SE 20 a 32):**  
   La serie temporal regional de 1,357 semanas continuas muestra una oscilación periódica regular con picos sistemáticos en el segundo trimestre de cada año (semanas epidemiológicas 20 a 32). Esta elevación estacional coincide temporalmente con los meses de invierno en el altiplano (mayo a agosto). La hipótesis médica tradicional atribuye este incremento al estrés por frío y al confinamiento en ambientes cerrados con biomasa; no obstante, el dataset no contiene lecturas meteorológicas, por lo que dicha relación es una hipótesis explicativa plausible y no una conclusión demostrada en esta etapa.
3. **Disrupción Histórica en el Período 2020–2022:**  
   Se observa una caída cuantitativa abrupta en 2020 (23,954 casos registrados frente a 76,077 en 2019, una reducción del 68.5%), seguida por 27,969 casos en 2021 y un rebote hacia 62,479 en 2022, estabilizándose en ~70,000 casos en 2023–2025. Aunque esta caída coincide con las restricciones por COVID-19, debe reconocerse que el sistema de vigilancia pasiva es susceptible a subconsulta, por lo que la magnitud exacta atribuible a menor circulación viral vs. menor acceso a centros de salud requiere cruces con encuestas de hogares (ENDES/ENAHO).
4. **Asimetría de la Carga Notificada por Provincia:**  
   San Román (569,203 episodios; 24.77%) y Puno (383,892 episodios; 16.71%) concentran el 41.48% del volumen acumulado regional. Esta concentración refleja fundamentalmente la distribución poblacional y la capacidad de captación de los complejos hospitalarios urbanos (Carlos Monge Medrano y Manuel Núñez Butrón). No constituye una prueba de mayor riesgo epidemiológico relativo en ausencia de denominadores censales.
5. **Comportamiento Estadístico y Esparsidad:**  
   - `ira_no_neumonia`: Mediana semanal distrital de 8 casos, media de 18.22, y únicamente **1.32% de registros en cero**, lo que confirma su aptitud para modelos autorregresivos continuos.
   - Formas graves (`neumonias_men5`, `hospitalizados_men5`, `defunciones_men5`): Presentan esparsidades de **88.87%**, **95.57%** y **99.10%** de ceros, respectivamente. Cualquier intento de modelarlas a nivel semanal distrital adolece de problemas de ceros excesivos que invalidan la regresión gaussiana tradicional.

---

## 2. Reconciliación de la Tabla de Episodios por Provincia

| Provincia | Total Episodios Notificados (2000–2025) | % del Total Regional | Semanas Teóricas | Semanas Presentes | Semanas Faltantes | % Cobertura Real |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SAN ROMAN** | 569,203 | 24.78% | 1,357 | 1,347 | 10 | 99.26% |
| **PUNO** | 383,892 | 16.71% | 1,357 | 1,355 | 2 | 99.85% |
| **AZANGARO** | 254,292 | 11.07% | 1,357 | 1,355 | 2 | 99.85% |
| **CARABAYA** | 180,457 | 7.85% | 1,357 | 1,354 | 3 | 99.78% |
| **MELGAR** | 177,949 | 7.75% | 1,357 | 1,339 | 18 | 98.67% |
| **LAMPA** | 138,722 | 6.04% | 1,357 | 1,355 | 2 | 99.85% |
| **SANDIA** | 132,521 | 5.77% | 1,357 | 1,352 | 5 | 99.63% |
| **CHUCUITO** | 116,127 | 5.05% | 1,357 | 1,347 | 10 | 99.26% |
| **HUANCANE** | 99,744 | 4.34% | 1,357 | 1,355 | 2 | 99.85% |
| **EL COLLAO** | 97,619 | 4.25% | 1,357 | 1,338 | 19 | 98.60% |
| **SAN ANTONIO DE PUTINA** | 68,673 | 2.99% | 1,357 | 1,336 | 21 | 98.45% |
| **YUNGUYO** | 54,161 | 2.36% | 1,357 | 1,337 | 20 | 98.53% |
| **MOHO** | 24,018 | 1.05% | 1,357 | 1,318 | 39 | 97.13% |
| **TOTAL REGIONAL** | **2,297,378** | **100.00%** | **17,641** | **17,488** | **153** | **99.13%** |

*(Nota de trazabilidad: La suma de la columna Total Episodios Notificados es rigurosamente 2,297,378. Queda superada la discrepancia anterior de 732 episodios provocada por la doble adición del registro N° 92,981).*

---

## 3. Matriz de Cobertura y Distinción entre Cero y Ausencia de Registro

El análisis del panel provincial de 17,641 celdas ($1,357 \times 13$) demuestra que:
- **17,476 celdas (99.06%)** contienen registros válidos con notificación de casos ($>0$).
- **12 celdas (0.07%)** corresponden a semanas donde la provincia remitió información a NotiSP pero la suma de episodios fue explícitamente 0 (`CERO_REPORTADO`).
- **153 celdas (0.87%)** corresponden a semanas donde no existe ninguna fila en el archivo NotiSP para esa provincia (`AUSENCIA_REGISTRO`).
- Esta distinción es crucial para la Fase 2: un `CERO_REPORTADO` se procesa como valor numérico 0 en la serie temporal, mientras que una `AUSENCIA_REGISTRO` requiere una estrategia explícita de imputación (e.g. interpolación lineal local o imputación estacional del mismo período histórico) para evitar sesgar el cálculo de retrasos autorregresivos (lags).
