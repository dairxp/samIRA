# Informe de Auditoría y Control de Calidad de Datos (Fase 1 – Versión Corregida y Reconciliada)
## Vigilancia Epidemiológica de Infecciones Respiratorias Agudas (IRAs) en Menores de 5 Años – Región Puno (2000–2025)

**Autor:** Científico de Datos Senior (Especialidad en Epidemiología, Calidad de Datos y BI)  
**Dataset Oficial:** `datos_abiertos_vigilancia_iras_2000_2025 PUNO.csv` (MINSA / CDC Perú – NotiSP)  
**Fecha de Emisión:** Noviembre 2026 (Versión 2.1 Auditada)  
**Trazabilidad:** Versión previa archivada en [`reports/reporte_calidad_datos_v1.md`](file:///F:/dxp-projects/DevBasics/R%20y%20statistics/samIRA/reports/reporte_calidad_datos_v1.md)  

---

## 1. Reconciliación de la Discrepancia Documental (732 Episodios)

### 1.1. Diagnóstico de la Discrepancia Previa (732 episodios)
En la revisión preliminar se citó un total regional de **2,298,110** episodios de `ira_no_neumonia`, mientras que la suma de las 13 provincias arrojaba **2,297,378** (diferencia de **732 episodios**).

Se procedió a la reconciliación forense consultando directamente el archivo CSV original (`data/raw/datos_abiertos_vigilancia_iras_2000_2025 PUNO.csv`) y el Parquet procesado:
1. **Suma directa en el CSV original:**
   $$\sum_{i=1}^{126,118} \text{ira\_no\_neumonia}_i = \mathbf{2,297,378}$$
2. **Suma agrupada por provincia en el CSV original:**
   $$\sum_{p=1}^{13} \left( \sum_{i \in p} \text{ira\_no\_neumonia}_i \right) = \mathbf{2,297,378}$$
   $$\text{Diferencia entre suma directa y suma provincial} = 2,297,378 - 2,297,378 = \mathbf{0}$$
3. **Calificación Metodológica:**  
   Esta discrepancia de 732 episodios se clasifica formalmente como una **discrepancia de consolidación documental** en el borrador v1 (un desfase de redacción entre la tabla desglosada y el texto narrativo de síntesis).  
   Si bien se identificó que la fila física N° **92,981** (Juliaca, 2011, SE 28) reporta un valor idéntico de 732 episodios, en ausencia de un registro de cálculo que demuestre que esa fila específica fue duplicada en la suma manual preliminar, **no se concluye como causa demostrada**, tratándose de una coincidencia numérica.
4. **Cifra Oficial y Definitiva:**  
   El volumen real, auditado e inalterado de episodios de `ira_no_neumonia` en el dataset para el período 2000–2025 es exactamente **2,297,378 episodios**.

---

## 2. Auditoría de Semanas Epidemiológicas y Cobertura (Calendario Oficial MINSA)

### 2.1. Semanas Oficialmente Esperadas según el CDC Perú / DGE
De acuerdo con las directivas de vigilancia del Centro Nacional de Epidemiología, Prevención y Control de Enfermedades (CDC Perú), el calendario epidemiológico contiene habitualmente 52 semanas, extendiéndose a una **Semana 53** en aquellos años en que el cómputo de semanas epidemiológicas finaliza en los primeros días de enero del año subsiguiente.
En la serie histórica 2000–2025:
- **Años con 52 semanas (21 años):** 2000–2002, 2004–2007, 2009–2013, 2015–2019, 2021–2024 $\rightarrow 21 \times 52 = 1,092 \text{ semanas}$.
- **Años con 53 semanas oficiales (5 años):** 2003, 2008, 2014, 2020 y **2025** $\rightarrow 5 \times 53 = 265 \text{ semanas}$.
  *(Nota técnica: El CDC Perú efectivamente publicó sala situacional oficial para la SE 53 de 2025).*
- **Total de Semanas Esperadas a Nivel Regional:** $1,092 + 265 = \mathbf{1,357 \text{ semanas}}$.
- **Semanas Observadas a Nivel Regional:** **1,357 semanas (100.0% de cobertura regional)**. No falta ninguna semana departamental en los 26 años.

### 2.2. Desagregación Teórica vs. Observada a Nivel Provincial
El producto de 1,357 semanas regionales por las 13 provincias oficiales arroja un **máximo teórico de combinaciones**:
$$\text{Total Teórico} = 1,357 \times 13 = \mathbf{17,641 \text{ pares provincia-semana}}$$

Al cruzar este universo teórico contra los registros empíricos del dataset, se obtiene la siguiente partición estricta:
1. **Pares provincia-semana presentes en el dataset:** **17,488 combinaciones** (**99.13%** de cobertura observada global).
   - Semanas con reporte de episodios ($>0$): **17,476** (99.06%).
   - Semanas con reporte explícito de cero casos ($=0$): **12** (0.07%).
2. **Pares provincia-semana ausentes (Ausencia de Registro):** **153 combinaciones** (**0.87%** de vacíos).

### 2.3. Desglose Exacto de Semanas por Provincia

| Provincia | Semanas Teóricas | Semanas Presentes | Ausencia de Registro | Ceros Reportados | % Cobertura Real | Total Episodios IRA |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AZANGARO** | 1,357 | 1,355 | 2 | 0 | 99.85% | 254,292 |
| **CARABAYA** | 1,357 | 1,354 | 3 | 0 | 99.78% | 180,457 |
| **CHUCUITO** | 1,357 | 1,347 | 10 | 2 | 99.26% | 116,127 |
| **EL COLLAO** | 1,357 | 1,338 | 19 | 5 | 98.60% | 97,619 |
| **HUANCANE** | 1,357 | 1,355 | 2 | 0 | 99.85% | 99,744 |
| **LAMPA** | 1,357 | 1,355 | 2 | 0 | 99.85% | 138,722 |
| **MELGAR** | 1,357 | 1,339 | 18 | 1 | 98.67% | 177,949 |
| **MOHO** | 1,357 | 1,318 | 39 | 2 | 97.13% | 24,018 |
| **PUNO** | 1,357 | 1,355 | 2 | 0 | 99.85% | 383,892 |
| **SAN ANTONIO DE PUTINA** | 1,357 | 1,336 | 21 | 1 | 98.45% | 68,673 |
| **SAN ROMAN** | 1,357 | 1,347 | 10 | 0 | 99.26% | 569,203 |
| **SANDIA** | 1,357 | 1,352 | 5 | 0 | 99.63% | 132,521 |
| **YUNGUYO** | 1,357 | 1,337 | 20 | 1 | 98.53% | 54,161 |
| **TOTAL CONSOLIDADO** | **17,641** | **17,488** | **153** | **12** | **99.13%** | **2,297,378** |

---

## 3. Preservación Estricta de Faltantes (NaN) y Criterios de Evaluación

### 3.1. Prohibición de Imputación de Ceros Sintéticos
Las 153 semanas sin registro no representan cero casos, sino la ausencia de información consolidada por el sistema. **Se eliminó cualquier sustitución por cero y se conservan estrictamente como `NaN` en el panel provincial.**

### 3.2. Bandera `es_evaluable`
- `es_evaluable == True` (17,488 semanas): Conjunto sobre el cual se calcularán las métricas de desempeño out-of-sample en Fase 2.
- `es_evaluable == False` (153 semanas): Excluidas de la evaluación de error para no calificar predicciones contra valores inventados.

### 3.3. Tratamiento Causal para Retardos (Lags)
Para el cálculo de variables de entrada en modelos de Machine Learning (LightGBM/XGBoost), los faltantes que impidan calcular un retardo se tratarán exclusivamente con información pasada (Lag estacional $y_{t-52}$, LOCF o media móvil retrospectiva), garantizando causalidad estricta.

---

## 4. Aditividad de Claves Repetidas y Notificación Múltiple

En el dataset existen **3,935 filas** correspondientes a **1,781 combinaciones** de `[ano, semana, ubigeo]` que presentan más de un registro, distinguidos por el código de la Subregión Notificante (`sub_reg_nt`).

### 4.1. Justificación de la Aditividad
La agregación aditiva ($\sum$) de estas filas al consolidar los casos del distrito o provincia es la práctica habitual en la gestión operativa del MINSA porque cada fila refleja una hoja de consolidación remitida por una Red de Salud distinta (e.g. atenciones en centros de salud locales de la DISA Puno vs. atenciones de emergencia en hospitales de EsSalud, Arequipa o Cusco).

### 4.2. Limitación Metodológica por Potencial Sobreconteo
Debe documentarse formalmente para el artículo científico la siguiente restricción del sistema de vigilancia pasiva:
- Si un menor con infección respiratoria acude el día lunes al puesto de salud de su comunidad (reportado por `sub_reg_nt = 21`), y el día jueves es llevado por sus padres al hospital de EsSalud en Juliaca (`sub_reg_nt = 50`) o al hospital de Arequipa (`sub_reg_nt = 4`), **el mismo episodio de enfermedad puede haber sido notificado por ambos establecimientos bajo el UBIGEO de residencia del paciente**.
- Dado que los datos abiertos no son nominales (no contienen DNI ni identificador unívoco de paciente), **es metodológicamente imposible desduplicar episodios a nivel de persona**. Por tanto, la suma agregada representa *atenciones o consultas notificadas*, pudiendo sobreestimar ligeramente la verdadera incidencia comunitaria de nuevos episodios clínicos.

---

## 5. Crítica Epistemológica de los Ceros y la Subnotificación

El dataset registra **1,660 filas donde `ira_no_neumonia == 0`** a nivel distrital.

### 5.1. Qué Significa un Cero Reportado
Una fila con valor cero indica que la institución notificadora procesó y remitió una plantilla de vigilancia correspondiente a ese distrito y semana, registrando la cifra numérica `0`.

### 5.2. Lo que NO Permite Concluir
Un valor de cero en la vigilancia pasiva **no equivale a ausencia de morbilidad en el territorio**, ni permite asumir que la totalidad de Instituciones Prestadoras de Servicios de Salud (IPRESS) del distrito cumplieron con notificar. Puede reflejar:
1. Puestos de salud periféricos cerrados temporalmente o con personal técnico no médico abocado a campañas de campo.
2. Dificultades de conectividad geográfica o climática (crecidas de ríos, nevadas) que impidieron el traslado del paciente o el envío de la hoja NotiSP.
3. Tratamiento domiciliario o recurso a medicina tradicional sin contacto con el sistema de salud oficial.

---

## 6. Delimitación Estricta entre Hechos Observados, Hipótesis y Causalidad

| Fenómeno Analizado | 1. Resultado Observado en Datos | 2. Hipótesis Explicativa Plausible | 3. Límites de Causalidad (Datos Requeridos) |
| :--- | :--- | :--- | :--- |
| **Estacionalidad (SE 20–32)** | Picos repetitivos de casos entre mayo y agosto en todos los años. | Coincide con el descenso de temperaturas mínimas y heladas meteorológicas en el altiplano. | **No demostrable con este dataset**. Se requiere vincular series climáticas diarias de SENAMHI o ERA5-Land (temperatura, heladas) y ajustar modelos de retardos no lineales (DLNM). |
| **Colapso en 2020 (-68.5%)** | Caída de 76,077 casos en 2019 a 23,954 en 2020. | Cuarentenas, cierre de cunas Cuna Más, mascarillas, temor a acudir a hospitales y desvío de personal de vigilancia. | **No demostrable aisladamente**. No se dispone de datos de movilidad poblacional ni de operatividad de postas médicas para deslindar el porcentaje de reducción biológica real del porcentaje de subconsulta. |
| **Declive de Mortalidad Infantil** | Defunciones por neumonía en <5 años cayeron de 121 en 2002 a 6 en 2025. | Despliegue de vacuna conjugada antineumocócica (iniciada en 2009), protocolos de abrigo y oxigenoterapia. | **No demostrable con esta fuente**. El dataset no contiene el estatus vacunal de los pacientes. Se requeriría enlazar microdatos nominales del HIS-MINSA/Padrón Nominal de Vacunación. |
| **Hospitalizaciones > Neumonías** | 150 filas semanales donde hospitalizados superan a los episodios ambulatorios de neumonía. | Desfase temporal de reporte (ingreso hospitalario en semana posterior al diagnóstico ambulatorio) o pacientes derivados. | **No es un error de digitación**, pero no se puede comprobar la trayectoria clínica sin un identificador único del paciente. |
