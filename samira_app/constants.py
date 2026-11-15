"""
Constantes, paleta de colores y estilos CSS estilo AdminLTE sólido y limpio para samIRA.
Diseño profesional, minimalista y sobrio: sin gradientes, sin sombras, cuadros limpios,
y arquitectura 100% responsive para dispositivos móviles y escritorios.
"""

from pathlib import Path

# Directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas de datos
PATH_PANEL_PROVINCIAL = BASE_DIR / "data" / "processed" / "panel_semanal_provincial_puno.parquet"
PATH_PANEL_REGIONAL = BASE_DIR / "data" / "processed" / "panel_semanal_regional_puno.parquet"
PATH_PREDICCIONES = BASE_DIR / "results" / "predicciones_completas.parquet"
PATH_METRICAS_VAL = BASE_DIR / "results" / "metricas_validacion_modelos.csv"
PATH_METRICAS_TEST = BASE_DIR / "results" / "metricas_prueba_2025.csv"
PATH_COBERTURA = BASE_DIR / "results" / "cobertura_intervalos_95.csv"
PATH_CALIBRACION_PROV = (
    BASE_DIR / "results" / "experiments" / "calibracion" / "comparacion_provincial_calibracion.csv"
)
PATH_ROLLING_ORIGIN = (
    BASE_DIR / "results" / "experiments" / "rolling_origin" / "metricas_rolling_origin.csv"
)
PATH_ALERTAS = BASE_DIR / "results" / "evaluacion_alertas_epidemiologicas.csv"
PATH_LOGO = BASE_DIR / "www" / "logo.png"

# Lista canónica de las 13 provincias de Puno (mayúsculas sin tilde)
PROVINCIAS = [
    "AZANGARO",
    "CARABAYA",
    "CHUCUITO",
    "EL COLLAO",
    "HUANCANE",
    "LAMPA",
    "MELGAR",
    "MOHO",
    "PUNO",
    "SAN ANTONIO DE PUTINA",
    "SAN ROMAN",
    "SANDIA",
    "YUNGUYO",
]

# Centroides aproximados (latitud, longitud) para cada provincia
CENTROIDES = {
    "PUNO": (-15.840, -70.020),
    "SAN ROMAN": (-15.500, -70.130),
    "AZANGARO": (-14.910, -70.190),
    "CARABAYA": (-14.070, -70.430),
    "CHUCUITO": (-16.050, -69.680),
    "EL COLLAO": (-16.080, -69.640),
    "HUANCANE": (-15.200, -69.450),
    "LAMPA": (-15.360, -70.370),
    "MELGAR": (-14.880, -70.590),
    "MOHO": (-15.360, -69.500),
    "SAN ANTONIO DE PUTINA": (-14.910, -69.870),
    "SANDIA": (-14.320, -69.460),
    "YUNGUYO": (-16.240, -69.090),
}

# Modelos predictivos evaluados
MODELOS = [
    "XGBoost Global",
    "LightGBM Global",
    "XGBoost Global Directo",
    "LightGBM Global Directo",
    "LightGBM Independiente",
    "ETS / Holt-Winters",
    "Seasonal Naive",
    "Naive",
]

# Horizontes de predicción disponibles
HORIZONTES = [1, 4]

# Paleta sólida contemporánea, sobria y profesional
COLOR_PRIMARIO = "#1d5b8c"     # Azul corporativo balanceado y sereno
COLOR_SECUNDARIO = "#298bac"   # Azul cerúleo / turquesa armónico con el logo
COLOR_OBSERVADO = "#1d5b8c"    # Azul nítido para la serie real observada
COLOR_PRONOSTICO = "#dc2626"   # Rojo coral para pronóstico (máximo contraste)
COLOR_EXITO = "#16a34a"        # Verde sobrio
COLOR_ADVERTENCIA = "#d97706"  # Ámbar suave
COLOR_PELIGRO = "#dc2626"      # Rojo profesional
COLOR_NEUTRO = "#64748b"       # Gris slate
COLOR_FONDO_APP = "#f8fafc"    # Fondo slate ultra-claro y limpio
COLOR_FONDO_CARD = "#ffffff"   # Blanco puro
COLOR_BORDE = "#e2e8f0"        # Borde sutil y fino

# Plantilla CSS unificada: limpia, sin sombras, con soporte responsive completo
STYLE_ADMINLTE = """
/* Reset y estilos generales */
body {
    background-color: #f8fafc !important;
    font-family: 'Source Sans 3', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    font-size: 14px !important;
    color: #0f172a !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow-x: hidden !important;
}

/* Header limpio que maximiza el contraste del logo */
.adminlte-header {
    min-height: 56px;
    background-color: #ffffff !important;
    color: #0f172a !important;
    display: flex;
    align-items: center;
    padding: 8px 20px;
    border-bottom: 1px solid #e2e8f0;
}

.adminlte-header-logo {
    height: 34px;
    margin-right: 12px;
    display: inline-block;
    vertical-align: middle;
}

.adminlte-header-title {
    font-size: 18px;
    font-weight: 700;
    color: #1d5b8c;
    letter-spacing: 0.3px;
    margin: 0;
}

.adminlte-header-subtitle {
    font-size: 12px;
    color: #64748b;
    margin-left: 10px;
    font-weight: 400;
}

/* Barra lateral / Sidebar limpia */
.adminlte-sidebar {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    padding: 16px !important;
}

.adminlte-sidebar label {
    font-weight: 600 !important;
    font-size: 11px !important;
    color: #334155 !important;
    margin-bottom: 4px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.3px !important;
}

.form-control,
.form-select,
.shiny-input-container input {
    border-radius: 2px !important;
    border: 1px solid #cbd5e1 !important;
    font-size: 13px !important;
    color: #1e293b !important;
    background-color: #ffffff !important;
    box-shadow: none !important;
}

.form-control:focus,
.form-select:focus {
    border-color: #1d5b8c !important;
    outline: none !important;
}

/* Botones cuadrados planos */
.btn, .btn-primary, .btn-default, .btn-secondary {
    border-radius: 2px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.3px !important;
    transition: background-color 150ms ease-in-out !important;
    box-shadow: none !important;
}

.btn-primary {
    background-color: #1d5b8c !important;
    border-color: #1d5b8c !important;
    color: #ffffff !important;
}

.btn-primary:hover {
    background-color: #16466c !important;
    border-color: #16466c !important;
}

/* Tarjetas (Cards) limpias: sin sombra y sin línea superior */
.adminlte-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    box-shadow: none !important;
    border-top: none !important;
    margin-bottom: 14px;
    padding: 16px;
}

.adminlte-card-title {
    font-size: 14px;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 12px;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 6px;
}

/* Tarjetas KPI limpias: sin sombras y sin líneas arriba */
.kpi-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    box-shadow: none !important;
    border-top: none !important;
    padding: 12px 14px;
    height: 100%;
}

.kpi-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748b;
    margin-bottom: 2px;
}

.kpi-value {
    font-size: 24px;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
}

.kpi-subtext {
    font-size: 11px;
    color: #94a3b8;
    margin-top: 2px;
}

/* Pestañas (Nav Tabs) limpias con scroll horizontal suave en móviles */
.nav-tabs {
    display: flex !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    -webkit-overflow-scrolling: touch !important;
    scrollbar-width: none !important;
    border-bottom: 1px solid #e2e8f0 !important;
    margin-bottom: 14px !important;
}

.nav-tabs::-webkit-scrollbar {
    display: none !important;
}

.nav-tabs .nav-link {
    white-space: nowrap !important;
    flex-shrink: 0 !important;
    border: none !important;
    border-radius: 0 !important;
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 18px !important;
    background-color: transparent !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
}

.nav-tabs .nav-link:hover {
    color: #1d5b8c !important;
    border-bottom: 2px solid #cbd5e1 !important;
}

.nav-tabs .nav-link.active {
    color: #1d5b8c !important;
    background-color: transparent !important;
    border-bottom: 2px solid #1d5b8c !important;
}

/* Cuadros de notas analíticas: simples, sin líneas laterales de color */
.adminlte-note {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 13px;
    color: #334155;
    line-height: 1.5;
    margin-top: 14px;
    margin-bottom: 14px;
}

.adminlte-warning-box {
    background-color: #fffdf5;
    border: 1px solid #fef3c7;
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 13px;
    color: #92400e;
    line-height: 1.5;
    margin-top: 14px;
    margin-bottom: 14px;
}

/* Tablas con scroll horizontal para móviles */
.table-responsive,
.dataTables_wrapper,
.shiny-datatable-output {
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
    max-width: 100% !important;
}

.table {
    font-size: 13px !important;
    border: 1px solid #e2e8f0 !important;
    margin-bottom: 0 !important;
    width: 100% !important;
}

.table th {
    background-color: #f8fafc !important;
    color: #475569 !important;
    font-weight: 600 !important;
    border-bottom: 1px solid #e2e8f0 !important;
}

.table td {
    background-color: #ffffff !important;
    border-bottom: 1px solid #f1f5f9 !important;
}

.table-striped tbody tr:nth-of-type(odd) td {
    background-color: #fafbfc !important;
}

/* ==========================================================================
   REGLAS DE RESPONSIVIDAD PARA CELULARES (Móviles < 768px)
   ========================================================================== */
@media (max-width: 768px) {
    .adminlte-header {
        height: auto !important;
        padding: 8px 12px !important;
    }
    .adminlte-header-logo {
        height: 28px !important;
        margin-right: 8px !important;
    }
    .adminlte-header-title {
        font-size: 16px !important;
    }
    .adminlte-header-subtitle {
        display: none !important; /* Ocultar texto largo para evitar desbordes */
    }
    .kpi-card {
        padding: 8px 10px !important;
    }
    .kpi-label {
        font-size: 10px !important;
    }
    .kpi-value {
        font-size: 19px !important;
    }
    .kpi-subtext {
        font-size: 9.5px !important;
    }
    .adminlte-card {
        padding: 10px !important;
        margin-bottom: 10px !important;
    }
    .nav-tabs .nav-link {
        padding: 8px 12px !important;
        font-size: 12px !important;
    }
    .adminlte-note, .adminlte-warning-box {
        padding: 10px 12px !important;
        font-size: 12px !important;
    }
}
"""
