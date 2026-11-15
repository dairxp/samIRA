"""
Definición de interfaz de usuario (UI) para la aplicación samIRA.
Diseño panorámico de ancho completo (Full Width), limpio y 100% responsive:
- Barra de filtros superior horizontal compacta y adaptable a móviles.
- Pestañas directas con scroll horizontal táctil nativo tipo App móvil.
- Tablas con envoltura table-responsive para evitar desbordes.
REGLA: Cero pandas, cero cálculos en este módulo.
"""

from pathlib import Path
from shiny import ui
from shinywidgets import output_widget

from samira_app.constants import (
    MODELOS,
    PATH_LOGO,
    PROVINCIAS,
    STYLE_ADMINLTE,
)

# Detección segura del logo institucional
tiene_logo = PATH_LOGO.is_file()

# Elementos del encabezado
head_elements = [
    ui.tags.meta(charset="utf-8"),
    ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1, shrink-to-fit=no"),
    ui.tags.title("samIRA · Vigilancia y Pronóstico de IRA en Puno"),
    ui.tags.style(STYLE_ADMINLTE),
]

if tiene_logo:
    head_elements.append(ui.tags.link(rel="icon", href="logo.png", type="image/png"))

header_logo_tag = (
    ui.tags.img(src="logo.png", class_="adminlte-header-logo", alt="Logo samIRA")
    if tiene_logo
    else ui.HTML("")
)

header_tag = ui.tags.header(
    ui.tags.div(
        header_logo_tag,
        ui.tags.span("samIRA", class_="adminlte-header-title"),
        ui.tags.span(
            "· Vigilancia Epidemiológica y Pronóstico de Infecciones Respiratorias Agudas (Puno)",
            class_="adminlte-header-subtitle",
        ),
        class_="adminlte-header",
    ),
    style="margin-bottom: 12px;",
)

# Barra de Filtros Superior Horizontal Compacta y Responsive
filtros_toolbar = ui.tags.div(
    ui.row(
        ui.column(
            3,
            ui.input_select(
                "provincia",
                "Provincia:",
                choices=["TODAS"] + PROVINCIAS,
                selected="TODAS",
            ),
            class_="col-12 col-md-3",
        ),
        ui.column(
            3,
            ui.input_slider(
                "anios",
                "Rango de Años:",
                min=2000,
                max=2025,
                value=[2000, 2025],
                step=1,
                sep="",
            ),
            class_="col-12 col-md-3",
        ),
        ui.column(
            2,
            ui.input_radio_buttons(
                "h",
                "Horizonte (h):",
                choices={"1": "1 sem", "4": "4 sem"},
                selected="1",
                inline=True,
            ),
            class_="col-6 col-md-2",
        ),
        ui.column(
            3,
            ui.input_select(
                "modelo",
                "Modelo Predictivo:",
                choices=MODELOS,
                selected="XGBoost Global",
            ),
            class_="col-6 col-md-3",
        ),
        ui.column(
            1,
            ui.download_button(
                "download_metricas",
                "CSV",
                class_="btn btn-primary w-100",
                style="margin-top: 18px;",
            ),
            class_="col-12 col-md-1",
        ),
        class_="g-2 align-items-center",
    ),
    class_="adminlte-card mb-2",
    style="padding: 10px 14px; margin-bottom: 12px;",
)

# Tarjetas KPI superiores: responsive 2x2 en móviles, 4x1 en escritorio
kpi_row = ui.row(
    ui.column(
        3,
        ui.tags.div(
            ui.tags.div("Episodios Totales", class_="kpi-label"),
            ui.tags.div(ui.output_text("kpi_episodios"), class_="kpi-value"),
            ui.tags.div("Casos en selección activa", class_="kpi-subtext"),
            class_="kpi-card",
        ),
        class_="col-6 col-md-3 mb-2",
    ),
    ui.column(
        3,
        ui.tags.div(
            ui.tags.div("Provincias Evaluadas", class_="kpi-label"),
            ui.tags.div(ui.output_text("kpi_provincias"), class_="kpi-value"),
            ui.tags.div("Ámbito territorial activo", class_="kpi-subtext"),
            class_="kpi-card",
        ),
        class_="col-6 col-md-3 mb-2",
    ),
    ui.column(
        3,
        ui.tags.div(
            ui.tags.div("WAPE Modelo (2025)", class_="kpi-label"),
            ui.tags.div(ui.output_text("kpi_wape"), class_="kpi-value"),
            ui.tags.div("Error ponderado global", class_="kpi-subtext"),
            class_="kpi-card",
        ),
        class_="col-6 col-md-3 mb-2",
    ),
    ui.column(
        3,
        ui.tags.div(
            ui.tags.div("Cobertura Calibrada", class_="kpi-label"),
            ui.tags.div(ui.output_text("kpi_cobertura"), class_="kpi-value"),
            ui.tags.div("Nivel nominal 95%", class_="kpi-subtext"),
            class_="kpi-card",
        ),
        class_="col-6 col-md-3 mb-2",
    ),
    class_="g-2 mb-2",
)

# ==============================================================================
# PESTAÑAS PRINCIPALES (Flujo continuo y responsive)
# ==============================================================================

# Pestaña 1: Panorama
tab_panorama = ui.nav_panel(
    "Panorama",
    ui.row(
        ui.column(
            8,
            ui.tags.div(
                ui.tags.div("Evolución Temporal", class_="adminlte-card-title"),
                output_widget("plot_evolution"),
                class_="adminlte-card",
            ),
            class_="col-12 col-lg-8",
        ),
        ui.column(
            4,
            ui.tags.div(
                ui.tags.div("Concentración Provincial", class_="adminlte-card-title"),
                output_widget("plot_distribution"),
                class_="adminlte-card",
            ),
            class_="col-12 col-lg-4",
        ),
        class_="g-2",
    ),
    ui.row(
        ui.column(
            6,
            ui.tags.div(
                ui.tags.div("Distribución Territorial", class_="adminlte-card-title"),
                output_widget("plot_map"),
                class_="adminlte-card",
            ),
            class_="col-12 col-lg-6",
        ),
        ui.column(
            6,
            ui.tags.div(
                ui.tags.div("Ciclo Estacional Semanal", class_="adminlte-card-title"),
                output_widget("plot_seasonality"),
                class_="adminlte-card",
            ),
            class_="col-12 col-lg-6",
        ),
        class_="g-2",
    ),
)

# Pestaña 2: Diagnóstico
tab_diagnostico = ui.nav_panel(
    "Diagnóstico",
    ui.row(
        ui.column(
            6,
            ui.tags.div(
                ui.tags.div("Carga Semanal por Período", class_="adminlte-card-title"),
                output_widget("plot_periods"),
                class_="adminlte-card",
            ),
            class_="col-12 col-lg-6",
        ),
        ui.column(
            6,
            ui.tags.div(
                ui.tags.div("Sobredispersión Provincial (VMR)", class_="adminlte-card-title"),
                output_widget("plot_vmr"),
                class_="adminlte-card",
            ),
            class_="col-12 col-lg-6",
        ),
        class_="g-2",
    ),
    ui.tags.div(
        ui.tags.strong("Nota analítica fundamental: "),
        "El descenso en los episodios notificados durante el bienio 2020–2021 responde a la disrupción del sistema asistencial "
        "y a las restricciones de movilidad por COVID-19. Dicha caída se interpreta como una hipótesis epidemiológica y "
        "no como una causalidad determinística estricta ni un cambio permanente en la dinámica poblacional.",
        class_="adminlte-note",
    ),
)

# Pestaña 3: Pronóstico
tab_pronostico = ui.nav_panel(
    "Pronóstico",
    ui.tags.div(
        ui.tags.div("Pronóstico 2025 vs. Observado", class_="adminlte-card-title"),
        output_widget("plot_forecast"),
        class_="adminlte-card",
    ),
    ui.row(
        ui.column(
            6,
            ui.tags.div(
                ui.tags.div("Benchmark de Modelos (WAPE)", class_="adminlte-card-title"),
                output_widget("plot_model_comparison"),
                class_="adminlte-card",
            ),
            class_="col-12 col-lg-6",
        ),
        ui.column(
            6,
            ui.tags.div(
                ui.tags.div("WAPE por Provincia", class_="adminlte-card-title"),
                output_widget("plot_wape_province"),
                class_="adminlte-card",
            ),
            class_="col-12 col-lg-6",
        ),
        class_="g-2",
    ),
    ui.row(
        ui.column(
            6,
            ui.tags.div(
                ui.tags.div("Métricas de Evaluación (Prueba 2025)", class_="adminlte-card-title"),
                ui.tags.div(ui.output_data_frame("tabla_metricas"), class_="table-responsive"),
                class_="adminlte-card",
            ),
            class_="col-12 col-lg-6",
        ),
        ui.column(
            6,
            ui.tags.div(
                ui.tags.div("Estabilidad Temporal Rolling-Origin (2020–2024)", class_="adminlte-card-title"),
                output_widget("plot_rolling"),
                class_="adminlte-card",
            ),
            class_="col-12 col-lg-6",
        ),
        class_="g-2",
    ),
    ui.tags.div(
        ui.tags.strong("Nota metodológica de evaluación: "),
        "Diferencias menores a 0.5 puntos porcentuales (p.p.) en WAPE entre los modelos líderes (XGBoost vs. LightGBM) "
        "no constituyen evidencia de superioridad estadística sin un contraste emparejado (Diebold-Mariano). Ambos modelos "
        "demuestran paridad técnica operacional. Asimismo, la validación Rolling-Origin evidencia convergencia estable en el período post-pandemia.",
        class_="adminlte-note",
    ),
)

# Pestaña 4: Incertidumbre y Escenarios
tab_incertidumbre = ui.nav_panel(
    "Incertidumbre y Escenarios",
    ui.row(
        ui.column(
            7,
            ui.tags.div(
                ui.tags.div("Cobertura Conformal (95%)", class_="adminlte-card-title"),
                output_widget("plot_calibration"),
                class_="adminlte-card",
            ),
            class_="col-12 col-lg-7",
        ),
        ui.column(
            5,
            ui.tags.div(
                ui.tags.div("Métricas de Calibración Provincial", class_="adminlte-card-title"),
                ui.tags.div(ui.output_data_frame("tabla_calibracion"), class_="table-responsive"),
                class_="adminlte-card",
            ),
            class_="col-12 col-lg-5",
        ),
        class_="g-2",
    ),
    ui.tags.div(
        ui.row(
            ui.column(
                4,
                ui.tags.div(
                    ui.tags.div("Configuración Asistencial", class_="adminlte-card-title"),
                    ui.input_slider(
                        "umbral_capacidad",
                        "Capacidad Semanal (casos/semana):",
                        min=50,
                        max=1000,
                        value=300,
                        step=25,
                    ),
                    ui.tags.div(
                        "Ajuste el umbral para simular la capacidad operativa semanal de la red asistencial y estimar la presión de atención.",
                        style="font-size: 11px; color: #64748b; margin-top: 6px;",
                    ),
                ),
                class_="col-12 col-md-4",
            ),
            ui.column(
                8,
                ui.tags.div(
                    ui.tags.div("Demanda Prevista vs. Capacidad Asistencial (2025)", class_="adminlte-card-title"),
                    output_widget("plot_scenario"),
                ),
                class_="col-12 col-md-8",
            ),
            class_="g-2 align-items-center",
        ),
        class_="adminlte-card",
    ),
    ui.tags.div(
        ui.tags.strong("ADVERTENCIA IMPORTANTE: "),
        "Las brechas aquí visualizadas corresponden a simulaciones matemáticas hipotéticas frente a umbrales paramétricos de prueba, "
        "no a déficits sanitarios reales. El modelo de vigilancia no incorpora variables de capacidad instalada física, "
        "disponibilidad de oxígeno medicinal ni dotación de recursos humanos de la red de salud de Puno.",
        class_="adminlte-warning-box",
    ),
)

# Layout Panorámico de Ancho Completo con Padding Responsive
app_ui = ui.page_fluid(
    *head_elements,
    header_tag,
    filtros_toolbar,
    kpi_row,
    ui.navset_tab(
        tab_panorama,
        tab_diagnostico,
        tab_pronostico,
        tab_incertidumbre,
        id="main_navset",
    ),
    style="padding: 0 14px 20px 14px; max-width: 100vw; overflow-x: hidden;",
)
