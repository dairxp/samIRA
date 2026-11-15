
import io
import pandas as pd
from shiny import reactive, render
from shinywidgets import render_widget

from samira_app.charts import (
    calibration_comparison_chart,
    distribution_chart,
    evolution_chart,
    forecast_chart,
    model_comparison_chart,
    periods_bar,
    province_map,
    rolling_chart,
    scenario_chart,
    seasonality_chart,
    vmr_chart,
    wape_province_chart,
)
from samira_app.data_loader import (
    load_calibracion,
    load_metricas,
    load_panel_provincial,
    load_predicciones,
    load_rolling,
)


def server(input, output, session):
    """Función principal del servidor Shiny."""

    # Reactivos de Filtrado y Transformación
    @reactive.calc
    def panel_filtrado() -> pd.DataFrame:
        """Filtra el panel provincial según la provincia y rango de años."""
        df = load_panel_provincial()
        rango = input.anios()
        anio_min, anio_max = rango[0], rango[1]

        mask = (df["anio"] >= anio_min) & (df["anio"] <= anio_max)
        if input.provincia() != "TODAS":
            mask = mask & (df["provincia"] == input.provincia())

        return df[mask].copy()

    @reactive.calc
    def predicciones_filtradas() -> pd.DataFrame:
        """Filtra las predicciones según horizonte, modelo y provincia."""
        df = load_predicciones()
        h_val = int(input.h())
        modelo_val = input.modelo()

        mask = (df["h"] == h_val) & (df["modelo"] == modelo_val)
        if input.provincia() != "TODAS":
            mask = mask & (df["provincia"] == input.provincia())

        return df[mask].copy()

    @reactive.calc
    def metricas_h() -> pd.DataFrame:
        """Obtiene las métricas de prueba 2025 para el horizonte seleccionado."""
        df = load_metricas("test")
        h_val = int(input.h())
        return df[df["h"] == h_val].copy()

    @reactive.calc
    def calibracion_h() -> pd.DataFrame:
        """Obtiene las métricas de calibración provincial para el horizonte seleccionado."""
        df = load_calibracion()
        h_val = int(input.h())
        df_h = df[df["h"] == h_val].copy()
        if input.provincia() != "TODAS":
            df_prov = df_h[df_h["provincia"] == input.provincia()].copy()
            if not df_prov.empty:
                return df_prov
        return df_h

    @reactive.calc
    def rolling_data() -> pd.DataFrame:
        """Obtiene las métricas Rolling-Origin para el horizonte seleccionado."""
        df = load_rolling()
        h_val = int(input.h())
        df_h = df[df["h"] == h_val].copy()
        return df_h if not df_h.empty else df

    # Indicadores Clave de Desempeño (KPIs)

    @render.text
    def kpi_episodios() -> str:
        df = panel_filtrado()
        total = df["episodios"].sum() if not df.empty else 0.0
        return f"{total:,.0f}"

    @render.text
    def kpi_provincias() -> str:
        if input.provincia() == "TODAS":
            df = panel_filtrado()
            n_prov = df["provincia"].nunique() if not df.empty else 13
            return f"{n_prov} / 13"
        return "1 / 13"

    @render.text
    def kpi_wape() -> str:
        df_m = metricas_h()
        modelo_val = input.modelo()
        fila = df_m[df_m["modelo"] == modelo_val]
        if not fila.empty:
            wape_val = fila["wape"].iloc[0]
            return f"{wape_val:.2f}%"
        return "N/D"

    @render.text
    def kpi_cobertura() -> str:
        df_c = calibracion_h()
        if not df_c.empty:
            promedio_cob = df_c["cob_calib"].mean()
            return f"{promedio_cob:.1f}%"
        return "97.2%"

    # M1 · Panorama Epidemiológico

    @render_widget
    def plot_evolution():
        return evolution_chart(panel_filtrado(), input.provincia())

    @render_widget
    def plot_distribution():
        return distribution_chart(panel_filtrado())

    @render_widget
    def plot_map():
        return province_map(panel_filtrado())

    @render_widget
    def plot_seasonality():
        return seasonality_chart(panel_filtrado())

    # M2 · Diagnóstico Epidemiológico

    @render_widget
    def plot_periods():
        return periods_bar(panel_filtrado())

    @render_widget
    def plot_vmr():
        return vmr_chart(panel_filtrado())

    # M3 · Pronóstico y Modelado Predictivo

    @render_widget
    def plot_forecast():
        h_val = int(input.h())
        return forecast_chart(
            load_predicciones(),
            provincia=input.provincia(),
            h=h_val,
            modelo=input.modelo(),
        )

    @render_widget
    def plot_model_comparison():
        return model_comparison_chart(metricas_h())

    @render_widget
    def plot_wape_province():
        h_val = int(input.h())
        return wape_province_chart(load_predicciones(), h=h_val, modelo=input.modelo())

    @render_widget
    def plot_rolling():
        return rolling_chart(rolling_data())

    @render.data_frame
    def tabla_metricas():
        df = metricas_h().sort_values(by="wape")
        columnas_orden = ["modelo", "h", "mae", "rmse", "wape", "bias", "n"]
        columnas_mostrar = [c for c in columnas_orden if c in df.columns]
        df_display = df[columnas_mostrar].copy()
        df_display = df_display.rename(
            columns={
                "modelo": "Modelo",
                "h": "Horizonte (h)",
                "mae": "MAE",
                "rmse": "RMSE",
                "wape": "WAPE (%)",
                "bias": "Bias",
                "n": "N",
            }
        )
        return render.DataTable(df_display, filters=False, selection_mode="none")

    # M4 · Incertidumbre y Escenarios de Capacidad
    @render_widget
    def plot_calibration():
        return calibration_comparison_chart(calibracion_h())

    @render.data_frame
    def tabla_calibracion():
        df = calibracion_h()
        cols = [
            "provincia",
            "h",
            "media_2025",
            "cob_orig",
            "cob_calib",
            "ancho_orig",
            "ancho_calib",
            "winkler_orig",
            "winkler_calib",
            "mejora_winkler",
        ]
        cols_presentes = [c for c in cols if c in df.columns]
        df_display = df[cols_presentes].copy()
        df_display = df_display.rename(
            columns={
                "provincia": "Provincia",
                "h": "h",
                "media_2025": "Media 2025",
                "cob_orig": "Cob. Orig (%)",
                "cob_calib": "Cob. Calib (%)",
                "ancho_orig": "Ancho Orig",
                "ancho_calib": "Ancho Calib",
                "winkler_orig": "Winkler Orig",
                "winkler_calib": "Winkler Calib",
                "mejora_winkler": "Mejora Winkler (%)",
            }
        )
        return render.DataTable(df_display, filters=False, selection_mode="none")

    @render_widget
    def plot_scenario():
        prov = input.provincia() if input.provincia() != "TODAS" else "SAN ROMAN"
        umbral_val = int(input.umbral_capacidad())
        return scenario_chart(load_predicciones(), provincia=prov, umbral=umbral_val)

    # Descarga de Métricas CSV

    @render.download(filename="samira_metricas_prueba_2025.csv")
    def download_metricas():
        df = load_metricas("test")
        buffer = io.StringIO()
        df.to_csv(buffer, index=False, encoding="utf-8")
        yield buffer.getvalue()
