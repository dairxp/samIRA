"""
Funciones puras de visualización Plotly para la aplicación samIRA.
Diseño profesional, limpio y minimalista:
- Sin títulos duplicados (el encabezado de la tarjeta ya identifica el gráfico).
- Visualización territorial rápida y limpia vía Treemap (cero tiempo de carga, cero textos encimados).
- Líneas con alto contraste cromático (Azul vs Rojo para Observado vs Pronóstico).
- Sin gridlines verticales, gridlines horizontales sutiles y diseño responsive.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from samira_app.constants import (
    COLOR_ADVERTENCIA,
    COLOR_BORDE,
    COLOR_EXITO,
    COLOR_NEUTRO,
    COLOR_OBSERVADO,
    COLOR_PELIGRO,
    COLOR_PRIMARIO,
    COLOR_PRONOSTICO,
    COLOR_SECUNDARIO,
)


def _base_layout(fig: go.Figure, height: int = 350) -> go.Figure:
    """Aplica formato visual estándar limpio, minimalista y responsive a figuras Plotly."""
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        height=height,
        margin={"l": 45, "r": 20, "t": 20, "b": 35},
        font={"family": "Segoe UI, sans-serif", "size": 12, "color": "#334155"},
        hoverlabel={"font_family": "Segoe UI, sans-serif", "font_size": 12},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 11},
        },
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=COLOR_BORDE,
        tickfont={"size": 11, "color": COLOR_NEUTRO},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#f1f5f9",
        gridwidth=1,
        linecolor=COLOR_BORDE,
        tickfont={"size": 11, "color": COLOR_NEUTRO},
    )
    return fig


# ==============================================================================
# M1 · PANORAMA EPIDEMIOLÓGICO
# ==============================================================================


def evolution_chart(df: pd.DataFrame, provincia: str = "TODAS") -> go.Figure:
    """
    Fig 01: Evolución temporal histórica continua de episodios de IRA (2000-2025).
    """
    if provincia != "TODAS" and "provincia" in df.columns:
        df_sub = df[df["provincia"] == provincia].copy()
    else:
        df_sub = df.copy()

    # Agrupar por anio y semana
    serie = (
        df_sub.groupby(["anio", "semana"])["episodios"]
        .sum()
        .reset_index()
    )
    serie["indice_semanal"] = (
        serie["anio"].astype(str) + "-S" + serie["semana"].astype(str).str.zfill(2)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=serie["indice_semanal"],
            y=serie["episodios"],
            mode="lines",
            line={"color": COLOR_PRIMARIO, "width": 1.5},
            name="Episodios Semanales",
            hovertemplate="Semana: %{x}<br>Episodios: %{y:,.0f}<extra></extra>",
        )
    )

    fig = _base_layout(fig, height=350)
    fig.update_xaxes(nticks=14)
    fig.update_yaxes(title_text="Episodios Notificados")
    return fig


def province_map(df: pd.DataFrame) -> go.Figure:
    """
    Visualización territorial rápida, limpia y proporcional de las 13 provincias de Puno.
    Implementada como Treemap territorial: carga instantánea, cero textos amontonados
    y jerarquía visual directa.
    """
    totales = (
        df.groupby("provincia")["episodios"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = go.Figure(
        go.Treemap(
            labels=totales["provincia"],
            parents=[""] * len(totales),
            values=totales["episodios"],
            texttemplate="<b>%{label}</b><br>%{value:,.0f} casos<br>(%{percentEntry:.1%})",
            textfont={"size": 11, "family": "Segoe UI, sans-serif", "color": "#ffffff"},
            marker={
                "colors": totales["episodios"],
                "colorscale": [[0, "#38bdf8"], [0.5, "#298bac"], [1.0, "#1d5b8c"]],
                "showscale": False,
                "line": {"width": 1, "color": "#ffffff"},
            },
            hovertemplate="<b>%{label}</b><br>Total Episodios: %{value:,.0f}<br>Participación: %{percentEntry:.1%}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        autosize=True,
        height=350,
        margin={"l": 5, "r": 5, "t": 10, "b": 10},
        font={"family": "Segoe UI, sans-serif"},
    )
    return fig


def seasonality_chart(df: pd.DataFrame) -> go.Figure:
    """
    Fig 03: Ciclo estacional semanal y banda crítica de invierno (SE 20-32).
    Polígono continuo cerrado P25-P75 para evitar deformaciones visuales.
    """
    stats = (
        df.groupby("semana")["episodios"]
        .agg(
            mediana="median",
            p25=lambda x: np.percentile(x.dropna(), 25),
            p75=lambda x: np.percentile(x.dropna(), 75),
        )
        .reset_index()
        .sort_values(by="semana")
    )

    fig = go.Figure()

    # Polígono continuo cerrado para P25 - P75
    x_band = list(stats["semana"]) + list(stats["semana"][::-1])
    y_band = list(stats["p75"]) + list(stats["p25"][::-1])

    fig.add_trace(
        go.Scatter(
            x=x_band,
            y=y_band,
            fill="toself",
            fillcolor="rgba(41, 139, 172, 0.18)",
            line={"color": "rgba(255,255,255,0)"},
            name="Rango Intercuartil (P25–P75)",
            hoverinfo="skip",
        )
    )

    # Mediana estacional
    fig.add_trace(
        go.Scatter(
            x=stats["semana"],
            y=stats["mediana"],
            mode="lines+markers",
            line={"color": COLOR_PRIMARIO, "width": 2},
            marker={"size": 4, "color": COLOR_PRIMARIO},
            name="Mediana Histórica Semanal",
            hovertemplate="SE %{x}: %{y:.1f} casos<extra></extra>",
        )
    )

    # Banda crítica invernal (SE 20 a 32)
    fig.add_vrect(
        x0=20,
        x1=32,
        fillcolor="rgba(226, 232, 240, 0.5)",
        line_width=1,
        line_dash="dot",
        line_color=COLOR_NEUTRO,
        layer="below",
        annotation_text="Banda Crítica Invernal (SE 20–32)",
        annotation_position="top left",
        annotation_font={"size": 10, "color": COLOR_NEUTRO},
    )

    fig = _base_layout(fig, height=350)
    fig.update_xaxes(title_text="Semana Epidemiológica (SE)", dtick=4, range=[1, 53])
    fig.update_yaxes(title_text="Episodios Notificados")
    return fig


def distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Fig 02: Distribución y concentración territorial de episodios por provincia.
    """
    totales = (
        df.groupby("provincia")["episodios"]
        .sum()
        .sort_values(ascending=True)
        .reset_index()
    )

    total_general = totales["episodios"].sum()
    totales["pct"] = (totales["episodios"] / total_general) * 100

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=totales["episodios"],
            y=totales["provincia"],
            orientation="h",
            marker={"color": COLOR_PRIMARIO, "line": {"width": 0}},
            text=totales["pct"].apply(lambda p: f"{p:.1f}%"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Casos: %{x:,.0f} (%{text})<extra></extra>",
        )
    )

    fig = _base_layout(fig, height=350)
    fig.update_xaxes(title_text="Episodios Acumulados")
    fig.update_yaxes(title_text="")
    return fig


# ==============================================================================
# M2 · DIAGNÓSTICO EPIDEMIOLÓGICO
# ==============================================================================


def periods_bar(df: pd.DataFrame) -> go.Figure:
    """
    Comparación de la carga media semanal por períodos epidemiológicos:
    Pre-Pandemia (2000-2019: 1,889.6), Pandemia (2020-2021: 728.7), Post-Pandemia (2022-2025: 1,350.9).
    """
    df_p = df.copy()
    conditions = [
        (df_p["anio"] < 2020),
        (df_p["anio"] >= 2020) & (df_p["anio"] <= 2021),
        (df_p["anio"] >= 2022),
    ]
    choices = ["Pre-Pandemia (2000–2019)", "Pandemia (2020–2021)", "Post-Pandemia (2022–2025)"]
    df_p["periodo"] = np.select(conditions, choices, default="Otro")

    semanal = (
        df_p.groupby(["periodo", "anio", "semana"])["episodios"]
        .sum()
        .reset_index()
    )
    resumen = semanal.groupby("periodo")["episodios"].mean().reindex(choices).reset_index()

    colores = [COLOR_SECUNDARIO, COLOR_PELIGRO, COLOR_PRIMARIO]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=resumen["periodo"],
            y=resumen["episodios"],
            marker={"color": colores, "line": {"width": 0}},
            text=resumen["episodios"].apply(lambda v: f"{v:,.1f} casos/sem"),
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Media Semanal: %{y:.1f}<extra></extra>",
        )
    )

    fig = _base_layout(fig, height=350)
    fig.update_yaxes(title_text="Episodios Promedio por Semana")
    fig.update_xaxes(title_text="")
    return fig


def vmr_chart(df: pd.DataFrame) -> go.Figure:
    """
    Razón Varianza/Media (VMR) por provincia y valor regional de referencia.
    """
    stats = (
        df.groupby("provincia")["episodios"]
        .agg(media="mean", varianza="var")
        .reset_index()
    )
    stats["vmr"] = stats["varianza"] / stats["media"]
    stats = stats.sort_values(by="vmr", ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=stats["vmr"],
            y=stats["provincia"],
            orientation="h",
            marker={"color": COLOR_PRIMARIO},
            text=stats["vmr"].apply(lambda v: f"{v:.2f}"),
            textposition="outside",
            name="VMR Provincial",
            hovertemplate="<b>%{y}</b><br>VMR: %{x:.2f}<extra></extra>",
        )
    )

    fig.add_vline(
        x=1.0,
        line_width=1.5,
        line_dash="dash",
        line_color=COLOR_NEUTRO,
        annotation_text="Poisson Teórico (VMR=1.0)",
        annotation_position="bottom right",
        annotation_font={"size": 10, "color": COLOR_NEUTRO},
    )

    fig = _base_layout(fig, height=350)
    fig.update_xaxes(title_text="VMR (Varianza / Media)")
    fig.update_yaxes(title_text="")
    return fig


# ==============================================================================
# M3 · PRONÓSTICO Y MODELADO PREDICTIVO
# ==============================================================================


def model_comparison_chart(metricas: pd.DataFrame) -> go.Figure:
    """
    Fig 04: Benchmark comparativo de modelos ordenados por WAPE (%) para el horizonte seleccionado.
    """
    df_m = metricas.sort_values(by="wape", ascending=True).copy()
    colores = [COLOR_PRIMARIO if i == 0 else COLOR_SECUNDARIO for i in range(len(df_m))]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_m["modelo"],
            y=df_m["wape"],
            marker={"color": colores},
            text=df_m["wape"].apply(lambda w: f"{w:.2f}%"),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>WAPE: %{y:.2f}%<extra></extra>",
        )
    )

    fig = _base_layout(fig, height=350)
    fig.update_yaxes(title_text="WAPE (%)", range=[0, max(df_m["wape"].max() * 1.15, 30)])
    fig.update_xaxes(title_text="", tickangle=-20)
    return fig


def forecast_chart(
    predicciones: pd.DataFrame,
    provincia: str = "SAN ROMAN",
    h: int = 1,
    modelo: str = "XGBoost Global",
) -> go.Figure:
    """
    Fig 05: Serie temporal 2025 - Real (Azul) vs Pronóstico (Rojo) con Intervalos de Incertidumbre (95%).
    Máximo contraste cromático para identificación inmediata.
    """
    df_sub = predicciones[
        (predicciones["anio"] == 2025)
        & (predicciones["particion"] == "test")
        & (predicciones["h"] == h)
        & (predicciones["modelo"] == modelo)
    ].copy()

    if provincia != "TODAS":
        df_sub = df_sub[df_sub["provincia"] == provincia].copy()
    else:
        df_sub = (
            df_sub.groupby("semana")[["observado", "pronostico", "lim_inf", "lim_sup"]]
            .sum(min_count=1)
            .reset_index()
        )

    df_sub = df_sub.sort_values(by="semana")

    fig = go.Figure()

    # Intervalo de confianza 95% (sombreado sutil y limpio)
    tiene_ic = (
        "lim_inf" in df_sub.columns
        and "lim_sup" in df_sub.columns
        and df_sub["lim_inf"].notna().any()
        and df_sub["lim_sup"].notna().any()
    )

    if tiene_ic:
        x_ic = list(df_sub["semana"]) + list(df_sub["semana"][::-1])
        y_ic = list(df_sub["lim_sup"]) + list(df_sub["lim_inf"][::-1])
        fig.add_trace(
            go.Scatter(
                x=x_ic,
                y=y_ic,
                fill="toself",
                fillcolor="rgba(29, 91, 140, 0.08)",
                line={"color": "rgba(255,255,255,0)"},
                name="Intervalo 95%",
                hoverinfo="skip",
            )
        )

    # Observado (Real) en AZUL institucional nítido
    fig.add_trace(
        go.Scatter(
            x=df_sub["semana"],
            y=df_sub["observado"],
            mode="lines+markers",
            line={"color": COLOR_OBSERVADO, "width": 2},
            marker={"size": 5, "color": COLOR_OBSERVADO, "symbol": "circle"},
            name="Observado (Real)",
            hovertemplate="SE %{x}<br>Real: %{y:,.0f}<extra></extra>",
        )
    )

    # Pronóstico en ROJO coral de alto contraste
    fig.add_trace(
        go.Scatter(
            x=df_sub["semana"],
            y=df_sub["pronostico"],
            mode="lines+markers",
            line={"color": COLOR_PRONOSTICO, "width": 2},
            marker={"size": 5, "color": COLOR_PRONOSTICO, "symbol": "diamond"},
            name="Pronóstico",
            hovertemplate="SE %{x}<br>Pronóstico: %{y:,.1f}<extra></extra>",
        )
    )

    fig = _base_layout(fig, height=360)
    fig.update_xaxes(title_text="Semana Epidemiológica 2025", dtick=4)
    fig.update_yaxes(title_text="Episodios Semanales")
    return fig


def wape_province_chart(
    predicciones: pd.DataFrame,
    h: int = 1,
    modelo: str = "XGBoost Global",
) -> go.Figure:
    """
    Fig 06: Error WAPE provincial calculado en Prueba 2025 con semáforo de colores.
    """
    df_sub = predicciones[
        (predicciones["particion"] == "test")
        & (predicciones["h"] == h)
        & (predicciones["modelo"] == modelo)
    ].copy()

    resultados = []
    for prov, grp in df_sub.groupby("provincia"):
        obs = grp["observado"].values
        pred = grp["pronostico"].values
        denom = np.sum(obs)
        wape = 100.0 * np.sum(np.abs(obs - pred)) / denom if denom > 0 else 0.0
        resultados.append({"provincia": prov, "wape": wape})

    df_wape = pd.DataFrame(resultados).sort_values(by="wape", ascending=True)

    colores = []
    for w in df_wape["wape"]:
        if w < 15.0:
            colores.append(COLOR_EXITO)
        elif w <= 30.0:
            colores.append(COLOR_ADVERTENCIA)
        else:
            colores.append(COLOR_PELIGRO)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_wape["wape"],
            y=df_wape["provincia"],
            orientation="h",
            marker={"color": colores},
            text=df_wape["wape"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>WAPE: %{x:.2f}%<extra></extra>",
        )
    )

    fig = _base_layout(fig, height=360)
    fig.update_xaxes(title_text="WAPE (%)", range=[0, max(df_wape["wape"].max() * 1.15, 35)])
    fig.update_yaxes(title_text="")
    return fig


def rolling_chart(rolling: pd.DataFrame) -> go.Figure:
    """
    Fig 08: Validación temporal Rolling-Origin (2020-2024) con colores claramente contrastados por modelo.
    """
    fig = go.Figure()
    # Paleta de 4 colores vivas y diferenciadas
    colores_modelo = {
        "XGBoost Global": "#dc2626",          # Rojo vivo
        "LightGBM Global": "#2563eb",         # Azul brillante
        "XGBoost Global Directo": "#d97706",  # Ámbar / naranja
        "LightGBM Global Directo": "#059669", # Verde esmeralda
    }

    df_sorted = rolling.sort_values(by=["modelo", "anio_test"])

    for mod, grp in df_sorted.groupby("modelo"):
        color = colores_modelo.get(mod, COLOR_NEUTRO)
        fig.add_trace(
            go.Scatter(
                x=grp["anio_test"],
                y=grp["wape"],
                mode="lines+markers",
                name=f"{mod} (h={grp['h'].iloc[0]})",
                line={"color": color, "width": 2},
                marker={"size": 6, "color": color},
                hovertemplate="<b>%{fullData.name}</b><br>Año: %{x}<br>WAPE: %{y:.2f}%<extra></extra>",
            )
        )

    fig.add_vrect(
        x0=2019.8,
        x1=2020.2,
        fillcolor="rgba(220, 38, 38, 0.12)",
        line_width=1,
        line_dash="dot",
        line_color=COLOR_PELIGRO,
        layer="below",
        annotation_text="Shock COVID-19 (2020)",
        annotation_position="top left",
        annotation_font={"size": 10, "color": COLOR_PELIGRO},
    )

    fig = _base_layout(fig, height=360)
    fig.update_xaxes(title_text="Año de Evaluación Out-of-Sample", dtick=1)
    fig.update_yaxes(title_text="WAPE (%)")
    return fig


# ==============================================================================
# M4 · INCERTIDUMBRE Y ESCENARIOS DE CAPACIDAD
# ==============================================================================


def calibration_comparison_chart(calibracion: pd.DataFrame) -> go.Figure:
    """
    Comparación de Cobertura empírica Original vs Calibrada (Conformal Prediction).
    """
    df_c = calibracion.sort_values(by="provincia")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_c["provincia"],
            y=df_c["cob_orig"],
            name="Cobertura Original",
            marker={"color": "#94a3b8"},
            hovertemplate="<b>%{x}</b><br>Cob. Original: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df_c["provincia"],
            y=df_c["cob_calib"],
            name="Cobertura Calibrada",
            marker={"color": COLOR_PRIMARIO},
            hovertemplate="<b>%{x}</b><br>Cob. Calibrada: %{y:.1f}%<extra></extra>",
        )
    )

    fig.add_hline(
        y=95.0,
        line_width=2,
        line_dash="dash",
        line_color=COLOR_PELIGRO,
        annotation_text="Nivel Nominal 95%",
        annotation_position="top right",
        annotation_font={"size": 11, "color": COLOR_PELIGRO},
    )

    fig = _base_layout(fig, height=360)
    fig.update_yaxes(title_text="Cobertura Empírica (%)", range=[80, 102])
    fig.update_xaxes(title_text="", tickangle=-25)
    return fig


def scenario_chart(
    predicciones: pd.DataFrame,
    provincia: str = "SAN ROMAN",
    umbral: int = 300,
) -> go.Figure:
    """
    Fig 07: Simulación prescriptiva: Demanda prevista 2025 vs Capacidad Asistencial Hipotética.
    Demanda en Azul y Capacidad en Rojo.
    """
    df_sub = predicciones[
        (predicciones["anio"] == 2025)
        & (predicciones["particion"] == "test")
        & (predicciones["h"] == 1)
        & (predicciones["provincia"] == provincia)
    ].sort_values(by="semana")

    if df_sub.empty:
        df_sub = (
            predicciones[
                (predicciones["anio"] == 2025)
                & (predicciones["particion"] == "test")
                & (predicciones["h"] == 1)
            ]
            .groupby("semana")[["observado", "pronostico"]]
            .sum()
            .reset_index()
        )

    demanda = df_sub["pronostico"].values
    semanas = df_sub["semana"].values
    capacidad = np.full_like(demanda, float(umbral))
    brecha = np.maximum(0, demanda - capacidad)

    fig = go.Figure()

    # Demanda prevista (Azul)
    fig.add_trace(
        go.Scatter(
            x=semanas,
            y=demanda,
            mode="lines+markers",
            line={"color": COLOR_PRIMARIO, "width": 2},
            marker={"size": 4, "color": COLOR_PRIMARIO},
            name="Demanda Prevista",
            hovertemplate="SE %{x}<br>Demanda: %{y:,.1f}<extra></extra>",
        )
    )

    # Capacidad asistencial hipotética (Rojo punteado)
    fig.add_trace(
        go.Scatter(
            x=semanas,
            y=capacidad,
            mode="lines",
            line={"color": COLOR_PELIGRO, "width": 2, "dash": "dot"},
            name=f"Capacidad ({umbral} casos)",
            hovertemplate="Capacidad: %{y:,.0f}<extra></extra>",
        )
    )

    semanas_brecha = semanas[brecha > 0]
    brechas_valor = brecha[brecha > 0]

    if len(semanas_brecha) > 0:
        fig.add_trace(
            go.Bar(
                x=semanas_brecha,
                y=brechas_valor,
                base=capacidad[brecha > 0],
                marker={"color": "rgba(220, 38, 38, 0.4)", "line": {"color": COLOR_PELIGRO, "width": 1}},
                name="Brecha de Atención",
                hovertemplate="SE %{x}<br>Brecha: +%{y:.1f} casos<extra></extra>",
            )
        )

    fig = _base_layout(fig, height=360)
    fig.update_xaxes(title_text="Semana Epidemiológica 2025", dtick=4)
    fig.update_yaxes(title_text="Episodios de IRA")
    return fig
