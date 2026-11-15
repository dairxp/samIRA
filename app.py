"""
samIRA - Aplicación Shiny for Python.
Vigilancia epidemiológica y pronóstico de Infecciones Respiratorias Agudas en Puno (2000–2025).

Para ejecutar localmente con recarga en vivo:
    uv run shiny run --reload app.py
"""

from pathlib import Path
from shiny import App

from samira_app.server import server
from samira_app.ui import app_ui

# Directorio de recursos estáticos (logo.png, etc.)
www_dir = Path(__file__).resolve().parent / "www"

app = App(app_ui, server, static_assets=www_dir)
