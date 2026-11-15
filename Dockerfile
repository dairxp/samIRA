# Dockerfile para Hugging Face Spaces (samIRA Shiny for Python)
FROM python:3.12-slim

# Evitar buffers en stdout/stderr y prevenir creación de archivos .pyc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Instalar dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario estándar para Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copiar configuración de dependencias e instalarlas
COPY --chown=user:user pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copiar estructura de la aplicación y datos
COPY --chown=user:user app.py ./
COPY --chown=user:user samira_app/ ./samira_app/
COPY --chown=user:user data/ ./data/
COPY --chown=user:user results/ ./results/
COPY --chown=user:user www/ ./www/

# Exponer el puerto predeterminado de Hugging Face Spaces
EXPOSE 7860

# Comando de inicio del servidor Shiny
CMD ["shiny", "run", "--host", "0.0.0.0", "--port", "7860", "app.py"]
