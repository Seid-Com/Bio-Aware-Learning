FROM python:3.12-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy simulation code
COPY simulation/ ./simulation/

# Create non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import streamlit; print('ok')" || exit 1

ENTRYPOINT ["streamlit", "run", "simulation/app.py", \
    "--server.port=5000", \
    "--server.headless=true", \
    "--server.address=0.0.0.0"]
