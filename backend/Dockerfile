# Build context is the repository root (see docker-compose.yml) so that both
# backend/ and the shared ml/ package land in the image. The workers import
# `ml.inference.*`, so ml/ must be importable from /app.
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir setuptools==69.5.1 wheel
RUN pip install --no-cache-dir --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Application code
COPY backend/ /app/
# Shared ML package — the single source of truth for inference code
COPY ml/ /app/ml/

# Runtime directories. Ownership is set here so bind/named volumes mounted
# over them are writable by the non-root user without a manual chown.
RUN mkdir -p /app/storage /app/ml_models \
    && useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

COPY backend/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

USER appuser

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
