# =============================================================================
# AI Anime Video Creator - Azure CPU Dockerfile
# =============================================================================
# Optimized for Azure Free VM (NO GPU)
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base Image
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3.10-dev \
    git \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Ensure python points to python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

RUN python -m pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# Stage 2: Python Dependencies
# -----------------------------------------------------------------------------
FROM base AS dependencies

WORKDIR /app

COPY requirements.txt .

# CPU-only PyTorch (SAFE on Azure)
RUN pip install torch torchvision torchaudio

RUN pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 3: Production Image
# -----------------------------------------------------------------------------
FROM dependencies AS production

WORKDIR /app

# Non-root user
RUN useradd -m -u 1000 appuser

# Create runtime directories (CPU SAFE)
RUN mkdir -p \
    /app/data \
    /app/outputs \
    /app/temp \
    /app/logs \
    /app/workflows \
    && chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . .

# Entrypoint
COPY --chown=appuser:appuser docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER appuser

ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "frontend/app.py"]
