# =============================================================================
# AI Video Creator - Dockerfile
# =============================================================================
# Multi-stage build for optimized production image
# Supports NVIDIA GPU acceleration via nvidia-docker
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base image with CUDA support
# -----------------------------------------------------------------------------
FROM python:3.10-slim


# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set Python environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for Python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# Stage 2: Install Python dependencies
# -----------------------------------------------------------------------------
FROM base AS dependencies

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining Python dependencies
RUN pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 3: Production image
# -----------------------------------------------------------------------------
FROM dependencies AS production

WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Create required directories
RUN mkdir -p \
    /app/data \
    /app/outputs \
    /app/temp \
    /app/logs \
    /app/models/checkpoints \
    /app/models/motion_modules \
    /app/models/loras \
    /app/models/vae \
    /app/workflows \
    && chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . .

# Copy entrypoint script
COPY --chown=appuser:appuser docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=7860

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "frontend/app.py"]

