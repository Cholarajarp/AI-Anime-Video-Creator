#!/bin/bash
# =============================================================================
# AI Anime Video Creator - Docker Entrypoint (Azure CPU Safe)
# =============================================================================

set -e

echo "=============================================="
echo "  AI Anime Video Creator - Starting"
echo "=============================================="

# -----------------------------------------------------------------------------
# Environment Defaults
# -----------------------------------------------------------------------------
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-7860}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export WAIT_FOR_COMFYUI=${WAIT_FOR_COMFYUI:-false}

echo "Configuration:"
echo "  - Host        : $HOST:$PORT"
echo "  - Log Level  : $LOG_LEVEL"
echo "  - CPU Mode   : Enabled"

# -----------------------------------------------------------------------------
# Directory Setup (CPU SAFE)
# -----------------------------------------------------------------------------
echo "Setting up directories..."

mkdir -p /app/data
mkdir -p /app/outputs
mkdir -p /app/temp
mkdir -p /app/logs
mkdir -p /app/workflows

# -----------------------------------------------------------------------------
# Hardware Detection (Informational Only)
# -----------------------------------------------------------------------------
echo "Checking hardware availability..."

python - <<EOF
import torch
if torch.cuda.is_available():
    print("  - GPU detected (unexpected on Azure Free VM)")
else:
    print("  - No GPU detected (CPU mode)")
EOF

# -----------------------------------------------------------------------------
# ComfyUI Handling (Disabled on Azure)
# -----------------------------------------------------------------------------
if [ "$WAIT_FOR_COMFYUI" = "true" ]; then
    echo "WARNING: WAIT_FOR_COMFYUI=true but ComfyUI is not deployed."
    echo "Disabling ComfyUI wait to prevent startup hang."
fi

# -----------------------------------------------------------------------------
# Start Application
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Launching Application"
echo "=============================================="
echo ""

exec "$@"
