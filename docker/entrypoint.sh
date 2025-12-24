#!/bin/bash
# =============================================================================
# AI Video Creator - Docker Entrypoint Script
# =============================================================================
# Handles initialization, environment setup, and application startup
# =============================================================================

set -e

echo "=============================================="
echo "  AI Video Creator - Starting Up"
echo "=============================================="

# -----------------------------------------------------------------------------
# Environment Configuration
# -----------------------------------------------------------------------------

# Set default values if not provided
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-7860}
export COMFYUI_HOST=${COMFYUI_HOST:-comfyui}
export COMFYUI_PORT=${COMFYUI_PORT:-8188}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

echo "Configuration:"
echo "  - Host: $HOST:$PORT"
echo "  - ComfyUI: $COMFYUI_HOST:$COMFYUI_PORT"
echo "  - Log Level: $LOG_LEVEL"

# -----------------------------------------------------------------------------
# Directory Setup
# -----------------------------------------------------------------------------

echo "Setting up directories..."

# Ensure all required directories exist
mkdir -p /app/data
mkdir -p /app/outputs
mkdir -p /app/temp
mkdir -p /app/logs
mkdir -p /app/models/checkpoints
mkdir -p /app/models/motion_modules
mkdir -p /app/models/loras
mkdir -p /app/models/vae
mkdir -p /app/workflows

# -----------------------------------------------------------------------------
# GPU Detection
# -----------------------------------------------------------------------------

echo "Checking GPU availability..."

python -c "
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'  - GPU: {gpu_name}')
    print(f'  - VRAM: {gpu_memory:.1f} GB')
else:
    print('  - WARNING: No GPU detected, running on CPU')
    print('  - Video generation will be extremely slow!')
" || echo "  - Failed to detect GPU"

# -----------------------------------------------------------------------------
# Wait for Dependencies
# -----------------------------------------------------------------------------

if [ "$WAIT_FOR_COMFYUI" = "true" ]; then
    echo "Waiting for ComfyUI to be ready..."

    MAX_RETRIES=60
    RETRY_COUNT=0

    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -s "http://$COMFYUI_HOST:$COMFYUI_PORT/system_stats" > /dev/null 2>&1; then
            echo "  - ComfyUI is ready!"
            break
        fi

        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "  - Waiting for ComfyUI... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 5
    done

    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "  - WARNING: ComfyUI not available after $MAX_RETRIES attempts"
        echo "  - Starting anyway, but video generation will fail"
    fi
fi

# -----------------------------------------------------------------------------
# Start Application
# -----------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "  Starting Application..."
echo "=============================================="
echo ""

# Execute the main command
exec "$@"

