#!/usr/bin/env bash
# Starts the LEMoE API server.
# Creates a virtualenv on first run if one does not already exist.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
HOST="${LEMOE_HOST:-0.0.0.0}"
PORT="${LEMOE_PORT:-11435}"
WORKERS="${LEMOE_WORKERS:-1}"

cd "$SCRIPT_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "[LEMoE] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "[LEMoE] Installing dependencies..."
    "$VENV_DIR/bin/pip" install --upgrade pip -q
    "$VENV_DIR/bin/pip" install -r requirements.txt -q
    echo "[LEMoE] Installing PyTorch (CPU)..."
    "$VENV_DIR/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu -q
    echo "[LEMoE] Dependencies installed."
fi

echo "[LEMoE] Activating virtual environment: $VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[LEMoE] Starting API server on http://${HOST}:${PORT}"
echo "[LEMoE] OpenAI-compatible endpoint: http://${HOST}:${PORT}/v1"
echo "[LEMoE] Ollama-compatible endpoint: http://${HOST}:${PORT}/api"
echo "[LEMoE] Server: Gunicorn (workers=${WORKERS})"

exec "$VENV_DIR/bin/gunicorn" \
    --workers "$WORKERS" \
    --worker-class sync \
    --bind "${HOST}:${PORT}" \
    --timeout 120 \
    --keep-alive 5 \
    --log-level warning \
    --access-logfile - \
    --error-logfile - \
    api_server:app
