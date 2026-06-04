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

# Cargar variables de entorno desde .env si existe
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "[LEMoE] Cargando variables de entorno desde .env..."
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

# Comprobar actualizaciones
if command -v git &> /dev/null && [ -d ".git" ]; then
    echo "[LEMoE] Comprobando actualizaciones..."
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "master")
    git fetch https://github.com/lemoelink/LeMoE.git "$CURRENT_BRANCH" -q 2>/dev/null || true
    if [ $(git rev-list HEAD..FETCH_HEAD 2>/dev/null | wc -l) -gt 0 ]; then
        echo -e "\033[1;32m"
        echo "==========================================================="
        echo "¡Hay una nueva actualización de LEMoE disponible en la rama $CURRENT_BRANCH!"
        echo "Para actualizar, ejecuta el comando:"
        echo "  git pull"
        echo "==========================================================="
        echo -e "\033[0m"
    fi
fi


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
