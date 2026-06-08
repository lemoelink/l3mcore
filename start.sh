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
    echo -e "\033[32m[L3MCOre] Cargando variables de entorno desde .env...\033[0m"
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

# Comprobar actualizaciones
if command -v git &> /dev/null && [ -d ".git" ]; then
    echo -e "\033[32m[L3MCOre] Comprobando actualizaciones...\033[0m"
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "master")
    git fetch https://github.com/lemoelink/LeMoE.git "$CURRENT_BRANCH" -q 2>/dev/null || true
    if [ $(git rev-list HEAD..FETCH_HEAD 2>/dev/null | wc -l) -gt 0 ]; then
        echo -e "\033[33m"
        echo "==========================================================="
        echo "¡Hay una nueva actualización de L3MCOre disponible en la rama $CURRENT_BRANCH!"
        echo "Para actualizar, ejecuta el comando:"
        echo "  git pull"
        echo "==========================================================="
        echo -e "\033[0m"
    fi
fi


if [ ! -d "$VENV_DIR" ]; then
    echo -e "\033[32m[L3MCOre] Creating virtual environment...\033[0m"
    python3 -m venv "$VENV_DIR"
    echo -e "\033[32m[L3MCOre] Installing dependencies...\033[0m"
    "$VENV_DIR/bin/pip" install --upgrade pip -q
    "$VENV_DIR/bin/pip" install -r requirements.txt -q
    echo -e "\033[32m[L3MCOre] Installing PyTorch (CPU)...\033[0m"
    "$VENV_DIR/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu -q
    echo -e "\033[32m[L3MCOre] Dependencies installed.\033[0m"
fi

echo -e "\033[32m[L3MCOre] Activating virtual environment: $VENV_DIR\033[0m"
source "$VENV_DIR/bin/activate"

echo -e "\033[32m[L3MCOre] Starting API server on http://${HOST}:${PORT}\033[0m"
echo -e "\033[32m[L3MCOre] OpenAI-compatible endpoint: http://${HOST}:${PORT}/v1\033[0m"
echo -e "\033[32m[L3MCOre] Ollama-compatible endpoint: http://${HOST}:${PORT}/api\033[0m"
echo -e "\033[32m[L3MCOre] Server: Gunicorn (workers=${WORKERS})\033[0m"

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
