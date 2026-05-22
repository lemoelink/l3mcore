#!/usr/bin/env bash
# LEMoE - Script de arranque del servidor API
# Crea el entorno virtual si no existe e inicia el servidor OpenAI/Ollama compatible

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
HOST="${LEMOE_HOST:-0.0.0.0}"
PORT="${LEMOE_PORT:-11435}"

cd "$SCRIPT_DIR"

# Crear venv si no existe
if [ ! -d "$VENV_DIR" ]; then
    echo "[LEMoE] Creando entorno virtual..."
    python3 -m venv "$VENV_DIR"
    echo "[LEMoE] Instalando dependencias base..."
    "$VENV_DIR/bin/pip" install --upgrade pip -q
    "$VENV_DIR/bin/pip" install flask transformers sentencepiece onnxruntime numpy -q
    echo "[LEMoE] Instalando torch CPU..."
    "$VENV_DIR/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu -q
    echo "[LEMoE] Dependencias instaladas."
fi

source "$VENV_DIR/bin/activate"
echo "[LEMoE] Servidor API arrancando en http://${HOST}:${PORT}"
echo "[LEMoE] Compatible con OpenAI: http://${HOST}:${PORT}/v1"
echo "[LEMoE] Compatible con Ollama: http://${HOST}:${PORT}/api"
exec python3 api_server.py --host "$HOST" --port "$PORT"
