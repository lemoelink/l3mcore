#!/usr/bin/env bash
# LEMoE setup script.
# Creates a virtualenv, installs dependencies, and downloads the router model.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for updates
if command -v git &> /dev/null && [ -d ".git" ]; then
    echo "[LEMoE] Checking for updates..."
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "master")
    git fetch https://github.com/lemoelink/LeMoE.git "$CURRENT_BRANCH" -q 2>/dev/null || true
    if [ $(git rev-list HEAD..FETCH_HEAD 2>/dev/null | wc -l) -gt 0 ]; then
        echo -e "\033[1;32m"
        echo "==========================================================="
        echo "A new LEMoE update is available on branch $CURRENT_BRANCH!"
        echo "To update, run the command:"
        echo "  git pull"
        echo "==========================================================="
        echo -e "\033[0m"
    fi
fi


echo "LEMoE Setup"
echo "==========="
echo ""

# Check prerequisites
MISSING_PREREQS=0

if ! command -v curl &> /dev/null; then
    echo "Error: 'curl' is required but not installed."
    MISSING_PREREQS=1
fi

if ! command -v python3 &> /dev/null; then
    echo "Error: 'python3' is required but not installed."
    MISSING_PREREQS=1
fi

if ! command -v gcc &> /dev/null && ! command -v clang &> /dev/null; then
    echo "Error: A C/C++ compiler ('gcc' or 'clang') is required to compile native dependencies (e.g., llama-cpp-python) but none was found."
    MISSING_PREREQS=1
fi

if ! command -v make &> /dev/null; then
    echo "Error: 'make' is required to compile native dependencies but was not found."
    MISSING_PREREQS=1
fi

if [ $MISSING_PREREQS -eq 1 ]; then
    echo "Please install the missing dependencies and run the setup again."
    echo ""
    echo "Installation commands for common distributions:"
    echo "  Debian/Ubuntu: sudo apt update && sudo apt install curl python3 python3-venv python3-pip build-essential"
    echo "  Fedora:        sudo dnf install curl python3 python3-pip gcc gcc-c++ make"
    echo "  Arch Linux:    sudo pacman -S curl python python-pip base-devel"
    echo ""
    exit 1
fi

echo ""
echo "Setting up Python virtual environment..."
if [ ! -f "venv/bin/activate" ]; then
    echo "Virtual environment not found or incomplete. Recreating venv..."
    rm -rf venv
    python3 -m venv venv
fi
source venv/bin/activate
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Check Ollama
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed."
    read -p "Install Ollama automatically? [y/N]: " install_ollama
    if [[ "$install_ollama" =~ ^[Yy]$ ]]; then
        echo "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        if [ $? -ne 0 ]; then
            echo "Ollama installation failed. Please install it manually from https://ollama.com"
            exit 1
        fi
        echo "Ollama installed."
    else
        echo "Skipping Ollama installation."
    fi
else
    echo "Ollama is already installed."
fi

echo ""

# Router language / embedding model
echo "Select the primary language for the semantic router:"
echo "  1) Spanish / Multilingual"
echo "  2) English"
read -p "Choice [1/2]: " lang_choice

if [ "$lang_choice" = "1" ]; then
    model_name="intfloat/multilingual-e5-small"
    echo "Using multilingual model: $model_name"
elif [ "$lang_choice" = "2" ]; then
    model_name="BAAI/bge-small-en-v1.5"
    echo "Using English model: $model_name"
else
    echo "Invalid choice. Defaulting to multilingual (intfloat/multilingual-e5-small)."
    model_name="intfloat/multilingual-e5-small"
fi

echo ""

# Plugin directory
echo "Enable plugin system? (creates plugins/ directory)"
read -p "Enable plugins? [y/N]: " enable_plugins
if [[ "$enable_plugins" =~ ^[Yy]$ ]]; then
    mkdir -p plugins
    echo "plugins/ directory created."
else
    echo "Plugins disabled."
fi

echo ""

# Write config
CONFIG_FILE="config/config.json"
mkdir -p config

python3 - <<PYEOF
import json, os

config_file = '$CONFIG_FILE'
model_name  = '$model_name'

data = {}
if os.path.exists(config_file):
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f'Warning: could not read {config_file}: {e}')

data.setdefault('router', {})
data['router']['model_path']  = model_name
data['router']['router_type'] = 'embedding'

try:
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print('config.json updated.')
except Exception as e:
    print(f'Error writing {config_file}: {e}')
PYEOF

echo ""
echo "Downloading router model (this may take a moment)..."

python3 - <<PYEOF
import sys
try:
    from sentence_transformers import SentenceTransformer
    print('Fetching: $model_name ...')
    SentenceTransformer('$model_name')
    print('Model ready.')
except ImportError:
    print('Error: sentence-transformers is not installed. Run: pip install sentence-transformers')
    sys.exit(1)
except Exception as e:
    print(f'Error downloading model: {e}')
    sys.exit(1)
PYEOF

echo ""
echo "Do you want to download the generic fallback model (Qwen3.5-0.8B-Q4_K_M.gguf)?"
echo "If you choose no, you must configure your own fallback model (API, Ollama, etc.) in config/experts.json."
read -p "Download fallback model? [y/N]: " dl_fallback
if [[ "$dl_fallback" =~ ^[Yy]$ ]]; then
    echo "Downloading Qwen3.5-0.8B-Q4_K_M.gguf (approx 500MB)..."
    mkdir -p models
    curl -L "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf" -o models/Qwen3.5-0.8B-Q4_K_M.gguf
    if [ $? -eq 0 ]; then
        echo "Fallback model downloaded successfully."
    else
        echo "Failed to download the fallback model."
    fi
else
    echo "Skipping fallback model download."
fi



if [ $? -eq 0 ]; then
    echo ""
    echo "Setup complete. Run ./start.sh to start LEMoE."
else
    echo ""
    echo "Setup failed during model download."
fi
