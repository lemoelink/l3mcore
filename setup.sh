#!/usr/bin/env bash
# LEMoE setup script.
# Creates a virtualenv, installs dependencies, and downloads the router model.

set -e

# Detect if piped from web or run locally
if [ ! -f "api_server.py" ] && [ ! -d "modules" ]; then
    echo -e "\033[32m[L3MCOre] Downloading and installing from GitHub...\033[0m"
    if [ -d "LeMoE" ]; then
        echo "Directory 'LeMoE' already exists. Please remove it or run setup from inside it."
        exit 1
    fi
    git clone https://github.com/lemoelink/l3mcore.git
    cd L3mcore
else
    # Check for updates if we are already inside the local repo
    if command -v git &> /dev/null && [ -d ".git" ]; then
        echo -e "\033[32m[L3MCOre] Checking for updates...\033[0m"
        CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "master")
        git fetch https://github.com/lemoelink/l3mcore.git "$CURRENT_BRANCH" -q 2>/dev/null || true
        if [ $(git rev-list HEAD..FETCH_HEAD 2>/dev/null | wc -l) -gt 0 ]; then
            echo -e "\033[33m"
            echo "==========================================================="
            echo "A new L3MCOre update is available on branch $CURRENT_BRANCH!"
            echo "To update, run the command:"
            echo "  git pull"
            echo "==========================================================="
            echo -e "\033[0m"
        fi
    fi
fi


echo "L3MCOre Setup"
echo "============="
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
    read -p "Install Ollama automatically? [y/N]: " install_ollama < /dev/tty
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

# Semantic router model configuration
echo "Do you want to download and enable the semantic router model?"
echo "If disabled, the system will use keyword matching exclusively."
read -p "Enable semantic router? [Y/n]: " enable_router < /dev/tty

# Default to Yes if empty or Y/y
if [[ -z "$enable_router" || "$enable_router" =~ ^[Yy]$ ]]; then
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    echo "Using semantic router model: $model_name"
else
    model_name=""
    echo "Semantic router model disabled."
fi

echo ""

# Plugin directory
echo "Enable plugin system? (creates plugins/ directory)"
read -p "Enable plugins? [y/N]: " enable_plugins < /dev/tty
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

if [ -n "$model_name" ]; then
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
else
    echo "Skipping router model download."
fi

echo ""
echo "Do you want to download the generic fallback model (Qwen3.5-0.8B-Q4_K_M.gguf)?"
echo "If you choose no, you must configure your own fallback model (API, Ollama, etc.) in config/experts.json."
read -p "Download fallback model? [y/N]: " dl_fallback < /dev/tty
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


# Custom Paperless Search models
if [[ "$enable_plugins" =~ ^[Yy]$ ]] && [ -f "plugins/paperless_search.py" ]; then
    echo ""
    echo "Do you want to download the custom BERT and DeBERTa models for the Paperless Search plugin?"
    read -p "Download Paperless Search models? [y/N]: " dl_paperless < /dev/tty
    if [[ "$dl_paperless" =~ ^[Yy]$ ]]; then
        echo "Downloading custom BERT and DeBERTa models (this may take a few minutes)..."
        python3 - <<PYEOF
import sys
import os
sys.path.insert(0, os.getcwd())
try:
    import plugins.paperless_search as p
    p._is_testing = False
    print("Downloading BERT Classifier...")
    p._perform_update_for(
        model_name_log="Clasificador BERT",
        model_dir=p.MODEL_DIR,
        hf_api_url=p.HF_API_URL,
        hf_resolve_url=p.HF_RESOLVE_URL,
        files_list=p.FILES_TO_DOWNLOAD,
        reload_callback=lambda: None
    )
    print("Downloading DeBERTa Distiller...")
    p._perform_update_for(
        model_name_log="Destilador DeBERTa",
        model_dir=p.DISTILLER_DIR,
        hf_api_url=p.HF_API_URL_DISTILLER,
        hf_resolve_url=p.HF_RESOLVE_URL_DISTILLER,
        files_list=p.FILES_TO_DOWNLOAD_DISTILLER,
        reload_callback=lambda: None
    )
    print("Paperless Search models ready.")
except Exception as e:
    print(f"Error downloading Paperless Search models: {e}")
    sys.exit(1)
PYEOF
    fi
fi




if [ $? -eq 0 ]; then
    echo ""
    echo "Setup complete. Run ./start.sh to start L3MCOre."
else
    echo ""
    echo "Setup failed during model download."
fi
