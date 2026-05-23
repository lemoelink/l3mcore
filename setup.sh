#!/usr/bin/env bash
# LEMoE setup script.
# Creates a virtualenv, installs dependencies, and downloads the router model.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

if [ $MISSING_PREREQS -eq 1 ]; then
    echo "Please install the missing dependencies and run the setup again."
    exit 1
fi

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
read -p "Enable plugin system? (creates plugins/ directory) [y/N]: " enable_plugins
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

if [ $? -eq 0 ]; then
    echo ""
    echo "Setup complete. Run ./start.sh to start LEMoE."
else
    echo ""
    echo "Setup failed during model download."
fi
