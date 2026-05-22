#!/bin/bash

echo "=================================="
echo "      LEMoE Setup Script          "
echo "=================================="
echo ""

# 1. Comprobar instalación de Ollama
if ! command -v ollama &> /dev/null; then
    echo "Ollama no está instalado / Ollama is not installed."
    read -p "¿Desea instalar Ollama automáticamente? / Install Ollama automatically? [y/N]: " install_ollama
    if [[ "$install_ollama" =~ ^[Yy]$ ]]; then
        echo "Instalando Ollama / Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        if [ $? -ne 0 ]; then
            echo "Error instalando Ollama. Por favor, instálelo manualmente desde https://ollama.com"
            exit 1
        fi
        echo "Ollama se instaló correctamente / Ollama installed successfully."
    else
        echo "Omitiendo instalación de Ollama / Skipping Ollama installation."
    fi
else
    echo "Ollama ya está instalado / Ollama is already installed."
fi

echo ""
# 2. Elegir idioma para el router
echo "Seleccione el idioma principal para el router semántico:"
echo "Select your primary language for the semantic router:"
echo "  1) Español (ES)"
echo "  2) English (EN)"
read -p "Opción / Choice [1/2]: " lang_choice

model_name=""
if [ "$lang_choice" = "1" ]; then
    model_name="intfloat/multilingual-e5-small"
    echo "Idioma seleccionado: Español. Modelo: $model_name"
elif [ "$lang_choice" = "2" ]; then
    model_name="BAAI/bge-small-en-v1.5"
    echo "Selected language: English. Model: $model_name"
else
    echo "Opción no válida. Usando por defecto Español (intfloat/multilingual-e5-small)"
    model_name="intfloat/multilingual-e5-small"
fi

echo ""
echo "Actualizando config.json / Updating config.json..."
CONFIG_FILE="config/config.json"

if [ ! -d "config" ]; then
    mkdir -p config
fi

python3 -c "
import json
import os

config_file = '$CONFIG_FILE'
model_name = '$model_name'

data = {}
if os.path.exists(config_file):
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f'Error leyendo {config_file}: {e}')

if 'router' not in data:
    data['router'] = {}

data['router']['model_path'] = model_name
data['router']['router_type'] = 'embedding'

try:
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print('config.json actualizado correctamente / config.json updated successfully.')
except Exception as e:
    print(f'Error escribiendo en {config_file}: {e}')
"

echo ""
echo "Descargando modelo del router (puede tardar un momento) / Downloading router model (may take a moment)..."
python3 -c "
import sys
try:
    from sentence_transformers import SentenceTransformer
    print('Descargando/Verificando: $model_name ...')
    model = SentenceTransformer('$model_name')
    print('Modelo descargado y listo / Model downloaded and ready.')
except ImportError:
    print('Error: sentence-transformers no está instalado. Ejecute: pip install sentence-transformers')
    sys.exit(1)
except Exception as e:
    print(f'Error descargando el modelo / Error downloading model: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "¡Configuración completada con éxito! / Setup completed successfully!"
    echo "Ya puedes ejecutar LEMoE / You can now run LEMoE."
else
    echo ""
    echo "La configuración falló durante la descarga del modelo / Setup failed during model download."
fi
