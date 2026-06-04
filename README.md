# L3mcore - Light Easy Mix Of Experts

> By downloading this software you agree to the Polyform license terms / Al descargar este software aceptas los términos de la licencia PolyForm Noncommercial 1.0.0 [https://polyformproject.org/licenses/noncommercial/1.0.0]


> [!WARNING]
> **Fase de Desarrollo / In Development**
> Este proyecto se encuentra actualmente en fase de desarrollo activo. Sin embargo, su funcionalidad principal (enrutamiento, carga de modelos y backends) es completamente operativa y estable para su uso.
> 
> This project is currently under active development. However, its core functionality (routing, model loading, and backends) is fully operational and stable for use.

A lightweight Mixture of Experts (MoE) system that acts as an intelligent router for AI requests. It exposes an API fully compatible with OpenAI and Ollama, classifies user input, and automatically dispatches it to the appropriate expert model.

### 👀 Watch it in action!
https://github.com/user-attachments/assets/e97e1481-a0a3-4f25-a3de-7ed25936e2b3

## Table of Contents

1. [Features](#features)
2. [Quick Start](#quick-start)
3. [Manual Installation](#manual-installation)
4. [Configuration](#configuration)
   - [Router (config.json)](#1-router-configjson)
   - [Experts (experts.json)](#2-experts-expertsjson)
   - [Expert Backends](#3-expert-backends)
5. [Router Architecture](#router-architecture)
   - [Hybrid Scoring System](#hybrid-scoring-system)
   - [Scoring Weights](#scoring-weights-tuning)
   - [Fallback Chain](#fallback-chain)
6. [API Usage](#api-usage)
7. [Security](#security)
8. [Project Structure](#project-structure)

---

## Features

- **OpenAI and Ollama Compatible API**: Drop-in replacement for Ollama (port 11435) or an OpenAI-compatible server (`/v1/chat/completions`). Works with Open WebUI, Continue (VSCode), LiteLLM, and any OpenAI client.
- **Precision Hybrid Router**: Three-tier routing system:
  - *Semantic Embedding*: Multi-vector comparison using SentenceTransformers with 4-signal hybrid scoring and softmax normalization.
  - *ML Classifier*: Standard BERT/RoBERTa classification for ultra-fast routing when a fine-tuned model is available.
  - *Keyword Fallback*: Fuzzy matching algorithm (rapidfuzz) when AI methods are unavailable or below the confidence threshold.
- **Multi-Backend Expert Dispatcher**: Connects up to 15 simultaneous experts across different backends:
  - External APIs (OpenAI, Anthropic, Gemini, etc. via LiteLLM)
  - Local Ollama instances
  - Local ONNX or GGUF models (Llama.cpp)
- **Cascading Contextual Routing**: Stateless conversation-aware routing. When a user message is ambiguous (e.g. "Make it shorter"), the router evaluates the last 2-3 user messages from the conversation history to maintain topic continuity. No database or session state required -- each request carries its own context via the standard OpenAI messages array. Configurable via `context_messages` and `context_max_chars` in `config.json`.
- **Silent Self-Correction**: If any expert fails (timeout, API down, connection refused), the system automatically and silently redirects the request to the fallback model. The user receives a normal response without knowing an error occurred. Administrators can monitor failures via `[Auto-Correction]` log entries.
- **Smart Memory Management**: Limits RAM usage for local models (maximum 3 simultaneous), with LRU eviction and TTL-based automatic unloading after 5 minutes of inactivity.
- **Rate Limiting**: Built-in per-IP sliding-window rate limiter (60 req/min by default). Supports `X-Forwarded-For` for reverse proxy setups.
- **Request Size Protection**: Incoming request bodies capped at 1 MB to prevent memory exhaustion.
- **Extensible Plugin System**: Customize pre- and post-processing steps or override routing rules using Python hooks (`override_route`, `before_routing`, `after_generation`). Official and community plugins are hosted at the [plugins repository](https://github.com/lemoelink/plugins), with new ones being released gradually.

---

## Quick Start

Requires Python 3.10 or higher.

```bash
git clone https://github.com/lemoelink/LeMoE
cd "LeMoE"
```

Run the setup script to check dependencies, select your router language, and download the embedding model. *(Note: This script will automatically create and activate an isolated Python virtual environment `venv` to install all requirements securely without breaking your system packages).*

```bash
./setup.sh
```

Then start the server:

```bash
./start.sh
```

The server will be available at `http://0.0.0.0:11435`.

---

## Manual Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Docker Installation

LEMoE is available on Docker Hub and can be run entirely via Docker, avoiding the need to install Python dependencies on your host.

```bash
# Run the default Debian Slim image
docker run -d -p 11435:11435 \
  -v ./config:/app/config \
  -v ./models:/app/models \
  -v ./data:/app/data \
  --name lemoe \
  lemoelink/lemoe:general
```

### Available Tags

- `lemoelink/lemoe:general` (Default): Built on Debian Slim. Optimized for CPU inference. Smallest footprint with maximum compatibility.
- `lemoelink/lemoe:debian`: Built on standard Debian. Identical to `general` but includes full OS libraries.
- `lemoelink/lemoe:cuda`: (Beta) Built on Nvidia CUDA. Use this if you are running Llama.cpp models with GPU acceleration. Requires the `--gpus all` flag when running the container.

**Volume Mounts:**
- `/app/config`: Maps your `config.json` and `experts.json` so you can edit routing rules on the fly without rebuilding.
- `/app/models`: Persists downloaded GGUF/ONNX models and HuggingFace caches across container restarts.
- `/app/data`: Persists LRU/TTL runtime metrics.

---

## Configuration

All configuration lives in the `config/` folder.

### 1. Router (config.json)

The file `config/config.json` controls the routing engine.

```json
{
    "router": {
        "mode": "generic",
        "router_type": "embedding",
        "model_path": "intfloat/multilingual-e5-small",
        "categories_file": "config/experts.json",
        "confidence_threshold": 0.4,
        "keyword_fallback": true,
        "softmax_temperature": 0.15,
        "scoring_weights": {
            "max_keyword":  0.40,
            "description":  0.30,
            "mean_keyword": 0.20,
            "top3_vote":    0.10
        }
    }
}
```

| Field | Type | Description |
|---|---|---|
| `mode` | string | `generic` uses `experts.json`. `model` uses a trained classifier. |
| `router_type` | string | `embedding` (SentenceTransformers) or `classification` (fine-tuned BERT). |
| `model_path` | string | HuggingFace repo or local path to the router model. Empty string disables AI routing and uses keyword-only mode. |
| `categories_file` | string | Path to `experts.json`. Must stay within the project directory. |
| `confidence_threshold` | float | Minimum score (0-1) to accept a router prediction. Queries below this fall through to keyword fallback or return `null`. |
| `keyword_fallback` | bool | Enable keyword + fuzzy matching as secondary routing method. |
| `softmax_temperature` | float | Controls sharpness of softmax normalization. Lower = more decisive (try 0.10-0.20). |
| `scoring_weights` | object | Hybrid scoring signal weights. See [Scoring Weights](#scoring-weights-tuning). |
| `context_messages` | int | Number of recent user messages to concatenate for cascade routing when the last message alone has low confidence. Default: `3`. |
| `context_max_chars` | int | Maximum characters for concatenated context text, to respect the embedding model token window. Default: `1600`. |

**Recommended router models:**

| Language | Model |
|---|---|
| Spanish / Multilingual | `intfloat/multilingual-e5-small` |
| English | `BAAI/bge-small-en-v1.5` |

---

### 2. Experts (experts.json)

Defines the list of specialist models the router dispatches to.

```json
{
  "max_experts": 16,
  "experts": [
    {
      "id": 0,
      "label": "fallback",
      "description": "General fallback model used when the router cannot classify the request.",
      "type": "local",
      "format": "gguf",
      "model_path": "models/Qwen3.5-0.8B-Q4_K_M.gguf"
    },
    {
      "id": 1,
      "label": "programador",
      "description": "Expert in writing and reviewing code in any programming language",
      "keywords": ["codigo", "programar", "python", "javascript", "funcion",
                   "script", "error", "bug", "html", "css", "clase", "objeto",
                   "modulo", "api", "refactorizar"],
      "type": "ollama",
      "url": "http://127.0.0.1:11434",
      "model_name": "qwen2.5:7b"
    }
  ]
}
```

**IMPORTANT: The Fallback Expert**
The `fallback` expert (ID `0`) is mandatory and must not have any `keywords`. When the router's confidence falls below the `confidence_threshold`, or no keywords match, the system routes the request to this fallback expert. You can configure this fallback to use any backend (local GGUF, Ollama, API) just like any other expert.

**Keyword quality is critical.** The embedding router builds individual semantic vectors for each keyword and combines them using a 4-signal scoring formula. A minimum of **15 descriptive keywords per expert** is required for reliable routing. With fewer keywords the `mean_keyword` and `top3_vote` signals are too sparse and accuracy drops.

Write keywords as concrete, domain-specific terms the user would actually say — not abstract categories. For example, for a coding expert use `"funcion"`, `"clase"`, `"loop"`, `"variable"` rather than just `"programacion"`.

---

### 3. Expert Backends

Each expert entry in `experts.json` supports three `type` values:

**`ollama`** — Local or remote Ollama instance:
```json
{
  "type": "ollama",
  "url": "http://127.0.0.1:11434",
  "model_name": "qwen2.5:7b"
}
```

**`api`** — External API via LiteLLM (OpenAI, Anthropic, Gemini, etc.):
```json
{
  "type": "api",
  "provider": "openai",
  "model_name": "gpt-4o-mini",
  "api_key_env": "OPENAI_API_KEY"
}
```
The API key is read from the environment variable named in `api_key_env`. Never put keys directly in the JSON file.

**`local`** — Local ONNX or GGUF model:
```json
{
  "type": "local",
  "format": "onnx",
  "label": "my_model",
  "model_path": "models/my_model"
}
```

---

## Router Architecture

### Hybrid Scoring System

The default `embedding` router builds three data structures per expert at startup and combines four signals at inference time.

**At startup** (`_precompute_category_embeddings`):

```
Expert "programador"
  ├── kw_vecs  : [embed("codigo"), embed("python"), embed("script"), ...]  ← one vector per keyword
  ├── centroid : L2-normalised mean of all keyword vectors
  └── desc_vec : embed("Expert in writing and reviewing code...")
```

**At inference** (`_embed_score`):

The user query is encoded as `"query: <user text>"` and compared against each expert's data using four signals:

| Signal | Weight | What it captures |
|---|---|---|
| `max_keyword` | 40% | Maximum cosine similarity between the query and any single keyword. High when the user uses an exact domain term. |
| `description` | 30% | Cosine similarity between the query and the full expert description. Captures intent that keywords alone might miss. |
| `mean_keyword` | 20% | Average similarity across all keywords. Measures overall domain alignment. |
| `top3_vote` | 10% | Fraction of the top-3 keyword scores that exceed 0.40. Penalises lucky single-keyword matches and rewards consensus. |

After computing a raw hybrid score for every expert, the scores are passed through **softmax normalization** so the final score is a real probability (0-1) rather than a raw cosine value. This makes the `confidence_threshold` meaningful regardless of the embedding model used.

```
raw_scores: {programador: 0.61, sysadmin: 0.53, traductor: 0.41, ...}
    ↓ softmax(temperature=0.15)
norm_scores: {programador: 0.82, sysadmin: 0.14, traductor: 0.02, ...}
    → winner: "programador" at 0.82
```

### Scoring Weights Tuning

You can adjust weights in `config.json` without touching code:

```json
"scoring_weights": {
    "max_keyword":  0.40,
    "description":  0.30,
    "mean_keyword": 0.20,
    "top3_vote":    0.10
}
```

Tuning guidelines:
- **Raise `max_keyword`** if users tend to use exact technical terms.
- **Raise `description`** if expert descriptions are detailed and precise.
- **Lower `softmax_temperature`** (toward 0.05) to make the router more decisive when experts are clearly distinct.
- **Raise `softmax_temperature`** (toward 0.30) if experts overlap and you want smoother transitions.

### Fallback Chain

```
User query
    │
    ▼
Cascade Step 1: Evaluate last user message
    │ score < confidence_threshold
    ▼
Cascade Step 2: Evaluate last 2-3 user messages (concatenated)
    │ score < confidence_threshold
    ▼
Keyword + Fuzzy matching (rapidfuzz)
    │ score < threshold * 0.5
    ▼
Fallback expert (ID 0)
```

---

## API Usage

The server starts on `http://0.0.0.0:11435` by default.

**OpenAI-compatible endpoint:**
```bash
curl -X POST http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Escribe un script en Python para listar archivos"}],
    "stream": false
  }'
```

**Route directly to a specific expert** (skips the router):
```bash
curl -X POST http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "programador",
    "messages": [{"role": "user", "content": "Refactoriza esta funcion"}]
  }'
```

**Ollama-compatible endpoints:**
```bash
# List available models
curl http://localhost:11435/api/tags

# Chat
curl -X POST http://localhost:11435/api/chat \
  -d '{"model": "lemoe", "messages": [{"role": "user", "content": "Hola"}]}'
```

**Available routes:**
| Method | Path | Description |
|---|---|---|
| GET | `/` | Server info |
| GET | `/v1/models` | List models (OpenAI format) |
| POST | `/v1/chat/completions` | Inference (OpenAI format, streaming supported) |
| GET | `/api/tags` | List models (Ollama format) |
| POST | `/api/chat` | Inference (Ollama format, streaming supported) |
| GET | `/api/version` | Server version |

---

## Security

LEMoE includes hardened defaults for production-adjacent use:

| Protection | Details |
|---|---|
| **Path Traversal** | Model paths and labels are canonicalized and confined to the `models/` directory. Labels containing `/`, `\`, or `..` are rejected. |
| **SSRF Protection** | Ollama URLs are validated: only `http`/`https` schemes accepted. Cloud metadata endpoints (`169.254.0.0/16`) are blocked. Private/loopback IPs are allowed for internal deployments. |
| **Request Size Limit** | Incoming request bodies are capped at 1 MB (`MAX_CONTENT_LENGTH`). |
| **Rate Limiting** | Sliding-window rate limiter: 60 requests per minute per IP. Respects `X-Forwarded-For` when behind a reverse proxy. Returns HTTP 429 when exceeded. |
| **Log Sanitization** | User input is stripped of control characters and ANSI escape sequences before being written to log files. |
| **Config Path Validation** | `categories_file` paths in `config.json` are resolved and must stay within the project directory. |
| **Silent Error Isolation** | Internal exceptions from expert backends are never exposed to the end user. Error details are logged server-side only. The user receives a generic fallback response. |
| **Atomic File Writes** | Model stats are written atomically (write to `.tmp`, then `os.replace()`) to prevent corruption on crash. |

---


## Project Structure

```
LEMoE - Light Easy Mix Of Experts/
├── api_server.py          # Flask API server (OpenAI + Ollama compatible)
├── main.py                # CLI entry point (stdin loop)
├── setup.sh               # Interactive setup: Ollama check + model download
├── start.sh               # Starts the API server (creates venv if needed)
├── requirements.txt       # Python dependencies
├── config/
│   ├── config.json        # Router and server configuration
│   └── experts.json       # Expert definitions (labels, keywords, backends)
├── models/                # Local ONNX or GGUF model directories
├── data/                  # Runtime data (model usage stats)
├── logs/                  # Application logs
├── docker/                # Dockerfiles for various platforms
│   ├── Dockerfile         # Default Debian Slim (CPU)
│   ├── Dockerfile.debian  # Standard Debian (CPU)
│   └── Dockerfile.cuda    # Nvidia CUDA (GPU)
├── tests/
│   └── test_adversarial.py  # Adversarial test suite (42 tests)
└── modules/
    ├── generic_router.py  # Hybrid embedding/classification router
    ├── decision_router.py # Fine-tuned classifier router (model mode)
    ├── router_factory.py  # Router selection factory
    ├── expert_runner.py   # Expert dispatcher (API / Ollama / Local)
    ├── onnx_runner.py     # ONNX model runner with LRU + TTL memory management
    ├── ai_engine.py       # GGUF fallback engine (llama-cpp-python)
    ├── config_manager.py  # JSON config loader
    └── logger.py          # Shared application logger
```
