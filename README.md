# l3mcore – Light Easy Mix Of Experts

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

**l3mcore** is a lightweight, open-source Mixture of Experts (MoE) intelligent routing proxy. It acts as a drop-in replacement for OpenAI and Ollama APIs, routing each request to the most suitable LLM expert based on semantic understanding of the input.

## Features

- **OpenAI & Ollama compatible API** – Any client that speaks either protocol works out of the box
- **Real-time token streaming** – Low-latency token-to-token streaming via Server-Sent Events (SSE) and ndjson
- **vLLM & OpenAI-compatible support** – Connect local or remote vLLM, TGI, LocalAI, or Ollama instances natively
- **Semantic hybrid router** – Multi-signal scoring (embeddings + keywords + fuzzy matching) with softmax normalization
- **Multi-backend expert dispatch** – Route to Ollama, OpenAI, Anthropic, Gemini (via LiteLLM), local ONNX, or local GGUF models
- **Circuit Breaker & Silent self-correction** – Automatic failover and fast circuit tripping for degraded backends
- **Hot-reload** – Edit `experts.json` or `config.json` and changes apply without restart
- **Rate limiting** – Sliding-window per-IP rate limiter (configurable)
- **Security headers** – XSS protection, nosniff, DENY framing, no-referrer
- **Input sanitization** – Prompt injection and canary token detection
- **CORS support** – Configurable cross-origin resource sharing
- **Health endpoints** – System status and expert backend health probes

## Quick Start

```bash
# Clone and setup
git clone https://github.com/lemoelink/l3mcore.git
cd l3mcore
bash setup.sh

# Start the server
bash start.sh
```

The server starts on `http://0.0.0.0:11435` by default.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Server info |
| `GET` | `/v1/models` | List available experts (OpenAI format) |
| `GET` | `/v1/models/<model_id>` | Retrieve specific model details (OpenAI format) |
| `POST` | `/v1/chat/completions` | Chat inference (OpenAI format, real streaming supported) |
| `GET` | `/api/tags` | List models (Ollama format) |
| `POST` | `/api/chat` | Chat inference (Ollama format, real streaming supported) |
| `GET` | `/api/version` | Server version |
| `GET` | `/health` | System health status |
| `GET` | `/health/experts` | Expert backend health probe |

## Configuration

### `config/config.json`

Core settings for the router, rate limiting, health auth, and CORS.

### `config/experts.json`

Define your expert models. Each expert has:
- `label` – Unique identifier used for routing
- `description` – What the expert does (used by the semantic router)
- `keywords` – Keywords that trigger routing to this expert
- `type` – Backend type: `ollama`, `api`, or `local`
- `url` / `model_name` – Backend connection details

Example:
```json
{
  "max_experts": 4,
  "experts": [
    {
      "id": 0,
      "label": "fallback",
      "description": "General fallback model.",
      "type": "ollama",
      "url": "http://127.0.0.1:11434",
      "model_name": "qwen2.5:1.5b"
    },
    {
      "id": 1,
      "label": "programador_python",
      "description": "Python development expert.",
      "keywords": ["python", "django", "flask", "bug", "codigo"],
      "type": "ollama",
      "url": "http://127.0.0.1:11434",
      "model_name": "qwen2.5-coder:1.5b"
    },
    {
      "id": 2,
      "label": "experto_vllm",
      "description": "High throughput expert running on local or remote vLLM server.",
      "keywords": ["analisis", "complejo", "razonamiento", "investigacion"],
      "type": "api",
      "api_base": "http://localhost:8000/v1",
      "model_name": "meta-llama/Meta-Llama-3-8B-Instruct"
    }
  ]
}
```

## Docker Deployment

Official images are published to [Docker Hub (`lemoelink/l3mcore`)](https://hub.docker.com/r/lemoelink/l3mcore).

### Available Image Tags

| Image Tag | Base Image | Acceleration | Description |
|-----------|------------|--------------|-------------|
| `latest`, `cpu`, `1.0.0`, `1.0.0-cpu` | Debian Slim (`python:3.10-slim`) | CPU | **Default / Recommended.** Optimized for CPU inference (semantic routing model). Smallest footprint with maximum portability and compatibility across servers, desktops, and edge devices. |
| `debian` | Debian Bullseye (`python:3.10-bullseye`) | CPU | Built on standard Debian. Identical to the general CPU build but includes standard OS libraries, utilities, and debugging tools. |
| `cuda`, `1.0.0-cuda` | NVIDIA CUDA 12.1.1 (`ubuntu22.04`) | NVIDIA GPU (CUDA) | Built on NVIDIA CUDA runtime. Use this if you are running local GGUF/llama.cpp fallback models inside the container with GPU acceleration. Requires the `--gpus all` flag. |
| `rocm`, `1.0.0-rocm` | AMD ROCm 6.1.2 (`ubuntu22.04`) | AMD GPU (ROCm/HIP) | Enables GPU acceleration on supported AMD Radeon and Instinct hardware via HIP. Requires passing GPU devices (`--device=/dev/kfd --device=/dev/dri`). |
| `bundle`, `1.0.0-bundle` | Open WebUI + l3mcore | CPU | **Unified All-In-One Bundle.** Runs both `l3mcore` and **Open WebUI** in a single container. Pre-configured with internal loopback connections for instant plug-and-play chat. Exposes port `8080` (Web UI) and `11435` (API). |

### Running with Docker

#### 1. General CPU (Default)
```bash
docker run -d \
  --name l3mcore \
  -p 11435:11435 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  lemoelink/l3mcore:latest
```

#### 2. NVIDIA GPU (CUDA)
```bash
docker run -d \
  --name l3mcore-cuda \
  --gpus all \
  -p 11435:11435 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  lemoelink/l3mcore:cuda
```

#### 3. AMD ROCm
```bash
docker run -d \
  --name l3mcore-rocm \
  --device=/dev/kfd \
  --device=/dev/dri \
  -p 11435:11435 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  lemoelink/l3mcore:rocm
```

#### 4. Unified Pack (l3mcore + Open WebUI)
```bash
docker run -d \
  --name l3mcore-bundle \
  -p 3000:8080 \
  -p 11435:11435 \
  -v $(pwd)/config:/app/lemoe/config \
  -v open-webui-data:/app/backend/data \
  lemoelink/l3mcore:bundle
```
Access the Open WebUI interface at `http://localhost:3000`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LEMOE_HOST` | `0.0.0.0` | Bind address |
| `LEMOE_PORT` | `11435` | Bind port |
| `LEMOE_WORKERS` | `1` | Gunicorn workers |

## Project Structure

```
l3mcore/
├── api_server.py          # Flask API server (OpenAI + Ollama endpoints)
├── main.py                # CLI entry point
├── config/
│   ├── config.json        # Router and server configuration
│   └── experts.json       # Expert definitions
├── modules/
│   ├── ai_engine.py       # GGUF fallback engine
│   ├── config_manager.py  # Configuration loader
│   ├── decision_router.py # Decision routing logic
│   ├── expert_runner.py   # Expert dispatch to backends
│   ├── generic_router.py  # Semantic embedding router
│   ├── logger.py          # Logging
│   ├── onnx_runner.py     # ONNX model runner
│   ├── router_factory.py  # Router factory
│   ├── session_store.py   # Session/telemetry tracking
│   ├── utils_router.py    # Router utilities
│   └── utils_text.py      # Text sanitization
├── models/                # Local model files (ONNX/GGUF)
├── setup.sh               # Interactive setup script
├── start.sh               # Server startup script
└── requirements.txt       # Python dependencies
```

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).
