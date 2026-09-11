# l3mcore – Light Easy Mix Of Experts

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

**l3mcore** is a lightweight, open-source Mixture of Experts (MoE) intelligent routing proxy. It acts as a drop-in replacement for OpenAI and Ollama APIs, routing each request to the most suitable LLM expert based on semantic understanding of the input.

## Features

- **OpenAI & Ollama compatible API** – Any client that speaks either protocol works out of the box
- **Semantic hybrid router** – Multi-signal scoring (embeddings + keywords + fuzzy matching) with softmax normalization
- **Multi-backend expert dispatch** – Route to Ollama, OpenAI, Anthropic, Gemini (via LiteLLM), local ONNX, or local GGUF models
- **Silent self-correction** – Failed expert requests automatically redirect to the fallback model
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
| `POST` | `/v1/chat/completions` | Chat inference (OpenAI format, streaming supported) |
| `GET` | `/api/tags` | List models (Ollama format) |
| `POST` | `/api/chat` | Chat inference (Ollama format, streaming supported) |
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
    }
  ]
}
```

## Usage with OpenAI clients

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11435/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="l3mcore",
    messages=[{"role": "user", "content": "Write a Python function to sort a list"}]
)
print(response.choices[0].message.content)
```

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
