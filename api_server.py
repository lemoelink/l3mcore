"""
LEMoE API Server - OpenAI & Ollama compatible

Exposes an HTTP API so any client (Open WebUI, Continue, LiteLLM, curl)
can use LEMoE as a drop-in LLM backend.

Security:
  - Request body capped at 1 MB (MAX_CONTENT_LENGTH).
  - Sliding-window rate limiter: 60 req/min per IP (X-Forwarded-For aware).
  - User input sanitized before writing to logs.

Endpoints:
  GET  /                    -> Server info
  GET  /v1/models           -> List available experts (OpenAI format)
  POST /v1/chat/completions -> Inference (OpenAI format, streaming)
  GET  /api/tags            -> List models (Ollama format)
  POST /api/chat            -> Inference (Ollama format, streaming)
  GET  /api/version         -> Server version
"""

import json
import os
import re
import time
import uuid
import threading
from collections import defaultdict

# Ensure cwd is always the script directory
# (necessary for relative model paths to resolve correctly)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)

from flask import Flask, request, jsonify, Response, stream_with_context

from modules.logger import app_logger
from modules.config_manager import ConfigManager
from modules.router_factory import create_router
from modules.onnx_runner import SpecificModelRunner
from modules.ai_engine import AIEngine
from modules.expert_runner import ExpertDispatcher

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVER_VERSION = "0.1.0"
DEFAULT_MODEL  = "lemoe"


def _load_available_models(config_manager) -> list:
    """
    Builds the list of models announced in /v1/models and /api/tags.
    Generic mode: reads config/experts.json.
    Model mode: uses static list of known labels.
    """
    import json as _json
    cfg = config_manager.get('router', {})
    mode = cfg.get('mode', 'generic').lower()

    if mode == 'generic':
        cats_file = cfg.get('categories_file', 'config/experts.json')
        models = [DEFAULT_MODEL]
        if os.path.exists(cats_file):
            with open(cats_file, encoding='utf-8') as f:
                data = _json.load(f)
                experts = data.get('experts', [])
                for entry in experts:
                    label = entry.get('label', '').strip()
                    if label:
                        models.append(label)
        return models
    else:
        return [DEFAULT_MODEL, "malbec", "syrah", "pinot", "chardonnay", "grape-route"]


# ---------------------------------------------------------------------------
# Core MoE Initialization (singleton shared across requests)
# ---------------------------------------------------------------------------

class _Core:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls._init()
        return cls._instance

    @staticmethod
    def _init():
        app_logger.info("Initializing LEMoE Core...")
        config = ConfigManager()
        router = create_router(config)
        runner = SpecificModelRunner(
            models_base_path="models",
            stats_path="data/model_stats.json"
        )
        ai_engine = AIEngine()
        dispatcher = ExpertDispatcher(runner, ai_engine)
        
        # Dynamic list of available models
        available = _load_available_models(config)
        app_logger.info(f"LEMoE Core ready. Models: {available}")
        return {
            "config": config,
            "router": router,
            "runner": runner,
            "ai_engine": ai_engine,
            "dispatcher": dispatcher,
            "available_models": available,
        }


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def _run_inference(user_text: str, model_hint: str) -> tuple[str, str]:
    """
    Executes inference based on the requested model.
    Returns: (response_text, used_model)
    """
    core = _Core.get()
    router        = core["router"]
    dispatcher    = core["dispatcher"]
    ai_engine     = core["ai_engine"]
    available     = core["available_models"]
    expert_models = [m for m in available if m != DEFAULT_MODEL]

    # Helper to execute an expert knowing its label
    def _execute_expert(label: str) -> str:
        # 'generic' mode: use dispatcher with config json
        if hasattr(router, 'get_expert_config'):
            expert_config = router.get_expert_config(label)
            if expert_config:
                return dispatcher.run(user_text, expert_config)
        
        # Fallback 'model' mode (grape-route) or if no config exists
        # In this case, we assume local ONNX model
        cfg = {"type": "local", "format": "onnx", "label": label}
        return dispatcher.run(user_text, cfg)

    # 1. If client asks for a specific expert and it exists, use it directly
    if model_hint and model_hint in expert_models:
        try:
            result = _execute_expert(model_hint)
            return result, model_hint
        except Exception as e:
            app_logger.error(f"Error in explicit expert '{model_hint}': {e}")

    # 2. Semantic router
    label, score = router.predict(user_text)
    if label and label != 'null':
        try:
            result = _execute_expert(label)
            return result, label
        except Exception as e:
            app_logger.error(f"Error in engine for label '{label}': {e}")

    # 3. Generic GGUF fallback if everything fails or router yields 'null'
    app_logger.info("Using general AIEngine fallback")
    result = ai_engine.generate_response(user_text)
    return result, "ai_engine"


def _extract_user_text(messages: list) -> str:
    """Extracts content from the last 'user' role message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multimodal format: [{type: "text", text: "..."}]
                return " ".join(
                    part["text"] for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            return str(content)
    return ""


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
# SEC-3: Cap incoming request body to 1 MB to prevent memory exhaustion DoS
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024


# -- Rate limiter (SEC-4) ---------------------------------------------------
# Fixed-window in-memory limiter. Default: 60 requests per 60 seconds per IP.
# Configurable via config.json -> server.rate_limit_requests / rate_limit_window.

_rl_lock   = threading.Lock()
_rl_counts: dict[str, list] = defaultdict(list)  # ip -> [timestamp, ...]


def _get_client_ip() -> str:
    """Return client IP, honouring X-Forwarded-For if present."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        # Take the leftmost (original client) IP
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _rate_limit_check(max_requests: int = 60, window_seconds: int = 60) -> bool:
    """
    Returns True if the request should be allowed, False if rate-limited.
    Uses a sliding-window algorithm per client IP.
    """
    ip  = _get_client_ip()
    now = time.time()
    with _rl_lock:
        timestamps = _rl_counts[ip]
        # Prune old entries outside the window
        cutoff     = now - window_seconds
        _rl_counts[ip] = [t for t in timestamps if t > cutoff]
        if len(_rl_counts[ip]) >= max_requests:
            return False
        _rl_counts[ip].append(now)
        return True


# -- Log sanitization (SEC-7) -----------------------------------------------

_NON_PRINTABLE = re.compile(r'[\x00-\x1f\x7f]')


def _safe_log(text: str, max_len: int = 120) -> str:
    """Strip control characters and truncate user text before writing to logs."""
    cleaned = _NON_PRINTABLE.sub(' ', text)
    return cleaned[:max_len] if len(cleaned) > max_len else cleaned


# -- OpenAI format helpers --------------------------------------------------

def _openai_model_object(name: str) -> dict:
    return {
        "id": name,
        "object": "model",
        "created": 1700000000,
        "owned_by": "lemoe",
    }


def _openai_chat_chunk(content: str, model: str, finish_reason=None) -> str:
    """Creates an SSE chunk in OpenAI streaming format."""
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _openai_chat_response(content: str, model: str) -> dict:
    """Full response (no streaming) in OpenAI format."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1,
        },
    }


# -- OpenAI endpoints -------------------------------------------------------

@app.route("/v1/models", methods=["GET"])
def list_models_openai():
    available = _Core.get()["available_models"]
    return jsonify({
        "object": "list",
        "data": [_openai_model_object(m) for m in available],
    })


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    if not _rate_limit_check():
        return jsonify({"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}), 429

    body = request.get_json(force=True, silent=True) or {}
    messages    = body.get("messages", [])
    model_hint  = body.get("model", DEFAULT_MODEL)
    do_stream   = body.get("stream", False)

    user_text = _extract_user_text(messages)
    if not user_text:
        return jsonify({"error": {"message": "No user message found", "type": "invalid_request_error"}}), 400

    # SEC-7: sanitize user text before logging
    app_logger.info(f"[API] /v1/chat/completions model='{model_hint}' stream={do_stream} text='{_safe_log(user_text)}'")

    if do_stream:
        def generate():
            try:
                response_text, used_model = _run_inference(user_text, model_hint)
                # Emit content in a single chunk (ONNX experts don't do real streaming)
                yield _openai_chat_chunk(response_text, used_model)
                yield _openai_chat_chunk("", used_model, finish_reason="stop")
                yield "data: [DONE]\n\n"
            except Exception as e:
                app_logger.error(f"Streaming error: {e}")
                err = {"error": {"message": str(e), "type": "server_error"}}
                yield f"data: {json.dumps(err)}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        try:
            response_text, used_model = _run_inference(user_text, model_hint)
            return jsonify(_openai_chat_response(response_text, used_model))
        except Exception as e:
            app_logger.error(f"Inference error: {e}")
            return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500


# -- Ollama-compatible endpoints --------------------------------------------

@app.route("/api/version", methods=["GET"])
def ollama_version():
    return jsonify({"version": SERVER_VERSION})


@app.route("/api/tags", methods=["GET"])
def ollama_tags():
    """Ollama /api/tags - lists models in Ollama format."""
    available = _Core.get()["available_models"]
    expert_set = set(available) - {DEFAULT_MODEL}
    models = []
    for name in available:
        models.append({
            "name": name,
            "model": name,
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "",
            "details": {
                "parent_model": "",
                "format": "onnx" if name in expert_set else "mixed",
                "family": "lemoe",
                "families": ["lemoe"],
                "parameter_size": "unknown",
                "quantization_level": "Q4",
            }
        })
    return jsonify({"models": models})


@app.route("/api/chat", methods=["POST"])
def ollama_chat():
    """
    Ollama POST /api/chat
    Body: { model, messages: [{role, content}], stream }
    """
    if not _rate_limit_check():
        return jsonify({"error": "Rate limit exceeded"}), 429

    body = request.get_json(force=True, silent=True) or {}
    messages   = body.get("messages", [])
    model_hint = body.get("model", DEFAULT_MODEL)
    do_stream  = body.get("stream", True)  # Ollama defaults to stream=true

    user_text = _extract_user_text(messages)
    if not user_text:
        return jsonify({"error": "No user message found"}), 400

    # SEC-7: sanitize user text before logging
    app_logger.info(f"[API] /api/chat model='{model_hint}' stream={do_stream} text='{_safe_log(user_text)}'")

    def _ollama_chunk(content: str, model: str, done: bool) -> str:
        obj = {
            "model": model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": content},
            "done": done,
        }
        if done:
            obj.update({
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "eval_count": 0,
                "eval_duration": 0,
            })
        return json.dumps(obj) + "\n"

    if do_stream:
        def generate():
            try:
                response_text, used_model = _run_inference(user_text, model_hint)
                yield _ollama_chunk(response_text, used_model, done=False)
                yield _ollama_chunk("", used_model, done=True)
            except Exception as e:
                app_logger.error(f"Error in /api/chat streaming: {e}")
                yield json.dumps({"error": str(e)}) + "\n"

        return Response(
            stream_with_context(generate()),
            mimetype="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        try:
            response_text, used_model = _run_inference(user_text, model_hint)
            return Response(
                _ollama_chunk(response_text, used_model, done=True),
                mimetype="application/json",
            )
        except Exception as e:
            app_logger.error(f"Error in /api/chat: {e}")
            return jsonify({"error": str(e)}), 500


# -- Additional compatibility routes ----------------------------------------

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "name": "LEMoE",
        "version": SERVER_VERSION,
        "description": "Light Easy Mix Of Experts – OpenAI & Ollama compatible API",
        "endpoints": ["/v1/models", "/v1/chat/completions", "/api/tags", "/api/chat", "/api/version"],
    })


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run(host: str = "0.0.0.0", port: int = 11435, debug: bool = False):
    """
    Starts the API server.
    Default port 11435 (Ollama uses 11434 - LEMoE avoids collision).
    To use as a drop-in replacement for Ollama start with --port 11434.
    """
    # Pre-load core before accepting requests
    _Core.get()
    app_logger.info(f"LEMoE API listening on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LEMoE API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run(host=args.host, port=args.port, debug=args.debug)
