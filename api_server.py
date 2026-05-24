"""
LEMoE API Server

OpenAI- and Ollama-compatible HTTP API. Any client that speaks either protocol
can use LEMoE as a drop-in backend by pointing its base URL to this server.

Endpoints:
  GET  /                    -> Server info
  GET  /v1/models           -> List available experts (OpenAI format)
  POST /v1/chat/completions -> Inference (OpenAI format, streaming supported)
  GET  /api/tags            -> List models (Ollama format)
  POST /api/chat            -> Inference (Ollama format, streaming supported)
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
from modules.plugin_manager import PluginManager

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
        plugin_mgr = PluginManager()

        available = _load_available_models(config)
        expert_models = [m for m in available if m != DEFAULT_MODEL]
        app_logger.info(f"LEMoE Core ready. Models: {available}")
        return {
            "config": config,
            "router": router,
            "runner": runner,
            "ai_engine": ai_engine,
            "dispatcher": dispatcher,
            "plugin_mgr": plugin_mgr,
            "available_models": available,
            "expert_models": expert_models,
        }


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def _extract_routing_context(messages: list, max_messages: int = 3,
                              max_chars: int = 1600) -> dict:
    """
    Extracts routing context from the conversation history.

    Returns a dict with:
      - last_user_text:  text of the most recent user message (Step 1).
      - context_text:    concatenation of the last N user messages (Step 2).
    """
    if not messages or not isinstance(messages, list):
        return {"last_user_text": "", "context_text": ""}

    user_messages = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            text = str(content)
        text = text.strip()
        if text:
            user_messages.append(text)

    last_user_text = user_messages[-1] if user_messages else ""

    recent = user_messages[-max_messages:] if len(user_messages) > 1 else []
    context_text = " ".join(recent)
    if len(context_text) > max_chars:
        context_text = context_text[-max_chars:]

    if not context_text:
        context_text = last_user_text
    if len(context_text) > max_chars:
        context_text = context_text[-max_chars:]

    return {
        "last_user_text": last_user_text,
        "context_text": context_text,
    }


def _run_inference(messages: list, model_hint: str) -> tuple[str, str]:
    """
    Executes inference with cascading contextual routing.
    Returns: (response_text, used_model)
    """
    core = _Core.get()
    router        = core["router"]
    config        = core["config"]
    dispatcher    = core["dispatcher"]
    ai_engine     = core["ai_engine"]
    expert_models = core["expert_models"]
    plugin_mgr    = core["plugin_mgr"]

    router_cfg   = config.get('router', {})
    ctx_messages = router_cfg.get('context_messages', 3)
    ctx_chars    = router_cfg.get('context_max_chars', 1600)
    threshold    = router_cfg.get('confidence_threshold', 0.4)

    routing_ctx = _extract_routing_context(messages, ctx_messages, ctx_chars)

    override_label = plugin_mgr.hook_override_route(messages)

    last_text = plugin_mgr.hook_before_routing(routing_ctx["last_user_text"])
    ctx_text  = plugin_mgr.hook_before_routing(routing_ctx["context_text"])

    def _execute_expert(label: str) -> str:
        if hasattr(router, 'get_expert_config'):
            expert_config = router.get_expert_config(label)
            if expert_config:
                result = dispatcher.run(messages, expert_config)
                return plugin_mgr.hook_after_generation(result, label)
        cfg = {"type": "local", "format": "onnx", "label": label}
        result = dispatcher.run(messages, cfg)
        return plugin_mgr.hook_after_generation(result, label)

    def _do_fallback() -> tuple[str, str]:
        try:
            result = _execute_expert("fallback")
            return result, "fallback"
        except Exception as e:
            app_logger.error(f"Critical error in fallback engine: {e}")
            return "An internal error occurred. Please try again later.", "error"

    # Phase 0: Plugin forced route
    if override_label:
        try:
            app_logger.info(f"Plugin forced route to: {override_label}")
            result = _execute_expert(override_label)
            return result, override_label
        except Exception as e:
            app_logger.error(f"[Auto-Correction] Plugin forced route '{override_label}' failed: {e}. Redirecting to fallback.")
            return _do_fallback()

    # Phase 1: Client explicitly requests a known expert
    if model_hint and model_hint in expert_models:
        try:
            result = _execute_expert(model_hint)
            return result, model_hint
        except Exception as e:
            app_logger.error(f"[Auto-Correction] Explicit expert '{model_hint}' failed: {e}. Redirecting to fallback.")
            return _do_fallback()

    # Phase 2: Cascade Step 1 - evaluate last user message only
    label, score = router.predict(last_text)
    if label and label not in ('null', 'fallback') and score >= threshold:
        app_logger.info(f"[Cascade] Step 1: '{last_text[:60]}' -> {label} ({score:.2f})")
        try:
            result = _execute_expert(label)
            return result, label
        except Exception as e:
            app_logger.error(f"[Auto-Correction] Routed expert '{label}' failed: {e}. Redirecting to fallback.")
            return _do_fallback()
    else:
        app_logger.info(f"[Cascade] Step 1: score {score:.2f} below threshold ({threshold}). Escalating to Step 2.")

    # Phase 3: Cascade Step 2 - evaluate concatenated recent user messages
    if ctx_text != last_text:
        label, score = router.predict(ctx_text)
        if label and label not in ('null', 'fallback') and score >= threshold:
            app_logger.info(f"[Cascade] Step 2: '{ctx_text[:60]}' -> {label} ({score:.2f})")
            try:
                result = _execute_expert(label)
                return result, label
            except Exception as e:
                app_logger.error(f"[Auto-Correction] Routed expert '{label}' failed: {e}. Redirecting to fallback.")
                return _do_fallback()
        else:
            app_logger.info(f"[Cascade] Step 2: score {score:.2f} below threshold ({threshold}). Using fallback.")

    # Phase 4: Fallback
    app_logger.info("[Cascade] No expert matched. Using fallback.")
    return _do_fallback()


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB cap


@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response



_rl_lock   = threading.Lock()
_rl_counts: dict[str, list] = defaultdict(list)


def _get_client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _rate_limit_check(max_requests: int = 100, window_seconds: int = 60) -> bool:
    """Sliding-window rate limiter. Returns True if the request is allowed."""
    ip  = _get_client_ip()
    now = time.time()
    with _rl_lock:
        cutoff = now - window_seconds
        _rl_counts[ip] = [t for t in _rl_counts[ip] if t > cutoff]
        if len(_rl_counts[ip]) >= max_requests:
            return False
        _rl_counts[ip].append(now)

        # Purge stale IPs to prevent unbounded memory growth
        if len(_rl_counts) > 10000:
            stale = [k for k, v in _rl_counts.items() if not v]
            for k in stale:
                del _rl_counts[k]

        return True


_NON_PRINTABLE = re.compile(r'[\x00-\x1f\x7f]')


def _safe_log(text: str, max_len: int = 120) -> str:
    """Strip control characters and truncate before writing to logs."""
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
    messages    = body.get("messages") or []
    model_hint  = body.get("model", DEFAULT_MODEL)
    do_stream   = body.get("stream", False)

    routing_ctx = _extract_routing_context(messages)
    user_text = routing_ctx["last_user_text"]
    if not user_text:
        return jsonify({"error": {"message": "No user message found", "type": "invalid_request_error"}}), 400

    app_logger.info(f"[/v1/chat] model={model_hint!r} stream={do_stream} text={_safe_log(user_text)!r}")

    if do_stream:
        def generate():
            try:
                response_text, used_model = _run_inference(messages, model_hint)
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
            response_text, used_model = _run_inference(messages, model_hint)
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
    messages   = body.get("messages") or []
    model_hint = body.get("model", DEFAULT_MODEL)
    do_stream  = body.get("stream", True)  # Ollama defaults to stream=true

    routing_ctx = _extract_routing_context(messages)
    user_text = routing_ctx["last_user_text"]
    if not user_text:
        return jsonify({"error": "No user message found"}), 400

    app_logger.info(f"[/api/chat] model={model_hint!r} stream={do_stream} text={_safe_log(user_text)!r}")

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
                response_text, used_model = _run_inference(messages, model_hint)
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
            response_text, used_model = _run_inference(messages, model_hint)
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
# WSGI entrypoint (Gunicorn / uWSGI) and dev entrypoint
# ---------------------------------------------------------------------------

def _bootstrap():
    """Pre-load core + plugins once before the first request."""
    _Core.get()
    PluginManager()


# Gunicorn calls this module-level; bootstrap when the module is imported
_bootstrap()


def run(host: str = "0.0.0.0", port: int = 11435, debug: bool = False):
    """
    Dev-only entrypoint (Flask built-in server).
    In production use Gunicorn: gunicorn -w 1 -b 0.0.0.0:11435 api_server:app
    """
    app_logger.info(f"[DEV] LEMoE API listening on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LEMoE API Server (dev mode)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run(host=args.host, port=args.port, debug=args.debug)
