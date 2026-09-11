"""
l3mcore API Server

OpenAI- and Ollama-compatible HTTP API. Any client that speaks either protocol
can use l3mcore as a drop-in backend by pointing its base URL to this server.

Endpoints:
  GET  /                    -> Server info
  GET  /v1/models           -> List available experts (OpenAI format)
  POST /v1/chat/completions -> Inference (OpenAI format, streaming supported)
  GET  /api/tags            -> List models (Ollama format)
  POST /api/chat            -> Inference (Ollama format, streaming supported)
  GET  /api/version         -> Server version
  GET  /health              -> System health status
  GET  /health/experts      -> Expert backend health probe
"""

import json
import os
import re
import time
import uuid
import threading
from collections import deque

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
# Rate Limiter (sliding window per IP)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Simple sliding-window rate limiter per IP address."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self._max_requests = max_requests
        self._window = window_seconds
        self._requests: dict[str, deque] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # seconds

    def _cleanup_stale(self):
        """Remove IPs with no recent activity to prevent unbounded memory growth."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        stale_ips = [
            ip for ip, timestamps in self._requests.items()
            if not timestamps or timestamps[-1] < now - self._window * 2
        ]
        for ip in stale_ips:
            del self._requests[ip]

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        with self._lock:
            self._cleanup_stale()
            if client_ip not in self._requests:
                self._requests[client_ip] = deque()
            timestamps = self._requests[client_ip]
            while timestamps and timestamps[0] < now - self._window:
                timestamps.popleft()
            if len(timestamps) >= self._max_requests:
                return False
            timestamps.append(now)
            return True

_rate_limiter = _RateLimiter()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVER_VERSION = "1.0.0"
DEFAULT_MODEL  = "l3mcore"


def _load_available_models(config_manager) -> list:
    """
    Builds the list of models announced in /v1/models and /api/tags.
    Reads config/experts.json.
    """
    cfg = config_manager.get('router', {})
    cats_file = cfg.get('categories_file', 'config/experts.json')
    models = [DEFAULT_MODEL]
    if os.path.exists(cats_file):
        with open(cats_file, encoding='utf-8') as f:
            data = json.load(f)
        from modules.config_manager import deobfuscate_value
        data = deobfuscate_value(data)
        experts = data.get('experts', [])
        for entry in experts:
            label = entry.get('label', '').strip()
            if label:
                models.append(label)
    return models


def _validate_api_keys(config_manager) -> None:
    """Logs warnings for API-type experts missing their configured API keys."""
    try:
        cats_file = config_manager.get('router', {}).get('categories_file', 'config/experts.json')
        if not os.path.exists(cats_file):
            return
        with open(cats_file, encoding='utf-8') as f:
            data = json.load(f)
        from modules.config_manager import deobfuscate_value
        data = deobfuscate_value(data)
        for expert in data.get('experts', []):
            if expert.get('type', '').lower() == 'api':
                env_var = expert.get('api_key_env', '')
                if env_var and not os.environ.get(env_var):
                    label = expert.get('label', 'unknown')
                    app_logger.warning(
                        f"API expert '{label}': environment variable '{env_var}' is not set. "
                        f"Inference will fail until the key is configured."
                    )
    except Exception as e:
        app_logger.debug(f"_validate_api_keys: {e}")


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

    @classmethod
    def reload_experts(cls):
        instance = cls.get()
        with cls._lock:
            app_logger.info("l3mcore Core: Hot-reloading expert configurations...")
            try:
                router = instance.get("router")
                if hasattr(router, "reload_categories"):
                    router.reload_categories()

                config = instance.get("config")
                available = _load_available_models(config)

                instance["available_models"] = available
                instance["expert_models"] = [m for m in available if m != DEFAULT_MODEL]

                app_logger.info(f"l3mcore Core: Hot-reload complete. Routable models: {available}")
            except Exception as e:
                app_logger.error(f"l3mcore Core: Error during hot-reload: {e}")

    @staticmethod
    def _init():
        app_logger.info("Initializing l3mcore Core...")
        config = ConfigManager()
        router = create_router(config)
        runner = SpecificModelRunner(
            models_base_path="models",
            stats_path="data/model_stats.json"
        )
        ai_engine = AIEngine(config_manager=config)
        dispatcher = ExpertDispatcher(runner, ai_engine, config_manager=config)

        _validate_api_keys(config)

        available = _load_available_models(config)
        expert_models = [m for m in available if m != DEFAULT_MODEL]
        app_logger.info(f"l3mcore Core ready. Models: {available}")
        return {
            "config": config,
            "router": router,
            "runner": runner,
            "ai_engine": ai_engine,
            "dispatcher": dispatcher,
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


def _clean_assistant_response(text: str) -> str:
    """
    Cleans up the assistant's final response to remove raw JSON tool calls
    (e.g., {"name": "...", "parameters": {...}}).
    """
    if not text:
        return text

    i = 0
    n = len(text)
    ranges_to_remove = []
    while i < n:
        if text[i] == '{':
            depth = 1
            j = i + 1
            in_string = False
            escaped = False
            while j < n and depth > 0:
                char = text[j]
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    in_string = not in_string
                elif not in_string:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                j += 1
            if depth == 0:
                candidate = text[i:j]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict) and "name" in obj and ("parameters" in obj or "arguments" in obj or "parameter" in obj):
                        ranges_to_remove.append((i, j))
                        i = j - 1
                except Exception:
                    pass
        i += 1

    new_text = text
    for start, end in reversed(ranges_to_remove):
        prefix = new_text[:start]
        suffix = new_text[end:]
        prefix = prefix.rstrip()
        suffix = suffix.lstrip()
        if prefix and suffix:
            new_text = prefix + "\n\n" + suffix
        else:
            new_text = prefix + suffix

    return new_text.strip()


def _resolve_expert_and_config(messages: list, model_hint: str) -> tuple[str, dict]:
    """
    Evaluates router cascade to select the best expert label and its configuration.
    Integrates Circuit Breaker check to skip failing backends immediately.
    """
    core = _Core.get()
    router = core["router"]
    config = core["config"]
    dispatcher = core["dispatcher"]

    router_cfg = config.get('router', {})
    ctx_messages = router_cfg.get('context_messages', 3)
    ctx_chars = router_cfg.get('context_max_chars', 1600)
    threshold = router_cfg.get('confidence_threshold', 0.4)

    routing_ctx = _extract_routing_context(messages, ctx_messages, ctx_chars)
    last_text = routing_ctx["last_user_text"]

    # 1. Direct model hint if specified and not default
    if model_hint and model_hint != DEFAULT_MODEL:
        if hasattr(router, 'get_expert_config'):
            cfg = router.get_expert_config(model_hint)
            if cfg and dispatcher.circuit_breaker.is_available(model_hint):
                return model_hint, cfg

    # 2. Regex triggers matching
    if last_text:
        try:
            if hasattr(router, 'categories') and router.categories:
                for label, cat_data in router.categories.items():
                    cfg = cat_data.get('config', {})
                    triggers = cfg.get("regex_triggers", [])
                    if not isinstance(triggers, list):
                        continue
                    for pattern in triggers:
                        if isinstance(pattern, str) and pattern:
                            if re.search(pattern, last_text, re.IGNORECASE):
                                if dispatcher.circuit_breaker.is_available(label):
                                    app_logger.info(f"[Regex] Matched '{pattern}' -> '{label}'")
                                    return label, cfg
        except Exception as e:
            app_logger.error(f"[Regex] Error evaluating triggers: {e}")

    # 3. Semantic / embedding router
    label, score = router.predict(last_text)
    if label and label not in ('null', 'fallback') and score >= threshold:
        if dispatcher.circuit_breaker.is_available(label):
            app_logger.info(f"[Router] '{last_text[:60]}' -> {label} ({score:.2f})")
            if hasattr(router, 'get_expert_config'):
                cfg = router.get_expert_config(label)
                if cfg:
                    return label, cfg
            return label, {"type": "local", "format": "onnx", "label": label}
        else:
            app_logger.warning(f"[CircuitBreaker] Expert '{label}' circuit is open. Routing to fallback.")
    else:
        app_logger.info(f"[Router] Score {score:.2f} below threshold ({threshold}). Using fallback.")

    # 4. Fallback expert
    if hasattr(router, 'get_expert_config'):
        fb_cfg = router.get_expert_config("fallback")
        if fb_cfg:
            return "fallback", fb_cfg
    return "fallback", {"type": "local", "format": "gguf", "label": "fallback"}


def _run_inference(messages: list, model_hint: str, gen_params: dict | None = None) -> tuple[str, str]:
    """
    Non-streaming inference execution with security sanitization and telemetry.
    """
    core = _Core.get()
    dispatcher = core["dispatcher"]
    label, cfg = _resolve_expert_and_config(messages, model_hint)

    routing_ctx = _extract_routing_context(messages)
    last_text = routing_ctx["last_user_text"]

    # 1. Security interceptor
    try:
        from modules.utils_text import sanitize
        intercepted = sanitize(last_text)
        if intercepted is not None:
            return intercepted, "canary_interceptor"
    except Exception as e:
        app_logger.warning(f"Security interceptor failed: {e}")

    t0 = time.monotonic()

    # 2. Dispatch with fallback
    try:
        res_text = dispatcher.run(messages, cfg, gen_params)
        used_lbl = label
    except Exception as e:
        app_logger.error(f"Execution failed on '{label}': {e}. Triggering fallback.")
        try:
            fb_cfg = core["router"].get_expert_config("fallback") if hasattr(core["router"], 'get_expert_config') else None
            fb_cfg = fb_cfg or {"type": "local", "format": "gguf", "label": "fallback"}
            res_text = dispatcher.run(messages, fb_cfg, gen_params)
            used_lbl = "fallback"
        except Exception as fb_err:
            app_logger.error(f"Fallback also failed: {fb_err}")
            return "An internal error occurred. Please try again later.", "error"

    res_text = _clean_assistant_response(res_text)

    # 3. Telemetry tracking
    duration = time.monotonic() - t0
    try:
        from modules.session_store import push_context
        m_type = cfg.get("type", "local")
        if m_type == "local":
            m_type = f"local-{cfg.get('format', 'onnx')}"
        push_context(used_lbl, m_type, last_text, res_text, duration)
    except Exception as te:
        app_logger.warning(f"Telemetry tracking failed: {te}")

    return res_text, used_lbl


def _run_inference_stream(messages: list, model_hint: str, gen_params: dict | None = None):
    """
    Yields chunks token-to-token in real time.
    Returns generator of chunks and model label.
    """
    core = _Core.get()
    dispatcher = core["dispatcher"]
    label, cfg = _resolve_expert_and_config(messages, model_hint)

    routing_ctx = _extract_routing_context(messages)
    last_text = routing_ctx["last_user_text"]

    # 1. Security interceptor
    try:
        from modules.utils_text import sanitize
        intercepted = sanitize(last_text)
        if intercepted is not None:
            def _interceptor_stream():
                yield intercepted
            return _interceptor_stream(), "canary_interceptor"
    except Exception as e:
        app_logger.warning(f"Security interceptor failed: {e}")

    # 2. Return real streaming generator with fallback
    try:
        stream_gen = dispatcher.run_stream(messages, cfg, gen_params)
        return stream_gen, label
    except Exception as e:
        app_logger.error(f"Failed to start stream on '{label}': {e}. Using fallback stream.")
        fb_cfg = core["router"].get_expert_config("fallback") if hasattr(core["router"], 'get_expert_config') else None
        fb_cfg = fb_cfg or {"type": "local", "format": "gguf", "label": "fallback"}
        stream_gen = dispatcher.run_stream(messages, fb_cfg, gen_params)
        return stream_gen, "fallback"


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
    response.headers["X-RateLimit-Limit"] = str(_rate_limiter._max_requests)
    response.headers["X-RateLimit-Window"] = str(_rate_limiter._window)

    try:
        core = _Core.get()
        cors_cfg = core["config"].get("cors", {})
        if cors_cfg.get("enabled", False):
            origin = cors_cfg.get("origin", "*")
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            response.headers["Access-Control-Max-Age"] = "86400"
    except Exception:
        pass

    return response


@app.before_request
def check_rate_limit():
    client_ip = request.remote_addr or "unknown"
    if not _rate_limiter.is_allowed(client_ip):
        return jsonify({
            "error": {
                "message": "Rate limit exceeded. Please slow down.",
                "type": "rate_limit_error"
            }
        }), 429


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
        "owned_by": "l3mcore",
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
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
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


@app.route("/v1/models/<path:model_id>", methods=["GET"])
def get_model_openai(model_id):
    """Retrieves a specific model instance (OpenAI standard)."""
    available = _Core.get()["available_models"]
    if model_id in available:
        return jsonify(_openai_model_object(model_id))
    return jsonify({
        "error": {
            "message": f"The model '{model_id}' does not exist",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found"
        }
    }), 404


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    content_type = request.content_type or ""
    if "application/json" not in content_type and "multipart/form-data" not in content_type:
        return jsonify({
            "error": {"message": "Content-Type must be application/json", "type": "invalid_request_error"}
        }), 415

    body = request.get_json(force=True, silent=True) or {}
    messages    = body.get("messages") or []
    model_hint  = body.get("model", DEFAULT_MODEL)
    do_stream   = body.get("stream", False)

    # Extract standard LLM generation parameters
    gen_params = {}
    for param in ("temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "presence_penalty", "frequency_penalty"):
        if param in body and body[param] is not None:
            gen_params[param] = body[param]
    if "max_completion_tokens" in gen_params and "max_tokens" not in gen_params:
        gen_params["max_tokens"] = gen_params.pop("max_completion_tokens")

    routing_ctx = _extract_routing_context(messages)
    user_text = routing_ctx["last_user_text"]
    if not user_text:
        return jsonify({"error": {"message": "No user message found", "type": "invalid_request_error"}}), 400

    app_logger.info(f"[/v1/chat] model={model_hint!r} stream={do_stream} text={_safe_log(user_text)!r}")

    if do_stream:
        def generate():
            try:
                stream_gen, used_model = _run_inference_stream(messages, model_hint, gen_params)
                for chunk in stream_gen:
                    if chunk:
                        yield _openai_chat_chunk(chunk, used_model)
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
            response_text, used_model = _run_inference(messages, model_hint, gen_params)
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
                "family": "l3mcore",
                "families": ["l3mcore"],
                "parameter_size": "unknown",
                "quantization_level": "Q4",
            }
        })
    return jsonify({"models": models})


@app.route("/api/chat", methods=["POST"])
def ollama_chat():
    """
    Ollama POST /api/chat
    Body: { model, messages: [{role, content}], stream, options }
    """
    content_type = request.content_type or ""
    if "application/json" not in content_type:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    body = request.get_json(force=True, silent=True) or {}
    messages   = body.get("messages") or []
    model_hint = body.get("model", DEFAULT_MODEL)
    do_stream  = body.get("stream", True)  # Ollama defaults to stream=true

    # Extract options and generation params
    gen_params = {}
    options = body.get("options") or {}
    for param in ("temperature", "top_p", "stop"):
        if param in options and options[param] is not None:
            gen_params[param] = options[param]
        elif param in body and body[param] is not None:
            gen_params[param] = body[param]
    if "num_predict" in options and options["num_predict"] is not None:
        gen_params["max_tokens"] = options["num_predict"]
    elif "max_tokens" in body and body["max_tokens"] is not None:
        gen_params["max_tokens"] = body["max_tokens"]

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
                stream_gen, used_model = _run_inference_stream(messages, model_hint, gen_params)
                for chunk in stream_gen:
                    if chunk:
                        yield _ollama_chunk(chunk, used_model, done=False)
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
            response_text, used_model = _run_inference(messages, model_hint, gen_params)
            return Response(
                _ollama_chunk(response_text, used_model, done=True),
                mimetype="application/json",
            )
        except Exception as e:
            app_logger.error(f"Error in /api/chat: {e}")
            return jsonify({"error": str(e)}), 500


# -- Info and health endpoints -----------------------------------------------

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "name": "l3mcore",
        "version": SERVER_VERSION,
        "description": "Light Easy Mix Of Experts – OpenAI & Ollama compatible API",
        "endpoints": ["/v1/models", "/v1/chat/completions", "/api/tags", "/api/chat", "/api/version", "/health"],
    })


@app.route("/health", methods=["GET"])
def health():
    """Returns the operational status of every core component."""
    core = _Core.get()
    config = core["config"]

    health_cfg = config.get("health", {})
    if health_cfg.get("auth_required", False):
        auth_header = request.headers.get("Authorization", "")
        expected_token = health_cfg.get("auth_token", "")
        if expected_token and auth_header != f"Bearer {expected_token}":
            return jsonify({"error": {"message": "Unauthorized", "type": "auth_error"}}), 401

    router = core["router"]
    runner = core["runner"]
    ai_engine = core["ai_engine"]

    router_status = "ok"
    router_mode = getattr(router, 'router_type', 'model')
    router_enabled = getattr(router, 'enabled', False)
    if not router_enabled:
        router_status = "degraded (keyword fallback only)"

    ai_ready = getattr(ai_engine, 'is_ready', False)
    models_in_memory = list(getattr(runner, 'sessions', {}).keys())

    config.check_for_changes()

    status = {
        "status": "ok",
        "version": SERVER_VERSION,
        "router": {
            "mode": router_mode,
            "status": router_status,
            "cache_size": len(getattr(router, '_predict_cache', {})),
        },
        "onnx_runner": {
            "models_in_memory": models_in_memory,
            "max_models": getattr(runner, 'max_models', 3),
        },
        "ai_engine": {
            "model": getattr(ai_engine, 'model_path', 'unknown'),
            "loaded": ai_ready,
        },
        "available_models": core["available_models"],
    }
    return jsonify(status)


@app.route("/v1/health/experts", methods=["GET"])
@app.route("/health/experts", methods=["GET"])
def health_experts():
    """
    Checks the connectivity and status of all configured experts.
    Returns a JSON report showing which backends are active/reachable.
    """
    core = _Core.get()
    router_cfg_file = core["config"].get("router", {}).get("categories_file", "config/experts.json")

    try:
        with open(router_cfg_file, encoding="utf-8") as f:
            data = json.load(f)
        from modules.config_manager import deobfuscate_value
        data = deobfuscate_value(data)
        experts_list = data.get("experts", [])
    except Exception as e:
        return jsonify({"status": "error", "message": f"Could not read experts file: {e}"}), 500

    results = {}

    import urllib.request as _ur

    for exp in experts_list:
        label = exp.get("label")
        if not label:
            continue

        expert_type = exp.get("type", "local").lower()
        status = "unknown"
        details = ""

        if expert_type == "ollama":
            url = exp.get("url", "http://127.0.0.1:11434").rstrip("/")
            try:
                req = _ur.Request(f"{url}/api/tags")
                with _ur.urlopen(req, timeout=2.0) as resp:
                    if resp.status == 200:
                        status = "reachable"
                        details = f"Ollama responds on {url}"
                    else:
                        status = "unreachable"
                        details = f"HTTP status {resp.status}"
            except Exception as e:
                status = "unreachable"
                details = str(e)

        elif expert_type == "api":
            env_var = exp.get("api_key_env", "")
            if env_var:
                if os.environ.get(env_var):
                    status = "configured"
                    details = f"API key set in environment ({env_var})"
                else:
                    status = "missing_key"
                    details = f"Environment variable '{env_var}' is not set"
            else:
                status = "configured"
                details = "No specific API key environment variable required"

        elif expert_type == "local":
            model_format = exp.get("format", "onnx").lower()
            model_path = exp.get("model_path", "")
            if model_path and os.path.exists(model_path):
                status = "ready"
                details = f"Local {model_format.upper()} model path exists ({model_path})"
            elif model_path:
                status = "missing_model"
                details = f"Model path not found ({model_path})"
            else:
                status = "configured"
                details = f"Local expert with format: {model_format}"

        results[label] = {
            "type": expert_type,
            "status": status,
            "details": details
        }

    return jsonify({
        "status": "ok",
        "experts": results
    })


# ---------------------------------------------------------------------------
# WSGI entrypoint (Gunicorn / uWSGI) and dev entrypoint
# ---------------------------------------------------------------------------

def _start_experts_watcher():
    """Starts background thread to watch experts.json and config.json changes."""
    def watch():
        exp_path = "config/experts.json"
        cfg_path = "config/config.json"

        last_exp_mtime = os.path.getmtime(exp_path) if os.path.exists(exp_path) else 0.0
        last_cfg_mtime = os.path.getmtime(cfg_path) if os.path.exists(cfg_path) else 0.0

        while True:
            time.sleep(2)
            try:
                if os.path.exists(exp_path):
                    current_exp_mtime = os.path.getmtime(exp_path)
                    if current_exp_mtime > last_exp_mtime:
                        last_exp_mtime = current_exp_mtime
                        app_logger.info("Watcher: experts.json modification detected. Hot-reloading experts...")
                        _Core.reload_experts()

                if os.path.exists(cfg_path):
                    current_cfg_mtime = os.path.getmtime(cfg_path)
                    if current_cfg_mtime > last_cfg_mtime:
                        last_cfg_mtime = current_cfg_mtime
                        app_logger.info("Watcher: config.json modification detected. Hot-reloading configuration...")
                        ConfigManager().load()
                        _Core.reload_experts()
            except Exception as e:
                app_logger.error(f"Watcher error: {e}")

    watcher_thread = threading.Thread(target=watch, daemon=True)
    watcher_thread.start()
    app_logger.info("Watcher: Started background thread for configuration and experts monitoring.")


def _print_startup_summary(core: dict) -> None:
    """Prints a summary of the loaded configuration at startup."""
    config = core["config"]
    router = core["router"]
    available = core["available_models"]

    router_cfg = config.get('router', {})
    rl_cfg = config.get('rate_limiting', {})

    app_logger.info("=" * 50)
    app_logger.info("l3mcore v1.0.0 - Startup Summary")
    app_logger.info("=" * 50)
    app_logger.info(f"  Router mode:     {router_cfg.get('mode', 'generic')}")
    app_logger.info(f"  Router type:     {router_cfg.get('router_type', 'embedding')}")
    app_logger.info(f"  Confidence:      {router_cfg.get('confidence_threshold', 0.4)}")
    app_logger.info(f"  Experts loaded:  {len(available) - 1}")
    app_logger.info(f"  Models:          {', '.join(available)}")
    rl_status = "enabled" if rl_cfg.get('enabled', True) else "disabled"
    app_logger.info(f"  Rate limiting:   {rl_status} ({rl_cfg.get('max_requests', 60)}/{rl_cfg.get('window_seconds', 60)}s)")
    app_logger.info("=" * 50)


def _bootstrap():
    """Pre-load core once before the first request."""
    core = _Core.get()
    _start_experts_watcher()
    _print_startup_summary(core)


# Gunicorn calls this module-level; bootstrap when the module is imported
_bootstrap()


def run(host: str = "0.0.0.0", port: int = 11435, debug: bool = False):
    """
    Dev-only entrypoint (Flask built-in server).
    In production use Gunicorn: gunicorn -w 1 -b 0.0.0.0:11435 api_server:app
    """
    app_logger.info(f"[DEV] l3mcore API listening on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="l3mcore API Server (dev mode)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run(host=args.host, port=args.port, debug=args.debug)
