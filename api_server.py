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
DEFAULT_MODEL  = "l3mcore"


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

    @classmethod
    def reload_experts(cls):
        instance = cls.get()
        with cls._lock:
            app_logger.info("l3mcore Core: Recargando configuraciones de expertos en el aire...")
            try:
                # 1. Recargar GenericRouter
                router = instance.get("router")
                if hasattr(router, "reload_categories"):
                    router.reload_categories()
                
                # 2. Recargar modelos disponibles en api_server
                config = instance.get("config")
                available = _load_available_models(config)
                
                instance["available_models"] = available
                instance["expert_models"] = [m for m in available if m != DEFAULT_MODEL]
                
                app_logger.info(f"l3mcore Core: Recarga automática finalizada. Modelos enrutables: {available}")
                _start_keyword_enrichment()
            except Exception as e:
                app_logger.error(f"l3mcore Core: Error durante la recarga en caliente: {e}")

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
        plugin_mgr = PluginManager()

        available = _load_available_models(config)
        expert_models = [m for m in available if m != DEFAULT_MODEL]
        app_logger.info(f"l3mcore Core ready. Models: {available}")
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



def _notify_transparency(label: str, score: float) -> None:
    """
    Pushes the resolved label+score into the routing_transparency plugin
    if it is loaded, so the footer can display the confidence value.
    This is a best-effort call; any failure is silently ignored.
    """
    try:
        import sys
        plugin_module = sys.modules.get("l3mcore_plugin.routing_transparency")
        if plugin_module and hasattr(plugin_module, "set_route_score"):
            plugin_module.set_route_score(label, score)
    except Exception:
        pass


def _fire_failure_webhook(failed_label: str, reason: str) -> None:
    """
    Sends a POST notification to the webhook URL configured under
    config.json > expert_runner.failure_webhook_url when an expert
    fails and the system falls back automatically.

    The payload is a small JSON object:
      {"event": "expert_failure", "expert": "<label>", "reason": "<msg>"}

    The call is fire-and-forget (best-effort): any error is logged and
    swallowed so it never interrupts the fallback chain.
    """
    try:
        import urllib.request as _ur
        import json as _json
        core = _Core.get()
        cfg  = core["config"].get("expert_runner", {})
        url  = cfg.get("failure_webhook_url", "")
        if not url or not isinstance(url, str):
            return
        # Minimal scheme check — no SSRF protection needed here because
        # the operator sets this value themselves in config.json.
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return
        payload = _json.dumps({
            "event":  "expert_failure",
            "expert": str(failed_label)[:64],
            "reason": str(reason)[:256],
        }).encode("utf-8")
        req = _ur.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with _ur.urlopen(req, timeout=3):
            pass
    except Exception as exc:
        app_logger.warning(f"[Webhook] Failed to notify failure for '{failed_label}': {exc}")


def _run_inference(messages: list, model_hint: str) -> tuple[str, str]:
    """
    Wrapper for _run_inference_impl that checks security interceptor
    and records local telemetry.
    """
    core = _Core.get()
    config = core["config"]
    router_cfg = config.get('router', {})
    ctx_messages = router_cfg.get('context_messages', 3)
    ctx_chars = router_cfg.get('context_max_chars', 1600)
    
    routing_ctx = _extract_routing_context(messages, ctx_messages, ctx_chars)
    plugin_mgr = core["plugin_mgr"]
    last_text = plugin_mgr.hook_before_routing(routing_ctx["last_user_text"])

    # 1. Check security interceptor
    try:
        from modules.utils_text import sanitize
        intercepted = sanitize(last_text)
        if intercepted is not None:
            return intercepted, "canary_interceptor"
    except Exception as e:
        app_logger.warning(f"Security interceptor failed: {e}")

    t0 = time.monotonic()

    # 2. Run core inference
    res_text, used_lbl = _run_inference_impl(messages, model_hint)

    # 3. Record telemetry
    duration = time.monotonic() - t0
    try:
        from modules.session_store import push_context
        m_type = "unknown"
        if used_lbl == "fallback":
            m_type = "local-gguf"
        elif used_lbl == "error":
            m_type = "error"
        else:
            try:
                if hasattr(core["router"], 'get_expert_config'):
                    cfg = core["router"].get_expert_config(used_lbl)
                    if cfg:
                        m_type = cfg.get("type", "local")
                        if m_type == "local":
                            m_type = f"local-{cfg.get('format', 'onnx')}"
            except Exception:
                pass
        push_context(used_lbl, m_type, last_text, res_text, duration)
    except Exception as te:
        app_logger.warning(f"Telemetry tracking failed: {te}")

    return res_text, used_lbl


def _run_inference_impl(messages: list, model_hint: str) -> tuple[str, str]:
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

    def _execute_expert(label: str, score: float = 0.0) -> str:
        # Notify the routing_transparency plugin of the selected label+score
        # before running the expert so the footer is always accurate.
        _notify_transparency(label, score)
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
            result = _execute_expert("fallback", score=0.0)
            return result, "fallback"
        except Exception as e:
            app_logger.error(f"Critical error in fallback engine: {e}")
            return "An internal error occurred. Please try again later.", "error"

    # Phase 0: Plugin forced route
    if override_label:
        try:
            app_logger.info(f"Plugin forced route to: {override_label}")
            result = _execute_expert(override_label, score=1.0)
            return result, override_label
        except Exception as e:
            app_logger.error(f"[Auto-Correction] Plugin forced route '{override_label}' failed: {e}. Redirecting to fallback.")
            _fire_failure_webhook(override_label, str(e))
            return _do_fallback()

    # Phase 1.5: Regex triggers matching
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
                                app_logger.info(f"[Regex] Matched '{pattern}' -> routing directly to '{label}'")
                                result = _execute_expert(label, score=1.0)
                                return result, label
        except Exception as e:
            app_logger.error(f"[Regex] Error evaluating triggers: {e}")

    # Phase 2: Cascade Step 1 - evaluate last user message only
    label, score = router.predict(last_text)
    if label and label not in ('null', 'fallback') and score >= threshold:
        app_logger.info(f"[Cascade] Step 1: '{last_text[:60]}' -> {label} ({score:.2f})")
        try:
            result = _execute_expert(label, score=score)
            return result, label
        except Exception as e:
            app_logger.error(f"[Auto-Correction] Routed expert '{label}' failed: {e}. Redirecting to fallback.")
            _fire_failure_webhook(label, str(e))
            return _do_fallback()
    else:
        app_logger.info(f"[Cascade] Step 1: score {score:.2f} below threshold ({threshold}). Escalating to Step 2.")

    # Phase 3: Cascade Step 2 - evaluate concatenated recent user messages
    if ctx_text != last_text:
        label, score = router.predict(ctx_text)
        if label and label not in ('null', 'fallback') and score >= threshold:
            app_logger.info(f"[Cascade] Step 2: '{ctx_text[:60]}' -> {label} ({score:.2f})")
            try:
                result = _execute_expert(label, score=score)
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

# -- Routing diagnostic endpoint -------------------------------------------

_ROUTE_TEXT_MAX = 2000  # characters accepted in the query parameter
_ROUTE_TEXT_RE  = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')  # strip control chars


@app.route("/v1/route", methods=["GET", "POST"])
def route_inspect():
    """
    Diagnostic endpoint: runs the router against a text and returns the
    full scoring breakdown without generating any model response.

    GET  /v1/route?text=<your+prompt>
    POST /v1/route  body: {"text": "your prompt"}

    Response:
      {
        "expert":       "programador",
        "score":        0.87,
        "method":       "embedding",
        "cascade_step": 1,
        "top_experts":  [{"expert": "...", "score": ...}, ...]
      }
    """
    if not _rate_limit_check():
        return jsonify({"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}), 429

    # --- Extract input text -------------------------------------------------
    if request.method == "POST":
        body = request.get_json(force=True, silent=True) or {}
        raw_text = body.get("text", "")
    else:
        raw_text = request.args.get("text", "")

    if not isinstance(raw_text, str):
        return jsonify({"error": {"message": "'text' must be a string", "type": "invalid_request_error"}}), 400

    # Strip control characters and cap length
    text = _ROUTE_TEXT_RE.sub(' ', raw_text).strip()[:_ROUTE_TEXT_MAX]

    if not text:
        return jsonify({"error": {"message": "'text' is required and cannot be empty", "type": "invalid_request_error"}}), 400

    app_logger.info(f"[/v1/route] text={_safe_log(text)!r}")

    # --- Run router ---------------------------------------------------------
    core   = _Core.get()
    router = core["router"]
    config = core["config"]

    router_cfg = config.get('router', {})
    threshold  = router_cfg.get('confidence_threshold', 0.4)

    result: dict = {
        "expert":       "fallback",
        "score":        0.0,
        "method":       "fallback",
        "cascade_step": None,
        "top_experts":  [],
    }

    try:
        # Step 0: Regex triggers matching
        matched_label = None
        if hasattr(router, 'categories') and router.categories:
            for label, cat_data in router.categories.items():
                cfg = cat_data.get('config', {})
                triggers = cfg.get("regex_triggers", [])
                if not isinstance(triggers, list):
                    continue
                for pattern in triggers:
                    if isinstance(pattern, str) and pattern:
                        if re.search(pattern, text, re.IGNORECASE):
                            matched_label = label
                            break
                if matched_label:
                    break

        if matched_label:
            result["expert"]       = matched_label
            result["score"]        = 1.0
            result["method"]       = "regex"
            result["cascade_step"] = 1
        else:
            # Step 1: direct prediction
            label, score = router.predict(text)
            if label and label not in ('null', 'fallback') and score >= threshold:
                result["expert"]       = label
                result["score"]        = round(score, 4)
                result["method"]       = getattr(router, 'router_type', 'unknown')
                result["cascade_step"] = 1
            else:
                result["method"] = "fallback"

        # Build top_experts from category_embeddings if available
        # (embedding mode only — gives the full breakdown to the caller)
        cat_emb = getattr(router, 'category_embeddings', {})
        if cat_emb:
            import math
            sw  = getattr(router, 'scoring_weights', {})
            tmp = getattr(router, 'softmax_temperature', 0.15)
            from modules.utils_router import clean_text
            from sentence_transformers import util as st_util
            clean = clean_text(text)
            if clean and router._model is not None:
                query_vec = router._model.encode(
                    "query: " + clean, convert_to_tensor=True, show_progress_bar=False
                )
                raw: dict[str, float] = {
                    lbl: router._embed_score(query_vec, data)
                    for lbl, data in cat_emb.items()
                }
                max_raw = max(raw.values()) if raw else 0.0
                exp_s   = {l: math.exp((s - max_raw) / tmp) for l, s in raw.items()}
                total   = sum(exp_s.values()) or 1.0
                norm    = {l: v / total for l, v in exp_s.items()}
                top = sorted(norm.items(), key=lambda x: -x[1])[:5]
                result["top_experts"] = [
                    {"expert": lbl, "score": round(sc, 4)} for lbl, sc in top
                ]

    except Exception as e:
        app_logger.error(f"[/v1/route] Router error: {e}")
        result["error"] = "router_error"

    return jsonify(result)


# -- Ollama model discovery --------------------------------------------------

@app.route("/v1/discover", methods=["GET"])
def discover_ollama():
    """
    Queries the local Ollama instance and returns models that are not yet
    configured as experts, along with a ready-to-paste experts.json snippet.

    Query params:
      url  — Ollama base URL (default: http://127.0.0.1:11434)
    """
    if not _rate_limit_check():
        return jsonify({"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}), 429

    import urllib.request as _ur

    raw_url = request.args.get("url", "http://127.0.0.1:11434").strip().rstrip("/")

    # Scheme guard
    if not raw_url.startswith(("http://", "https://")):
        return jsonify({"error": {"message": "url must start with http:// or https://", "type": "invalid_request_error"}}), 400

    core            = _Core.get()
    configured      = set(core["expert_models"])
    suggestions     = []
    raw_models      = []
    error_msg       = None

    try:
        req = _ur.Request(f"{raw_url}/api/tags", headers={"Accept": "application/json"})
        with _ur.urlopen(req, timeout=5) as resp:
            data       = json.loads(resp.read().decode("utf-8"))
            raw_models = data.get("models", [])
    except Exception as exc:
        error_msg = str(exc)

    for entry in raw_models:
        name = entry.get("name", "") or entry.get("model", "")
        if not name:
            continue
        label = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.split(":")[0])[:48]
        if label in configured:
            continue
        suggestions.append({
            "model_name": name,
            "suggested_label": label,
            "snippet": {
                "label":       label,
                "description": f"Expert using {name}. Add keywords and a description.",
                "keywords":    [],
                "type":        "ollama",
                "url":         raw_url,
                "model_name":  name,
            },
        })

    result: dict = {
        "ollama_url":         raw_url,
        "configured_experts": sorted(configured),
        "unconfigured_models": suggestions,
    }
    if error_msg:
        result["error"] = error_msg
        app_logger.warning(f"[/v1/discover] Could not reach Ollama at {raw_url}: {error_msg}")

    return jsonify(result)


# -- Additional compatibility routes ----------------------------------------

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "name": "l3mcore",
        "version": SERVER_VERSION,
        "description": "Light Easy Mix Of Experts – OpenAI & Ollama compatible API",
        "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/route", "/v1/discover", "/api/tags", "/api/chat", "/api/version", "/health"],
    })



@app.route("/health", methods=["GET"])
def health():
    """Returns the operational status of every core component."""
    core = _Core.get()
    router = core["router"]
    runner = core["runner"]
    ai_engine = core["ai_engine"]
    plugin_mgr = core["plugin_mgr"]
    config = core["config"]

    router_status = "ok"
    router_mode = getattr(router, 'router_type', 'model')
    router_enabled = getattr(router, 'enabled', False)
    if not router_enabled:
        router_status = "degraded (keyword fallback only)"

    plugins_loaded = len(getattr(plugin_mgr, '_plugins', []))

    ai_ready = getattr(ai_engine, 'is_ready', False)
    models_in_memory = list(getattr(runner, 'sessions', {}).keys())

    # Check if config.json has been modified since startup
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
        "plugins": {
            "loaded": plugins_loaded,
            "names": [
                getattr(p, '__name__', '?').replace('l3mcore_plugin.', '')
                for p in getattr(plugin_mgr, '_plugins', [])
            ],
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
    if not _rate_limit_check():
        return jsonify({"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}), 429

    core = _Core.get()
    router_cfg_file = core["config"].get("router", {}).get("categories_file", "config/experts.json")
    
    try:
        with open(router_cfg_file, encoding="utf-8") as f:
            data = json.load(f)
        experts_list = data.get("experts", [])
    except Exception as e:
        return jsonify({"status": "error", "message": f"Could not read experts file: {e}"}), 500

    results = {}
    
    import urllib.request as _ur
    import urllib.error as _ue
    import os
    
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
            if model_format == "onnx":
                if model_path and os.path.exists(model_path):
                    status = "ready"
                    details = f"Local ONNX model path exists ({model_path})"
                else:
                    status = "missing_model"
                    details = f"Model path not found ({model_path})"
            elif model_format == "gguf":
                if model_path and os.path.exists(model_path):
                    status = "ready"
                    details = f"Local GGUF model path exists ({model_path})"
                else:
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


_enrichment_active = False
_enrichment_lock = threading.Lock()


def _apply_enriched_keywords(router, enriched_map):
    with router._cache_lock:
        for label, new_words in enriched_map.items():
            if label in router.categories:
                config_keywords = [k.lower() for k in router.categories[label]['config'].get('keywords', [])]
                combined = []
                seen = set()
                for k in config_keywords:
                    if k not in seen:
                        combined.append(k)
                        seen.add(k)
                for w in new_words:
                    if w not in seen:
                        combined.append(w)
                        seen.add(w)
                router.categories[label]['keywords'] = combined
                router.categories[label]['kw_tokens'] = frozenset(w for kw in combined for w in kw.split())
        
        if router.enabled and router.router_type == 'embedding':
            app_logger.info("[Keyword Enrichment] Recomputing semantic embeddings with enriched keywords...")
            router._precompute_category_embeddings()
        router._predict_cache.clear()


def _start_keyword_enrichment():
    """Starts background thread to semantically enrich keywords for experts using an active LLM."""
    global _enrichment_active
    
    def enrich():
        global _enrichment_active
        with _enrichment_lock:
            if _enrichment_active:
                return
            _enrichment_active = True
            
        try:
            time.sleep(3)
            app_logger.info("[Keyword Enrichment] Checking categories for semantic enrichment...")
            
            core = _Core.get()
            router = core.get("router")
            dispatcher = core.get("dispatcher")
            
            if not router or not hasattr(router, "categories"):
                return
            
            if getattr(router, "router_type", "") != "embedding" or not getattr(router, "enabled", False):
                app_logger.info("[Keyword Enrichment] Embedding router is disabled or not in embedding mode. Skipping.")
                return
                
            exp_path = "config/experts.json"
            if not os.path.exists(exp_path):
                return
                
            try:
                with open(exp_path, encoding='utf-8') as f:
                    data = json.load(f)
                experts = data.get('experts', [])
            except Exception as e:
                app_logger.error(f"[Keyword Enrichment] Error loading experts.json: {e}")
                return
                
            import hashlib
            norm_list = []
            for exp in experts:
                norm_list.append({
                    "label": exp.get("label", ""),
                    "description": exp.get("description", ""),
                    "keywords": sorted(exp.get("keywords", []))
                })
            experts_str = json.dumps(norm_list, sort_keys=True)
            current_hash = hashlib.sha256(experts_str.encode('utf-8')).hexdigest()
            
            cache_path = "config/.experts_enriched_cache.json"
            cache_data = {}
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, encoding='utf-8') as f:
                        cache_data = json.load(f)
                except Exception:
                    pass
                    
            if cache_data.get("hash") == current_hash:
                app_logger.info("[Keyword Enrichment] Enriched keywords cache is up-to-date. Applying from cache...")
                enriched_map = cache_data.get("enriched", {})
                if enriched_map:
                    _apply_enriched_keywords(router, enriched_map)
                return
                
            gen_expert = None
            for exp in experts:
                if exp.get("label") == "fallback":
                    if exp.get("type") == "local" and exp.get("format") == "gguf":
                        model_file = exp.get("model_path", "")
                        if not model_file or not os.path.exists(model_file):
                            continue
                    gen_expert = exp
                    break
                    
            if not gen_expert:
                for exp in experts:
                    if exp.get("type") in ["ollama", "api"]:
                        gen_expert = exp
                        break
                        
            if not gen_expert:
                for exp in experts:
                    if exp.get("type") == "local":
                        gen_expert = exp
                        break
                        
            if not gen_expert:
                app_logger.warning("[Keyword Enrichment] No LLM expert available to generate semantic keywords.")
                return
                
            app_logger.info(f"[Keyword Enrichment] Using expert '{gen_expert.get('label')}' for keyword generation...")
            
            enriched_map = {}
            for exp in experts:
                label = exp.get("label")
                if not label or label == "fallback":
                    continue
                    
                keywords = exp.get("keywords", [])
                description = exp.get("description", "")
                
                prompt = (
                    f"You are an assistant for a Mixture of Experts router.\n"
                    f"Category Name: {label}\n"
                    f"Description: {description}\n"
                    f"Existing Keywords: {', '.join(keywords)}\n\n"
                    f"Generate a list of 20 additional relevant keywords, search terms, synonyms, or short phrasing patterns (2-3 words max) in Spanish and English that users would use when asking about this category.\n"
                    f"Format each keyword or phrase naturally with spaces between words (e.g., use 'computer science' instead of 'computerscience', or 'react developer' instead of 'reactdeveloper'). Do not create long concatenated compound words.\n"
                    f"Output ONLY the keywords as a comma-separated list. Do not include numbering, explanations, colons, headers, markdown formatting, or introductory text. Just the comma-separated words."
                )
                
                messages = [{"role": "user", "content": prompt}]
                try:
                    app_logger.info(f"[Keyword Enrichment] Generating semantic synonyms for '{label}'...")
                    response = dispatcher.run(messages, gen_expert)
                    raw_parts = []
                    for line in response.split('\n'):
                        for part in line.split(','):
                            part_mod = part.replace('/', ' ').replace('\\', ' ')
                            raw_parts.append(part_mod)
                    
                    new_words = []
                    seen_words = set()
                    for part in raw_parts:
                        part_clean = part.strip()
                        part_clean = re.sub(r'^\d+[\.\)\s]+', '', part_clean)
                        part_clean = re.sub(r'^[\-\*\•\+]\s*', '', part_clean)
                        part_clean = part_clean.strip().strip('"').strip("'").strip(".").strip("-").strip("*").strip()
                        
                        if part_clean and len(part_clean) > 1 and len(part_clean) < 45 and ":" not in part_clean:
                            if len(part_clean) > 16 and " " not in part_clean:
                                continue  # Skip concatenated compound words
                            w_lower = part_clean.lower()
                            if w_lower not in seen_words:
                                if not any(stop in w_lower for stop in ["here is", "here are", "sure", "additional", "keywords:", "list of", "spanish", "english"]):
                                    new_words.append(w_lower)
                                    seen_words.add(w_lower)
                    if new_words:
                        enriched_map[label] = new_words
                        app_logger.info(f"[Keyword Enrichment] Generated {len(new_words)} enriched keywords for '{label}': {new_words[:5]}...")
                except Exception as e:
                    app_logger.error(f"[Keyword Enrichment] Error generating synonyms for '{label}': {e}")
                    return
                    
            try:
                with open(cache_path, "w", encoding='utf-8') as f:
                    json.dump({
                        "hash": current_hash,
                        "enriched": enriched_map
                    }, f, indent=4, ensure_ascii=False)
                app_logger.info("[Keyword Enrichment] Enriched keywords cached successfully.")
                _apply_enriched_keywords(router, enriched_map)
            except Exception as e:
                app_logger.error(f"[Keyword Enrichment] Error saving enrichment cache: {e}")
                
        finally:
            with _enrichment_lock:
                _enrichment_active = False


    threading.Thread(target=enrich, daemon=True).start()


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
                # Watch experts.json
                if os.path.exists(exp_path):
                    current_exp_mtime = os.path.getmtime(exp_path)
                    if current_exp_mtime > last_exp_mtime:
                        last_exp_mtime = current_exp_mtime
                        app_logger.info("Watcher: experts.json modification detected. Hot-reloading experts...")
                        _Core.reload_experts()

                # Watch config.json
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


def _bootstrap():
    """Pre-load core + plugins once before the first request."""
    _Core.get()
    PluginManager()
    _start_experts_watcher()
    _start_keyword_enrichment()


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
