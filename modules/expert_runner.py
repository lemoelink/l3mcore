import os
import gc
import json
import time
import ipaddress
import threading
import urllib.request
import urllib.error
from urllib.parse import urlparse
from typing import Generator
from modules.logger import app_logger

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    app_logger.warning("litellm is not installed. External API calls may fail.")

try:
    import urllib3
    _HTTP_POOL = urllib3.PoolManager(maxsize=10)
    URLLIB3_AVAILABLE = True
except ImportError:
    _HTTP_POOL = None
    URLLIB3_AVAILABLE = False


_ALLOWED_SCHEMES = {"http", "https"}

# Cloud metadata / link-local ranges always blocked
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("169.254.0.0/16"),  # AWS/GCP/Azure metadata + link-local
    ipaddress.ip_network("100.64.0.0/10"),   # Carrier-grade NAT
]

# Default Ollama hostname allowlist. Add more entries in config.json under
# expert_runner.ollama_allowed_hosts if needed.
_DEFAULT_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1"}

_DEFAULT_API_TIMEOUT = 60  # seconds


# ---------------------------------------------------------------------------
# Circuit Breaker: fast failover when backends are unreachable
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Lightweight in-memory circuit breaker to prevent hanging on degraded backends."""
    def __init__(self, failure_threshold: int = 3, recovery_time: float = 30.0):
        self.threshold = failure_threshold
        self.recovery_time = recovery_time
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}
        self._lock = threading.Lock()

    def is_available(self, label: str) -> bool:
        with self._lock:
            if label not in self._opened_at:
                return True
            if time.time() - self._opened_at[label] > self.recovery_time:
                # Cooldown period elapsed, allow probe request
                del self._opened_at[label]
                self._failures[label] = 0
                app_logger.info(f"CircuitBreaker: Cooldown elapsed for '{label}', probing backend.")
                return True
            return False

    def record_success(self, label: str) -> None:
        with self._lock:
            self._failures[label] = 0
            self._opened_at.pop(label, None)

    def record_failure(self, label: str) -> None:
        with self._lock:
            count = self._failures.get(label, 0) + 1
            self._failures[label] = count
            if count >= self.threshold:
                self._opened_at[label] = time.time()
                app_logger.warning(
                    f"CircuitBreaker: Backend '{label}' tripped ({count} consecutive errors). "
                    f"Fast-failing to fallback for {self.recovery_time}s."
                )


circuit_breaker = CircuitBreaker()


def _get_runner_config(config_manager=None) -> dict:
    if config_manager is None:
        return {}
    return config_manager.get("expert_runner", {})


def _validate_ollama_url(url: str, allowed_hosts: set | None = None) -> str:
    """
    Validates an Ollama endpoint URL.
    """
    if allowed_hosts is None:
        allowed_hosts = _DEFAULT_ALLOWED_HOSTS

    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise ValueError(f"Malformed Ollama URL: {url}") from exc

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"Unsafe URL scheme '{parsed.scheme}' in Ollama config. "
            f"Only {_ALLOWED_SCHEMES} are allowed."
        )

    hostname = parsed.hostname or ""
    try:
        addr = ipaddress.ip_address(hostname)
        for net in _BLOCKED_NETWORKS:
            if addr in net:
                raise ValueError(f"Ollama URL points to a blocked network ({net}): {url}")
    except ValueError as exc:
        if "blocked network" in str(exc) or "scheme" in str(exc):
            raise
        if hostname not in allowed_hosts:
            raise ValueError(
                f"Ollama hostname '{hostname}' is not in the allowed hosts list. "
                f"Add it to expert_runner.ollama_allowed_hosts in config.json."
            )

    return url


def _extract_text_from_messages(messages) -> str:
    """Extracts a plain text string from a messages list for local model inference."""
    if isinstance(messages, str):
        return messages

    parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    if part.get("type", "text") == "text":
                        parts.append(str(part.get("text", part.get("content", ""))))
    return " ".join(parts)


_SYS_PROMPT_MAX = 4000  # characters cap


def _inject_system_prompt(messages, expert_config: dict) -> list:
    """
    Prepends a system message from the expert's 'system_prompt' field.
    """
    raw = expert_config.get("system_prompt", "")
    if not isinstance(raw, str) or not raw.strip():
        return messages if isinstance(messages, list) else list(messages)

    prompt = raw.strip()[:_SYS_PROMPT_MAX]
    msgs = list(messages) if isinstance(messages, list) else []
    system_msg = {"role": "system", "content": prompt}

    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        return [system_msg] + msgs
    return [system_msg] + msgs


class ExpertDispatcher:
    """
    Routes inference to the correct backend based on the expert config dict.

    Supported backends:
      'api'    -> External REST API via litellm (OpenAI, Anthropic, vLLM, LocalAI, ...).
                  Compatible with vLLM via provider='openai' and api_base='http://.../v1'.
      'ollama' -> Local or remote Ollama instance with HTTP Keep-Alive pooling.
      'local'  -> Local ONNX model (via SpecificModelRunner) or GGUF (via AIEngine).
    """

    def __init__(self, onnx_runner, ai_engine, config_manager=None):
        self.onnx_runner = onnx_runner
        self.ai_engine = ai_engine
        self._config_manager = config_manager
        self._gguf_lock = threading.Lock()
        self.circuit_breaker = circuit_breaker

    def _runner_cfg(self) -> dict:
        return _get_runner_config(self._config_manager)

    def run(self, messages, expert_config: dict, gen_params: dict | None = None) -> str:
        """
        Runs non-streaming inference for the given expert.
        """
        label = expert_config.get('label', 'unknown')
        if not self.circuit_breaker.is_available(label):
            raise RuntimeError(f"Circuit breaker open for expert '{label}'")

        expert_type = expert_config.get('type', 'local').lower()
        messages = _inject_system_prompt(messages, expert_config)
        gen_params = gen_params or {}

        try:
            if expert_type == 'api':
                result = self._run_api(messages, expert_config, gen_params)
            elif expert_type == 'ollama':
                result = self._run_ollama(messages, expert_config, gen_params)
            elif expert_type == 'local':
                result = self._run_local(messages, expert_config)
            else:
                raise ValueError(f"Unknown expert type: {expert_type}")

            self.circuit_breaker.record_success(label)
            return result
        except Exception as e:
            self.circuit_breaker.record_failure(label)
            app_logger.error(f"Error executing expert '{label}': {e}")
            raise

    def run_stream(self, messages, expert_config: dict, gen_params: dict | None = None) -> Generator[str, None, None]:
        """
        Runs real token-to-token streaming inference for the given expert.
        Yields text chunks as they arrive from the backend.
        """
        label = expert_config.get('label', 'unknown')
        if not self.circuit_breaker.is_available(label):
            raise RuntimeError(f"Circuit breaker open for expert '{label}'")

        expert_type = expert_config.get('type', 'local').lower()
        messages = _inject_system_prompt(messages, expert_config)
        gen_params = gen_params or {}

        try:
            if expert_type == 'api':
                for chunk in self._run_api_stream(messages, expert_config, gen_params):
                    yield chunk
            elif expert_type == 'ollama':
                for chunk in self._run_ollama_stream(messages, expert_config, gen_params):
                    yield chunk
            elif expert_type == 'local':
                # Local models (ONNX/GGUF) yield full result in one piece
                yield self._run_local(messages, expert_config)
            else:
                raise ValueError(f"Unknown expert type: {expert_type}")

            self.circuit_breaker.record_success(label)
        except Exception as e:
            self.circuit_breaker.record_failure(label)
            app_logger.error(f"Error executing streaming expert '{label}': {e}")
            raise

    def _prepare_api_kwargs(self, messages, config: dict, gen_params: dict, stream: bool = False) -> dict:
        provider = config.get('provider', '')
        model_name = config.get('model_name', '')
        if not model_name:
            raise ValueError("model_name required for 'api' expert")

        # Support for custom api_base (vLLM, LocalAI, Ollama OpenAI endpoint, TGI)
        api_base = config.get('api_base') or config.get('url')

        if provider and provider.lower() != 'openai':
            litellm_model = f"{provider}/{model_name}"
        elif api_base:
            # When api_base is supplied (like vLLM), prefix with openai/ so litellm treats it as OpenAI-compatible
            litellm_model = f"openai/{model_name}" if not model_name.startswith("openai/") else model_name
        else:
            litellm_model = model_name

        env_var = config.get('api_key_env', '')
        api_key = os.environ.get(env_var) if env_var else config.get('api_key')
        if not api_key:
            if api_base:
                # Local servers like vLLM do not require real keys, dummy string suffices
                api_key = "dummy-vllm-key"
            else:
                app_logger.warning(f"API key not found in env var '{env_var}'. litellm will try defaults.")

        cfg = self._runner_cfg()
        timeout = cfg.get("api_timeout", _DEFAULT_API_TIMEOUT)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        formatted_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                formatted_messages.append(msg)
                continue

            new_msg = {"role": msg.get("role")}
            content = msg.get("content")
            images = msg.get("images")

            if isinstance(content, list):
                new_msg["content"] = content
                formatted_messages.append(new_msg)
                continue

            if isinstance(images, list) and images:
                parts = []
                if content:
                    parts.append({"type": "text", "text": str(content)})
                for img in images:
                    if isinstance(img, str):
                        if not img.startswith("data:image/"):
                            img = f"data:image/png;base64,{img}"
                        parts.append({"type": "image_url", "image_url": {"url": img}})
                new_msg["content"] = parts
            else:
                new_msg["content"] = str(content) if content is not None else ""

            formatted_messages.append(new_msg)

        kwargs = {
            "model": litellm_model,
            "messages": formatted_messages,
            "api_key": api_key,
            "timeout": timeout,
            "stream": stream,
        }
        if api_base:
            kwargs["api_base"] = api_base

        # Propagate standard LLM generation parameters
        for p in ("temperature", "top_p", "max_tokens", "stop", "presence_penalty", "frequency_penalty"):
            val = gen_params.get(p)
            if val is not None:
                kwargs[p] = val

        return kwargs

    def _run_api(self, messages, config: dict, gen_params: dict) -> str:
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm required for 'api' type experts")

        kwargs = self._prepare_api_kwargs(messages, config, gen_params, stream=False)
        app_logger.info(f"ExpertDispatcher [api]: calling {kwargs.get('model')}")
        response = litellm.completion(**kwargs)
        choice = response.choices[0]
        return choice.message.content.strip()

    def _run_api_stream(self, messages, config: dict, gen_params: dict) -> Generator[str, None, None]:
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm required for 'api' type experts")

        kwargs = self._prepare_api_kwargs(messages, config, gen_params, stream=True)
        app_logger.info(f"ExpertDispatcher [api-stream]: streaming {kwargs.get('model')}")
        response = litellm.completion(**kwargs)
        for chunk in response:
            if not chunk or not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, 'content', None)
            if content:
                yield content

    def _prepare_ollama_payload(self, messages, config: dict, gen_params: dict, stream: bool = False) -> tuple[str, dict, float]:
        raw_url = config.get('url', 'http://127.0.0.1:11434').rstrip('/')
        model_name = config.get('model_name', 'llama3')

        cfg = self._runner_cfg()
        allowed_hosts = set(cfg.get("ollama_allowed_hosts", [])) | _DEFAULT_ALLOWED_HOSTS
        timeout = cfg.get("ollama_timeout", _DEFAULT_API_TIMEOUT)

        url = _validate_ollama_url(raw_url, allowed_hosts=allowed_hosts)
        endpoint = f"{url}/api/chat"

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        formatted_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                formatted_messages.append(msg)
                continue

            new_msg = {"role": msg.get("role")}
            content = msg.get("content")
            images = msg.get("images") or []
            if not isinstance(images, list):
                images = [images]

            clean_images = []
            for img in images:
                if isinstance(img, str):
                    if img.startswith("data:image/") and ";base64," in img:
                        img = img.split(";base64,", 1)[1]
                    clean_images.append(img)

            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type")
                        if part_type == "text":
                            text_parts.append(part.get("text", ""))
                        elif part_type == "image_url":
                            img_url_dict = part.get("image_url")
                            if isinstance(img_url_dict, dict):
                                img_url = img_url_dict.get("url", "")
                                if isinstance(img_url, str) and img_url.startswith("data:image/"):
                                    if ";base64," in img_url:
                                        raw_b64 = img_url.split(";base64,", 1)[1]
                                        clean_images.append(raw_b64)
                new_msg["content"] = "\n".join(text_parts)
            else:
                new_msg["content"] = str(content) if content is not None else ""

            if clean_images:
                new_msg["images"] = clean_images

            formatted_messages.append(new_msg)

        data = {
            "model": model_name,
            "messages": formatted_messages,
            "stream": stream,
        }

        # Ollama generation options
        options = {}
        if "temperature" in gen_params and gen_params["temperature"] is not None:
            options["temperature"] = float(gen_params["temperature"])
        if "top_p" in gen_params and gen_params["top_p"] is not None:
            options["top_p"] = float(gen_params["top_p"])
        if "max_tokens" in gen_params and gen_params["max_tokens"] is not None:
            options["num_predict"] = int(gen_params["max_tokens"])
        if "stop" in gen_params and gen_params["stop"] is not None:
            options["stop"] = gen_params["stop"]

        if options:
            data["options"] = options

        return endpoint, data, timeout

    def _run_ollama(self, messages, config: dict, gen_params: dict) -> str:
        endpoint, data, timeout = self._prepare_ollama_payload(messages, config, gen_params, stream=False)
        payload_bytes = json.dumps(data).encode('utf-8')

        try:
            if URLLIB3_AVAILABLE and _HTTP_POOL is not None:
                resp = _HTTP_POOL.request(
                    'POST',
                    endpoint,
                    body=payload_bytes,
                    headers={'Content-Type': 'application/json'},
                    timeout=timeout
                )
                if resp.status != 200:
                    raise RuntimeError(f"Ollama returned HTTP status {resp.status}: {resp.data.decode('utf-8')}")
                result = json.loads(resp.data.decode('utf-8'))
            else:
                req = urllib.request.Request(
                    endpoint,
                    data=payload_bytes,
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    result = json.loads(resp.read().decode('utf-8'))

            return result.get('message', {}).get('content', '').strip()
        except Exception as e:
            raise RuntimeError(f"Error communicating with Ollama at {endpoint}: {e}")

    def _run_ollama_stream(self, messages, config: dict, gen_params: dict) -> Generator[str, None, None]:
        endpoint, data, timeout = self._prepare_ollama_payload(messages, config, gen_params, stream=True)
        payload_bytes = json.dumps(data).encode('utf-8')

        try:
            if URLLIB3_AVAILABLE and _HTTP_POOL is not None:
                resp = _HTTP_POOL.request(
                    'POST',
                    endpoint,
                    body=payload_bytes,
                    headers={'Content-Type': 'application/json'},
                    timeout=timeout,
                    preload_content=False
                )
                try:
                    for line in resp.stream():
                        if not line:
                            continue
                        line_str = line.decode('utf-8').strip()
                        if not line_str:
                            continue
                        chunk = json.loads(line_str)
                        content = chunk.get('message', {}).get('content', '')
                        if content:
                            yield content
                finally:
                    resp.release_conn()
            else:
                req = urllib.request.Request(
                    endpoint,
                    data=payload_bytes,
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    for line in resp:
                        if not line:
                            continue
                        line_str = line.decode('utf-8').strip()
                        if not line_str:
                            continue
                        chunk = json.loads(line_str)
                        content = chunk.get('message', {}).get('content', '')
                        if content:
                            yield content
        except Exception as e:
            raise RuntimeError(f"Error streaming from Ollama at {endpoint}: {e}")

    def _run_local(self, messages, config: dict) -> str:
        model_format = config.get('format', 'onnx').lower()
        text = _extract_text_from_messages(messages)
        label = config.get('label', '')
        model_path = config.get('model_path')

        if model_format == 'onnx':
            return self.onnx_runner.generate_command(text, label, model_path)

        elif model_format == 'gguf':
            with self._gguf_lock:
                original_path = self.ai_engine.model_path
                try:
                    if model_path and os.path.exists(model_path):
                        self.ai_engine.model_path = model_path
                        if getattr(self.ai_engine, 'llm', None):
                            self.ai_engine.llm = None
                            gc.collect()
                    return self.ai_engine.generate_response(text)
                finally:
                    self.ai_engine.model_path = original_path

        elif model_format == 'huggingface':
            raise NotImplementedError("Local 'huggingface' format not implemented.")

        else:
            raise ValueError(f"Unknown local format: {model_format}")
