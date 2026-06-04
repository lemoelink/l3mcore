import os
import gc
import json
import ipaddress
import urllib.request
import urllib.error
from urllib.parse import urlparse
from modules.logger import app_logger

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    app_logger.warning("litellm is not installed. External API calls may fail.")


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


def _get_runner_config(config_manager=None) -> dict:
    if config_manager is None:
        return {}
    return config_manager.get("expert_runner", {})


def _validate_ollama_url(url: str, allowed_hosts: set | None = None) -> str:
    """
    Validates an Ollama endpoint URL.

    - Only http/https schemes are accepted.
    - Cloud metadata IP ranges (169.254.x.x etc.) are always blocked.
    - Hostname validation: if the value resolves to an IP it is checked against
      the blocked networks; if it is a plain hostname it must be in allowed_hosts.
    - Private/loopback IPs are allowed by default.

    Raises ValueError on invalid URLs.
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
        # It's a plain hostname — check against the allowlist
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


class ExpertDispatcher:
    """
    Routes inference to the correct backend based on the expert config dict.

    Supported backends:
      'api'    -> External REST API via litellm (OpenAI, Anthropic, Groq, ...).
                  API key is read from the environment variable named in api_key_env.
      'ollama' -> Local or remote Ollama instance.
                  URL is validated for scheme and blocked networks before each call.
      'local'  -> Local ONNX model (via SpecificModelRunner) or GGUF (via AIEngine).
    """

    def __init__(self, onnx_runner, ai_engine, config_manager=None):
        self.onnx_runner = onnx_runner
        self.ai_engine = ai_engine
        self._config_manager = config_manager

    def _runner_cfg(self) -> dict:
        return _get_runner_config(self._config_manager)

    def run(self, messages, expert_config: dict) -> str:
        expert_type = expert_config.get('type', 'local').lower()
        try:
            if expert_type == 'api':
                return self._run_api(messages, expert_config)
            elif expert_type == 'ollama':
                return self._run_ollama(messages, expert_config)
            elif expert_type == 'local':
                return self._run_local(messages, expert_config)
            else:
                raise ValueError(f"Unknown expert type: {expert_type}")
        except Exception as e:
            app_logger.error(f"Error executing expert '{expert_config.get('label')}': {e}")
            raise

    def _run_api(self, messages, config: dict) -> str:
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm required for 'api' type experts")

        provider = config.get('provider', '')
        model_name = config.get('model_name', '')
        if not model_name:
            raise ValueError("model_name required for 'api' expert")

        litellm_model = f"{provider}/{model_name}" if provider and provider != 'openai' else model_name

        env_var = config.get('api_key_env', '')
        api_key = os.environ.get(env_var) if env_var else None
        if not api_key:
            app_logger.warning(f"API key not found in env var '{env_var}'. litellm will try its defaults.")

        cfg = self._runner_cfg()
        timeout = cfg.get("api_timeout", _DEFAULT_API_TIMEOUT)

        app_logger.info(f"ExpertDispatcher [api]: calling {litellm_model} (timeout={timeout}s)")

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        response = litellm.completion(
            model=litellm_model,
            messages=messages,
            api_key=api_key,
            timeout=timeout,
        )
        return response.choices[0].message.content.strip()

    def _run_ollama(self, messages, config: dict) -> str:
        raw_url = config.get('url', 'http://127.0.0.1:11434').rstrip('/')
        model_name = config.get('model_name', 'llama3')

        cfg = self._runner_cfg()
        allowed_hosts = set(cfg.get("ollama_allowed_hosts", [])) | _DEFAULT_ALLOWED_HOSTS
        timeout = cfg.get("ollama_timeout", _DEFAULT_API_TIMEOUT)

        url = _validate_ollama_url(raw_url, allowed_hosts=allowed_hosts)
        endpoint = f"{url}/api/chat"
        app_logger.info(f"ExpertDispatcher [ollama]: POST {endpoint} ({model_name})")

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        data = {"model": model_name, "messages": messages, "stream": False}
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result.get('message', {}).get('content', '').strip()
        except urllib.error.URLError as e:
            raise RuntimeError(f"Error connecting to Ollama at {url}: {e}")

    def _run_local(self, messages, config: dict) -> str:
        model_format = config.get('format', 'onnx').lower()
        text = _extract_text_from_messages(messages)
        label = config.get('label', '')
        model_path = config.get('model_path')

        if model_format == 'onnx':
            return self.onnx_runner.generate_command(text, label, model_path)

        elif model_format == 'gguf':
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
            raise NotImplementedError("Local 'huggingface' format not implemented yet.")

        else:
            raise ValueError(f"Unknown local format: {model_format}")
