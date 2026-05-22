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


# Schemes accepted for Ollama endpoints
_ALLOWED_SCHEMES = {"http", "https"}

# Cloud metadata / link-local ranges always blocked (SEC-2)
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("169.254.0.0/16"),  # AWS/GCP/Azure metadata + link-local
    ipaddress.ip_network("100.64.0.0/10"),   # Carrier-grade NAT
]


def _validate_ollama_url(url: str) -> str:
    """
    SEC-2: Validate an Ollama endpoint URL.
    - Only http/https schemes allowed.
    - Blocks cloud metadata IP ranges (169.254.x.x etc).
    - Private/loopback IPs allowed by default (internal use case).
    Raises ValueError on invalid URLs.
    """
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
                raise ValueError(
                    f"Ollama URL points to a blocked network ({net}): {url}"
                )
    except ValueError as exc:
        # If it's not an IP but the ValueError is ours, re-raise
        if "blocked network" in str(exc) or "scheme" in str(exc):
            raise
        # Otherwise it's a hostname — allow it (DNS resolution not checked here)
        pass

    return url


class ExpertDispatcher:
    """
    Central dispatcher: receives an expert config dict and delegates
    inference to the right backend engine.

    Supported backends:
      'api'    -> External REST API via litellm (OpenAI, Anthropic, Gemini, ...).
                  API key read from the environment variable in api_key_env.
      'ollama' -> Local or remote Ollama instance.
                  URL validated for scheme and blocked networks before each call.
      'local'  -> Local ONNX model (via SpecificModelRunner) or GGUF (via AIEngine).

    Security:
      - Ollama URLs are validated by _validate_ollama_url() before every request:
        only http/https accepted; 169.254.x.x (cloud metadata) is blocked.
      - Private/loopback IPs are allowed (internal deployment use case).
    """

    def __init__(self, onnx_runner, ai_engine):
        self.onnx_runner = onnx_runner
        self.ai_engine = ai_engine

    def run(self, text: str, expert_config: dict) -> str:
        expert_type = expert_config.get('type', 'local').lower()

        try:
            if expert_type == 'api':
                return self._run_api(text, expert_config)
            elif expert_type == 'ollama':
                return self._run_ollama(text, expert_config)
            elif expert_type == 'local':
                return self._run_local(text, expert_config)
            else:
                raise ValueError(f"Unknown expert type: {expert_type}")
        except Exception as e:
            app_logger.error(f"Error executing expert '{expert_config.get('label')}': {e}")
            raise

    # ----------------------------------------------------------------------- API

    def _run_api(self, text: str, config: dict) -> str:
        """
        Calls an external API (OpenAI, Gemini, Claude, etc) using litellm.
        """
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm required for 'api' type experts")

        provider   = config.get('provider', '')
        model_name = config.get('model_name', '')
        if not model_name:
            raise ValueError("model_name required for 'api' expert")

        # litellm format: "provider/model" or just "model" if openai
        litellm_model = f"{provider}/{model_name}" if provider and provider != 'openai' else model_name

        # Load API key from environment variable
        env_var = config.get('api_key_env', '')
        api_key = os.environ.get(env_var) if env_var else None

        if not api_key:
            app_logger.warning(f"API key not found in {env_var}. litellm will try defaults.")

        app_logger.info(f"ExpertDispatcher [API]: calling {litellm_model}")

        response = litellm.completion(
            model=litellm_model,
            messages=[{"role": "user", "content": text}],
            api_key=api_key
        )

        return response.choices[0].message.content.strip()

    # -------------------------------------------------------------------- OLLAMA

    def _run_ollama(self, text: str, config: dict) -> str:
        """
        Performs an HTTP POST request to an Ollama instance.
        """
        raw_url    = config.get('url', 'http://127.0.0.1:11434').rstrip('/')
        model_name = config.get('model_name', 'llama3')

        # SEC-2: validate URL before making the request
        url      = _validate_ollama_url(raw_url)
        endpoint = f"{url}/api/generate"
        app_logger.info(f"ExpertDispatcher [Ollama]: POST {endpoint} ({model_name})")

        data = {
            "model":  model_name,
            "prompt": text,
            "stream": False
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('response', '').strip()
        except urllib.error.URLError as e:
            raise RuntimeError(f"Error connecting to Ollama at {url}: {e}")

    # --------------------------------------------------------------------- LOCAL

    def _run_local(self, text: str, config: dict) -> str:
        """
        Executes local ONNX or GGUF models.
        """
        fmt        = config.get('format', 'onnx').lower()
        label      = config.get('label', '')
        model_path = config.get('model_path')

        if fmt == 'onnx':
            # Delegate to SpecificModelRunner
            return self.onnx_runner.generate_command(text, label, model_path)

        elif fmt == 'gguf':
            # Temporarily reconfigure AIEngine
            original_path = self.ai_engine.model_path
            try:
                if model_path and os.path.exists(model_path):
                    self.ai_engine.model_path = model_path
                    # Force reload if path changed
                    if getattr(self.ai_engine, 'llm', None):
                        # Unload current
                        self.ai_engine.llm = None
                        gc.collect()
                return self.ai_engine.generate_response(text)
            finally:
                self.ai_engine.model_path = original_path

        elif fmt == 'huggingface':
            raise NotImplementedError("Local 'huggingface' format not implemented yet.")

        else:
            raise ValueError(f"Unknown local format: {fmt}")
