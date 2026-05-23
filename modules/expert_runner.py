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
    Routes inference to the correct backend based on the expert config dict.

    Supported backends:
      'api'    -> External REST API via litellm (OpenAI, Anthropic, Groq, ...).
                  API key is read from the environment variable named in api_key_env.
      'ollama' -> Local or remote Ollama instance.
                  URL is validated for scheme and blocked networks before each call.
      'local'  -> Local ONNX model (via SpecificModelRunner) or GGUF (via AIEngine).
    """

    def __init__(self, onnx_runner, ai_engine):
        self.onnx_runner = onnx_runner
        self.ai_engine = ai_engine

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

        provider   = config.get('provider', '')
        model_name = config.get('model_name', '')
        if not model_name:
            raise ValueError("model_name required for 'api' expert")

        litellm_model = f"{provider}/{model_name}" if provider and provider != 'openai' else model_name

        env_var = config.get('api_key_env', '')
        api_key = os.environ.get(env_var) if env_var else None
        if not api_key:
            app_logger.warning(f"API key not found in env var '{env_var}'. litellm will try its defaults.")

        app_logger.info(f"ExpertDispatcher [api]: calling {litellm_model}")

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        response = litellm.completion(
            model=litellm_model,
            messages=messages,
            api_key=api_key
        )
        return response.choices[0].message.content.strip()

    def _run_ollama(self, messages, config: dict) -> str:
        raw_url    = config.get('url', 'http://127.0.0.1:11434').rstrip('/')
        model_name = config.get('model_name', 'llama3')

        url      = _validate_ollama_url(raw_url)
        endpoint = f"{url}/api/chat"
        app_logger.info(f"ExpertDispatcher [ollama]: POST {endpoint} ({model_name})")

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        data = {"model": model_name, "messages": messages, "stream": False}
        req  = urllib.request.Request(
            endpoint,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result.get('message', {}).get('content', '').strip()
        except urllib.error.URLError as e:
            raise RuntimeError(f"Error connecting to Ollama at {url}: {e}")

    # --------------------------------------------------------------------- LOCAL

    def _run_local(self, messages, config: dict) -> str:
        """
        Runs local inference via SpecificModelRunner (ONNX) or AIEngine (GGUF).
        """
        model_format = config.get('format', 'onnx').lower()
        
        text = messages
        if isinstance(messages, list):
            # Extract text for local models that only accept strings
            text = " ".join(
                str(part.get("text", part.get("content", ""))) if isinstance(part, dict) else str(part)
                for msg in messages for part in ([msg["content"]] if isinstance(msg.get("content"), str) else msg.get("content", []))
                if isinstance(part, dict) and part.get("type", "text") == "text" or isinstance(part, str)
            )
        label      = config.get('label', '')
        model_path = config.get('model_path')

        if model_format == 'onnx':
            # Delegate to SpecificModelRunner
            return self.onnx_runner.generate_command(text, label, model_path)

        elif model_format == 'gguf':
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

        elif model_format == 'huggingface':
            raise NotImplementedError("Local 'huggingface' format not implemented yet.")

        else:
            raise ValueError(f"Unknown local format: {model_format}")
