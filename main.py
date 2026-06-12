"""
l3mcore - Light Easy Mix Of Experts
Main entry point: loads the classifier router and specialized ONNX models.
"""

import sys
import os
import re
import signal

from modules.logger import app_logger
from modules.config_manager import ConfigManager
from modules.router_factory import create_router
from modules.onnx_runner import SpecificModelRunner
from modules.ai_engine import AIEngine
from modules.expert_runner import ExpertDispatcher

_NON_PRINTABLE = re.compile(r'[\x00-\x1f\x7f]')


def _safe_log(text: str, max_len: int = 120) -> str:
    """Strip control characters and truncate user input before logging."""
    cleaned = _NON_PRINTABLE.sub(' ', text)
    return cleaned[:max_len] if len(cleaned) > max_len else cleaned


class l3mcore:
    """
    Main orchestrator of the MoE system.
    Flow: text input -> Router -> Label -> ExpertDispatcher -> Local Model / API / Ollama
    If the router returns 'null', AIEngine (GGUF) is used as fallback.
    """

    def __init__(self):
        app_logger.info("Starting l3mcore...")

        self.config = ConfigManager()
        self.router = create_router(self.config)
        self.runner = SpecificModelRunner(
            models_base_path="models",
            stats_path="data/model_stats.json"
        )
        self.ai_engine = AIEngine()
        self.dispatcher = ExpertDispatcher(self.runner, self.ai_engine)

        app_logger.info("l3mcore ready.")

    def process(self, text: str) -> str:
        """
        Processes text input:
        1. Router classifies input and gets model label.
        2. If valid label, ExpertDispatcher executes request.
        3. If low confidence, default GGUF AIEngine is used.
        """
        if not text or not text.strip():
            return ""

        # Check security interceptor
        try:
            from modules.utils_text import sanitize
            intercepted = sanitize(text)
            if intercepted is not None:
                return intercepted
        except Exception as e:
            app_logger.warning(f"Security interceptor failed: {e}")

        import time
        t0 = time.monotonic()

        label, score = self.router.predict(text)
        app_logger.info(f"Router: label='{label}' score={score:.3f}")

        used_model = "fallback"
        model_type = "local-gguf"
        result = None

        if label and label != "null":
            try:
                # If router is generic and has config
                if hasattr(self.router, 'get_expert_config'):
                    cfg = self.router.get_expert_config(label)
                    if cfg:
                        result = self.dispatcher.run(text, cfg)
                        used_model = label
                        model_type = cfg.get("type", "local")
                        if model_type == "local":
                            model_type = f"local-{cfg.get('format', 'onnx')}"
                
                if result is None:
                    # Fallback for pure ML router assuming local model
                    cfg = {"type": "local", "format": "onnx", "label": label}
                    result = self.dispatcher.run(text, cfg)
                    used_model = label
                    model_type = "local-onnx"
            except Exception as e:
                app_logger.error(f"Error in expert ({label}): {e}. Using GGUF fallback.")

        if result is None:
            # Fallback: general GGUF model
            result = self.ai_engine.generate_response(text)
            app_logger.info(f"AIEngine (GGUF) -> '{_safe_log(result)}'")

        # Record telemetry
        duration = time.monotonic() - t0
        try:
            from modules.session_store import push_context
            push_context(used_model, model_type, text, result, duration)
        except Exception as e:
            app_logger.warning(f"Telemetry tracking failed: {e}")

        return result

    def shutdown(self):
        app_logger.info("Shutting down l3mcore...")
        if hasattr(self.router, 'clear_cache'):
            self.router.clear_cache()
        app_logger.info("Shutdown complete.")


def _handle_signal(sig, frame, lemoe_instance):
    lemoe_instance.shutdown()
    sys.exit(0)


def main():
    core = l3mcore()

    # Register clean shutdown signals
    signal.signal(signal.SIGINT,  lambda s, f: _handle_signal(s, f, core))
    signal.signal(signal.SIGTERM, lambda s, f: _handle_signal(s, f, core))

    app_logger.info("l3mcore waiting for input. Send text via stdin or import l3mcore class.")

    # Stdin read loop (useful for manual testing or piping from another process)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            break
        app_logger.info(f"stdin input: '{_safe_log(line)}'")
        response = core.process(line)
        print(response, flush=True)

    core.shutdown()


if __name__ == "__main__":
    main()
