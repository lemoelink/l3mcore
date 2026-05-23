import gc
import logging
import os
import time
import threading
from modules.logger import app_logger

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    app_logger.warning("llama-cpp-python is not installed. AIEngine will not work.")


class AIEngine:
    """
    Inference engine for GGUF models (general fallback).
    Loads the model lazily on first use.
    """

    def __init__(self, model_path=None):
        self.default_path = "models/gemma-2-2b-it-Q4_K_M.gguf"

        if model_path and os.path.exists(model_path):
            self.model_path = model_path
        elif os.path.exists("models/gemma-2-2b-it-Q8_0.gguf"):
            self.model_path = "models/gemma-2-2b-it-Q8_0.gguf"
        else:
            self.model_path = self.default_path

        app_logger.info(f"AIEngine configured with: {self.model_path} (Lazy Loading)")
        self.llm = None
        self.is_ready = False
        
        self.last_access = 0
        self.ttl_seconds = 300 # 5 minutes
        self._cleanup_lock = threading.Lock()
        self._stop_cleanup = False
        
        if LLAMA_AVAILABLE:
            self._cleanup_thread = threading.Thread(
                target=self._ttl_cleanup_loop,
                daemon=True,
                name="AIEngine_TTL_Cleanup"
            )
            self._cleanup_thread.start()

    def _ttl_cleanup_loop(self):
        while not self._stop_cleanup:
            time.sleep(60)
            if self.llm and self.last_access > 0 and (time.time() - self.last_access) > self.ttl_seconds:
                with self._cleanup_lock:
                    if self.llm and (time.time() - self.last_access) > self.ttl_seconds:
                        app_logger.info(f"AIEngine TTL Cleanup: Unloading inactive model (idle {self.ttl_seconds}s)")
                        self.llm = None
                        self.is_ready = False
                        gc.collect()

    def _ensure_model_loaded(self):
        """Loads the model if it is not already in memory (lazy load)."""
        self.last_access = time.time()
        if not self.llm and LLAMA_AVAILABLE:
            with self._cleanup_lock:
                if not self.llm:
                    self.load_model()

    def load_model(self):
        """Loads the GGUF model into memory."""
        if not os.path.exists(self.model_path):
            app_logger.error(f"Model not found at {self.model_path}.")
            return

        try:
            app_logger.info(f"Loading GGUF model from {self.model_path}...")
            n_ctx = 4096 if "llama-3" in self.model_path.lower() else 2048
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=3,
                n_batch=512,
                use_mmap=True,
                verbose=False
            )
            self.is_ready = True
            app_logger.info(f"Model {os.path.basename(self.model_path)} loaded.")
            gc.collect()
        except Exception as e:
            app_logger.error(f"Error loading model: {e}")
            self.is_ready = False

    def generate_response(self, prompt, max_tokens=150):
        """Generates a response using the model (Raw Completion)."""
        self._ensure_model_loaded()

        if not self.is_ready:
            return "Sorry, the AI model is not available at the moment."

        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                stop=["<end_of_turn>"],
                echo=False,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1
            )
            return output['choices'][0]['text'].strip()
        except Exception as e:
            app_logger.error(f"Error generating response: {e}")
            return "Error generating response."

    def generate_response_stream(self, prompt, max_tokens=150):
        """Generates a streaming response (generator)."""
        self._ensure_model_loaded()

        if not self.is_ready:
            yield "Sorry, the AI model is not available."
            return

        try:
            stream = self.llm(
                prompt,
                max_tokens=max_tokens,
                stop=["<end_of_turn>"],
                echo=False,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                stream=True
            )
            for output in stream:
                yield output['choices'][0]['text']
        except Exception as e:
            app_logger.error(f"Error generating stream: {e}")
            yield " Error."


# Compatibility alias
GemmaEngine = AIEngine
