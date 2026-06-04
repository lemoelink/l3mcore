import os
import threading
from collections import OrderedDict
from modules.logger import app_logger
from modules.utils_router import clean_text, load_classification_model

try:
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    app_logger.error("transformers/torch are not installed. DecisionRouter disabled.")


class DecisionRouter:
    """
    Discriminative Semantic Router using Transformers (Classification Pipeline).
    Classifies user intent directly using the model's labels.
    """

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.enabled = False
        self._tokenizer = None
        self._model = None
        self._id2label = {}
        # LRU cache: OrderedDict keeps insertion order; oldest entry is evicted first.
        self._predict_cache: OrderedDict[str, tuple] = OrderedDict()
        self._cache_max_size = 128
        self._cache_lock = threading.Lock()

        self._load_config()
        if self.enabled and TRANSFORMERS_AVAILABLE:
            self._load_model()

    def _load_config(self):
        config = self.config_manager.get('model_router', {})
        self.enabled = config.get('enabled', True)
        model_path = config.get('model_path', "models/grape-route-local")
        self.model_path = os.path.abspath(model_path)
        self.confidence_threshold = config.get('confidence_threshold', 0.4)

    def _load_model(self):
        try:
            self._tokenizer, self._model, self._id2label = load_classification_model(self.model_path)
        except Exception as e:
            app_logger.error(f"Error loading Router Model: {e}")
            self.enabled = False

    def _predict_cached(self, text: str) -> tuple:
        with self._cache_lock:
            if text in self._predict_cache:
                # Move to end to mark as recently used
                self._predict_cache.move_to_end(text)
                return self._predict_cache[text]

        if not self.enabled or self._model is None:
            return None, 0.0

        try:
            inputs = self._tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=128, padding=True
            )
            with torch.no_grad():
                logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            best_idx = int(probs.argmax())
            result = (self._id2label[best_idx], float(probs[best_idx]))
        except Exception as e:
            app_logger.error(f"Error in Router Predict: {e}")
            return None, 0.0

        with self._cache_lock:
            if len(self._predict_cache) >= self._cache_max_size:
                self._predict_cache.popitem(last=False)  # evict LRU (oldest)
            self._predict_cache[text] = result
        return result

    def predict(self, text: str) -> tuple:
        """
        Classifies the input text using the model.
        Returns (label, score) or ('null', score) if below threshold.
        """
        clean = clean_text(text)
        if not clean:
            return "null", 0.0

        best_label, best_score = self._predict_cached(clean)

        if best_label and best_score >= self.confidence_threshold:
            app_logger.info(f"Router Prediction: '{clean}' -> {best_label} ({best_score:.2f})")
            return best_label, best_score
        return "null", best_score

    def clear_cache(self):
        with self._cache_lock:
            self._predict_cache.clear()
        app_logger.info("Router cache cleared.")
