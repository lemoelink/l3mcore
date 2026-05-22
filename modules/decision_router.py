import os
import logging
from modules.logger import app_logger

# Fallback if transformers is not present
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
        self.classifier = None
        # OPT-2: instance-level cache avoids lru_cache memory leak on self
        self._predict_cache: dict[str, tuple] = {}
        self._cache_max_size = 128

        self._load_config()
        if self.enabled and TRANSFORMERS_AVAILABLE:
            self._load_model()

    def _load_config(self):
        # 'model_router' key in config.json (RouterFactory model mode)
        config = self.config_manager.get('model_router', {})
        self.enabled = config.get('enabled', True)
        model_path = config.get('model_path', "models/grape-route-local")
        self.model_path = os.path.abspath(model_path)
        self.confidence_threshold = config.get('confidence_threshold', 0.4)

    def _load_model(self):
        """Loads the classification model directly (no pipeline)."""
        try:
            from pathlib import Path
            model_dir = Path(self.model_path)
            app_logger.info(f"Loading Router Model from: {model_dir}...")

            # XLMRobertaTokenizer avoids tokenizer.json bug with empty chars in transformers 5.x
            try:
                from transformers import XLMRobertaTokenizer
                self._tokenizer = XLMRobertaTokenizer.from_pretrained(
                    str(model_dir), local_files_only=True
                )
            except Exception:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(model_dir), local_files_only=True, use_fast=False
                )

            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(model_dir), local_files_only=True
            )
            self._model.eval()
            self._id2label = self._model.config.id2label
            app_logger.info(f"Router Model loaded. Labels: {list(self._id2label.values())}")
        except Exception as e:
            app_logger.error(f"Error loading Router Model: {e}")
            self.enabled = False

    def _predict_cached(self, text: str) -> tuple:
        """
        Cached prediction for repeated queries.
        Returns: tuple (label, score)
        """
        if text in self._predict_cache:
            return self._predict_cache[text]

        if not self.enabled or not hasattr(self, '_model'):
            return None, 0.0

        try:
            inputs = self._tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=128, padding=True
            )
            with torch.no_grad():
                logits = self._model(**inputs).logits
            probs     = torch.softmax(logits, dim=-1)[0]
            best_idx  = int(probs.argmax())
            best_label = self._id2label[best_idx]
            best_score = float(probs[best_idx])

        except Exception as e:
            app_logger.error(f"Error in Router Predict: {e}")
            return None, 0.0

        # Store in instance cache, evict oldest entry if at capacity
        result = (best_label, best_score)
        if len(self._predict_cache) >= self._cache_max_size:
            oldest = next(iter(self._predict_cache))
            del self._predict_cache[oldest]
        self._predict_cache[text] = result
        return result

    def _clean_text(self, text: str) -> str:
        """Removes markdown formatting that confuses the classification model."""
        import re
        # Remove markdown links [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove headers
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        # Remove backticks, asterisks, and tildes globally
        text = text.replace('*', '').replace('`', '').replace('~', '')
        return text.strip()

    def predict(self, text):
        """
        Classifies the input text using the model.
        Returns: (label, score) or ('null', score) if below threshold.
        """
        clean_text = self._clean_text(text)
        if not clean_text:
            return "null", 0.0

        best_label, best_score = self._predict_cached(clean_text)

        if best_label and best_score >= self.confidence_threshold:
            app_logger.info(f"Router Prediction: '{clean_text}' -> {best_label} ({best_score:.2f})")
            return best_label, best_score
        else:
            return "null", best_score
    def clear_cache(self):
        """Clears the prediction cache to free memory."""
        self._predict_cache.clear()
        app_logger.info("Router prediction cache cleared")
