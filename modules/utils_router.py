"""
Shared utilities for l3mcore routers.

Centralises helpers that were previously duplicated between
DecisionRouter and GenericRouter.
"""

import re
from pathlib import Path
from modules.logger import app_logger

_MARKDOWN_LINK_RE = re.compile(r'\[([^\]]+)\]\([^)]+\)')
_HEADING_RE = re.compile(r'^#+\s+', re.MULTILINE)
_LOOSE_UNDERSCORE_RE = re.compile(r'(?<!\w)_+|_+(?!\w)')


def clean_text(text: str) -> str:
    """Strips common markdown formatting before passing text to a router model."""
    text = _MARKDOWN_LINK_RE.sub(r'\1', text)
    text = _HEADING_RE.sub('', text)
    text = text.replace('*', '').replace('`', '').replace('~', '')
    text = _LOOSE_UNDERSCORE_RE.sub(' ', text)
    return text.strip()


def load_classification_model(model_path: str):
    """
    Loads a HuggingFace classification model (AutoModelForSequenceClassification)
    from a local directory. Returns (tokenizer, model, id2label) or raises on error.

    Tries XLMRobertaTokenizer first to avoid a known bug with empty chars in
    transformers 5.x, then falls back to AutoTokenizer with use_fast=False.
    """
    try:
        from transformers import (
            XLMRobertaTokenizer,
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )
        import torch  # noqa: F401 — imported here to surface the error early
    except ImportError as exc:
        raise ImportError("transformers and torch are required for classification routers") from exc

    model_dir = Path(model_path)

    try:
        tokenizer = XLMRobertaTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, use_fast=False
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True
    )
    model.eval()
    id2label = model.config.id2label

    app_logger.info(f"Classification model loaded from {model_dir}. Labels: {list(id2label.values())}")
    return tokenizer, model, id2label
