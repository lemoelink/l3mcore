"""
LEMoE GenericRouter
-------------------
Precision hybrid router with three-tier fallback:

  1. Embedding mode (default): encodes each keyword individually as a passage
     vector, then scores user queries against all experts using a 4-signal
     hybrid formula (max_keyword, description, mean_keyword, top3_vote).
     Final scores are softmax-normalised so the output is a real probability.

  2. Classification mode: fine-tuned BERT/RoBERTa model (AutoModelForSequenceClassification)
     for ultra-fast routing when a pre-trained classifier is available.

  3. Keyword fallback: token overlap + rapidfuzz fuzzy matching when neither
     AI method produces a score above the confidence threshold.

config.json (router section):
  model_path           -> HuggingFace repo or local path to the router model.
  router_type          -> 'embedding' | 'classification'
  categories_file      -> path to experts.json (must stay inside project dir).
  confidence_threshold -> minimum score (0-1) to accept a prediction.
  keyword_fallback     -> true/false to enable fuzzy matching tier.
  softmax_temperature  -> sharpness of softmax normalization (default 0.15).
  scoring_weights      -> per-signal weights for hybrid scoring (see _embed_score).
"""

import os
import json
import logging
from modules.logger import app_logger

# rapidfuzz for fuzzy matching in keyword fallback
try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    app_logger.warning("rapidfuzz is not installed. Keyword fallback is disabled.")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    app_logger.error("transformers/torch are not installed. GenericRouter disabled.")

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False



class GenericRouter:
    """
    Precision hybrid router for LEMoE.

    Routing pipeline:
      1. Embedding model (SentenceTransformer): multi-vector comparison
         using 4 signals: max_keyword, description, mean_keyword, top3_vote.
         Scores are softmax-normalised before threshold comparison.
      2. Classification model (BERT/RoBERTa): direct label classification.
      3. Keyword + fuzzy fallback (rapidfuzz): token overlap + partial ratio.

    Each expert in experts.json needs at minimum 15 keywords. The router
    builds one embedding vector per keyword at startup (not a concatenated soup),
    which allows precise single-term matching at inference time.
    """

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.enabled = False
        self._model = None
        self._tokenizer = None
        self._id2label = {}     # index -> model label (classification)
        # Per-expert embedding data (embedding mode)
        # Structure: {label: {kw_vecs: list[Tensor], centroid: Tensor, desc_vec: Tensor|None}}
        self.category_embeddings = {}
        self.categories = {}    # label -> full expert config dict
        self.max_experts = 15   # Default limit
        self.keyword_fallback = True
        self.router_type = "classification"
        # Scoring weights for hybrid embedding mode (see _embed_score)
        self.scoring_weights: dict = {}
        self.softmax_temperature: float = 0.15
        # OPT-2: instance-level prediction cache (avoids lru_cache memory leak on self)
        self._predict_cache: dict[str, tuple] = {}
        self._cache_max_size = 256

        self._load_config()
        self._load_categories()
        if self.enabled and TRANSFORMERS_AVAILABLE:
            self._load_model()

    # ---------------------------------------------------------------- config

    def _load_config(self):
        cfg = self.config_manager.get('router', {})
        
        raw_model_path = cfg.get('model_path', '')
        self.router_type = cfg.get('router_type', 'classification').lower()

        # If embedding mode and string is not local path, keep as is (e.g. HuggingFace repo)
        if self.router_type == 'embedding' and not os.path.exists(raw_model_path) and '/' in raw_model_path:
            self.model_path = raw_model_path
        else:
            self.model_path = os.path.abspath(raw_model_path) if raw_model_path else ''

        # SEC-5: validate categories_file path stays inside the project directory
        raw_cats = cfg.get('categories_file', 'config/experts.json')
        cats_canonical = os.path.realpath(raw_cats)
        project_root   = os.path.realpath('.')
        if not cats_canonical.startswith(project_root + os.sep) and cats_canonical != project_root:
            app_logger.error(
                f"categories_file '{raw_cats}' points outside the project directory. Rejected."
            )
            self.categories_file = 'config/experts.json'  # Fall back to safe default
        else:
            self.categories_file = raw_cats

        self.confidence_threshold = cfg.get('confidence_threshold', 0.4)
        self.keyword_fallback = cfg.get('keyword_fallback', True)
        self.enabled = bool(raw_model_path)

        # Hybrid scoring weights (configurable, must sum to 1.0)
        default_weights = {
            "max_keyword":   0.40,  # Best single keyword match (precision)
            "description":   0.30,  # Expert description semantic match (context)
            "mean_keyword":  0.20,  # Average keyword match (coverage)
            "top3_vote":     0.10,  # Fraction of top-3 keywords above 0.4 (consensus)
        }
        self.scoring_weights = cfg.get('scoring_weights', default_weights)
        self.softmax_temperature = cfg.get('softmax_temperature', 0.15)

    # ----------------------------------------------------------- categories

    def _load_categories(self):
        """
        Loads the experts.json file.
        Builds the router categories dictionary (label -> expert_config).
        """
        path = self.categories_file
        if not os.path.exists(path):
            app_logger.warning(f"experts.json not found at: {path}")
            return

        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
                
            self.max_experts = data.get('max_experts', 15)
            experts_list = data.get('experts', [])
            
            # User can define more than 15, we load all but warn
            if len(experts_list) > self.max_experts:
                app_logger.warning(f"Defined {len(experts_list)} experts, exceeding design limit of {self.max_experts}. Loading all.")
                
            for entry in experts_list:
                label = entry.get('label', '').strip()
                if label:
                    # Store full expert config
                    self.categories[label] = {
                        'config': entry,
                        'keywords': [k.lower() for k in entry.get('keywords', [])],
                    }
                    
            app_logger.info(
                f"GenericRouter: {len(self.categories)} categories loaded: "
                f"{list(self.categories.keys())}"
            )
        except Exception as e:
            app_logger.error(f"Error loading experts.json: {e}")

    def get_expert_config(self, label: str) -> dict | None:
        """Returns the full expert configuration for the given label."""
        return self.categories.get(label, {}).get('config')

    def get_model_path(self, label: str) -> str | None:
        """Legacy compatibility: returns model_path if it exists."""
        cfg = self.get_expert_config(label)
        return cfg.get('model_path') if cfg else None

    # ----------------------------------------------------------------- model

    def _load_model(self):
        """Loads the generic model trained by the user (Classifier or Embedding)."""
        if not self.model_path or not os.path.exists(self.model_path):
            # If not local, embedding mode allows downloading direct from HF
            # assuming model_path could be a HuggingFace repo like "intfloat/multilingual-e5-small".
            if self.router_type == 'embedding' and self.model_path:
                pass # Allow load from HuggingFace
            else:
                app_logger.warning(
                    f"GenericRouter: model not found at '{self.model_path}'. "
                    f"Keyword fallback will be used exclusively."
                )
                self.enabled = False
                return

        try:
            if self.router_type == 'embedding':
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    app_logger.error("sentence-transformers not installed. Cannot use router_type='embedding'.")
                    self.enabled = False
                    return
                
                app_logger.info(f"Loading Semantic Embedding Router from: {self.model_path}...")
                self._model = SentenceTransformer(self.model_path)

                # Precompute multi-vector representations per expert
                app_logger.info("Precomputing multi-vector representations per expert...")
                self._precompute_category_embeddings()
                app_logger.info(f"Multi-vector embeddings ready for {len(self.category_embeddings)} experts.")

            else:
                from pathlib import Path
                model_dir = Path(self.model_path)
                app_logger.info(f"Loading GenericRouter Classification Model from: {model_dir}...")

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
                app_logger.info(
                    f"GenericRouter Model loaded. "
                    f"Model labels: {list(self._id2label.values())}"
                )
                
        except Exception as e:
            app_logger.error(f"Error loading GenericRouter Model: {e}")
            self.enabled = False

    def _precompute_category_embeddings(self):
        """
        Builds a multi-vector representation for every expert.

        For each expert we store:
          kw_vecs   – one embedding per keyword (max 32), encoded as passages.
          centroid  – L2-normalised mean of all keyword vectors.
          desc_vec  – embedding of the expert description sentence (or None).

        This replaces the old single-vector (concatenated keyword soup) approach
        and is what enables the 4-signal hybrid scoring at inference time.
        """
        import math
        for label, info in self.categories.items():
            keywords = info.get('keywords', [])[:32]  # Cap at 32 to bound memory
            description = info.get('config', {}).get('description', '')

            # --- Individual keyword vectors ---
            if keywords:
                passages = ["passage: " + kw for kw in keywords]
                kw_vecs = self._model.encode(passages, convert_to_tensor=True, show_progress_bar=False)
            else:
                app_logger.warning(f"Expert '{label}' has no keywords — embedding quality will be poor.")
                kw_vecs = None

            # --- Centroid (normalised mean) ---
            if kw_vecs is not None:
                try:
                    import torch
                    import torch.nn.functional as F
                    centroid = kw_vecs.mean(dim=0)
                    centroid = F.normalize(centroid, dim=0)
                except Exception:
                    centroid = kw_vecs.mean(dim=0)  # no-op if torch not available
            else:
                centroid = None

            # --- Description vector ---
            desc_vec = None
            if description:
                desc_vec = self._model.encode(
                    "passage: " + description,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )

            self.category_embeddings[label] = {
                'kw_vecs':  kw_vecs,   # Tensor[N, dim] or None
                'centroid': centroid,  # Tensor[dim] or None
                'desc_vec': desc_vec,  # Tensor[dim] or None
                'n_kw':     len(keywords),
            }

    def _embed_score(self, query_vec, label_data: dict) -> float:
        """
        Computes a single precision-calibrated score for one expert.

        4 signals:
          max_keyword  – max cosine similarity between query and any single keyword.
                         High when the user says exactly one of the expert's words.
          mean_keyword – mean cosine similarity across all keywords.
                         High when the query is semantically close to the whole domain.
          description  – cosine similarity with the expert description sentence.
                         Captures the intent of the expert, not just surface words.
          top3_vote    – fraction of the top-3 keyword scores that beat 0.40.
                         Acts as a soft "consensus" vote to penalise lucky single matches.

        Final score = weighted sum of the four signals (weights from config.json).
        """
        w = self.scoring_weights
        sims_list = []

        kw_vecs = label_data.get('kw_vecs')
        centroid = label_data.get('centroid')
        desc_vec = label_data.get('desc_vec')

        # --- Keyword similarities ---
        if kw_vecs is not None:
            # Batch cosine similarity: shape [N]
            sims = util.cos_sim(query_vec, kw_vecs)[0].tolist()
            sims_list = sims

            max_kw  = max(sims)
            mean_kw = sum(sims) / len(sims)

            top3  = sorted(sims, reverse=True)[:3]
            vote  = sum(1.0 for s in top3 if s >= 0.40) / max(len(top3), 1)
        else:
            max_kw = mean_kw = vote = 0.0

        # --- Description similarity ---
        desc_sim = util.cos_sim(query_vec, desc_vec).item() if desc_vec is not None else mean_kw

        score = (
            w.get('max_keyword',  0.40) * max_kw  +
            w.get('description',  0.30) * desc_sim +
            w.get('mean_keyword', 0.20) * mean_kw  +
            w.get('top3_vote',    0.10) * vote
        )
        return score

    # ------------------------------------------------------------ inference

    def _model_predict(self, text: str) -> tuple:
        """ML model prediction. Returns (label, score)."""
        # OPT-2: instance-level cache — no lru_cache on self (memory leak)
        if text in self._predict_cache:
            return self._predict_cache[text]

        if not self._model:
            return None, 0.0

        result: tuple = (None, 0.0)
        try:
            if self.router_type == 'embedding':
                if not self.category_embeddings:
                    return None, 0.0

                # Encode query with e5 prefix
                query_vec = self._model.encode(
                    "query: " + text,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )

                # Compute hybrid score for every expert
                raw_scores: dict[str, float] = {}
                for label, label_data in self.category_embeddings.items():
                    raw_scores[label] = self._embed_score(query_vec, label_data)

                # Softmax normalization — sharpens differences between experts
                # so the winner's score becomes a real probability rather than
                # an arbitrary cosine value that varies by model.
                import math
                temp = self.softmax_temperature
                exp_scores = {l: math.exp(s / temp) for l, s in raw_scores.items()}
                total      = sum(exp_scores.values())
                norm_scores = {l: v / total for l, v in exp_scores.items()}

                best_label = max(norm_scores, key=norm_scores.__getitem__)
                best_score = norm_scores[best_label]

                app_logger.debug(
                    f"Router raw scores: { {l: f'{s:.3f}' for l,s in sorted(raw_scores.items(), key=lambda x: -x[1])[:3]} }"
                )
                result = (best_label, best_score)

            else:
                # Classic Classification
                if not self._tokenizer:
                    return None, 0.0

                inputs = self._tokenizer(
                    text, return_tensors='pt',
                    truncation=True, max_length=128, padding=True
                )
                with torch.no_grad():
                    logits = self._model(**inputs).logits
                probs   = torch.softmax(logits, dim=-1)[0]
                best_idx = int(probs.argmax())
                result  = (self._id2label[best_idx], float(probs[best_idx]))

        except Exception as e:
            app_logger.error(f"Error in GenericRouter model predict: {e}")
            return None, 0.0

        # Store in instance cache, evict oldest entry if at capacity
        if len(self._predict_cache) >= self._cache_max_size:
            oldest = next(iter(self._predict_cache))
            del self._predict_cache[oldest]
        self._predict_cache[text] = result
        return result

    def _keyword_predict(self, text: str) -> tuple:
        """
        Fallback: keyword scoring + fuzzy matching.
        Returns (label, normalized_score_0_1).
        """
        if not self.categories:
            return None, 0.0

        text_lower = text.lower()
        text_tokens = set(text_lower.split())
        scores = {}

        for label, info in self.categories.items():
            keywords = info['keywords']
            if not keywords:
                continue

            # 1. Exact token overlap
            kw_tokens = set(w for kw in keywords for w in kw.split())
            overlap = len(text_tokens & kw_tokens) / max(len(kw_tokens), 1)

            # 2. Fuzzy matching against each keyword
            fuzzy_score = 0.0
            if FUZZY_AVAILABLE:
                best_fuzz = max(
                    fuzz.token_set_ratio(text_lower, kw) / 100.0
                    for kw in keywords
                )
                fuzzy_score = best_fuzz

            # Combined score (overlap 60%, fuzzy 40%)
            combined = overlap * 0.6 + fuzzy_score * 0.4
            scores[label] = combined

        if not scores:
            return None, 0.0

        best_label = max(scores, key=scores.__getitem__)
        best_score = scores[best_label]
        return best_label, best_score

    def _clean_text(self, text: str) -> str:
        """Removes markdown formatting that confuses the embedding model."""
        import re
        # Remove markdown links [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove headers
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        # Remove backticks, asterisks, and tildes globally
        text = text.replace('*', '').replace('`', '').replace('~', '')
        # Remove underscores ONLY at word boundaries (preserves variable_names)
        text = re.sub(r'(?<!\w)_+|_+(?!\w)', ' ', text)
        return text.strip()

    def predict(self, text: str) -> tuple:
        """
        Classifies the text.
        1. Tries ML model if available.
        2. If below threshold, uses keyword fallback.
        Returns: (label, score) or ('null', score).
        """
        label, score = None, 0.0
        
        # Clean markdown formatting before routing
        clean_text = self._clean_text(text)
        if not clean_text:
            return 'null', 0.0

        # Step 1: ML model
        if self.enabled and self._model:
            label, score = self._model_predict(clean_text)
            if label and score >= self.confidence_threshold:
                # Verify label exists in experts.json
                if label in self.categories:
                    app_logger.info(
                        f"GenericRouter [model]: '{clean_text[:60]}' -> {label} ({score:.2f})"
                    )
                    return label, score
                else:
                    app_logger.warning(
                        f"GenericRouter Model predicted unknown label '{label}'. Falling back..."
                    )
            else:
                app_logger.info(f"GenericRouter [model]: score {score:.2f} below threshold ({self.confidence_threshold}).")

        # Step 2: Keyword fallback
        if self.keyword_fallback:
            k_label, k_score = self._keyword_predict(clean_text)
            if k_label and k_score >= self.confidence_threshold:
                app_logger.info(
                    f"GenericRouter [keyword]: '{clean_text[:60]}' -> {k_label} ({k_score:.2f})"
                )
                return k_label, k_score

        app_logger.info(f"GenericRouter: No match found for '{clean_text[:60]}'")
        return 'null', 0.0

    def clear_cache(self):
        self._predict_cache.clear()
        app_logger.info("GenericRouter cache cleared")
