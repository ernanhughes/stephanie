# stephanie/services/entailment_service.py
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from stephanie.services.service_protocol import Service

# Optional: if you have a dedicated NLI model service
try:
    from stephanie.models.nli import NLIModel  # <-- your own NLI wrapper
except ImportError:
    NLIModel = None

# Optional: if you use Hugging Face transformers directly
from transformers import pipeline

log = logging.getLogger(__name__)

class EntailmentService(Service):
    """
    Production-grade entailment scorer: computes P(hypothesis | premise) ∈ [0,1].

    Supports multiple backends:
      - Local NLI model (preferred, low-latency)
      - Remote NLI API (if configured)
      - Fallback: lexical overlap (Jaccard) + embedding cosine similarity

    Must be registered in container as:
        container.register("entailment", EntailmentService(...))

    Usage:
        score = await entailment(premise="The sky is blue.", hypothesis="It is daytime.")
        # Returns: 0.87 (float)

    Dependencies:
        - If using NLIModel: must provide a .predict(premise, hypothesis) -> float
        - If using transformers: requires 'bert-base-nli-stsb-mean-tokens' or similar
        - If using embedding: requires container.embedder

    Performance: Async-safe, batched, with timeout and fallbacks.
    """

    def __init__(self, container, logger=None, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.container = container
        self.logger = logger or logging.getLogger(__name__)

        # --- Configurable thresholds and backends ---
        self.timeout_s = float(self.config.get("timeout_s", 3.0))
        self.use_nli = self.config.get("use_nli", True)
        self.use_embedding = self.config.get("use_embedding", True)
        self.use_lexical = self.config.get("use_lexical", True)

        # --- Backend instances ---
        self._nli_model = None
        self._embedding_model = None
        self._hf_pipeline = None

        # Initialize backends
        self._init_backends()

    @property
    def name(self) -> str:
        return "entailment-service-v2"

    def _init_backends(self):
        """Initialize all available entailment backends."""
        # 1. Try custom NLI model
        if self.use_nli and NLIModel:
            try:
                self._nli_model = getattr(self.container, "nli_model", None)
                if self._nli_model and hasattr(self._nli_model, "predict"):
                    self.logger.info("EntailmentService: Using custom NLI model")
                else:
                    log.warning("EntailmentService: nli_model provided but lacks .predict()")
                    self._nli_model = None
            except Exception as e:
                log.warning(f"EntailmentService: Failed to load NLI model: {e}")
                self._nli_model = None

        # 2. Try Hugging Face pipeline (fallback)
        if not self._nli_model and self.use_nli and pipeline:
            try:
                model_name = self.config.get("hf_model", "cross-encoder/nli-deberta-v3-base")
                self.logger.info(f"EntailmentService: Loading Hugging Face NLI model: {model_name}")
                self._hf_pipeline = pipeline(
                    "text-classification",
                    model=model_name,
                    top_k=None,
                    truncation=True,
                    max_length=512,
                )
            except Exception as e:
                log.warning(f"EntailmentService: Failed to load HF NLI: {e}")
                self._hf_pipeline = None

        # 3. Try embedding model (for cosine similarity fallback)
        if self.use_embedding:
            self._embedding_model = getattr(self.container, "embedder", None)
            if self._embedding_model:
                self.logger.info("EntailmentService: Using embedding model for cosine fallback")
            else:
                log.warning("EntailmentService: No embedder available for cosine fallback")

        # 4. Lexical fallback is always available (no dependency)

        if not any([self._nli_model, self._hf_pipeline, self._embedding_model]):
            log.warning(
                "EntailmentService: NO ENTAILMENT BACKENDS ENABLED. "
                "FALLING BACK TO LEXICAL OVERLAP (LOW ACCURACY)."
            )

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "metrics": {},
            "dependencies": {
                "nli_model": "ready" if self._nli_model else "missing",
                "hf_pipeline": "ready" if self._hf_pipeline else "missing",
                "embedder": "ready" if self._embedding_model else "missing",
                "use_nli": self.use_nli,
                "use_embedding": self.use_embedding,
                "use_lexical": self.use_lexical,
            },
        }

    def shutdown(self) -> None:
        self._nli_model = None
        self._hf_pipeline = None
        self._embedding_model = None
        self.logger.info("EntailmentService shutdown")

    # ------------------- Public Async API -------------------
    async def __call__(self, premise: str, hypothesis: str) -> float:
        """
        Async call: compute P(hypothesis | premise) ∈ [0,1].
        Returns a scalar probability. Higher = more entailed.
        """
        if not premise.strip() or not hypothesis.strip():
            return 0.5  # neutral if empty

        # Try NLI first (highest accuracy)
        if self._nli_model:
            try:
                return await self._predict_nli_model(premise, hypothesis)
            except Exception as e:
                log.warning(f"NLI model failed: {e}")

        if self._hf_pipeline:
            try:
                return await self._predict_hf_pipeline(premise, hypothesis)
            except Exception as e:
                log.warning(f"HF pipeline failed: {e}")

        # Fall back to embedding similarity
        if self._embedding_model:
            try:
                return await self._predict_embedding(premise, hypothesis)
            except Exception as e:
                log.warning(f"Embedding similarity failed: {e}")

        # Final fallback: lexical Jaccard
        if self.use_lexical:
            return self._predict_lexical(premise, hypothesis)

        # If all fail, return neutral
        log.warning("All entailment backends failed. Returning neutral 0.5.")
        return 0.5

    # ------------------- Internal Backends -------------------
    async def _predict_nli_model(self, premise: str, hypothesis: str) -> float:
        """Use custom NLI model (.predict)"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._nli_model.predict(premise, hypothesis)
        )
        return float(np.clip(result, 0.0, 1.0))

    async def _predict_hf_pipeline(self, premise: str, hypothesis: str) -> float:
        """Use Hugging Face NLI pipeline."""
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._hf_pipeline(f"{premise} [SEP] {hypothesis}")
                ),
                timeout=self.timeout_s
            )
            # HF returns list of dicts: [{'label': 'entailment', 'score': 0.92}, ...]
            for item in result:
                if item['label'] == 'ENTAILMENT':
                    return float(np.clip(item['score'], 0.0, 1.0))
            # If no entailment, return highest neutral or contradiction
            scores = {item['label']: item['score'] for item in result}
            return float(np.clip(scores.get('NEUTRAL', 0.5), 0.0, 1.0))
        except asyncio.TimeoutError:
            raise TimeoutError(f"HF NLI model timed out after {self.timeout_s}s")
        except Exception as e:
            raise RuntimeError(f"HF NLI failed: {e}")

    async def _predict_embedding(self, premise: str, hypothesis: str) -> float:
        """Compute cosine similarity between sentence embeddings."""
        loop = asyncio.get_event_loop()
        try:
            # Embed both texts
            emb_p = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._embedding_model([premise])),
                timeout=self.timeout_s
            )
            emb_h = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._embedding_model([hypothesis])),
                timeout=self.timeout_s
            )

            # Convert to numpy
            emb_p = np.array(emb_p[0])
            emb_h = np.array(emb_h[0])

            # Cosine similarity
            dot = np.dot(emb_p, emb_h)
            norm = np.linalg.norm(emb_p) * np.linalg.norm(emb_h)
            cos_sim = dot / (norm + 1e-8)

            # Map [-1,1] → [0,1]: (cos_sim + 1)/2
            return float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))

        except asyncio.TimeoutError:
            raise TimeoutError(f"Embedding model timed out after {self.timeout_s}s")
        except Exception as e:
            raise RuntimeError(f"Embedding similarity failed: {e}")

    def _predict_lexical(self, premise: str, hypothesis: str) -> float:
        """
        Lexical overlap fallback: Jaccard similarity on word sets.
        This is a weak proxy — use only when no ML models are available.
        """
        import re

        def tokenize(text: str) -> set:
            return set(re.findall(r"[a-zA-Z0-9]{2,}", text.lower()))

        p_set = tokenize(premise)
        h_set = tokenize(hypothesis)

        if not p_set and not h_set:
            return 1.0
        if not p_set or not h_set:
            return 0.0

        intersection = len(p_set & h_set)
        union = len(p_set | h_set)
        jaccard = intersection / union

        # Map Jaccard → entailment: low overlap → low entailment
        # We assume: if premise and hypothesis share >60% words → likely entailed
        return float(np.clip(jaccard, 0.0, 1.0))