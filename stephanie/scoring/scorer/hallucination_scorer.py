# stephanie/scoring/hallucination/hallucination_scorer.py

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorer.base_scorer import BaseScorer

log = logging.getLogger(__name__)


class HallucinationScorer(BaseScorer):
    """
    Geometric Hallucination Energy scorer.

    Computes projection residual of response embedding onto
    span of historical/context embeddings.

    Lower score = better grounding.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        self.model_type = "hallucination"
        self.dimension_name = cfg.get(
            "dimension_name", "hallucination_energy"
        )

        # Optional rank truncation for SVD
        self.max_rank = int(cfg.get("max_rank", 32))
        self.eps = float(cfg.get("eps", 1e-8))

    # ---------------------------------------------------------
    # Core scoring logic
    # ---------------------------------------------------------
    def _score_core(
        self,
        context: Dict[str, Any],
        scorable: Scorable,
        dimensions: List[str],
    ) -> ScoreBundle:

        # Only compute if requested
        if self.dimension_name not in dimensions:
            return ScoreBundle(results={})

        # -----------------------------------------------------
        # 1️⃣ Get response embedding
        # -----------------------------------------------------
        resp_vec = np.asarray(
            self.memory.embedding.get_or_create(scorable.text),
            dtype=np.float32,
        )

        if np.linalg.norm(resp_vec) < self.eps:
            score = 1.0  # Degenerate embedding → treat as maximal energy
            return self._build_bundle(score, resp_vec, [], rank=0)

        # -----------------------------------------------------
        # 2️⃣ Build evidence/history matrix
        # -----------------------------------------------------
        history_texts = self._extract_history(context)

        history_vecs = []
        for txt in history_texts:
            try:
                vec = np.asarray(
                    self.memory.embedding.get_or_create(txt),
                    dtype=np.float32,
                )
                if np.linalg.norm(vec) > self.eps:
                    history_vecs.append(vec)
            except Exception:
                continue

        if not history_vecs:
            # No grounding reference → maximal energy
            score = 1.0
            return self._build_bundle(score, resp_vec, [], rank=0)

        E = np.stack(history_vecs, axis=0)

        # -----------------------------------------------------
        # 3️⃣ Compute orthonormal basis via truncated SVD
        # -----------------------------------------------------
        try:
            U, S, Vt = np.linalg.svd(E, full_matrices=False)
            rank = min(self.max_rank, U.shape[1])
            U_r = U[:, :rank]
        except np.linalg.LinAlgError:
            score = 1.0
            return self._build_bundle(score, resp_vec, history_vecs, rank=0)

        # -----------------------------------------------------
        # 4️⃣ Projection
        # -----------------------------------------------------
        projection = U_r @ (U_r.T @ resp_vec)
        residual_vec = resp_vec - projection

        residual_norm = float(np.linalg.norm(residual_vec))
        resp_norm = float(np.linalg.norm(resp_vec))

        score = residual_norm / (resp_norm + self.eps)

        # Clip for safety
        score = float(np.clip(score, 0.0, 1.0))

        return self._build_bundle(
            score,
            resp_vec,
            history_vecs,
            rank=rank,
            residual_norm=residual_norm,
            resp_norm=resp_norm,
        )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _extract_history(self, context: Dict[str, Any]) -> List[str]:
        """
        Extract conversation history texts for grounding.
        Customize based on your context structure.
        """

        history = []

        # Common pattern: context["history"] = list of messages
        messages = context.get("history", [])

        for msg in messages:
            text = msg.get("text") if isinstance(msg, dict) else None
            if text:
                history.append(text)

        # Also include goal text if present
        goal = context.get("goal", {})
        goal_text = goal.get("goal_text")
        if goal_text:
            history.append(goal_text)

        return history

    def _build_bundle(
        self,
        score: float,
        resp_vec: np.ndarray,
        history_vecs: List[np.ndarray],
        rank: int,
        residual_norm: float | None = None,
        resp_norm: float | None = None,
    ) -> ScoreBundle:

        attributes = {
            "history_count": len(history_vecs),
            "rank": rank,
        }

        if residual_norm is not None:
            attributes["residual_norm"] = residual_norm

        if resp_norm is not None:
            attributes["response_norm"] = resp_norm

        result = ScoreResult(
            dimension=self.dimension_name,
            score=score,
            rationale=f"Projection residual energy={round(score, 4)}",
            weight=1.0,
            source=self.model_type,
            attributes=attributes,
        )

        return ScoreBundle(results={self.dimension_name: result})
