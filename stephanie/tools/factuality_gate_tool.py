# stephanie/tools/factuality_gate_tool.py
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from stephanie.scoring.scorable import Scorable
from stephanie.tools.base_tool import BaseTool  # same interface as EmbeddingTool

log = logging.getLogger(__name__)

try:
    # pip install summac
    from summac.model_summac import SummaCConv  # type: ignore
except Exception:
    SummaCConv = None
    log.warning(
        "SummaC is not installed. FactualityGateTool will degrade to a no-op "
        "and simply mark factuality as unavailable."
    )


class FactualityGateTool(BaseTool):
    """
    Source-grounded factuality gate for blog posts.

    Uses SummaCConv (NLI-based factuality metric) to score how consistent the
    blog text is with the source paper text. Summac already does:
      - sentence-level segmentation
      - document-level aggregation
    which matches the “claims vs source” pattern we want.

    Expected usage in Stephanie:
      - `scorable.text` = candidate blog post
      - `context["source_text"]` = source paper text (or packed summary of it)
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        self.backend: str = cfg.get("backend", "summac")  # currently only 'summac'
        self.threshold: float = float(cfg.get("threshold", 0.75))
        self.min_chars: int = int(cfg.get("min_chars", 200))

        # Device selection (12GB VRAM is plenty for SummaCConv)
        device_cfg = cfg.get("device", "auto")
        if device_cfg == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_cfg == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._available = False
        self._backend_name: Optional[str] = None

        if self.backend == "summac":
            self._init_summac(cfg)
        else:
            log.warning(
                "Unsupported factuality backend '%s'. Falling back to no-op.",
                self.backend,
            )

    # ------------------------------------------------------------------ #
    # Backend initialisation
    # ------------------------------------------------------------------ #
    def _init_summac(self, cfg: Dict[str, Any]) -> None:
        if SummaCConv is None:
            self._available = False
            self._backend_name = None
            return

        model_name = cfg.get("summac_model_name", "vitc")
        bins = cfg.get("bins", "percentile")
        granularity = cfg.get("granularity", "sentence")
        nli_labels = cfg.get("nli_labels", "e")
        start_file = cfg.get("start_file", "default")
        agg = cfg.get("agg", "mean")

        self.model = SummaCConv(
            models=[model_name],
            bins=bins,
            granularity=granularity,
            nli_labels=nli_labels,
            device=self.device,
            start_file=start_file,
            agg=agg,
        )
        self._available = True
        self._backend_name = f"SummaCConv({model_name})"
        log.info(
            "Initialized FactualityGateTool with backend=%s device=%s",
            self._backend_name,
            self.device,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def apply(self, scorable: Scorable, context: Dict[str, Any]) -> Scorable:
        """
        Main entry point for the pipeline.

        - Reads source text from context (or scorable.meta)
        - Uses SummaCConv to compute a factual consistency score
        - Adds `factuality_gate` + `metrics.factuality_score` to scorable.meta
        """
        text = getattr(scorable, "text", None) or ""
        source_text = (
            context.get("source_text")
            or context.get("paper_text")
            or context.get("source_document")
            or getattr(getattr(scorable, "meta", {}), "get", lambda *_: None)("source_text")
        )

        if not text.strip():
            return self._attach_meta(
                scorable,
                score=None,
                passed=False,
                reason="empty_candidate_text",
            )

        if not source_text or not str(source_text).strip():
            # We *could* still run a weak heuristic here, but better to be explicit.
            return self._attach_meta(
                scorable,
                score=None,
                passed=False,
                reason="missing_source_text",
            )

        if not self._available:
            # Degrade gracefully: mark as 'not checked', but don't block the pipeline.
            return self._attach_meta(
                scorable,
                score=None,
                passed=True,
                reason="backend_unavailable",
            )

        if len(text) < self.min_chars:
            # Too short to meaningfully evaluate – auto-pass but record it.
            return self._attach_meta(
                scorable,
                score=None,
                passed=True,
                reason="too_short_for_factuality_check",
            )

        score_info = await self._score_pair(source_text=str(source_text), candidate=text)
        score = score_info["score"]
        passed = score is not None and score >= self.threshold

        return self._attach_meta(
            scorable,
            score=score,
            passed=passed,
            reason="ok",
            extra=score_info,
        )

    async def _score_pair(self, source_text: str, candidate: str) -> Dict[str, Any]:
        """
        Evaluate factual consistency between `source_text` and `candidate`.

        Returns:
            {
              "score": float in [0,1],
              "raw": Dict[str, Any]   # backend-specific extra info
            }
        """
        try:
            result = self.model.score([source_text], [candidate])
            # SummaCConv returns scores in [0,1]
            score = float(result["scores"][0])
            return {
                "score": score,
                "raw": result,
                "backend": self._backend_name,
            }
        except Exception as e:
            log.exception("Factuality scoring failed: %s", e)
            return {
                "score": None,
                "raw": {"error": str(e)},
                "backend": self._backend_name,
            }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _attach_meta(
        self,
        scorable: Scorable,
        *,
        score: Optional[float],
        passed: bool,
        reason: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Scorable:
        meta = getattr(scorable, "meta", None) or {}
        metrics = meta.get("metrics") or {}

        metrics["factuality_score"] = score
        meta["metrics"] = metrics

        meta["factuality_gate"] = {
            "score": score,
            "threshold": self.threshold,
            "pass": passed,
            "reason": reason,
            "backend": self._backend_name,
            **(extra or {}),
        }

        scorable.meta = meta
        return scorable

    # Convenience for direct, non-scorable use (e.g., quick experiments)
    def score_texts(self, source_text: str, candidate: str) -> Dict[str, Any]:
        """
        Synchronous helper for quick experiments without a Scorable/context.
        """
        if not self._available:
            return {
                "score": None,
                "pass": True,
                "reason": "backend_unavailable",
                "backend": self._backend_name,
            }

        if not candidate.strip() or not source_text.strip():
            return {
                "score": None,
                "pass": False,
                "reason": "missing_source_or_candidate",
                "backend": self._backend_name,
            }

        result = self.model.score([source_text], [candidate])
        score = float(result["scores"][0])
        return {
            "score": score,
            "pass": score >= self.threshold,
            "backend": self._backend_name,
            "raw": result,
        }
