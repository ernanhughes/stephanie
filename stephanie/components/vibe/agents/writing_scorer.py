from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.vibe.rubrics.writing_quality import \
    build_writing_quality_rubric
from stephanie.components.vibe.writing_types import WritingScore

log = logging.getLogger(__name__)


class RubricEvaluatorService:
    """
    Thin abstraction over whatever rubric evaluation pipeline you use.

    You can replace this with:
      - a call into your existing RubricEvaluatorAgent,
      - a ScoringService method,
      - or a direct LLM call with a rubric-aware prompt.

    Contract:
      evaluate(rubric, text, meta) -> Dict[criterion_name, score_float]
    """

    def __init__(self, container: Any, logger: Any):
        self.container = container
        self.logger = logger

        # Example: look up a shared service from your DI container
        # self._evaluator = container.get("rubric_evaluator_service")

    async def evaluate(
        self,
        rubric,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Run the rubric over `text` and return scores per criterion.

        This is intentionally generic; wire it to your real evaluator.
        """
        if not text or not text.strip():
            return {c.name: 0.0 for c in rubric.criteria}

        # --- PSEUDOCODE: replace with your actual rubric evaluation ---
        # Example if you already have a rubric evaluator agent/service:
        #
        # result = await self._evaluator.evaluate(
        #     rubric=rubric,
        #     target_text=text,
        #     meta=meta or {},
        # )
        # return result.scores  # {criterion_name: score_float}
        #
        # For now, we just log and raise so you wire it correctly.
        self.logger.error(
            "RubricEvaluatorService.evaluate() is not wired yet. "
            "Please connect it to your real rubric evaluation pipeline."
        )
        raise NotImplementedError("Wire RubricEvaluatorService to your evaluator.")
        # ----------------------------------------------------------------


class WritingScorerAgent(BaseAgent):
    """
    Agent that scores a piece of writing (e.g., blog section, research summary)
    using the writing_quality_v1 rubric and returns a structured WritingScore.

    Expected context inputs:
      - context["text"]: str               # the writing to score
        OR
      - context["sections"]: List[str]     # optional: score each and average

      - context["writing_meta"] (optional): Dict[str, Any]
          e.g. {"goal": "...", "source_paper": "...", "section_id": ...}

    Outputs:
      - context["writing_score"]: Dict[str, Any]  # serialized WritingScore
      - context["writing_score_obj"]: WritingScore (if you want the object)
      - context["writing_rubric_scores"]: Dict[str, float]  # raw per-criterion

    You can plug this into Blossom, Nexus pipelines, or Arena as a scorer.
    """

    def __init__(self, cfg: Dict[str, Any], memory: Any, container: Any, logger: Any):
        super().__init__(cfg, memory, container, logger)

        self.cfg = cfg or {}
        self.rubric = build_writing_quality_rubric()
        self.rubric_evaluator = RubricEvaluatorService(container, logger)

        # Weights for overall score aggregation
        self.weights: Dict[str, float] = self.cfg.get(
            "weights",
            {
                "clarity": 0.2,
                "structure": 0.15,
                "technical_correctness": 0.25,
                "depth": 0.2,
                "actionability": 0.1,
                "vibe": 0.1,
            },
        )

    # ---------- Public entry point ----------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        text = self._extract_text(context)
        meta = context.get("writing_meta") or {}

        if not text:
            self.logger.warning("WritingScorerAgent: no text provided; returning zeros.")
            empty_score = self._empty_score()
            context["writing_score"] = empty_score.to_dict()
            context["writing_score_obj"] = empty_score
            context["writing_rubric_scores"] = {}
            return context

        rubric_scores = await self.rubric_evaluator.evaluate(
            rubric=self.rubric,
            text=text,
            meta=meta,
        )

        writing_score = self._build_writing_score(rubric_scores)

        context["writing_rubric_scores"] = rubric_scores
        context["writing_score"] = writing_score.to_dict()
        context["writing_score_obj"] = writing_score

        return context

    # ---------- Internal helpers ----------

    def _extract_text(self, context: Dict[str, Any]) -> str:
        """
        Extracts the text to score from the context.

        v1 behavior:
          - if context["text"] exists and is non-empty, use that.
          - elif context["sections"] is a non-empty list of str, join them.
        """
        text = context.get("text")
        if isinstance(text, str) and text.strip():
            return text

        sections = context.get("sections") or []
        if isinstance(sections, (list, tuple)):
            joined = "\n\n".join(s for s in sections if isinstance(s, str) and s.strip())
            return joined

        return ""

    def _empty_score(self) -> WritingScore:
        return WritingScore(
            clarity=0.0,
            structure=0.0,
            technical_correctness=0.0,
            depth=0.0,
            actionability=0.0,
            vibe=0.0,
            overall=0.0,
            breakdown={},
        )

    def _build_writing_score(self, rubric_scores: Dict[str, float]) -> WritingScore:
        """
        Map rubric criterion scores -> WritingScore fields and compute overall.
        rubric_scores is expected to be keyed by criterion.name
        (clarity, structure, technical_correctness, depth, actionability, vibe).
        """
        clarity = float(rubric_scores.get("clarity", 0.0))
        structure = float(rubric_scores.get("structure", 0.0))
        technical = float(rubric_scores.get("technical_correctness", 0.0))
        depth = float(rubric_scores.get("depth", 0.0))
        actionability = float(rubric_scores.get("actionability", 0.0))
        vibe = float(rubric_scores.get("vibe", 0.0))

        overall = (
            self.weights["clarity"] * clarity
            + self.weights["structure"] * structure
            + self.weights["technical_correctness"] * technical
            + self.weights["depth"] * depth
            + self.weights["actionability"] * actionability
            + self.weights["vibe"] * vibe
        )

        breakdown = {
            "clarity": clarity,
            "structure": structure,
            "technical_correctness": technical,
            "depth": depth,
            "actionability": actionability,
            "vibe": vibe,
        }

        return WritingScore(
            clarity=clarity,
            structure=structure,
            technical_correctness=technical,
            depth=depth,
            actionability=actionability,
            vibe=vibe,
            overall=overall,
            breakdown=breakdown,
        )
