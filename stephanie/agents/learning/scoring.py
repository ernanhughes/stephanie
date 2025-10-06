# stephanie/agents/learning/scoring.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


class Scoring:
    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        # Configure scoring weights with defaults but make them configurable
        scoring_cfg = cfg.get("scoring", {})
        self.knowledge_weight = scoring_cfg.get("knowledge_weight", 0.6)
        self.clarity_weight = scoring_cfg.get("clarity_weight", 0.25)
        self.grounding_weight = scoring_cfg.get("grounding_weight", 0.15)

        # Validate weights sum to 1.0
        weights_sum = (
            self.knowledge_weight + self.clarity_weight + self.grounding_weight
        )
        if abs(weights_sum - 1.0) > 0.01:
            self.logger.warning(
                f"Scoring weights don't sum to 1.0 (knowledge={self.knowledge_weight}, "
                f"clarity={self.clarity_weight}, grounding={self.grounding_weight}). "
                "Auto-normalizing weights."
            )
            # Auto-normalize weights
            total = weights_sum
            self.knowledge_weight /= total
            self.clarity_weight /= total
            self.grounding_weight /= total

        # Other scoring parameters
        self.min_verified_len = scoring_cfg.get("min_verified_len", 250)
        self.min_grounding_verified = scoring_cfg.get(
            "min_grounding_verified", 0.45
        )

        # Knowledge scorer
        try:
            from stephanie.scoring.scorer.knowledge_scorer import \
                KnowledgeScorer

            self.knowledge = KnowledgeScorer(
                cfg.get("knowledge_scorer", {}), memory, container, logger
            )
        except Exception:
            self.knowledge = None

    def rubric_dims(self, text: str, ref: str) -> Tuple[float, float]:
        sents = [s for s in re.split(r"[.!?]\s+", (text or "").strip()) if s]
        avg_len = sum(len(s.split()) for s in sents) / max(1, len(sents))
        clarity = max(0.0, min(1.0, 1.1 - (abs(avg_len - 22) / 22)))

        def toks(t):
            return set(re.findall(r"\b\w+\b", (t or "").lower()))

        inter = len(toks(text) & toks(ref))
        grounding = max(0.0, min(1.0, inter / max(30, len(toks(ref)) or 1)))
        return clarity, grounding

    def weaknesses(self, summary: str, ref: str) -> List[str]:
        out = []
        if len(summary or "") < 400:
            out.append("too short / thin detail")
        if (
            "we propose" in (ref or "").lower()
            and "we propose" not in (summary or "").lower()
        ):
            out.append("misses core claim language")
        if (summary or "").count("(") != (summary or "").count(")"):
            out.append("formatting/parens issues")
        return out

    def score_summary(
        self,
        text: str,
        paper: Dict[str, Any],
        section: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Score summary with configurable weights"""
        clarity, grounding = self.rubric_dims(
            text, section.get("section_text", "")
        )
        goal_text = (
            f"{paper.get('title', '')}\n\n{paper.get('abstract', '')}"
        )
        meta = {"text_len_norm": min(1.0, len(text) / 2000.0)}

            # Support different scoring API signatures
        p, comps = self.knowledge.model.predict(
            goal_text, text, meta=meta, return_components=True
        )
        knowledge = float((comps or {}).get("probability", p))

        # Calculate overall score using configurable weights
        overall = (
            self.knowledge_weight * knowledge
            + self.clarity_weight * clarity
            + self.grounding_weight * grounding
        )

        # Log scoring weights for traceability
        if self.logger:
            self.logger.log(
                "ScoringWeights",
                {
                    "knowledge_weight": self.knowledge_weight,
                    "clarity_weight": self.clarity_weight,
                    "grounding_weight": self.grounding_weight,
                    "knowledge_score": knowledge,
                    "clarity_score": clarity,
                    "grounding_score": grounding,
                    "overall_score": overall,
                },
            )

        return {
            "overall": overall,
            "knowledge_score": knowledge,
            "clarity": clarity,
            "grounding": grounding,
            "weaknesses": self.weaknesses(
                text, section.get("section_text", "")
            ),
            **(comps or {}),
        }

    def score_candidate(
        self, text: str, section_text: str
    ) -> Dict[str, float]:
        """Score candidate with configurable weights"""
        dims = self.score_summary(
            text,
            {"title": "", "abstract": ""},
            {"section_text": section_text},
            {},
        )

        k, c, g = dims["knowledge_score"], dims["clarity"], dims["grounding"]
        overall = (
            self.knowledge_weight * k
            + self.clarity_weight * c
            + self.grounding_weight * g
        )

        verified = (g >= self.min_grounding_verified) and (
            len(text) >= self.min_verified_len
        )

        return {
            "k": k,
            "c": c,
            "g": g,
            "overall": overall,
            "verified": bool(verified),
            "weights": {
                "knowledge": self.knowledge_weight,
                "clarity": self.clarity_weight,
                "grounding": self.grounding_weight,
                "min_grounding_verified": self.min_grounding_verified,
                "min_verified_len": self.min_verified_len,
            },
        }
