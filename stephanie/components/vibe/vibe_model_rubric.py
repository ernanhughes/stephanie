# stephanie/components/vibe/vibe_model_rubric.py
from __future__ import annotations

from typing import Dict

from stephanie.components.vibe.vibe import VibeCandidate


class LegacyVibeModelRubric:
    """
    Adapter that turns the older vibe-based model into a Rubric-like scorer.
    """

    def __init__(self, model, cfg, logger):
        self.model = model           # whatever we used before
        self.cfg = cfg or {}
        self.logger = logger

    async def score(self, candidate: VibeCandidate) -> Dict[str, float]:
        """
        Returns a dict of vibe dimensions, e.g.:
        {
            "style": 0.82,
            "clarity": 0.76,
            "safety": 0.94,
        }
        """
        # 1) Extract the text / code / trace the legacy model expects
        text = candidate.candidate_state.to_text()  # or your real adaptor

        # 2) Call the old vibe model
        raw_scores = await self.model.predict(text)

        # 3) Normalize to [0, 1] and rename keys if needed
        normalized = {
            "style":   float(raw_scores.get("style", 0.0)),
            "clarity": float(raw_scores.get("clarity", 0.0)),
            "safety":  float(raw_scores.get("safety", 0.0)),
        }
        return normalized
