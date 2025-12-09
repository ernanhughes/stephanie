from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from stephanie.agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class ExplainerRewardScorer(BaseAgent):
    """
    Uses the trained explainer reward model to score candidate drafts.
    Score ~ probability(draft is 'good').
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg=cfg, memory=memory, container=container, logger=logger)
        self.embedding = self.memory.embedding
        self.model_dir = cfg.get("model_dir", "models/explainer_reward")
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        path = Path(self.model_dir) / "logreg_explainer_reward.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Explainer reward model not found at {path}")
        self._model = joblib.load(path)
        return self._model

    async def score(
        self,
        paper_summary: str,
        draft: str,
    ) -> float:
        model = self._load_model()
        text = paper_summary + "\n\n" + draft

        emb = await self.embedding.get_or_create(text)
        if emb is None:
            return 0.0

        X = np.array(emb, dtype=np.float32).reshape(1, -1)
        prob = float(model.predict_proba(X)[0, 1])  # P(label=1)
        return prob
