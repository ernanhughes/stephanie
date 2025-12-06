# stephanie/components/information/agents/explainer_reward_trainer.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from stephanie.agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class ExplainerRewardTrainerAgent(BaseAgent):
    """
    Trains a small reward model for explainer quality using
    winner/loser data from training_events (dimension='explainer_quality').

    Input features: embedding of [paper_summary || draft].
    Target: label in {0,1} from pointwise events.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg=cfg, memory=memory, container=container, logger=logger)
        self.embedding = self.memory.embedding
        self.tes = self.memory.training_events
        self.model_dir = cfg.get("model_dir", "models/explainer_reward")

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.tes:
            log.warning("No training_events store available; skipping reward training.")
            return context

        dimension = "explainer_quality"
        min_events = int(self.cfg.get("min_events", 100))

        # 1) Load events
        events = self._fetch_events(dimension)
        if len(events) < min_events:
            log.info(f"Not enough events ({len(events)}/{min_events}) to train reward model.")
            return context

        # 2) Build dataset
        X, y = await self._build_dataset(events)
        if len(X) == 0:
            log.warning("Empty dataset after embedding; aborting training.")
            return context

        # 3) Train simple logistic regression
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)

        # 4) Persist model (simple .npz)
        self._save_model(clf)

        log.info(
            f"Trained ExplainerReward model on {len(X)} examples "
            f"(pos={int(y.sum())}, neg={int(len(y)-y.sum())})"
        )
        context["explainer_reward_trained"] = True
        context["explainer_reward_num_samples"] = len(X)
        return context

    def _fetch_events(self, dimension: str) -> List[Dict[str, Any]]:
        """
        Pull pointwise events for this dimension.
        Assumes training_events provides a fetch API; adjust to your actual interface.
        """
        try:
            return self.tes.fetch_pointwise(dimension=dimension)
        except AttributeError:
            log.warning("training_events.fetch_pointwise not implemented; returning empty list.")
            return []

    async def _build_dataset(
        self,
        events: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_list: List[np.ndarray] = []
        y_list: List[int] = []

        for ev in events:
            label = int(ev.get("label", 0))
            query = ev.get("query_text") or ""
            cand = ev.get("cand_text") or ""
            text = query + "\n\n" + cand

            try:
                emb = await self.embedding.get_or_create(text)
                if emb is None:
                    continue
                X_list.append(np.array(emb, dtype=np.float32))
                y_list.append(label)
            except Exception as e:
                log.warning(f"Failed to embed explainer text: {e}")

        if not X_list:
            return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        X = np.vstack(X_list)
        y = np.array(y_list, dtype=np.int64)
        return X, y

    def _save_model(self, clf: LogisticRegression) -> None:
        from pathlib import Path

        import joblib

        out_dir = Path(self.model_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "logreg_explainer_reward.joblib"
        joblib.dump(clf, path)
        log.info(f"Saved explainer reward model to {path}")
