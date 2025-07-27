# stephanie/scoring/svm/svm_scorer.py


import numpy as np
from joblib import load

from stephanie.models.score import ScoreORM
from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json


class SVMScorer(BaseScorer):
    def __init__(self, cfg: dict, memory, logger):
        super().__init__(cfg, memory, logger)
        self.model_type = "svm"
        self.models = {}        # dim -> (scaler, model)
        self.tuners = {}        # dim -> RegressionTuner
        self.metas = {}         # dim -> model metadata
        self._load_all_dimensions()

    def _load_all_dimensions(self):
        for dim in self.dimensions:
            locator = self.get_locator(dim) 
            self.models[dim] = (
                load(locator.scaler_file()),
                load(locator.model_file(suffix=".joblib")),
            )
            tuner = RegressionTuner(dimension=dim, logger=self.logger)
            tuner.load(locator.tuner_file())
            self.tuners[dim] = tuner
            self.metas[dim] = load_json(locator.meta_file())

    def score(self, goal: dict, scorable: Scorable, dimensions: list[str]) -> ScoreBundle:
        goal_text = goal.get("goal_text", "")
        ctx_emb = np.array(self.memory.embedding.get_or_create(goal_text))
        doc_emb = np.array(self.memory.embedding.get_or_create(scorable.text))
        input_vec = np.concatenate([ctx_emb, doc_emb]).reshape(1, -1)

        results = {}
        for dim in dimensions:
            scaler, model = self.models[dim]
            tuner = self.tuners[dim]
            meta = self.metas.get(dim, {"min_score": 0, "max_score": 100})

            scaled_input = scaler.transform(input_vec)
            raw_score = model.predict(scaled_input)[0]
            tuned_score = tuner.transform(raw_score)

            # Clip to min/max
            final_score = max(min(tuned_score, meta["max_score"]), meta["min_score"])
            results[dim] = ScoreResult(
                dimension=dim,
                score=final_score,
                rationale=f"SVM raw={round(raw_score, 4)}",
                weight=1.0,
                source=self.model_type,
                energy=0.0,
                target_type=scorable.target_type,
                prompt_hash=ScoreORM.compute_prompt_hash(goal_text, scorable),
            )

        return ScoreBundle(results=results)
