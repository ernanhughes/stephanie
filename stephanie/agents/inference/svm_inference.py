# stephanie/agents/inference/document_svm_inference.py
import os

import numpy as np
from joblib import load

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.scoring_manager import ScoringManager
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json
from stephanie.utils.model_utils import (discover_saved_dimensions,
                                         get_svm_file_paths)
from stephanie.models.score import ScoreORM

class SVMInferenceAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_path = cfg.get("model_path", "models")
        self.model_type = cfg.get("model_type", "svm")
        self.target_type = cfg.get("target_type", "document")
        self.model_version = cfg.get("model_version", "v1")
        self.dimensions = cfg.get("dimensions", [])
        self.models = {}
        self.model_meta = {}
        self.tuners = {}

        if not self.dimensions:
            self.dimensions = discover_saved_dimensions(
                model_type=self.model_type, target_type=self.target_type
            )

        self.logger.log(
            "SVMInferenceInitialized", {"dimensions": self.dimensions}
        )

        for dim in self.dimensions:
            paths = get_svm_file_paths(
                self.model_path,
                self.model_type,
                self.target_type,
                dim,
                self.model_version,
            )
            scaler_path = paths["scaler"]
            model_file = paths["model"]
            meta_path = paths["meta"]

            self.logger.log("LoadingSVMModel", {"dimension": dim, "model": model_file})

            self.models[dim] = (load(scaler_path), load(model_file))
            self.model_meta[dim] = (
                load_json(meta_path)
                if os.path.exists(meta_path)
                else {"min_score": 0, "max_score": 100}
            )
            self.tuners[dim] = RegressionTuner(dimension=dim, logger=logger)
            self.tuners[dim].load(paths["tuner"])

    def get_model_name(self) -> str:
        return f"{self.target_type}_{self.model_type}_{self.model_version}"

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")
        results = []

        for doc in context.get(self.input_key, []):
            doc_id = doc.get("id")
            self.logger.log("SVMScoringStarted", {"document_id": doc_id})

            scorable = Scorable(
                id=doc_id, text=doc.get("text", ""), target_type=TargetType.DOCUMENT
            )

            ctx_emb = self.memory.embedding.get_or_create(goal_text)
            doc_emb = self.memory.embedding.get_or_create(scorable.text)
            feature = np.array(ctx_emb + doc_emb).reshape(1, -1)

            dimension_scores = {}
            score_results = []

            for dim, (scaler, model) in self.models.items():
                X_scaled = scaler.transform(feature)
                raw_score = model.predict(X_scaled)[0]
                tuned_score = self.tuners[dim].transform(raw_score)

                meta = self.model_meta.get(dim, {"min_score": 0, "max_score": 100})
                min_s, max_s = meta["min_score"], meta["max_score"]
                final_score = max(min(tuned_score, max_s), min_s)
                final_score = round(final_score, 4)
                dimension_scores[dim] = final_score

                score_results.append(
                    ScoreResult(
                        dimension=dim,
                        score=final_score,
                        rationale=f"SVM raw={round(raw_score, 4)}",
                        weight=1.0,
                        source=self.model_type,
                        target_type=scorable.target_type,
                        prompt_hash = ScoreORM.compute_prompt_hash(goal_text, scorable)
                    )
                )

                self.logger.log(
                    "SVMScoreComputed",
                    {
                        "document_id": doc_id,
                        "dimension": dim,
                        "raw_score": round(raw_score, 4),
                        "tuned_score": round(tuned_score, 4),
                        "final_score": final_score,
                    },
                )

            score_bundle = ScoreBundle(results={r.dimension: r for r in score_results})

            ScoringManager.save_score_to_memory(
                score_bundle,
                scorable,
                context,
                self.cfg,
                self.memory,
                self.logger,
                source=self.model_type,
                model_name=self.get_model_name(),
            )

            results.append(
                {
                    "scorable": scorable.to_dict(),
                    "scores": dimension_scores,
                    "score_bundle": score_bundle.to_dict(),
                }
            )

            self.logger.log(
                "SVMScoringFinished",
                {
                    "document_id": doc_id,
                    "scores": dimension_scores,
                    "dimensions_scored": list(dimension_scores.keys()),
                },
            )

        context[self.output_key] = results
        self.logger.log(
            "SVMInferenceCompleted", {"total_documents_scored": len(results)}
        )
        return context
