# stephanie/agents/inference/document_mrq_inference.py
import os

import torch

from stephanie.agents.base_agent import BaseAgent
from stephanie.evaluator.hypothesis_value_predictor import \
    HypothesisValuePredictor
from stephanie.models.score import ScoreORM
from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.mrq.model import MRQModel
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.scoring_manager import ScoringManager
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json
from stephanie.utils.model_utils import (discover_saved_dimensions,
                                         get_model_path)

from stephanie.scoring.scorable_factory import ScorableFactory


class MRQInferenceAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim
        self.model_path = cfg.get("model_path", "models")
        self.model_type = "mrq"
        self.evaluator = "mrq"
        self.target_type = cfg.get("target_type", "document")
        self.model_version = cfg.get("model_version", "v1")
        self.embedding_type = self.memory.embedding.type
        self.dimensions = cfg.get("dimensions", [])
        self.models = {}
        self.model_meta = {}
        self.tuners = {}  # Added tuners dict
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if not self.dimensions:
            self.dimensions = discover_saved_dimensions(
                model_type=self.model_type, target_type=self.target_type
            )

        self.logger.log(
            "MRQInferenceAgentInitialized",
            {
                "model_type": self.model_type,
                "target_type": self.target_type,
                "dimensions": self.dimensions,
                "device": str(self.device),
            },
        )


    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")
        results = []

        docs = context.get(self.input_key, [])
        self.load_models(self.dimensions)
        for doc in docs:
            doc_id = doc.get("id")
            self.logger.log("MRQScoringStarted", {"document_id": doc_id})

            scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)

            dimension_scores = {}
            score_results = []  # For storing ScoreResult objects per dimension

            for dim, model in self.models.items():
                q_value = model.predict(goal_text, scorable.text)

                if dim in self.tuners:
                    scaled_score = self.tuners[dim].transform(q_value)
                else:
                    meta = self.model_meta.get(dim, {"min": 0, "max": 100})
                    normalized = torch.sigmoid(torch.tensor(q_value)).item()
                    scaled_score = (
                        normalized * (meta["max"] - meta["min"]) + meta["min"]
                    )

                final_score = round(scaled_score, 4)
                dimension_scores[dim] = final_score

                # Create ScoreResult
                score_results.append(
                    ScoreResult(
                        dimension=dim,
                        score=final_score,
                        rationale=f"Q={round(q_value, 4)}",  # Optional
                        weight=1.0,
                        source=self.name,
                        target_type=scorable.target_type,
                        prompt_hash = ScoreORM.compute_prompt_hash(goal_text, scorable)
                    )
                )

                self.logger.log(
                    "MRQScoreComputed",
                    {
                        "document_id": doc_id,
                        "dimension": dim,
                        "q_value": round(q_value, 4),
                        "final_score": final_score,
                    },
                )

            # Create ScoreBundle
            score_bundle = ScoreBundle(
                results={r.dimension: r for r in score_results}
            )

            model_name = (
                f"{self.target_type}_{self.model_type}_{self.model_version}"
            )
            # Save score bundle to memory via ScoringManager
            ScoringManager.save_score_to_memory(
                score_bundle,
                scorable,
                context,
                self.cfg,
                self.memory,
                self.logger,
                source="mrq",
                model_name=model_name,
            )

            # Store results for this document in output
            results.append(
                {
                    "scorable": scorable.to_dict(),
                    "scores": dimension_scores,
                    "score_bundle": score_bundle.to_dict(),
                }
            )

            self.logger.log(
                "MRQScoringFinished",
                {
                    "document_id": doc_id,
                    "scores": dimension_scores,
                    "dimensions_scored": list(dimension_scores.keys()),
                },
            )

        context[self.output_key] = results
        self.logger.log(
            "MRQInferenceCompleted", {"total_documents_scored": len(results)}
        )
        return context

    def load_models(self, dimensions: list):
        """
        Load MRQ models for the specified dimensions.
        This is called during initialization to ensure models are ready for scoring.
        """
        self.models = {}
        self.model_meta = {}
        self.tuners = {}

        for dim in dimensions:
            model_path = get_model_path(
                self.model_path,
                self.model_type,
                self.target_type,
                dim,
                version=self.model_version,
                embedding_type=self.embedding_type
            )
            encoder_path = f"{model_path}/{dim}_encoder.pt"
            predictor_path = f"{model_path}/{dim}.pt"
            meta_path = f"{model_path}/{dim}.meta.json"
            tuner_path = f"{model_path}/{dim}_model.tuner.json"

            encoder = TextEncoder(self.dim, self.hdim)
            predictor = HypothesisValuePredictor(self.dim, self.hdim)
            model = MRQModel(
                encoder, predictor, self.memory.embedding, device=self.device
            )
            model.load_weights(encoder_path, predictor_path)
            self.models[dim] = model

            if os.path.exists(meta_path):
                self.model_meta[dim] = load_json(meta_path)
            else:
                self.model_meta[dim] = {"min": 0, "max": 100}

            if os.path.exists(tuner_path):
                tuner = RegressionTuner(dimension=dim)
                tuner.load(tuner_path)
                self.tuners[dim] = tuner

        self.logger.log("AllMRQModelsLoaded", {"dimensions": dimensions})

    def score(self, context: dict, scorable: Scorable) -> dict:
        """
        Score a single document using all loaded MRQ models and return a dict of dimension -> score.
        This does NOT save to memory or return a ScoreBundle — just raw scores.
        """
        goal_text = context["goal"]["goal_text"]
        dimension_scores = {}

        for dim, model in self.models.items():
            q_value = model.predict(goal_text, scorable.text)

            if dim in self.tuners:
                scaled_score = self.tuners[dim].transform(q_value)
            else:
                meta = self.model_meta.get(dim, {"min": 0, "max": 100})
                normalized = torch.sigmoid(torch.tensor(q_value)).item()
                scaled_score = normalized * (meta["max"] - meta["min"]) + meta["min"]

            final_score = round(scaled_score, 4)
            dimension_scores[dim] = final_score

            if self.logger:
                self.logger.log(
                    "MRQScoreComputed",
                    {
                        "scorble": scorable.to_dict(),
                        "dimension": dim,
                        "q_value": round(q_value, 4),
                        "final_score": final_score,
                    },
                )

        return dimension_scores
