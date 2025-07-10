import torch
import os
from sklearn.model_selection import train_test_split
from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.model.mrq_model import MRQModel
from stephanie.evaluator.text_encoder import TextEncoder
from stephanie.evaluator.hypothesis_value_predictor import HypothesisValuePredictor
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.utils.model_utils import get_model_path, discover_saved_dimensions
from stephanie.utils.file_utils import load_json
from stephanie.scoring.scoring_manager import ScoringManager
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.score_bundle import ScoreBundle


class DocumentMRQInferenceAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_path = cfg.get("model_path", "models")
        self.model_type = cfg.get("model_type", "mrq")
        self.target_type = cfg.get("target_type", "document")
        self.model_version = cfg.get("model_version", "v1")
        self.dimensions = cfg.get("dimensions", [])
        self.models = {}
        self.model_meta = {}
        self.tuners = {}  # Added tuners dict
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.dimensions:
            self.dimensions = discover_saved_dimensions(
                model_type=self.model_type, target_type=self.target_type
            )

        self.logger.log(
            "DocumentMRQInferenceAgentInitialized",
            {
                "model_type": self.model_type,
                "target_type": self.target_type,
                "dimensions": self.dimensions,
                "device": str(self.device),
            },
        )

        for dim in self.dimensions:
            model_path = get_model_path(
                self.model_path,
                self.model_type,
                self.target_type,
                dim,
                version=self.model_version,
            )
            encoder_path = f"{model_path}/{dim}_encoder.pt"
            predictor_path = f"{model_path}/{dim}.pt"
            meta_path = f"{model_path}/{dim}.meta.json"
            tuner_path = f"{model_path}/{dim}_model.tuner.json"

            self.logger.log(
                "LoadingModelPaths",
                {
                    "dimension": dim,
                    "encoder_path": encoder_path,
                    "predictor_path": predictor_path,
                },
            )
            encoder = TextEncoder()
            predictor = HypothesisValuePredictor(512, 1024)
            model = MRQModel(encoder, predictor, device=self.device)
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

        self.logger.log("AllMRQModelsLoaded", {"dimensions": self.dimensions})

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")
        results = []

        docs = context.get(self.input_key, [])

        for doc in docs:
            doc_id = doc.get("id")
            self.logger.log("MRQScoringStarted", {"document_id": doc_id})

            scorable = Scorable(
                id=doc_id, text=doc.get("text", ""), target_type=TargetType.DOCUMENT
            )

            dimension_scores = {}
            score_results = []  # For storing ScoreResult objects per dimension

            for dim, model in self.models.items():
                q_value = model.predict(goal_text, scorable.text, self.memory.embedding)

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
                        source="mrq",
                        target_type=scorable.target_type,
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
            score_bundle = ScoreBundle(results={r.dimension: r for r in score_results})

            model_name =  f"{self.target_type}_{self.model_type}_{self.model_version}"
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
