import torch
import os
from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.utils.model_utils import get_model_path, discover_saved_dimensions
from stephanie.scoring.model.ebt_model import EBTModel
from stephanie.utils.file_utils import load_json
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.scoring_manager import ScoringManager


class DocumentEBTInferenceAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_path = cfg.get("model_path", "models")
        self.model_type = cfg.get("model_type", "ebt")
        self.target_type = cfg.get("target_type", "document")
        self.model_version = cfg.get("model_version", "v1")
        self.dimensions = cfg.get("dimensions", [])
        self.models = {}
        self.model_meta = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.dimensions:
            self.dimensions = discover_saved_dimensions(
                model_type=self.model_type, target_type=self.target_type
            )

        self.logger.log(
            "DocumentEBTInferenceAgentInitialized",
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
                self.model_version,
            )
            infer_path = f"{model_path}/{dim}.pt"
            meta_path = f"{model_path}/{dim}.meta.json"

            self.logger.log("LoadingEBTModel", {"dimension": dim, "path": infer_path})
            model = self._load_model(infer_path)
            self.models[dim] = model

            if os.path.exists(meta_path):
                self.model_meta[dim] = load_json(meta_path)
            else:
                self.model_meta[dim] = {"min": 40, "max": 100}

        self.logger.log("AllEBTModelsLoaded", {"dimensions": self.dimensions})

    def _load_model(self, path):
        model = EBTModel().to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model

    def get_model_name(self) -> str:
        return f"{self.target_type}_{self.model_type}_{self.model_version}"

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")
        results = []

        for doc in context.get(self.input_key, []):
            doc_id = doc.get("id")
            self.logger.log("EBTScoringStarted", {"document_id": doc_id})

            scorable = Scorable(
                id=doc_id, text=doc.get("text", ""), target_type=TargetType.DOCUMENT
            )

            ctx_emb = torch.tensor(self.memory.embedding.get_or_create(goal_text)).to(self.device)
            doc_emb = torch.tensor(self.memory.embedding.get_or_create(scorable.text)).to(self.device)

            dimension_scores = {}
            score_results = []

            for dim, model in self.models.items():
                with torch.no_grad():
                    raw_energy = model(ctx_emb, doc_emb).squeeze().cpu().item()
                    normalized_score = torch.sigmoid(torch.tensor(raw_energy)).item()
                    meta = self.model_meta.get(dim, {"min": 40, "max": 100})
                    real_score = normalized_score * (meta["max"] - meta["min"]) + meta["min"]
                    final_score = round(real_score, 4)
                    dimension_scores[dim] = final_score

                    score_results.append(
                        ScoreResult(
                            dimension=dim,
                            score=final_score,
                            rationale=f"Energy={round(raw_energy, 4)}",
                            weight=1.0,
                            source=self.model_type,
                            target_type=scorable.target_type,
                        )
                    )

                    self.logger.log(
                        "EBTScoreComputed",
                        {
                            "document_id": doc_id,
                            "dimension": dim,
                            "raw_energy": round(raw_energy, 4),
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

            results.append({
                "scorable": scorable.to_dict(),
                "scores": dimension_scores,
                "score_bundle": score_bundle.to_dict(),
            })

            self.logger.log(
                "EBTScoringFinished",
                {
                    "document_id": doc_id,
                    "scores": dimension_scores,
                    "dimensions_scored": list(dimension_scores.keys()),
                },
            )

        context[self.output_key] = results
        self.logger.log("EBTInferenceCompleted", {"total_documents_scored": len(results)})
        return context
