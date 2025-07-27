# stephanie/agents/inference/document_mrq_inference.py
import os

import torch
import torch.nn.functional as F

from stephanie.agents.base_agent import BaseAgent
from stephanie.evaluator.hypothesis_value_predictor import \
    HypothesisValuePredictor
from stephanie.models.score import ScoreORM
from stephanie.scoring.model.text_encoder import TextEncoder
from stephanie.scoring.mrq.model import MRQModel
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.scoring_manager import ScoringManager
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json
from stephanie.utils.model_locator import ModelLocator


class MRQInferenceAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim
        self.cfg = cfg
        self.model_path = cfg.get("model_path", "models")
        self.use_sicql = cfg.get("use_sicql_style", False)
        self.model_type = "sicql" if self.use_sicql else "mrq"
        self.evaluator = "sicql" if self.use_sicql else "mrq"
        self.target_type = cfg.get("target_type", "document")
        self.model_version = cfg.get("model_version", "v1")
        self.embedding_type = self.memory.embedding.type
        self.dimensions = cfg.get("dimensions", [])
        self.models = {}
        self.model_meta = {}
        self.tuners = {}

        if not self.dimensions:
            self.dimensions = ModelLocator.discover_dimensions(
                self.model_path, self.embedding_type, self.model_type, self.target_type
            )

        self.logger.log("MRQInferenceAgentInitialized", {
            "model_type": self.model_type,
            "target_type": self.target_type,
            "dimensions": self.dimensions,
            "device": str(self.device),
            "model_version": self.model_version,
            "embedding_type": self.embedding_type,
            "use_sicql": self.use_sicql,
        })

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
            score_results = []

            for dim, model in self.models.items():
                if self.use_sicql:
                    prompt_emb = torch.tensor(
                        self.memory.embedding.get_or_create(goal_text), device=self.device
                    ).unsqueeze(0)
                    output_emb = torch.tensor(
                        self.memory.embedding.get_or_create(scorable.text), device=self.device
                    ).unsqueeze(0)
                    result = model(prompt_emb, output_emb)


                    q_value = result["q_value"].item()
                    v_value = result["state_value"].item()
                    policy_logits = result["action_logits"].cpu().detach().numpy().tolist()

                    if isinstance(policy_logits, list) and len(policy_logits) == 1:
                        if isinstance(policy_logits[0], list):
                            # [[0.1166]] → [0.1166]
                            policy_logits = policy_logits[0]

                    print(f"Policy logits for {dim}: {policy_logits}")

                        # Calculate uncertainty (|Q - V|)
                    uncertainty = abs(q_value - v_value)
                    
                    # Calculate entropy from policy logits
                    policy_tensor = torch.tensor(policy_logits)
                    action_probs = F.softmax(policy_tensor, dim=-1)
                    entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8)).item()
                    
                    # Calculate advantage
                    advantage = q_value - v_value

                else:
                    q_value = model.predict(goal_text, scorable.text)

                meta = self.model_meta.get(dim, {"min": 0, "max": 100})
                if dim in self.tuners:
                    scaled_score = self.tuners[dim].transform(q_value)
                else:
                    normalized = torch.sigmoid(torch.tensor(q_value)).item()
                    scaled_score = normalized * (meta["max_score"] - meta["min_score"]) + meta["min_score"]

                scaled_score = max(min(scaled_score, meta["max_score"]), meta["min_score"])
                final_score = round(scaled_score, 4)
                dimension_scores[dim] = final_score
                prompt_hash = ScoreORM.compute_prompt_hash(goal_text, scorable)

                score_results.append(
                    ScoreResult(
                        dimension=dim,
                        score=final_score,
                        rationale=f"Q={round(q_value, 4)}",
                        weight=1.0,
                        q_value=q_value,
                        energy=q_value,
                        source=self.name,
                        target_type=scorable.target_type,
                        prompt_hash=prompt_hash,
                        state_value=v_value if self.use_sicql else None,
                        policy_logits=policy_logits if self.use_sicql else None,
                        uncertainty=uncertainty if self.use_sicql else None,
                        entropy=entropy if self.use_sicql else None,
                        advantage=advantage if self.use_sicql else None,
                    )
                )

                self.logger.log("MRQScoreComputed", {
                    "document_id": doc_id,
                    "dimension": dim,
                    "q_value": round(q_value, 4),
                    "final_score": final_score,
                })

            score_bundle = ScoreBundle(results={r.dimension: r for r in score_results})
            model_name = f"{self.target_type}_{self.model_type}_{self.model_version}"

            ScoringManager.save_score_to_memory(
                score_bundle, scorable, context, self.cfg, self.memory, self.logger,
                source="mrq", model_name=model_name
            )

            results.append({
                "scorable": scorable.to_dict(),
                "scores": dimension_scores,
                "score_bundle": score_bundle.to_dict(),
            })

            self.logger.log("MRQScoringFinished", {
                "document_id": doc_id,
                "scores": dimension_scores,
                "dimensions_scored": list(dimension_scores.keys()),
            })

        context[self.output_key] = results
        self.logger.log("MRQInferenceCompleted", {"total_documents_scored": len(results)})
        return context

    def load_models(self, dimensions: list):
        self.models = {}
        self.model_meta = {}
        self.tuners = {}

        for dim in dimensions:
            locator = ModelLocator(
                root_dir=self.model_path,
                embedding_type=self.embedding_type,
                model_type=self.model_type,
                target_type=self.target_type,
                dimension=dim,
                version=self.model_version,
            )

            encoder = TextEncoder(self.dim, self.hdim)
            predictor = HypothesisValuePredictor(self.dim, self.hdim)
            model = MRQModel(encoder, predictor, self.memory.embedding, device=self.device)
            model.load_weights(locator.encoder_file(), locator.model_file())
            self.models[dim] = model

            meta = load_json(locator.meta_file()) if os.path.exists(locator.meta_file()) else {"min": 0, "max": 100}
            tuner_path = locator.tuner_file()
            self.model_meta[dim] = meta

            if os.path.exists(tuner_path):
                tuner = RegressionTuner(dimension=dim)
                tuner.load(tuner_path)
                self.tuners[dim] = tuner

        self.logger.log("AllMRQModelsLoaded", {"dimensions": dimensions})
