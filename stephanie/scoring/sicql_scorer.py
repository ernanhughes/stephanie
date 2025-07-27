import os

import torch
import torch.nn.functional as F

from stephanie.models.score import ScoreORM
from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.model.in_context_q import InContextQModel
from stephanie.scoring.model.policy_head import PolicyHead
from stephanie.scoring.model.q_head import QHead
from stephanie.scoring.model.v_head import VHead
from stephanie.scoring.model.text_encoder import TextEncoder
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import load_json
from stephanie.utils.model_locator import ModelLocator


class SICQLScorer(BaseScorer):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.model_type = "sicql"
        self.embedding_type = memory.embedding.type
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim

        self.target_type = cfg.get("target_type", "document")
        self.model_path = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")

        self.models = {}
        self.model_meta = {}
        self.tuners = {}

        self.dimensions = cfg.get("dimensions", [])
        self._load_models(self.dimensions)

    def _load_models(self, dimensions):
        for dim in dimensions:
            locator = ModelLocator(
                root_dir=self.model_path,
                embedding_type=self.embedding_type,
                model_type=self.model_type,
                target_type=self.target_type,
                dimension=dim,
                version=self.version,
            )

            encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)
            q_head = QHead(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
            v_head = VHead(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
            pi_head = PolicyHead(zsa_dim=self.dim, hdim=self.hdim, num_actions=3).to(self.device)

            encoder.load_state_dict(torch.load(locator.encoder_file(), map_location=self.device))
            q_head.load_state_dict(torch.load(locator.q_head_file(), map_location=self.device))
            v_head.load_state_dict(torch.load(locator.v_head_file(), map_location=self.device))
            pi_head.load_state_dict(torch.load(locator.pi_head_file(), map_location=self.device))

            model = InContextQModel(
                encoder=encoder,
                q_head=q_head,
                v_head=v_head,
                pi_head=pi_head,
                embedding_store=self.memory.embedding,
                device=self.device,
            )
            self.models[dim] = model

            meta = load_json(locator.meta_file()) if os.path.exists(locator.meta_file()) else {"min_score": 0, "max_score": 100}
            self.model_meta[dim] = meta

            tuner_path = locator.tuner_file()
            if os.path.exists(tuner_path):
                tuner = RegressionTuner(dimension=dim)
                tuner.load(tuner_path)
                self.tuners[dim] = tuner


    def score(self, goal: dict, scorable: Scorable, dimensions: list[str]) -> ScoreBundle:
        goal_text = goal.get("goal_text")
        results = {}

        for dim in dimensions:
            model = self.models.get(dim)
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

            self.logger.log("PolicyLogits", {"dimension": dim, "logits": policy_logits})

                # Calculate uncertainty (|Q - V|)
            uncertainty = abs(q_value - v_value)
            
            # Calculate entropy from policy logits
            policy_tensor = torch.tensor(policy_logits)
            action_probs = F.softmax(policy_tensor, dim=-1)
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8)).item()
            
            # Calculate advantage
            advantage = q_value - v_value
            meta = self.model_meta.get(dim, {"min": 0, "max": 100})
            if dim in self.tuners:
                scaled_score = self.tuners[dim].transform(q_value)
            else:
                normalized = torch.sigmoid(torch.tensor(q_value)).item()
                scaled_score = normalized * (meta["max_value"] - meta["min_value"]) + meta["min_value"]

            scaled_score = max(min(scaled_score, meta["max_value"]), meta["min_value"])


            final_score = round(scaled_score, 4)
            prompt_hash = ScoreORM.compute_prompt_hash(goal_text, scorable)

            rationale = f"Q={q_value:.4f}, V={v_value:.4f}, Δ={uncertainty:.3f}, H={entropy:.3f}"

            results[dim] = ScoreResult(
                        dimension=dim,
                        score=final_score,
                        rationale=rationale,
                        weight=1.0,
                        q_value=q_value,
                        energy=q_value,
                        source=self.name,
                        target_type=scorable.target_type,
                        prompt_hash=prompt_hash,
                        state_value=v_value,
                        policy_logits=policy_logits,
                        uncertainty=uncertainty,
                        entropy=entropy,
                        advantage=advantage,
                    )
        return ScoreBundle(results=results)