import os

import torch
import torch.nn.functional as F

from stephanie.models.score import ScoreORM
from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.model.in_context_q import InContextQModel
from stephanie.scoring.model.policy_head import PolicyHead
from stephanie.scoring.model.q_head import QHead
from stephanie.scoring.model.text_encoder import TextEncoder
from stephanie.scoring.model.v_head import VHead
from stephanie.scoring.scorable import Scorable
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
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
        self.return_zsa = cfg.get("return_zsa", False)

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
            pi_head = PolicyHead(
                zsa_dim=self.dim, hdim=self.hdim, num_actions=3
            ).to(self.device)

            encoder.load_state_dict(
                torch.load(locator.encoder_file(), map_location=self.device)
            )
            q_head.load_state_dict(
                torch.load(locator.q_head_file(), map_location=self.device)
            )
            v_head.load_state_dict(
                torch.load(locator.v_head_file(), map_location=self.device)
            )
            pi_head.load_state_dict(
                torch.load(locator.pi_head_file(), map_location=self.device)
            )

            model = InContextQModel(
                encoder=encoder,
                q_head=q_head,
                v_head=v_head,
                pi_head=pi_head,
                embedding_store=self.memory.embedding,
                device=self.device,
            )
            self.models[dim] = model

            meta = (
                load_json(locator.meta_file())
                if os.path.exists(locator.meta_file())
                else {"min_value": 0, "max_value": 100}
            )
            self.model_meta[dim] = meta

            tuner_path = locator.tuner_file()
            if os.path.exists(tuner_path):
                tuner = RegressionTuner(dimension=dim)
                tuner.load(tuner_path)
                self.tuners[dim] = tuner

    def score(
        self, goal: dict, scorable: Scorable, dimensions: list[str]
    ) -> ScoreBundle:
        goal_text = goal.get("goal_text")
        results = {}

        for dim in dimensions:
            model = self.models.get(dim)
            prompt_emb_np = self.memory.embedding.get_or_create(goal_text)
            output_emb_np = self.memory.embedding.get_or_create(scorable.text)
            prompt_emb = torch.tensor(
                prompt_emb_np, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
            output_emb = torch.tensor(
                output_emb_np, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
            with torch.no_grad():
                model_outputs = model(prompt_emb, output_emb)
                # Standard outputs
                q_value_tensor = model_outputs[
                    "q_value"
                ]  # Shape: (1, 1) or (1,)
                v_value_tensor = model_outputs["state_value"]
                policy_logits_tensor = model_outputs[
                    "action_logits"
                ]  # Shape: (1, num_actions) or (num_actions,)

                # Extract scalar values
                q_value = q_value_tensor.squeeze().item()
                v_value = v_value_tensor.squeeze().item()
                # Handle policy_logits shape variations
                policy_logits_np = policy_logits_tensor.cpu().detach().numpy()
                if policy_logits_np.ndim > 1:
                    policy_logits = (
                        policy_logits_np.flatten().tolist()
                    )  # Shape: (num_actions,)
                else:
                    policy_logits = (
                        policy_logits_np.tolist()
                    )  # Shape: (num_actions,) already

                # Calculate metrics
                uncertainty = abs(q_value - v_value)
                policy_tensor = torch.tensor(policy_logits)
                action_probs = F.softmax(policy_tensor, dim=-1)
                entropy = -torch.sum(
                    action_probs * torch.log(action_probs + 1e-8)
                ).item()
                advantage = q_value - v_value

                zsa_tensor = None
                if self.return_zsa:
                    zsa_tensor = model.encoder(prompt_emb, output_emb)

            meta = self.model_meta.get(dim, {"min_value": 0, "max_value": 100})
            # Ensure meta has min/max values for scaling logic
            min_val = meta.get("min_value", meta.get("min_value", 0))
            max_val = meta.get("max_value", meta.get("max_value", 100))
            if dim in self.tuners:
                scaled_score = self.tuners[dim].transform(q_value)
            else:
                normalized = torch.sigmoid(torch.tensor(q_value)).item()
                scaled_score = normalized * (max_val - min_val) + min_val
            scaled_score = max(min(scaled_score, max_val), min_val)
            final_score = round(scaled_score, 4)

            # Rationale and hash
            rationale = f"Q={q_value:.4f}, V={v_value:.4f}, Δ={uncertainty:.3f}, H={entropy:.3f}"
            prompt_hash = ScoreORM.compute_prompt_hash(goal_text, scorable)

            # --- Create ScoreResult with optional zsa ---
            attributes = {
                "q_value": q_value,
                "energy": q_value,  # Keeping energy as q_value as in original
                "state_value": v_value,
                "policy_logits": policy_logits,
                "uncertainty": uncertainty,
                "entropy": entropy,
                "advantage": advantage,
                "prompt_hash": prompt_hash,
            }
            # Add zsa if it was calculated and return_zsa is True
            if self.return_zsa and zsa_tensor is not None:
                attributes["zsa"] = (
                    zsa_tensor  # Pass tensor directly (ScoreResult should handle)
                )
                rationale += f", zsa_dim={zsa_tensor.shape[-1] if zsa_tensor.ndim > 0 else 1}"

            results[dim] = ScoreResult(
                dimension=dim,
                source=self.name,
                score=final_score,
                rationale=rationale,
                weight=1.0,
                attributes=attributes,
            )

        return ScoreBundle(results=results)

    def __call__(self, goal: dict, scorable: Scorable, dimension: str):
        """
        Direct model access for a single dimension. Useful for quick inference.

        Args:
            scorable: The object to score (must have `.text`)
            dimension: The scoring dimension (e.g. "epistemic_quality")
            goal_text: Optional override for goal text (defaults to "")

        Returns:
            dict: The raw model outputs (including q_value, state_value, action_logits, potentially zsa)
        """
        model = self.models.get(dimension)
        if model is None:
            raise ValueError(f"Model for dimension '{dimension}' not loaded.")

        prompt_emb = torch.tensor(
            self.memory.embedding.get_or_create(goal["goal_text"]),
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        output_emb = torch.tensor(
            self.memory.embedding.get_or_create(scorable.text),
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)

        with torch.no_grad():
            outputs = model(prompt_emb, output_emb)
        if self.return_zsa and "zsa" not in outputs:
            outputs["zsa"] = model.encoder(prompt_emb, output_emb)
        return outputs  # Return the full outputs dict
