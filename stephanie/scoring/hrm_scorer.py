# stephanie/scoring/hrm/hrm_scorer.py

import os

import torch

from stephanie.models.score import ScoreORM  # For prompt_hash
from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.model.hrm_model import HRMModel
from stephanie.scoring.scorable import Scorable
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.utils.file_utils import load_json  # To load meta file


class HRMScorer(BaseScorer):
    """
    Scorer that uses a trained Hierarchical Reasoning Model (HRM) to evaluate
    goal/document pairs. The HRM performs internal multi-step reasoning to
    produce a quality score.
    """
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.model_type = "hrm" # This identifies the scorer type
        
        # Use the embedding details from memory
        self.embedding_type = self.memory.embedding.type
        self.dim = self.memory.embedding.dim
        # HRM might use a different internal dimension (h_dim), but input is based on self.dim
        # h_dim, l_dim, etc. are loaded from the model's meta file or config
        
        # Get target type and version from config, with defaults
        self.target_type = cfg.get("target_type", "document")
        self.model_path = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")
        self.dimensions = cfg.get("dimensions", [])

        # HRM dimension is a specific dimension for this scorer        # Dictionary to hold the loaded HRM model instance
        self.models = {}
        # Dictionary to hold model metadata (e.g., hyperparameters)
        self.model_meta = {}

        # Attempt to load the model during initialization
        self._load_models(self.dimensions)

    def _load_models(self, dimensions):
        """
        Loads the trained HRM model components and metadata using ModelLocator.
        """
        for dimension in dimensions:
            try:
                locator = self.get_locator(dimension)

                # Check if the model files exist Is right that is wrong
                model_file_path = locator.model_file(suffix="_hrm.pt") # Match the suffix used in saving
                meta_file_path = locator.meta_file()

                if not os.path.exists(model_file_path):
                    self.logger.log("HRMScorerModelError", {
                        "message": "HRM model file not found.",
                        "path": model_file_path,
                    })
                    return # Cannot load if file is missing

                # Load model metadata
                if os.path.exists(meta_file_path):
                    self.model_meta[dimension] = load_json(meta_file_path)
                    self.logger.log("HRMScorerMetaLoaded", {
                        "dimension": dimension,
                        "meta": self.model_meta[dimension] # Log key meta info if needed
                    })
                else:
                    self.logger.log("HRMScorerWarning", {
                        "message": "HRM meta file not found. Using defaults.",
                        "path": meta_file_path
                    })
                    self.model_meta[dimension] = {} # Use empty dict if meta is missing

                # --- Reconstruct HRM Model Configuration ---
                # Get HRM hyperparameters from meta or use defaults consistent with training
                hrm_cfg_from_meta = {
                    "input_dim": self.model_meta[dimension].get("input_dim", self.dim * 2), # Default concat
                    "h_dim": self.model_meta[dimension].get("h_dim", 256),
                    "l_dim": self.model_meta[dimension].get("l_dim", 128),
                    "output_dim": self.model_meta[dimension].get("output_dim", 1),
                    "n_cycles": self.model_meta[dimension].get("n_cycles", 4),
                    "t_steps": self.model_meta[dimension].get("t_steps", 4),
                    # lr, epochs are not needed for inference
                }
                
                # --- Instantiate HRM Model ---
                # Create an instance of the HRMModel with the loaded config
                self.models[dimension] = HRMModel(hrm_cfg_from_meta, logger=self.logger)
                
                # --- Load Model Weights ---
                # Load the saved state dictionary into the model instance
                # Make sure the device is consistent
                self.models[dimension].to(self.device)
                self.models[dimension].load_state_dict(torch.load(model_file_path, map_location=self.device))
                self.models[dimension].eval() # Set to evaluation mode

                self.logger.log("HRMScorerModelLoaded", {
                    "dimension": dimension,
                    "model_path": model_file_path,
                    "device": str(self.device)
                })

            except Exception as e:
                self.logger.log("HRMScorerInitError", {
                    "message": "Failed to load HRM model.",
                    "dimension": self.hrm_dimension,
                    "error": str(e)
                })
                self.model = None # Ensure model is None on failure

    def score(self, goal: dict, scorable: Scorable, dimensions: list[str]) -> ScoreBundle:
        """
        Scores a single scorable item against a goal using the HRM models per dimension.

        Returns:
            ScoreBundle with a ScoreResult for each applicable dimension.
        """
        results = {}
        goal_text = goal.get("goal_text", "")
        doc_text = scorable.text

        if not goal_text or not doc_text:
            self.logger.log("HRMScorerWarning", {
                "message": "Missing goal_text or scorable text.",
                "goal": goal.get("id", "unknown"),
                "scorable_id": scorable.id
            })
            return ScoreBundle(results={})

        # 1. Get embeddings
        ctx_emb_np = self.memory.embedding.get_or_create(goal_text)
        doc_emb_np = self.memory.embedding.get_or_create(doc_text)

        # 2. Convert to PyTorch tensors
        ctx_emb = torch.tensor(ctx_emb_np, dtype=torch.float32).to(self.device).unsqueeze(0)
        doc_emb = torch.tensor(doc_emb_np, dtype=torch.float32).to(self.device).unsqueeze(0)
        x_input = torch.cat([ctx_emb, doc_emb], dim=-1)

        for dimension in dimensions:
            model = self.models.get(dimension)
            if not model:
                self.logger.log("HRMScorerError", {
                    "message": f"HRM model not found for dimension '{dimension}'. Skipping.",
                    "goal_id": goal.get("id", "unknown"),
                    "scorable_id": scorable.id
                })
                continue

            try:
                with torch.no_grad():
                    y_pred, intermediate_states = model(x_input)

                raw_score = y_pred.squeeze().item()

                zL_mag = zH_mag = None
                if "zL_final" in intermediate_states:
                    zL_mag = torch.norm(intermediate_states["zL_final"], p=2).item()
                if "zH_final" in intermediate_states:
                    zH_mag = torch.norm(intermediate_states["zH_final"], p=2).item()

                rationale = (
                    f"HRM[{dimension}] raw={round(raw_score, 4)} | "
                    f"zL_mag={round(zL_mag, 4) if zL_mag else 'NA'}, "
                    f"zH_mag={round(zH_mag, 4) if zH_mag else 'NA'}"
                )

                attributes = {
                    "raw_score": round(raw_score, 4),
                    "zL_magnitude": zL_mag,
                    "zH_magnitude": zH_mag,
                    "q_value": raw_score,  # Using raw_score as q_value
                    "energy": raw_score,  # Keeping energy as q_value as in original
                }

                result = ScoreResult(
                    dimension=dimension,
                    score=raw_score,
                    source=self.model_type,
                    rationale=rationale,
                    weight=1.0,
                    attributes=attributes
                )

                results[dimension] = result

                self.logger.log("HRMScorerEvaluated", {
                    "dimension": dimension,
                    "goal_id": goal.get("id", "unknown"),
                    "scorable_id": scorable.id,
                    "raw_score": raw_score,
                    "zL_magnitude": zL_mag,
                    "zH_magnitude": zH_mag,
                })

            except Exception as e:
                self.logger.log("HRMScorerError", {
                    "message": "Exception during HRM scoring.",
                    "dimension": dimension,
                    "goal_id": goal.get("id", "unknown"),
                    "scorable_id": scorable.id,
                    "error": str(e)
                })

        return ScoreBundle(results=results)


    def __repr__(self):
        return f"<HRMScorer(model_type={self.model_type}, dimension={self.hrm_dimension}, loaded={self.model is not None})>"
