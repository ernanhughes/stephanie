# stephanie/scoring/hrm/hrm_scorer.py

import os
import torch

from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_result import ScoreResult
from stephanie.models.score import ScoreORM # For prompt_hash
from stephanie.utils.file_utils import load_json # To load meta file
from stephanie.scoring.model.hrm_model import HRMModel


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

        # The specific HRM task/dimension this scorer represents
        # This should match the `hrm_dimension` used during training
        self.hrm_dimension = cfg.get("hrm_dimension", "sicql_alignment") 
        
        # Dictionary to hold the loaded HRM model instance
        self.model = None
        # Dictionary to hold model metadata (e.g., hyperparameters)
        self.model_meta = None

        # Attempt to load the model during initialization
        self._load_model()

    def _load_model(self):
        """
        Loads the trained HRM model components and metadata using ModelLocator.
        """
        try:
            # Use the inherited get_locator method (from ModelLocatorMixin via BaseScorer)
            # This will create the path based on embedding_type, model_type (hrm), 
            # target_type, dimension (hrm_dimension), and version.
            locator = self.get_locator(self.hrm_dimension) 

            # Check if the model files exist Is right that is wrong
            model_file_path = locator.model_file(suffix="_hrm.pt") # Match the suffix used in saving
            meta_file_path = locator.meta_file()

            if not os.path.exists(model_file_path):
                self.logger.log("HRMScorerModelError", {
                    "message": "HRM model file not found.",
                    "path": model_file_path,
                    "dimension": self.hrm_dimension
                })
                return # Cannot load if file is missing

            # Load model metadata
            if os.path.exists(meta_file_path):
                self.model_meta = load_json(meta_file_path)
                self.logger.log("HRMScorerMetaLoaded", {
                    "dimension": self.hrm_dimension,
                    "meta": self.model_meta # Log key meta info if needed
                })
            else:
                self.logger.log("HRMScorerWarning", {
                    "message": "HRM meta file not found. Using defaults.",
                    "path": meta_file_path
                })
                self.model_meta = {} # Use empty dict if meta is missing

            # --- Reconstruct HRM Model Configuration ---
            # Get HRM hyperparameters from meta or use defaults consistent with training
            hrm_cfg_from_meta = {
                "hrm.input_dim": self.model_meta.get("input_dim", self.dim * 2), # Default concat
                "hrm.h_dim": self.model_meta.get("h_dim", 256),
                "hrm.l_dim": self.model_meta.get("l_dim", 128),
                "hrm.output_dim": self.model_meta.get("output_dim", 1),
                "hrm.n_cycles": self.model_meta.get("n_cycles", 4),
                "hrm.t_steps": self.model_meta.get("t_steps", 4),
                # lr, epochs are not needed for inference
            }
            
            # --- Instantiate HRM Model ---
            # Create an instance of the HRMModel with the loaded config
            self.model = HRMModel(hrm_cfg_from_meta, logger=self.logger)
            
            # --- Load Model Weights ---
            # Load the saved state dictionary into the model instance
            # Make sure the device is consistent
            self.model.to(self.device)
            self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
            self.model.eval() # Set to evaluation mode
            
            self.logger.log("HRMScorerModelLoaded", {
                "dimension": self.hrm_dimension,
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
        Scores a single scorable item against a goal using the trained HRM model.
        
        Args:
            goal: A dictionary containing goal information (e.g., {"goal_text": "..."})
            scorable: A Scorable object representing the item to be scored.
            dimensions: A list of dimension names. The HRM scorer typically
                        produces one primary score, but this list allows integration
                        into the standard scoring framework. It will score if 
                        self.hrm_dimension is in this list.
        
        Returns:
            ScoreBundle: Contains the HRM score result if applicable.
        """
        results = {}

        if not self.model:
             self.logger.log("HRMScorerError", {
                 "message": "HRM model not loaded. Cannot score.",
                 "dimension": self.hrm_dimension
             })
             return ScoreBundle(results={})

        try:
            goal_text = goal.get("goal_text", "")
            doc_text = scorable.text

            if not goal_text or not doc_text:
                self.logger.log("HRMScorerWarning", {
                    "message": "Missing goal_text or scorable text.",
                    "dimension": self.hrm_dimension
                })
                return ScoreBundle(results={})

            # 1. Get embeddings
            ctx_emb_np = self.memory.embedding.get_or_create(goal_text)
            doc_emb_np = self.memory.embedding.get_or_create(doc_text)

            # 2. Convert to PyTorch tensors and move to device
            ctx_emb = torch.tensor(ctx_emb_np, dtype=torch.float32).to(self.device).unsqueeze(0)
            doc_emb = torch.tensor(doc_emb_np, dtype=torch.float32).to(self.device).unsqueeze(0)

            # 3. Prepare input for HRM Model (concatenate)
            x_input = torch.cat([ctx_emb, doc_emb], dim=-1) # Shape: (1, input_dim)

            # 4. Run the HRM Model (in evaluation mode) - Capture intermediate states
            with torch.no_grad():
                # UNPACK the tuple returned by HRMModel.forward
                # y_pred is the output tensor, intermediate_states is the dict
                y_pred, intermediate_states = self.model(x_input) # Shapes: (1, 1), dict
            
            # 5. Extract the scalar score value
            raw_hrm_score = y_pred.squeeze().item()

            # 6. Process intermediate states for logging/rationale
            # Extract final states (they are tensors)
            zL_final_tensor = intermediate_states.get('zL_final')
            zH_final_tensor = intermediate_states.get('zH_final')

            # Example: Calculate magnitude (L2 norm) of final states as a simple metric
            zL_magnitude = None
            zH_magnitude = None
            if zL_final_tensor is not None:
                # .item() to get scalar value from single-element tensor
                zL_magnitude = torch.norm(zL_final_tensor, p=2).item() 
            if zH_final_tensor is not None:
                zH_magnitude = torch.norm(zH_final_tensor, p=2).item()

            # Example: Get the actual final hidden state values (useful for debugging small models)
            # Convert to list for JSON serialization if needed
            # zL_final_values = zL_final_tensor.flatten().tolist() if zL_final_tensor is not None else None
            # zH_final_values = zH_final_tensor.flatten().tolist() if zH_final_tensor is not None else None

            # 7. (Optional) Apply post-processing/clipping/normalization
            final_score = raw_hrm_score # Or apply clipping/transform

            # 8. Create ScoreResult with enhanced rationale and metadata
            prompt_hash = ScoreORM.compute_prompt_hash(goal_text, scorable)

            # Build a more detailed rationale using intermediate state info
            rationale_parts = [f"HRM prediction (raw={round(raw_hrm_score, 4)})"]
            if zL_magnitude is not None:
                rationale_parts.append(f"zL_mag={round(zL_magnitude, 4)}")
            if zH_magnitude is not None:
                rationale_parts.append(f"zH_mag={round(zH_magnitude, 4)}")
            rationale = f" after {self.model_meta.get('n_cycles', 'N')}/{self.model_meta.get('t_steps', 'T')} cycles/steps. " + ", ".join(rationale_parts)

            # Prepare extra metadata to store in ScoreResult (optional)
            # This could include the magnitudes or even the full state lists (if small/serializable)
            extra_metadata = {
                "hrm_zL_final_magnitude": zL_magnitude,
                "hrm_zH_final_magnitude": zH_magnitude,
                # "hrm_zL_final_values": zL_final_values, # Uncomment if storing full states
                # "hrm_zH_final_values": zH_final_values, # Uncomment if storing full states
                "hrm_cycles": self.model_meta.get('n_cycles'),
                "hrm_t_steps": self.model_meta.get('t_steps'),
            }

            score_result = ScoreResult(
                dimension=self.hrm_dimension,
                score=final_score,
                rationale=rationale, # Enhanced rationale
                weight=1.0,
                q_value=raw_hrm_score,
                energy=raw_hrm_score, # You might adjust this based on intermediate states if desired
                source=self.model_type,
                target_type=scorable.target_type,
                prompt_hash=prompt_hash,
                # Add the extra metadata. Ensure ScoreResult can handle this or adapt accordingly.
                # If ScoreResult doesn't have a generic metadata field, you might need to extend it
                # or find another way to log this information (e.g., via logger or a separate ORM entry).
                # For now, assuming ScoreResult can accept **kwargs or has a metadata attribute.
                # Let's pass it via rationale or log it separately for now.
                # metadata=extra_metadata, # If ScoreResult supports this directly
            )

            # 8a. (Alternative) If ScoreResult can't hold extra metadata easily,
            # log the intermediate state info separately
            self.logger.log("HRMScorerIntermediateStates", {
                "dimension": self.hrm_dimension,
                "goal_id": goal.get("id", "unknown"),
                "scorable_id": scorable.id,
                "zL_final_magnitude": zL_magnitude,
                "zH_final_magnitude": zH_magnitude,
                # "zL_final_values": zL_final_values, # Log full values if needed/debugging
                # "zH_final_values": zH_final_values,
            })

            # 9. Add to results dictionary
            results[self.hrm_dimension] = score_result

            # 10. Log the scoring event
            self.logger.log("HRMScorerEvaluated", {
                "dimension": self.hrm_dimension,
                "goal_id": goal.get("id", "unknown"),
                "scorable_id": scorable.id,
                "raw_score": raw_hrm_score,
                "final_score": final_score,
                "zL_final_magnitude": zL_magnitude, # Log key metrics here too
                "zH_final_magnitude": zH_magnitude,
            })

        except Exception as e:
            self.logger.log("HRMScorerError", {
                "message": "Error during HRM scoring.",
                "dimension": self.hrm_dimension,
                "goal_id": goal.get("id", "unknown"),
                "scorable_id": scorable.id,
                "error": str(e)
            })
            return ScoreBundle(results={})

        return ScoreBundle(results=results)


    def __repr__(self):
        return f"<HRMScorer(model_type={self.model_type}, dimension={self.hrm_dimension}, loaded={self.model is not None})>"
