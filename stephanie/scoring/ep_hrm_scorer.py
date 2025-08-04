# stephanie/scoring/hrm/hrm_scorer.py

import os

import torch

from stephanie.data.plan_trace import PlanTrace
from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.model.epistemic_trace_encoder import \
    EpistemicTraceEncoder
from stephanie.scoring.model.hrm_model import HRMModel
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.utils.file_utils import load_json  # To load meta file
from stephanie.utils.trace_utils import get_trace_score_stats


class EpistemicPlanHRMScorer(BaseScorer):
    """
    Scorer that uses a trained Hierarchical Reasoning Model (HRM) to evaluate
    goal/document pairs. The HRM performs internal multi-step reasoning to
    produce a quality score.
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.model_type = "epistemic_hrm"  # This identifies the scorer type

        # Use the embedding details from memory
        self.embedding_type = self.memory.embedding.type
        self.dim = self.memory.embedding.dim
        # HRM might use a different internal dimension (h_dim), but input is based on self.dim
        # h_dim, l_dim, etc. are loaded from the model's meta file or config

        # Get target type and version from config, with defaults
        self.target_type = cfg.get("target_type", "plan_trace")
        self.model_path = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")
        self.dimensions = cfg.get("dimensions", [])
        self.get_trace_score_stats = get_trace_score_stats

        # HRM dimension is a specific dimension for this scorer        # Dictionary to hold the loaded HRM model instance
        self.models = {}
        # Dictionary to hold model metadata (e.g., hyperparameters)
        self.model_meta = {}
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

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
                model_file_path = locator.model_file(
                    suffix="_hrm_epistemic.pt"
                )  # Match the suffix used in saving
                meta_file_path = locator.meta_file()

                if not os.path.exists(model_file_path):
                    self.logger.log(
                        "EpistemicPlanHRMScorerModelError",
                        {
                            "message": "HRM model file not found.",
                            "path": model_file_path,
                            "dimension": dimension,
                        },
                    )
                    return  # Cannot load if file is missing

                # Load model metadata
                if os.path.exists(meta_file_path):
                    self.model_meta[dimension] = load_json(meta_file_path)
                    self.logger.log(
                        "EpistemicPlanHRMScorerMetaLoaded",
                        {
                            "dimension": dimension,
                            "meta": self.model_meta[
                                dimension
                            ],  # Log key meta info if needed
                        },
                    )
                else:
                    self.logger.log(
                        "EpistemicPlanHRMScorerWarning",
                        {
                            "message": "HRM meta file not found. Using defaults.",
                            "path": meta_file_path,
                        },
                    )
                    self.model_meta[
                        dimension
                    ] = {}  # Use empty dict if meta is missing

                # --- Reconstruct HRM Model Configuration ---
                # Get HRM hyperparameters from meta or use defaults consistent with training
                hrm_cfg_from_meta = {
                    "input_dim": self.model_meta.get(
                        "input_dim", 256
                    ),  # Default concat
                    "h_dim": self.model_meta.get("h_dim", 256),
                    "l_dim": self.model_meta.get("l_dim", 128),
                    "output_dim": self.model_meta.get("output_dim", 1),
                    "n_cycles": self.model_meta.get("n_cycles", 4),
                    "t_steps": self.model_meta.get("t_steps", 4),
                    # lr, epochs are not needed for inference
                }

                # --- Instantiate HRM Model ---
                # Create an instance of the HRMModel with the loaded config
                self.models[dimension] = HRMModel(
                    hrm_cfg_from_meta, logger=self.logger
                )

                # --- Load Model Weights ---
                # Load the saved state dictionary into the model instance
                # Make sure the device is consistent
                self.models[dimension].to(self.device)
                self.models[dimension].load_state_dict(
                    torch.load(model_file_path, map_location=self.device)
                )
                self.models[dimension].eval()  # Set to evaluation mode

                self.logger.log(
                    "EpistemicPlanHRMScorerModelLoaded",
                    {
                        "dimension": dimension,
                        "model_path": model_file_path,
                        "device": str(self.device),
                    },
                )

            except Exception as e:
                self.logger.log(
                    "EpistemicPlanHRMScorerInitError",
                    {
                        "message": "Failed to load HRM model.",
                        "dimension": dimension,
                        "error": str(e),
                    },
                )

    def score(
        self, plan_trace: PlanTrace, dimensions: list[str]
    ) -> ScoreBundle:
        """
        Scores a PlanTrace using the trained Epistemic Plan HRM model(s).

        Args:
            trace: A PlanTrace object (or dict) representing the reasoning process to evaluate.
                This is the primary input for the Epistemic Plan HRM.
            dimensions: A list of dimension names. The scorer will produce a result for
                        each dimension it has a trained model for *and* that is requested.

        Returns:
            ScoreBundle: Contains ScoreResults for each applicable dimension.
                        The score represents the 'epistemic quality' of the trace.
        """
        # Note: No 'goal: dict' or 'scorable: Scorable' args, as they are not the primary input.

        results = {}

        # Check if trace is valid
        if not plan_trace or not plan_trace.execution_steps:
            self.logger.log(
                "EpistemicPlanHRMScorerWarning",
                {"message": "Empty or missing plan trace."},
            )
            return ScoreBundle(results={})

        try:
            # Step 1: Encode the trace

            encoder = EpistemicTraceEncoder(self.cfg.get("encoder", {})).to(
                self.device
            )
            x_input = (
                encoder(
                    trace=plan_trace,
                    embedding_lookup_fn=self.memory.embedding.get_or_create,
                    score_stats_fn=self.get_trace_score_stats,
                    dimensions=dimensions,
                )
                .unsqueeze(0)
                .to(self.device)
            )

        except Exception as e:
            self.logger.log(
                "EpistemicPlanHRMScorerEncodingError",
                {"message": "Failed to encode plan trace.", "error": str(e)},
            )

        for dimension in dimensions:
            model = self.models.get(dimension)
            if not model:
                self.logger.log(
                    "EpistemicPlanHRMScorerError",
                    {
                        "message": f"HRM model not found for dimension '{dimension}'"
                    },
                )
                continue

            try:
                with torch.no_grad():
                    y_pred, intermediate_states = model(x_input)
                raw_score = y_pred.squeeze().item()

                rationale = f"HRM[{dimension}] score={round(raw_score, 4)}"

                attributes = {
                    "raw_score": round(raw_score, 4),
                }

                result = ScoreResult(
                    dimension=dimension,
                    score=raw_score,
                    rationale=rationale,
                    weight=1.0,
                    source=self.model_type,
                    attributes=attributes,
                )
                results[dimension] = result

            except Exception as e:
                self.logger.log(
                    "EpistemicPlanHRMScorerEvalError",
                    {"dimension": dimension, "error": str(e)},
                )

        return ScoreBundle(results=results)

    def __repr__(self):
        return f"<EpistemicPlanHRMScorer(model_type={self.model_type}, loaded={self.models is not None})>"
