# stephanie/components/ssp/agent.py
from __future__ import annotations

import logging
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.ssp.services.state_service import StateService
from stephanie.components.ssp.services.vpm_control_service import VPMControlService
from stephanie.components.ssp.trainer import Trainer
from stephanie.components.ssp.types import EpisodeStatus
from omegaconf import OmegaConf

_logger = logging.getLogger(__name__)


class SSPAgent(BaseAgent):
    """
    Self-Play System (SSP) Agent — Entry point for autonomous research & improvement cycles.

    This agent orchestrates the full Self-Play loop:
        Propose → Solve → Verify → Learn → Adapt Difficulty

    It serves as the main interface for pipelines to initiate or integrate with Stephanie's
    self-improvement engine. The SSP continuously generates novel, verifiable research questions,
    solves them using agentic search, verifies solutions, and adapts its challenge level based
    on success—driving organic growth in reasoning quality, knowledge integration, and tool use.

    Key Responsibilities:
      - Initialize and manage the SSP Trainer
      - Execute one or more self-play training steps
      - Inject pipeline context (e.g., mission, constraints)
      - Return rich metrics and artifacts for monitoring and downstream use

    Configurable Parameters (via `cfg`):
        - self_play.qmax.initial_difficulty: Starting difficulty (0.0–1.0)
        - self_play.qmax.max_difficulty: Upper bound on difficulty
        - self_play.qmax.difficulty_step: Step size for curriculum adjustment
        - self_play.curriculum.min_success_rate: Target for competence-based progression
        - self_play.verifier.verification_threshold: Score threshold for solution validity
        - self_play.proposer.mission: Global mission statement guiding proposals

    Example Usage:
        ```python
        agent = SSPAgent(cfg, memory, container, logger)
        context = {"pipeline_run_id": "ssp_demo_001"}
        result = await agent.run(context)
        # Returns: {
        #   "ssp": {
        #     "episode_id": "ssp-1730529840123",
        #     "success": True,
        #     "metrics": {
        #       "reward": 0.87,
        #       "verification": 0.92,
        #       "novelty": 0.76,
        #       "success_rate": 0.68,
        #       "difficulty": 0.55,
        #       "threshold": 0.85,
        #       "duration_ms": 4823
        #     },
        #     "training_batch": { ... }  # GRPO batch if enabled
        #   }
        # }
        ```

    Key Outputs:
        - Episode ID and success status
        - Composite reward and per-dimension scores
        - Current curriculum difficulty and recent success rate
        - Training batch (for reinforcement learning updates)
        - Structured trace logs via `trace_logger`
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        """
        Initialize the SSP Agent with configuration and core services.

        Args:
            cfg: Configuration dictionary containing SSP parameters.
            memory: Memory interface for state persistence (optional).
            container: ServiceContainer providing access to LLMs, scoring, etc.
            logger: Structured JSON logger for observability.

        Initializes:
            - The core `Trainer` orchestrating Proposer/Solver/Verifier
            - Default output key 'ssp' for consistent result placement
        """
        super().__init__(cfg, memory, container, logger)

        base_cfg = OmegaConf.create(cfg)

        container.register(
            name="ssp_state",
            factory=lambda: StateService(cfg=cfg, memory=memory, container=container, logger=logger),
            dependencies=[],
            init_args={
            },
        )

        container.register(
            name="vpm_control",
            factory=lambda: VPMControlService(cfg=cfg, memory=memory, container=container, logger=logger),
            dependencies=[],
            init_args={
            },
        )



        # Initialize the core SSP orchestrator
        try:
            self.trainer = Trainer(base_cfg, container)
            _logger.info("SSPAgent initialized successfully with Trainer.")
        except Exception as e:
            _logger.error("Failed to initialize SSPAgent.Trainer", extra={"error": str(e)})
            raise

        # Define where results will be placed in the context
        self.output_key = "ssp"

        # Optional: Extract high-level config for quick access
        sp_cfg = cfg.get("self_play", {})
        self.mission = sp_cfg.get("mission", "Improve Stephanie’s reasoning & tooling.")
        self.initial_difficulty = float(sp_cfg.get("qmax", {}).get("initial_difficulty", 0.3))

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one step of the Self-Play loop within the given pipeline context.

        This is the main entry point called by the Stephanie pipeline. It:
        1. Logs the start of the SSP episode
        2. Runs a single `train_step` through the Trainer
        3. Handles errors gracefully with structured logging
        4. Attaches results under `context['ssp']` for downstream consumption

        Args:
            context: Pipeline execution context. May contain:
                - pipeline_run_id: Unique identifier for this pipeline run
                - Additional metadata or constraints to guide proposal generation

        Returns:
            The input `context` dictionary updated with SSP results under `context['ssp']`.
            The structure is:
            {
                "ssp": {
                    "episode_id": str,           # Unique ID for this SSP cycle
                    "success": bool,             # Whether the solution passed verification
                    "metrics": dict,             # Detailed performance and state metrics
                    "training_batch": dict|null  # Batch data for RL training (if GRPO enabled)
                }
            }

        Raises:
            RuntimeError: If the trainer fails catastrophically.
            Any exception from underlying components (Proposer, Solver, ScoringService).

        Logs:
            - "SSPStepStarted": When the training step begins
            - "SSPStepCompleted": On successful completion with outcome and metrics
            - "SSPStepFailed": If an error occurs during the step
        """
        run_id = context.get("pipeline_run_id", "unknown")
        try:
            # Log the beginning of the SSP step
            self.logger.log("SSPStepStarted", {
                "run_id": run_id,
                "mission": self.mission,
                "current_proposer_difficulty": getattr(self.trainer.proposer, "difficulty", "unknown"),
                "context_keys": list(context.keys())
            })

            # Run one full SSP cycle: Propose → Solve → Verify → Update Curriculum
            result = self.trainer.train_step(context)

            # Attach the result to the context under the standard key
            context[self.output_key] = result

            # Log successful completion
            self.logger.log("SSPStepCompleted", {
                "run_id": run_id,
                "episode_id": result.get("episode_id"),
                "success": result.get("success"),
                "final_difficulty": result.get("metrics", {}).get("difficulty"),
                "verification_score": result.get("metrics", {}).get("verification"),
                "duration_ms": result.get("metrics", {}).get("duration_ms")
            })

            return context

        except Exception as e:
            # Critical failure; log details and re-raise
            error_msg = f"SSP step failed for run_id={run_id}: {str(e)}"
            _logger.exception(error_msg)  # Also logs traceback at ERROR level
            self.logger.log("SSPStepFailed", {
                "run_id": run_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "context_snapshot": {k: str(v)[:200] for k, v in context.items()}  # Avoid huge logs
            })
            raise RuntimeError(error_msg) from e