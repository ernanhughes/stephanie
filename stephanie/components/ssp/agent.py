# stephanie/components/ssp/agent.py
from __future__ import annotations

import logging
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.ssp.services.state_service import StateService
from stephanie.components.ssp.services.vpm_control_service import VPMControlService
from stephanie.components.ssp.trainer import Trainer
from omegaconf import OmegaConf

_logger = logging.getLogger(__name__)


class SSPAgent(BaseAgent):

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
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

            seeds = [
                "permafrost thaw releasing methane increases radiative forcing",
                "insulin enables glucose uptake in muscle and adipose tissue",
                "backpropagation updates weights by gradient descent on loss",
            ]

            stats = Trainer(difficulty=0.3, verify_threshold=0.6).run_batch(seeds)
            print("== Summary ==", stats)

            # Attach the result to the context under the standard key
            context[self.output_key] = stats

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