# stephanie/components/ssp/agent.py
from __future__ import annotations

import logging
from typing import Any, Dict, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.ssp.core.algorithm import SSPAlgorithm
# Impl bindings
from stephanie.components.ssp.impl.proposers.searching_proposer import \
    SearchingProposer
from stephanie.components.ssp.impl.solvers.ats_solver import ATSSolver
from stephanie.components.ssp.impl.solvers.solution_search import \
    SolutionSearch
from stephanie.components.ssp.impl.verifiers.rag_verifier import RAGVerifier
from stephanie.components.ssp.services.state_service import StateService
from stephanie.components.ssp.services.vpm_control_service import \
    VPMControlService
from stephanie.components.ssp.services.vpm_visualization_service import \
    VPMVisualizationService
from stephanie.components.tree.events import TreeEventEmitter
from stephanie.utils.progress_mixin import ProgressMixin

_logger = logging.getLogger(__name__)


class SSPAgent(BaseAgent, ProgressMixin):
    """
    High-level entrypoint for SSP. Wires Proposer/Solver/Verifier into SSPAlgorithm,
    then runs a single training step over the configured seeds.
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        self.seeds: List[str] = list(cfg.get("seeds", []))

        # Register supporting services (as before)
        container.register(
            name="ssp_state",
            factory=lambda: StateService(
                cfg=cfg, memory=memory, container=container, logger=logger
            ),
            dependencies=[],
            init_args={},
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one SSP training step using the algorithm (paper-aligned A vs B).
        """
        run_id = context.get("pipeline_run_id", "unknown")
        self.container.register(
            name="vpm_control",
            factory=lambda: VPMControlService(
                cfg=self.cfg,
                memory=self.memory,
                container=self.container,
                logger=self.logger,
                run_id=run_id,
            ),
            dependencies=[],
            init_args={},
        )
        self.container.register(
            name="ssp_vpm_viz",
            factory=lambda: VPMVisualizationService(
                cfg=self.cfg,
                memory=self.memory,
                logger=self.logger,
                container=self.container,
                run_id=run_id,
            ),
            dependencies=[],
            init_args={},
        )

        emitter = TreeEventEmitter(topic="ssp.ats")
        solution_search = SolutionSearch(
            cfg=self.cfg,
            memory=self.memory,
            container=self.container,
            logger=self.logger,
            event_emitter=emitter,
        )

        # Construct SSPAlgorithm with concrete roles
        self._algorithm = SSPAlgorithm(
            proposer=SearchingProposer(
                cfg=self.cfg,
                memory=self.memory,
                container=self.container,
                logger=self.logger,
                solution_search=solution_search,
            ),
            solver=ATSSolver(
                cfg=self.cfg,
                memory=self.memory,
                container=self.container,
                logger=self.logger,
                searcher=solution_search,
                event_emitter=emitter,
            ),
            verifier=RAGVerifier(
                container=self.container,
                logger=self.logger,
                memory=self.memory,
                cfg=self.cfg,
            ),
            vpm_visualization=self.container.get("ssp_vpm_viz"),
        )
        try:
            _logger.info("SSP step started for run_id=%s", run_id)
            self._init_progress(self.container, _logger)

            task = f"SSP:{run_id}"
            self.pstart(
                task=task, total=len(self.seeds), meta={"run_id": run_id}
            )

            # Train step (parallel per-seed)
            stats = await self._algorithm.train_step(
                seed_answers=self.seeds, context=context
            )

            # Mark progress done
            self.pstage(task=task, stage="complete")
            self.pdone(task=task)

            _logger.info("== SSP Summary ==\n%s", stats)

            # Attach under standard output key for downstream pipeline stages
            context[self.output_key] = {
                "stats": stats,
                "metrics": self._algorithm.get_metrics().__dict__,
            }
            return context

        except Exception as e:
            error_msg = f"SSP step failed for run_id={run_id}: {str(e)}"
            _logger.exception(error_msg)
            raise RuntimeError(error_msg) from e
