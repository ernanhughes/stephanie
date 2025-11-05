# stephanie/components/nexus/agents/nexus_inline.py
from __future__ import annotations
from typing import Any, Dict
from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import PIPELINE_RUN_ID
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.services.zeromodel_service import ZeroModelService
from stephanie.services.scoring_service import ScoringService
from stephanie.services.workers.nexus_workers import NexusVPMWorkerInline, NexusMetricsWorkerInline

class NexusInlineAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self.zm: ZeroModelService = self.container.get("zeromodel")
        self.scoring: ScoringService = self.container.get("scoring")
        self.vpmw = NexusVPMWorkerInline(self.zm, logger=logger)
        self.mxw  = NexusMetricsWorkerInline(
            scoring=self.scoring,
            scorers=["sicql","hrm", "tiny"],
            dimensions=["alignment","clarity","relevance","coverage","faithfulness"],
            persist=False
        )
        self.vpm_out = self.cfg.get("vpm_out", "./runs/nexus_vpm/")
        self.rollout_steps = int(self.cfg.get("rollout_steps", 0))
        self.rollout_strategy = self.cfg.get("rollout_strategy", "none")
        self.target_type = self.cfg.get("target_type", ScorableType.CONVERSATION_TURN)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.zm.initialize()
        # decide timeline metric surface now (dims used by vpm scoring)
        dims_for_vpm = ["clarity","coherence","complexity","alignment","coverage"]

        run_id  =context.get(PIPELINE_RUN_ID)

        self.vpmw.start_run(run_id, metrics=dims_for_vpm, out_dir=self.vpm_out)

        # Build scorables from chat turns (example)
        scorables = context.get("scorables", [])

        # For each scorable: (A) dense text metrics row; (B) VPM + optional rollout rows
        for s in scorables:
            goal = s.get("goal_ref") or context.get("goal")
            merged_context = {**context, "goal": goal}
            scorable = Scorable.from_dict(s)
            await self.mxw.score_and_append(self.zm, scorable, context=merged_context, run_id=run_id)
            await self.vpmw.run_item(
                run_id,
                scorable,
                out_dir=self.vpm_out,
                dims_for_score=dims_for_vpm,
                rollout_steps=int(self.rollout_steps),     # 0 for now; raise later
                rollout_strategy=self.rollout_strategy,    # "none" | "zoom_max"
                save_channels=False,
                name_hint=scorable.id,
            )

        # finalize global timeline
        await self.vpmw.finalize(run_id, out_dir=self.vpm_out)

        return context
