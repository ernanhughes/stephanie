from __future__ import annotations
import logging
from typing import Any, Dict, List

from stephanie.core.app_context import AppContext
from stephanie.agents.base_agent import BaseAgent
from stephanie.services.scoring_service import ScoringService  # existing in your repo
from stephanie.memcube.memcube_factory import MemCubeFactory    # typical pattern
from stephanie.components.thoughts.vpm.thought_vpm_encoder import ThoughtVPMEncoder
from ..thought_types import Thought, ThoughtKind, Evidence
from ..thought_trace import ThoughtTrace

_logger = logging.getLogger(__name__)

class ThoughtProcessorAgent(BaseAgent):
    """
    Pipeline stage:
      1) Extract raw thoughts from inputs (LLM traces, tools, etc.)
      2) Normalize to Thought objects
      3) Score with ScoringService (SICQL/MRQ/EBT) → score, uncertainty, meta
      4) Commit to MemCube (governed)
      5) Emit VPM tiles (for dashboards/HRM)
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.scoring: ScoringService = self.container.get("scoring")
        self.vpm_enc = ThoughtVPMEncoder()

    async def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        goal_text: str = ctx.get("goal_text", "")
        run_id: str = ctx.get("run_id", "")
        raw: List[Dict[str, Any]] = ctx.get("raw_thoughts", [])

        trace = ThoughtTrace(goal_text=goal_text, run_id=run_id, context={k:v for k,v in ctx.items() if k not in {"raw_thoughts"}})

        # 1) Normalize
        thoughts: List[Thought] = []
        for r in raw:
            t = Thought(
                text=r.get("content", "").strip(),
                kind=ThoughtKind(r.get("kind", "think")),
                tags=r.get("tags", []),
                evidence=[Evidence(**e) for e in r.get("evidence", [])],
                meta=r.get("meta", {}),
            )
            thoughts.append(t)

        # 2) Score (vectorized where possible)
        #    We'll call a generic scorer that returns dicts keyed by index
        scores = await self.scoring.score([t.text for t in thoughts], goal_text=goal_text)
        for i, t in enumerate(thoughts): 
            s = scores.get(i, {})
            t.score = float(s.get("score", 0.0))
            t.uncertainty = float(s.get("uncertainty", 1.0))
            t.meta.update({k: float(v) for k, v in s.get("meta", {}).items()})
            trace.add(t)

        # 3) Commit to MemCube
        cube = self.memcube.create(
            kind="thought_trace",
            header={"goal_text": goal_text, "run_id": run_id},
            body=trace.to_records(),
            tags=["thoughts", "trace"],
        )
        cube_id = self.memcube.store(cube)
        _logger.info("ThoughtTrace committed: %s", cube_id)

        # 4) VPM tile(s)
        if self.vpm_enc is not None:
            vpm_path = self.vpm_enc().encode(trace)
            _logger.info("Thought VPM tile written: %s", vpm_path)

        # 5) Emit next-stage context
        out = {**ctx, "thought_trace": trace, "thought_cube_id": cube_id}
        if self.vpm_enc is not None:
            out["thought_vpm_path"] = vpm_path
        return out
