# stephanie/components/nexus/agent.py
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.knowledge.chat_analyze import ChatAnalyzeAgent
from stephanie.agents.knowledge.scorable_annotate import ScorableAnnotateAgent
from stephanie.components.nexus.services.graph_layout import NexusService
from stephanie.utils.progress_mixin import ProgressMixin

log = logging.getLogger(__name__)

# Type aliases for readability
ScorableTuple = Tuple[str, str, str, dict]  # (node_id, scorable_type, scorable_id, meta)

class NexusAgent(BaseAgent, ProgressMixin):
    """
    High-level entrypoint for Nexus.
    - Builds/updates the Nexus graph index from scorables and/or VPMs.
    - Finds a best path given a start node and (optional) goal text/vector.
    - Returns a standard output payload under self.output_key for downstream stages.
    
    Expected context keys (flexible; provide whichever you have):
      - "nexus_scorables": List[ScorableTuple]
      - "nexus_vpms": List[ScorableTuple]    # use scorable_type="vpm"
      - "nexus_start_node_id": str           # required to run path-finding
      - "nexus_goal_vec": List[float] | np.ndarray  # optional
      - "nexus_goal_text": str               # optional (domain/NER boosts)
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Sub-agents / utilities 
        self.annotate = ScorableAnnotateAgent(
            cfg.get("annotate", {}), memory, container, logger
        )
        self.analyze = ChatAnalyzeAgent(
            cfg.get("analyze", {}), memory, container, logger
        )

        # Lazily create the service (you can also container.register if preferred)
        self._nexus = NexusService(cfg=self.cfg, memory=self.memory)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        run_id = context.get("pipeline_run_id", "unknown")
        try:
            log.info("Nexus step started for run_id=%s", run_id)
            self._init_progress(self.container, log)

            # ----------------------------
            # 1) Collect inputs
            # ----------------------------
            scorables: List[ScorableTuple] = list(context.get("scorables", []))
            vpms: List[ScorableTuple] = list(context.get("nexus_vpms", []))

            # Validate tuple shapes early (nice-to-have)
            def _is_4tuple(x) -> bool:
                return isinstance(x, (list, tuple)) and len(x) == 4

            if scorables and not all(_is_4tuple(x) for x in scorables):
                raise ValueError("nexus_scorables must be an iterable of 4-tuples: (node_id, scorable_type, scorable_id, meta)")

            if vpms and not all(_is_4tuple(x) for x in vpms):
                raise ValueError("nexus_vpms must be an iterable of 4-tuples: (node_id, 'vpm', scorable_id, meta)")

            # ----------------------------
            # 2) Build / update index
            # ----------------------------
            total_to_index = len(scorables) + len(vpms)
            task = f"NEXUS:{run_id}"
            self.pstart(task=task, total=max(1, total_to_index), meta={"run_id": run_id})

            indexed_count = 0
            if scorables:
                self.pstage(task=task, stage="index:scorables", detail={"count": len(scorables)})
                indexed_count += self._nexus.build_index_from_scorables(scorables)
                self.ptick(task=task, n=len(scorables))

            if vpms:
                self.pstage(task=task, stage="index:vpms", detail={"count": len(vpms)})
                indexed_count += self._nexus.build_index_from_vpms(vpms)
                self.ptick(task=task, n=len(vpms))

            # ----------------------------
            # 3) Path-finding (optional but typical)
            # ----------------------------
            start_node_id: Optional[str] = context.get("nexus_start_node_id")
            goal_vec = context.get("nexus_goal_vec")
            goal_text: str = context.get("nexus_goal_text", "")

            # If goal_vec is a python list, convert to np.ndarray for the pathfinder
            if isinstance(goal_vec, list):
                goal_vec = np.asarray(goal_vec, dtype=np.float32)

            path_out: Dict[str, Any] = {}
            if start_node_id:
                self.pstage(task=task, stage="path:search", detail={"start_node_id": start_node_id})
                path_out = self._nexus.find_path(
                    start_node_id=start_node_id,
                    goal_vec=goal_vec,
                    goal_text=goal_text,
                )
            else:
                log.info("Nexus: no 'nexus_start_node_id' provided; skipping path search.")

            # ----------------------------
            # 4) Finalize
            # ----------------------------
            self.pstage(task=task, stage="complete", detail={"indexed": indexed_count})
            self.pdone(task=task)

            # Standardized output for downstream pipeline stages
            context[self.output_key] = {
                "indexed": indexed_count,
                "path": path_out,
                "run_id": run_id,
            }
            log.info("== Nexus Summary ==\nindexed=%s path=%s", indexed_count, path_out)
            return context

        except Exception as e:
            error_msg = f"Nexus step failed for run_id={run_id}: {str(e)}"
            log.exception(error_msg)
            raise RuntimeError(error_msg) from e