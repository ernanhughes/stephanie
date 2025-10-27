# stephanie/components/ssp/grpo_bridge.py 
from __future__ import annotations

from typing import Dict, Any
from omegaconf import DictConfig
from stephanie.components.tree.tree_grpo import (
    TreeGRPOConfig, TreeGRPOAdapter
)
from stephanie.components.tree.core import AgenticTreeSearch
from stephanie.components.ssp.tree_events import TreeEventEmitter
from stephanie.utils.json_sanitize import sanitize

def make_tree_grpo_adapter(cfg: DictConfig, base_agent) -> TreeGRPOAdapter:
    g = cfg.ssp.grpo
    base = AgenticTreeSearch(
        agent=base_agent,
        max_iterations=0,  # adapter drives expansion
        time_limit=cfg.ssp.solver.time_limit_sec,
        N_init=cfg.ssp.solver.N_init,
        emit_cb=TreeEventEmitter(),
    )
    return TreeGRPOAdapter(
        base, TreeGRPOConfig(
            M=g.M, N=g.N, L=g.L,
            use_zscore_intra=g.use_zscore_intra,
            use_zscore_inter=g.use_zscore_inter,
            value_alpha=g.value_alpha,
            scorer_name="sicql",
            dimensions=["alignment"],
        )
    )

async def rollout_training_batch(adapter: TreeGRPOAdapter, goal_text: str, value: float = 0.0) -> Dict[str, Any]:
    result = await adapter.rollout_forest({
        "goal": {"goal_text": goal_text},
        "value": float(value),
    })
    return sanitize(result["training_batch"])
