from __future__ import annotations
from typing import Any, Dict, Optional

from stephanie.components.tree.core import AgenticTreeSearch
from stephanie.components.tree.tree_grpo import TreeGRPOAdapter, TreeGRPOConfig


class TreeWrapper:
    """
    Very thin wrapper so SSP can depend on a small surface:
      - run(context)  -> {'answer', 'report', ...}
      - rollout(context) when GRPO is enabled -> forest + batch
    """

    def __init__(self, agent: Any, cfg: Dict):
        self.base = AgenticTreeSearch(
            agent=agent,
            max_iterations=int(cfg.get("search_iterations", 40)),
            time_limit=int(cfg.get("time_limit_sec", 60)),
            N_init=int(cfg.get("N_init", 3)),
            C_ucb=float(cfg.get("C_ucb", 1.2)),
            H_greedy=float(cfg.get("H_greedy", 0.3)),
            H_debug=float(cfg.get("H_debug", 0.5)),
            no_improve_patience=int(cfg.get("no_improve_patience", 25)),
            progress_every=int(cfg.get("progress_every", 5)),
            report_top_k=int(cfg.get("report_top_k", 5)),
        )
        self.adapter: Optional[TreeGRPOAdapter] = None
        if bool(cfg.get("use_grpo", False)):
            tc = cfg.get("tree", {})
            self.adapter = TreeGRPOAdapter(
                self.base,
                TreeGRPOConfig(
                    M=int(tc.get("M", 2)),
                    N=int(tc.get("N", 2)),
                    L=int(tc.get("L", 1)),
                    use_zscore_intra=bool(tc.get("use_zscore_intra", False)),
                    use_zscore_inter=bool(tc.get("use_zscore_inter", True)),
                    value_alpha=float(tc.get("value_alpha", 0.0)),
                    scorer_name=str(tc.get("scorer_name", "sicql")),
                    dimensions=list(tc.get("dimensions", ["alignment"])),
                ),
            )

    async def run(self, context: Dict) -> Dict:
        return await self.base.run(context)

    async def rollout(self, context: Dict) -> Dict:
        if not self.adapter:
            raise RuntimeError("TreeWrapper.rollout called but adapter is None (use_grpo=false)")
        return await self.adapter.rollout_forest(context)
