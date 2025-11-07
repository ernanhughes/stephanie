# stephanie/components/nexus/blossom_runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.tree.core import AgenticTreeSearch
from stephanie.components.tree.tree_grpo import TreeGRPOAdapter, TreeGRPOConfig
from stephanie.memory.blossom_store import BlossomStore
from stephanie.memory.prompt_store import PromptStore  # optional


@dataclass
class BlossomRunConfig:
    M: int = 2
    N: int = 2
    L: int = 1
    return_top_k: int = 1
    use_zscore_inter: bool = True
    use_zscore_intra: bool = False
    value_alpha: float = 0.0
    sharpen_top_k: int = 0  # 0 = disabled


class BlossomRunnerAgent(BaseAgent):
    name = "blossom_runner"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.blossoms: BlossomStore = self.memory.blossoms
        self.prompts: Optional[PromptStore] = getattr(memory, "prompts", None)
        self.scoring = container.get("scoring")

        # wire base search
        self.base_search: AgenticTreeSearch = container.get(
            "agentic_tree_search"
        )
        tcfg = TreeGRPOConfig(
            M=int(cfg.get("M", 2)),
            N=int(cfg.get("N", 2)),
            L=int(cfg.get("L", 1)),
            use_zscore_inter=bool(cfg.get("use_zscore_inter", True)),
            use_zscore_intra=bool(cfg.get("use_zscore_intra", False)),
            value_alpha=float(cfg.get("value_alpha", 0.0)),
            return_top_k=int(cfg.get("return_top_k", 1)),
        )
        self.adapter = TreeGRPOAdapter(self.base_search, tcfg)
        self.adapter.cfg.on_new_node = self._record_node_edge  # live write

    # ---- public entry -----------------------------------------------------
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # 0) open blossom episode
        seed_info = context.get("seed", {})
        goal = context.get("goal", {})
        blossom = self.blossoms.open_episode(
            goal_text=goal.get("goal_text", ""),
            seed_meta=seed_info,
            pipeline_run_id=context.get("pipeline_run_id"),
        )

        # 1) prepare seed plans (text) if provided
        seed_plans = []
        if "plan_text" in seed_info:
            seed_plans = [seed_info["plan_text"]]
        elif "plans" in seed_info and isinstance(seed_info["plans"], list):
            seed_plans = [str(p) for p in seed_info["plans"]]

        # 2) roll out forest
        forest_out = await self.adapter.rollout_forest(
            {**context, "seed_plans": seed_plans or None}
        )

        top_paths = forest_out["top_paths"]
        top_leaves = forest_out["top_leaves"]  # [(leaf_id, reward)]

        # 3) optional sharpening of top leaves
        sharpened = []
        if int(self.cfg.get("sharpen_top_k", 0)) > 0 and top_paths:
            sharpened = await self._sharpen_top_leaves(top_paths, context)

        # 4) persist winners to blossom + emit nexus merge request
        winners = []
        for i, (leaf_id, reward) in enumerate(top_leaves):
            path_ids = top_paths[i]
            winners.append(
                {
                    "leaf_id": leaf_id,
                    "reward": float(reward),
                    "path": path_ids,
                    "sharpened": (
                        sharpened[i] if i < len(sharpened) else None
                    ),
                }
            )
            self.blossoms.add_winner(
                blossom.id,
                path_ids,
                reward,
                sharpened=(sharpened[i] if i < len(sharpened) else None),
            )

        self.blossoms.close_episode(
            blossom.id,
            status="completed",
            stats={
                "num_nodes": len(forest_out["nodes"]),
                "top_reward": float(top_leaves[0][1]) if top_leaves else None,
            },
        )

        # 5) return a compact merge request for NexusWriter
        context["blossom_result"] = {
            "episode_id": blossom.id,
            "winners": winners,
            "training_batch": forest_out[
                "training_batch"
            ],  # for GRPO/DPO updates
        }
        return context

    # ---- live-recording hook ---------------------------------------------
    def _record_node_edge(self, node) -> None:
        """Called for each new node by adapter; write node/edge into blossom store."""
        try:
            self.blossoms.add_node(
                plan_text=node.plan,
                node_id=node.id,
                parent_id=node.parent_id,
                root_id=node.root_id or node.id,
                depth=node.depth,
                node_type=node.node_type,
                metric=node.metric,
            )
            if node.parent_id:
                self.blossoms.add_edge(
                    node.parent_id, node.id, kind=node.node_type
                )
        except Exception as e:
            self.logger.log(
                "BlossomRecordError",
                {"node_id": getattr(node, "id", None), "error": str(e)},
            )

    # ---- optional sharpening ---------------------------------------------
    async def _sharpen_top_leaves(
        self, top_paths: List[List[str]], context: Dict[str, Any]
    ):
        """Run your sharpening loop on the leaf plans (last plan of each path)."""
        out = []
        for path in top_paths:
            leaf = self.base_search.nodes_by_id[path[-1]]
            prompt = self._build_sharpen_prompt(
                context.get("goal", {}).get("goal_text", ""), leaf.plan
            )
            improved = self.call_llm(prompt, context=context).strip()
            # quick score
            r = (
                float(self.scoring.score_text(improved, context=context))
                if self.scoring
                else 0.0
            )
            out.append(
                {"original": leaf.plan, "sharpened": improved, "score": r}
            )
        return out

    def _build_sharpen_prompt(self, goal: str, plan: str) -> str:
        return f"""Improve the following plan for the goal:
GOAL: {goal}

PLAN:
\"\"\"{plan}\"\"\"

Rewrite the plan to be clearer, safer, and more goal-aligned. Keep the same structure and steps."""
