# stephanie/components/ssp/tree_bridge.py
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from omegaconf import DictConfig

from stephanie.components.ssp.config import ensure_cfg
from stephanie.components.tree.core import AgenticTreeSearch
from stephanie.components.tree.tree_grpo import TreeGRPOAdapter, TreeGRPOConfig
from stephanie.utils.trace_logger import get_trace_logger

# If you already have BaseAgent, import and subclass it; else use this thin shim.
try:
    from stephanie.agents.base_agent import BaseAgent  # has async_call_llm
except Exception:
    class BaseAgent:
        async def async_call_llm(self, prompt: str, context: dict) -> str:
            # very small fallback (replace with your real LLM call)
            return f"[DRAFT] {prompt[:256]}"

class SspSearchAgent(BaseAgent):
    """Tiny wrapper that provides async_call_llm and a DI container handle if needed."""
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text", "No goal provided.")
        response = await self.async_call_llm(prompt=goal_text, context=context)
        return {"response": response}

class SspTreeBridge:
    """
    Adapts the AgenticTreeSearch + TreeGRPO to SSP.
    - Builds small forests (M roots) and expansions (L rounds × N per tree)
    - Emits training-style batches (advantages) for your SSP trainer
    - Returns a report usable by Jitter/SIS dashboards
    """
    def __init__(self, cfg: DictConfig | dict, memory=None, container=None, logger=None):
        self.cfg = cfg
        self.tcfg = TreeGRPOConfig(
            M=int(self.cfg.self_play.get("M", 2)),
            N=int(self.cfg.self_play.get("N", 2)),
            L=int(self.cfg.self_play.get("L", 1)),
            scorer_name=self.cfg.self_play.get("scorer_name", "sicql"),
            dimensions=self.cfg.self_play.get("dimensions", ["alignment"]),
            use_zscore_intra=bool(self.cfg.self_play.get("use_zscore_intra", False)),
            use_zscore_inter=bool(self.cfg.self_play.get("use_zscore_inter", True)),
            value_alpha=float(self.cfg.self_play.get("value_alpha", 0.0)),
            prefer_non_buggy=bool(self.cfg.self_play.get("prefer_non_buggy", True)),
        )
        self.agent = SspSearchAgent(cfg, memory=memory,container=container, logger=logger)
        self.search = AgenticTreeSearch(
            agent=self.agent,
            max_iterations=0,   # TreeGRPO drives expansions explicitly
            N_init=0,          # roots are created by adapter
        )
        self.adapter = TreeGRPOAdapter(self.search, self.tcfg)
        self.log = get_trace_logger()

    async def rollout(self, goal_text: str, value: float = 0.0, **ctx) -> Dict[str, Any]:
        context = {
            "goal": {"goal_text": goal_text},
            "value": float(value),
            **ctx,
        }
        self.log.emit("ssp-tree-start", role="ssp.tree", status="started", meta={"goal": goal_text})
        out = await self.adapter.rollout_forest(context)
        # Light report for SIS/Jitter
        report = {
            "goal": goal_text,
            "summary": out.get("training_batch", {}).get("meta", {}),
            "root_count": len(out.get("root_ids", [])),
            "node_count": len(out.get("nodes", [])),
            "stats": {
                "adv_mean": float(sum(out["advantages"].values())/max(1,len(out["advantages"])) if out["advantages"] else 0.0),
                "rew_mean": float(sum(out["rewards"].values())/max(1,len(out["rewards"])) if out["rewards"] else 0.0),
            },
        }
        self.log.emit("ssp-tree-report", role="ssp.tree", status="completed", meta=report)
        return {"forest": out, "report": report}
