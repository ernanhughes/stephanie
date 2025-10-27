from __future__ import annotations
from typing import Dict, Any, Optional
from omegaconf import DictConfig
from stephanie.components.tree.core import AgenticTreeSearch
from stephanie.components.tree.output_verifier import OutputVerifier
from stephanie.components.ssp.tree_events import TreeEventEmitter
from stephanie.utils.json_sanitize import sanitize
from stephanie.data.plan_trace import PlanTrace
from stephanie.utils.trace_logger import trace_logger

class TreeSolverAdapter:
    """Thin wrapper that makes AgenticTreeSearch a ‘solver’ for SSP."""

    def __init__(self, cfg: DictConfig, base_agent):
        s = cfg.ssp.solver
        self.tree = AgenticTreeSearch(
            agent=base_agent,
            max_iterations=int(s.max_iterations),
            time_limit=int(s.time_limit_sec),
            N_init=int(s.N_init),
            C_ucb=float(s.C_ucb),
            H_greedy=float(s.H_greedy),
            H_debug=float(s.H_debug),
            no_improve_patience=int(s.no_improve_patience),
            emit_cb=TreeEventEmitter(),
            progress_every=int(s.progress_every),
            heartbeat_secs=float(s.heartbeat_secs),
            report_top_k=int(s.report_top_k),
        )
        v_cfg = cfg.ssp.verifier
        self.verifier = OutputVerifier(
            thresholds=dict(v_cfg.thresholds),
            require_all=bool(v_cfg.require_all),
        )
        self.task_type = cfg.ssp.solver.task_type

    async def solve(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = dict(context or {})
        ctx.setdefault("goal", {"goal_text": query})
        ctx.setdefault("task_type", self.task_type)

        # Run the tree search
        out_ctx = await self.tree.run(ctx)

        # Best solution + verification
        best = out_ctx.get("final_solution") or {}
        merged = {
            "metric": best.get("metric", 0.0),
            "summary": best.get("summary", ""),
            "merged_output": best.get("output", ""),
            "vector": {},  # optional: pass along scorer dims if present
        }
        v = self.verifier.verify(merged)
        valid = v.get("is_verified", False)
        score = float(v.get("metric", 0.0))

        # Trace
        trace_logger.log(PlanTrace(
            trace_id=f"solver-{abs(hash(query)) % 1_000_000}",
            role="solver",
            goal=query,
            status="completed" if valid else "partial",
            metadata=sanitize({
                "verification_score": score,
                "verified": valid,
                "tree_best_metric": best.get("metric"),
                "tree_size": out_ctx.get("search_tree_size"),
            }),
            input=query,
            output=best.get("summary") or best.get("output") or "",
            artifacts=sanitize({
                "final_solution": best,
                "report": out_ctx.get("search_report"),
            })
        ))

        return sanitize({
            "answer": best.get("output") or best.get("summary") or "",
            "verified": valid,
            "score": score,
            "tree_report": out_ctx.get("search_report"),
            "tree_best": best,
        })
