# stephanie/agents/prompt_compiler_agent.py
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, Optional, Tuple, List
from copy import deepcopy

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.agentic_tree_search import AgenticTreeSearch, ExecutionResult
from stephanie.agents.pipeline.pipeline_runner import PipelineRunnerAgent
from stephanie.constants import PIPELINE_RUN_ID
from stephanie.services.workers.metrics_worker import MetricsWorker
from stephanie.services.workers.vpm_worker import VPMWorker
from stephanie.utils.emit_broadcaster import EmitBroadcaster
import time

class PromptCompilerAgent(BaseAgent):
    """ 
    Unified, minimal prompt compiler using Agentic Tree Search (ATS) with ZeroModel timeline control.
    
    INPUT (context):
      context["goal"]["goal_text"] : natural-language description of what the final prompt should achieve
      context["knowledge"]         : optional list[str] of hints/templates to bias planning (used by ATS)

    OUTPUT (context):
      context["final_prompt"]         : best prompt string
      context["final_prompt_metric"]  : score in [0,1]
      context["final_prompt_summary"] : short excerpt or explanation
      context["prompt_search_stats"]  : {"iterations", "tree_size", "best_metric"}
      context["timeline_path"]        : path to generated timeline GIF
    """

    def __init__(self, cfg, memory=None, container=None, logger=None, full_cfg=None):
        super().__init__(cfg, memory, container, logger)
        self.runner = PipelineRunnerAgent(
            cfg, memory=memory, logger=logger, container=container, full_cfg=full_cfg
        )

        # ---- Resolve ZeroModelService from the container (robustly) ----
        self.zm = container.get("zeromodel")
        if not self.zm:
            raise RuntimeError(
                "ZeroModelService not found. Please ensure it is registered in the container."
            )

        # --- Config knobs (with sensible defaults) ---
        self.ats_cfg = {
            "N_init":               cfg.get("ats_N_init", 4),
            "max_iterations":       cfg.get("ats_max_iter", 80),
            "time_limit":           cfg.get("ats_time_limit", 1200),  # 20 min
            "no_improve_patience":  cfg.get("ats_patience", 25),
            "H_debug":              0.0,   # not executing user code here
            "H_greedy":             cfg.get("ats_H_greedy", 0.5),
            "C_ucb":                cfg.get("ats_C_ucb", 1.2),
            "random_seed":          cfg.get("random_seed", 42),
        }

        # How to extract score/text from the judge output (dot-paths)
        self.score_path: str = cfg.get("score_path", "selected.score")
        self.text_path:  str = cfg.get("text_path",  "selected.text")

        # Optional original score range -> normalize to [0,1], e.g., [0,100]
        self.score_range: Optional[Tuple[float, float]] = None
        rng = cfg.get("score_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            try:
                a, b = float(rng[0]), float(rng[1])
                if a != b:
                    self.score_range = (a, b)
            except Exception:
                pass

        # Optional: let caller inject explicit pipeline_stages
        self.pipeline_stages: Optional[List[Dict[str, Any]]] = cfg.get("pipeline_stages")

        # ---- ATS + timeline sink wiring ----
        self._goal_text: str = ""
        self.run_id: str = ""

        self.metrics_worker = MetricsWorker(cfg, memory, container, logger)
        self.vpm_worker = VPMWorker(cfg, memory, container, logger)
        msubj_req   = (cfg.get("metrics", {}) or {}).get("subjects", {}).get("request", "arena.metrics.request")
        ats_report  = (cfg.get("zeromodel", {}) or {}).get("subjects", {}).get("ats_report", "arena.ats.report")

        async def _timeline_sink(event: str, payload: Dict[str, Any]) -> None:
            """Append rows on 'node'; finalize on 'report'. Never raise."""
            if not self.run_id:
                return
            try:
                if event == "node":
                    # 1) append to ZeroModel timeline (optional, if you want the immediate line)
                    node = payload.get("node", payload)
                    extra = {
                        "value": payload.get("value"),
                        "best_metric": payload.get("best_metric"),
                    }
                    self.zm.timeline_append_row(self.run_id, node=node, extra=extra)

                    # 2) publish a metrics job for the worker
                    await self.memory.bus.publish(
                        subject=msubj_req,
                        payload={
                            "run_id": self.run_id,
                            "node_id": node.get("id"),
                            "parent_id": node.get("parent_id"),
                            "action_type": node.get("type", "draft"),
                            "goal_text": self._goal_text,
                            "prompt_text": node.get("code") or node.get("plan") or node.get("text") or payload.get("prompt") or "",
                            "best_metric": node.get("metric"),
                            "bug": bool(node.get("bug", False)),
                            "ts_enqueued": time.time(),
                        },
                    )

                elif event == "report":
                    await self.memory.bus.publish(
                        subject=ats_report,
                        payload={"run_id": self.run_id},
                    )
            except Exception:
                # Telemetry must never stop the run
                pass


        self.ats = AgenticTreeSearch(
            agent=self,
            N_init=self.ats_cfg["N_init"],
            max_iterations=self.ats_cfg["max_iterations"],
            time_limit=self.ats_cfg["time_limit"],
            no_improve_patience=self.ats_cfg["no_improve_patience"],
            H_debug=self.ats_cfg["H_debug"],
            H_greedy=self.ats_cfg["H_greedy"],
            C_ucb=self.ats_cfg["C_ucb"],
            metric_fn=lambda m: 0.0 if m is None else float(m),
            emit_cb=EmitBroadcaster(self._emit_to_logger, _timeline_sink),
            random_seed=self.ats_cfg["random_seed"],
        )

        # Override ATS execution & verification to use prompt scoring
        self.ats.execute_code = self._execute_prompt  # type: ignore[assignment]
        self.ats.verifier.verify = self._verify_prompt  # type: ignore[assignment]

        self.start_time: float = time.time()

    # ---------------------------------------------------------------------
    # Sink 1/2 for EmitBroadcaster: safe logger
    # ---------------------------------------------------------------------
    def _emit_to_logger(self, event: str, payload: Dict[str, Any]) -> None:
        """
        Lightweight logging sink. Must not raise. Works whether self.logger is stdlib-like or a custom reporter.
        """
        try:
            if not self.logger:
                return
            # Prefer a structured log method if present; fall back to info()
            if hasattr(self.logger, "log") and callable(getattr(self.logger, "log")):
                self.logger.log(f"PromptCompiler::{event}", payload)
            elif hasattr(self.logger, "info"):
                self.logger.info("PromptCompiler::%s %s", event, payload)
        except Exception:
            # swallow logging errors by design
            pass

    # ---------------------------------------------------------------------
    # ATS hook: "execute" a prompt candidate by sending it to your judge.
    # ---------------------------------------------------------------------
    async def _execute_prompt(self, prompt_candidate: str) -> ExecutionResult:
        try:
            request_ctx = deepcopy(getattr(self, "_base_context", {}))
            request_ctx["goal"] = {"goal_text": self._goal_text}
            request_ctx["current_thought"] = (prompt_candidate or "").strip()

            if self.pipeline_stages:
                request_ctx["pipeline_stages"] = self.pipeline_stages

            result = await self.runner.run(request_ctx)

            score = self._dig(result, self.score_path)
            text  = self._dig(result, self.text_path) or ""

            metric = self._normalize_score(score)
            payload = {
                "selected_text": text,
                "score": metric,   # already normalized in [0,1]
                "raw_score": score,
            }
            return ExecutionResult(
                stdout=json.dumps(payload),
                stderr="",
                returncode=0,
                has_submission_file=False,
            )
        except Exception as e:
            return ExecutionResult(stdout="", stderr=str(e), returncode=1, has_submission_file=False)

    # ---------------------------------------------------------------------
    # ATS hook: verify/parse the judge result
    # ---------------------------------------------------------------------
    def _verify_prompt(self, stdout: str, stderr: str, has_submission_file: bool) -> Dict[str, Any]:
        metric: Optional[float] = None
        summary: str = ""
        try:
            if stdout:
                obj = json.loads(stdout)
                metric = obj.get("score")
                summary = (obj.get("selected_text") or "")[:500]
        except Exception:
            summary = (stdout or "")[:500]

        if metric is not None:
            try:
                metric = max(0.0, min(1.0, float(metric)))
            except Exception:
                metric = 0.0

        return {
            "is_bug": False,
            "is_overfitting": False,
            "has_csv_submission": False,
            "metric": metric,
            "summary": summary,
            "merged_output": stdout if not stderr else f"{stdout}\n--- STDERR ---\n{stderr}",
        }

    # ---------------------------------------------------------------------
    # BaseAgent integration & public API
    # ---------------------------------------------------------------------
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        await self.metrics_worker.start()
        await self.vpm_worker.start()

        goal = context.get("goal", {}) or {}
        goal_text = (goal.get("goal_text") or "").strip()

        if not goal_text:
            raise ValueError("Missing 'goal.goal_text' in context")

        # Run ID: prefer given PIPELINE_RUN_ID, else generate
        self.run_id = context.get(PIPELINE_RUN_ID) or str(uuid.uuid4())
        context[PIPELINE_RUN_ID] = self.run_id

        # Open timeline session (direct, in-process)
        self.zm.timeline_open(
            self.run_id,
            metrics=["metric","value","visits","bug","action_draft","action_improve","action_debug"],
        )

        # Keep originals for the runner
        self._base_context = context
        self._goal_text = goal_text

        # Run the search
        result_ctx = await self.ats.run(context)

        # Finalize timeline (defensive: in case no 'report' was emitted)
        finalize_res = await self.zm.timeline_finalize(self.run_id)
        timeline_path = (
            finalize_res.get("gif") or finalize_res.get("output_path") or
            # fallback: look for last generated file in out_dir
            getattr(self.zm, "last_output_for", lambda _rid: "")(self.run_id)
        )

        # Best solution
        best = result_ctx.get("final_solution") or {}
        context["final_prompt"] = best.get("code") or best.get("plan") or ""
        context["final_prompt_metric"] = best.get("metric")
        context["final_prompt_summary"] = best.get("summary")
        context["timeline_path"] = timeline_path

        # Accurate stats
        context["prompt_search_stats"] = {
            "iterations": getattr(self.ats, "iteration", 0),
            "tree_size":  len(getattr(self.ats, "tree", []) or []),
            "best_metric": getattr(self.ats, "best_metric", None),
        }

        return context

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _dig(self, obj: Any, path: str) -> Any:
        try:
            cur = obj
            for part in path.split("."):
                if isinstance(cur, dict):
                    cur = cur.get(part)
                else:
                    return None
            return cur
        except Exception:
            return None

    def _normalize_score(self, score: Any) -> Optional[float]:
        if score is None:
            return None
        try:
            val = float(score)
        except Exception:
            return None
        if self.score_range:
            lo, hi = self.score_range
            if hi != lo:
                val = (val - lo) / (hi - lo)
            else:
                val = 0.0
        return max(0.0, min(1.0, val))


