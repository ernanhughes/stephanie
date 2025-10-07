# stephanie/agents/prompt_compiler_agent.py
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, Optional, Tuple, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.agentic_tree_search import AgenticTreeSearch, ExecutionResult
from stephanie.agents.pipeline.pipeline_runner import PipelineRunnerAgent
from copy import deepcopy


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
        self.runner = PipelineRunnerAgent(cfg, memory=memory, logger=logger, container=container, full_cfg=full_cfg)
        
        # Initialize ZeroModelService RPC client
        self.zm_service = self.memory.services.get("zeromodel-service-v1")
        if not self.zm_service:
            raise RuntimeError("ZeroModelService not available in memory")
        
        # --- Config knobs (with sensible defaults) ---
        self.ats_cfg = {
            "N_init":            cfg.get("ats_N_init", 4),
            "max_iterations":    cfg.get("ats_max_iter", 80),
            "time_limit":        cfg.get("ats_time_limit", 1200),  # 20 min
            "no_improve_patience": cfg.get("ats_patience", 25),
            "H_debug":           0.0,   # not executing code
            "H_greedy":          cfg.get("ats_H_greedy", 0.5),
            "C_ucb":             cfg.get("ats_C_ucb", 1.2),
            "random_seed":       cfg.get("random_seed", 42),
        }

        # How to extract score/text from the judge output (dot-paths)
        # Default aligns with: {"selected": {"text": "...", "score": <float 0..1>}}
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

        # Build ATS with ZeroModel timeline emitter
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
            emit_cb=self._emit_to_zero_model,  # New timeline-aware emitter
            random_seed=self.ats_cfg["random_seed"],
        )

        # Override ATS execution & verification to use prompt scoring
        self.ats.execute_code = self._execute_prompt  # type: ignore[assignment]
        self.ats.verifier.verify = self._verify_prompt  # type: ignore[assignment]

        self._goal_text: str = ""
        self.run_id: str = ""  # Will be set in run()

    # ---------------------------------------------------------------------
    # ATS hook: "execute" a prompt candidate by sending it to your judge.
    # ---------------------------------------------------------------------
    async def _execute_prompt(self, prompt_candidate: str) -> ExecutionResult:
        """
        Score a prompt candidate using your pipeline judge.
        - We pass through the *entire* input context (so the judge can see knowledge, etc.)
        - We add `current_thought` = the candidate prompt
        - Optionally include configured pipeline_stages
        """
        try:
            # Base context: carry the original inputs through to the runner
            request_ctx = deepcopy(getattr(self, "_base_context", {}))
            request_ctx["goal"] = {"goal_text": self._goal_text}
            request_ctx["current_thought"] = (prompt_candidate or "").strip()

            if self.pipeline_stages:
                request_ctx["pipeline_stages"] = self.pipeline_stages

            result = await self.runner.run(request_ctx)

            # Extract score and text via configured paths
            score = self._dig(result, self.score_path)
            text  = self._dig(result, self.text_path) or ""

            # Normalize score to [0,1]
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
        """Parse normalized score from JSON stdout; assume no code bugs."""
        metric: Optional[float] = None
        summary: str = ""
        try:
            if stdout:
                obj = json.loads(stdout)
                metric = obj.get("score")
                # prefer the judge-selected text as a brief summary
                summary = (obj.get("selected_text") or "")[:500]
        except Exception:
            summary = (stdout or "")[:500]

        # Clamp for safety
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
    # ZeroModel timeline control
    # ---------------------------------------------------------------------
    async def _emit_to_zero_model(self, event: str, payload: Dict[str, Any]) -> None:
        """Emit events to ZeroModelService for timeline generation"""
        # Only process node events for timeline
        if event == "node":
            # Add run_id to payload
            payload["run_id"] = self.run_id
            # Fire-and-forget event push
            asyncio.create_task(
                self.zm_service.request("add_event", {
                    "run_id": self.run_id,
                    "event": payload
                })
            )
        
        # Keep original logging
        if self.logger:
            try:
                self.logger.log(f"PromptCompiler::{event}", payload)
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # BaseAgent integration & public API
    # ---------------------------------------------------------------------
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        goal = context.get("goal", {}) or {}
        goal_text = (goal.get("goal_text") or "").strip()
        if not goal_text:
            raise ValueError("Missing 'goal.goal_text' in context")

        # Generate unique run ID
        self.run_id = str(uuid.uuid4())
        
        # Create timeline context
        await self.zm_service.request("create_timeline", {
            "run_id": self.run_id,
            "config": {
                "metrics": ["metric", "draft", "improve", "debug"],
                "options": {
                    "title": f"Prompt Search (Run: {self.run_id[:8]})",
                    "x_label": "Time (seconds)",
                    "y_label": "Metric Value",
                    "bar_alpha": 0.3
                }
            }
        })
        
        # Keep originals for the runner
        self._base_context = context
        self._goal_text = goal_text

        # Run the search
        result_ctx = await self.ats.run(context)

        # Finalize timeline
        timeline_result = await self.zm_service.request("finalize_timeline", {
            "run_id": self.run_id
        })
        
        # Best solution
        best = result_ctx.get("final_solution") or {}
        context["final_prompt"] = best.get("code") or best.get("plan") or ""
        context["final_prompt_metric"] = best.get("metric")
        context["final_prompt_summary"] = best.get("summary")
        context["timeline_path"] = timeline_result.get("timeline_path", "")
        context["prompt_search_stats"] = {
            "iterations": getattr(self.ats, "iteration", 0),
            "tree_size": len(getattr(self.ats, "tree", []) or []),
            "best_metric": getattr(self.ats, "best_metric", None),
        }

        return context

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _dig(self, obj: Any, path: str) -> Any:
        """
        Retrieve nested value by dot-path (e.g., "selected.score").
        Returns None if anything is missing.
        """
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
        """Map any numeric score to [0,1] using optional score_range, else assume already [0,1]."""
        if score is None:
            return None
        try:
            val = float(score)
        except Exception:
            return None

        if self.score_range:
            lo, hi = self.score_range
            if hi == lo:
                return 0.0
            # linear map to [0,1]
            val = (val - lo) / (hi - lo)
        # final clamp
        return max(0.0, min(1.0, val))