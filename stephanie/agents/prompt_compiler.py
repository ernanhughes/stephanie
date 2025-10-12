# stephanie/agents/prompt_compiler.py
"""
PromptCompilerAgent — Using Agentic Tree Search to Evolve Prompts
=================================================================
This module demonstrates a *real* integration of Agentic Tree Search (ATS) for
optimizing prompts against a scoring stack (e.g., SICQL/MRQ/EBT/HRM/LLM).

What it does
------------
- Takes a natural-language goal (what the final prompt should achieve).
- Uses ATS to explore candidate prompts (draft → improve cycles).
- Scores each candidate with your configured scorer(s).
- Streams telemetry to the event bus and records a ZeroModel timeline (VPM GIF).
- Returns the best prompt, its score, a summary, and search stats.

Inputs (context)
----------------
- context["goal"]["goal_text"] : str (required)
- context["knowledge"]         : Optional[List[str]] (optional hints for ATS)

Outputs (context)
-----------------
- context["final_prompt"]          : str (best discovered prompt)
- context["final_prompt_metric"]   : float in [0,1] (best score)
- context["final_prompt_summary"]  : str (short rationale/summary from verifier)
- context["prompt_search_stats"]   : Dict[str, Any] (iterations/tree_size/best_metric)
- context["timeline_path"]         : str (path to generated ZeroModel GIF)

Dependencies
------------
- AgenticTreeSearch (stephanie.components.tree.core)
- OutputVerifier, TaskExecutor, TaskHandler
- Scoring workers: MetricsWorker & VPMWorker (bus-driven)
- ZeroModelService available in container at key "zeromodel"
- A bus attached to `self.memory.bus` (NATS or in-process dev bus)

Quickstart
----------
agent = PromptCompilerAgent(cfg, memory, container, logger, full_cfg)
ctx = {"goal": {"goal_text": "Write a prompt to extract 5 claims from any paper."}}
result = await agent.run(ctx)
print(result["final_prompt"], result["final_prompt_metric"], result["timeline_path"])

Notes
-----
- This agent assumes a run_id is placed into the context under PIPELINE_RUN_ID
  upstream. If not present, you may want to generate one (e.g., uuid.uuid4().hex).
- Debugging is disabled (H_debug=0.0) because this workflow doesn’t execute
  untrusted user code; only prompts are being “executed” (scored).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.tree.core import AgenticTreeSearch
from stephanie.components.tree.output_verifier import OutputVerifier
from stephanie.components.tree.task_executor import TaskExecutor
from stephanie.components.tree.task_handler import TaskHandler
from stephanie.constants import PIPELINE_RUN_ID
from stephanie.scoring.scorable import ScorableType
from stephanie.services.workers.metrics_worker import MetricsWorker
from stephanie.services.workers.vpm_worker import VPMWorker
from stephanie.utils.emit_broadcaster import EmitBroadcaster

_logger = logging.getLogger(__name__)


class PromptCompilerAgent(BaseAgent):
    """
    Unified, minimal prompt compiler using Agentic Tree Search (ATS) with
    ZeroModel timeline control and bus-based telemetry.

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

    def __init__(self, cfg, memory, container, logger, full_cfg):
        # BaseAgent boot (wires cfg/memory/container/logger)
        super().__init__(cfg, memory, container, logger)

        # ---- ZeroModelService from the container ----
        # Used to open/append/finalize ATS timelines → produces a VPM GIF.
        self.zm = container.get("zeromodel")

        # --- Config knobs (with sensible defaults) ---
        # These propagate into ATS behavior; tune per domain/task.
        self.ats_cfg = {
            "N_init": cfg.get(
                "ats_N_init", 4
            ),  # how many initial drafts to seed frontier
            "max_iterations": cfg.get(
                "ats_max_iter", 80
            ),  # hard cap on iterations
            "time_limit": cfg.get(
                # "ats_time_limit", 1200
                "ats_time_limit", 120

            ),  # 20 min; wall-clock cap
            "no_improve_patience": cfg.get(
                "ats_patience", 25
            ),  # early stop if no better metric
            "H_debug": 0.0,  # debugging disabled (no code execution here)
            "H_greedy": cfg.get(
                "ats_H_greedy", 0.5
            ),  # probability to improve current best
            "C_ucb": cfg.get("ats_C_ucb", 1.2),  # exploration constant
            "random_seed": cfg.get("random_seed", 42),  # determinism for tests
        }

        # How to extract score/text from your judge output (if you parse nested outputs)
        # e.g., result = {"selected":{"score":0.83, "text":"..."}}
        self.score_path: str = cfg.get("score_path", "selected.score")
        self.text_path: str = cfg.get("text_path", "selected.text")

        # Optional original score range -> normalize to [0,1], e.g., [0,100]
        # If your scorer returns percentages or a custom range, set it in cfg.
        self.score_range: Optional[Tuple[float, float]] = None
        rng = cfg.get("score_range", "[0,100]")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            try:
                a, b = float(rng[0]), float(rng[1])
                if a != b:
                    self.score_range = (a, b)
            except Exception:
                # silently ignore malformed values; proceed unnormalized
                pass

        # Optional: let caller inject explicit pipeline_stages (not used below, but reserved)
        self.pipeline_stages: Optional[List[Dict[str, Any]]] = cfg.get(
            "pipeline_stages"
        )

        # ---- ATS + timeline sink wiring ----
        self._goal_text: str = ""
        self.run_id: str = (
            ""  # NOTE: expected to be provided via context[PIPELINE_RUN_ID]
        )

        # Background workers used for metrics & VPM (bus-connected)
        self.metrics_worker = MetricsWorker(cfg, memory, container, logger)
        self.vpm_worker = VPMWorker(cfg, memory, container, logger)

        # Subjects: where to publish metrics jobs and ATS final reports
        msubj_req = (
            (cfg.get("metrics", {}) or {})
            .get("subjects", {})
            .get("request", "arena.metrics.request")
        )
        ats_report = (
            (cfg.get("zeromodel", {}) or {})
            .get("subjects", {})
            .get("ats_report", "arena.ats.report")
        )

        async def _timeline_sink(event: str, payload: Dict[str, Any]) -> None:
            """
            EmitBroadcaster sink:
            - On each 'node' event: enqueue a metrics job to the bus.
            - On final 'report': emit a small ATS report notification.
            Never raises (telemetry must not interrupt the run).
            """
            if not self.run_id:
                return
            try:
                if event == "node":
                    # Payload may be a node dict or wrapped; normalize
                    node = payload.get("node", payload)

                    # Publish a metrics job for workers to process asynchronously
                    await self.memory.bus.publish(
                        subject=msubj_req,
                        payload={
                            "run_id": self.run_id,
                            "node_id": node.get("id"),
                            "parent_id": node.get("parent_id"),
                            "action_type": node.get("type", "draft"),
                            "goal_text": self._goal_text,
                            "prompt_text": payload.get(
                                "prompt_text", node.get("plan")
                            ),
                            "best_metric": node.get("metric"),
                            "bug": bool(node.get("bug", False)),
                            "ts_enqueued": time.time(),
                        },
                    )

                elif event == "report":
                    # Notify interested listeners that ATS report is available
                    await self.memory.bus.publish(
                        subject=ats_report,
                        payload={"run_id": self.run_id},
                    )
            except Exception as e:
                # Telemetry must never stop the run
                _logger.warning("Error publishing event: %s", e)

        # Instantiate ATS with broadcasting emit (logger + timeline sink)
        self.ats = AgenticTreeSearch(
            agent=self,
            N_init=self.ats_cfg["N_init"],
            max_iterations=self.ats_cfg["max_iterations"],
            time_limit=self.ats_cfg["time_limit"],
            no_improve_patience=self.ats_cfg["no_improve_patience"],
            H_debug=self.ats_cfg["H_debug"],
            H_greedy=self.ats_cfg["H_greedy"],
            C_ucb=self.ats_cfg["C_ucb"],
            metric_fn=lambda m: 0.0
            if m is None
            else float(m),  # reward mapping
            emit_cb=EmitBroadcaster(self._emit_to_logger, _timeline_sink),
            random_seed=self.ats_cfg["random_seed"],
        )

        # ---- Override ATS execution & verification to use prompt scoring ----
        # These flags bind the search to "prompt" artifacts and chosen scorers.
        self.user_scoring = cfg.get("user_scoring", True)
        self.scorable_type = ScorableType.PROMPT
        self.scorer = cfg.get("scorer", "sicql")  # default scorer name
        self.dimensions = cfg.get("dimensions", ["alignment"])

        # --- Unified task-based execution layer ---
        # Re-wire ATS with a fresh TaskExecutor/Handler tailored to this agent.
        self.ats.task_executor = TaskExecutor(
            agent=self,
            container=container,
            verifier=OutputVerifier(),  # extracts scalar metric + summary
        )

        self.ats.task_handler = TaskHandler(
            agent=self,
            task_executor=self.ats.task_executor,
            verifier=self.ats.verifier,
            plan_gen=self.ats.plan_generator,
        )

        # Track start time for any external monitoring
        self.start_time: float = time.time()

    # ---------------------------------------------------------------------
    # BaseAgent integration & public API
    # ---------------------------------------------------------------------
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for compiling/improving a prompt via ATS.
        - Starts workers
        - Opens a ZeroModel timeline
        - Runs ATS
        - Finalizes timeline and returns best prompt + stats
        """
        # Ensure workers are running (idempotent in most implementations)
        await self.metrics_worker.start()
        await self.vpm_worker.start()

        # Validate goal text
        goal = context.get("goal", {}) or {}
        goal_text = (goal.get("goal_text") or "").strip()
        if not goal_text:
            raise ValueError("Missing 'goal.goal_text' in context")

        # Run ID: prefer given PIPELINE_RUN_ID, else use existing in context
        # NOTE: If neither present, consider generating one:
        # self.run_id = context.get(PIPELINE_RUN_ID) or uuid.uuid4().hex
        self.run_id = context.get(PIPELINE_RUN_ID)
        context[PIPELINE_RUN_ID] = self.run_id

        # Open a new timeline session for this run_id
        # (In earlier versions you passed metrics list; now zero-arg open is fine.)
        self.zm.timeline_open(self.run_id)

        # Keep originals for downstream components (optional)
        self._base_context = context
        self._goal_text = goal_text

        # Prepare execution context for ATS
        context["task_type"] = "prompt_improvement"
        context["scorer"] = self.scorer
        context["dimensions"] = self.dimensions
        context["goal"]["goal_text"] = goal_text
        context["user_scoring"] = self.user_scoring

        # Run the ATS search loop
        result_ctx = await self.ats.run(context)

        # Finalize the timeline (defensive: also done on 'report', but idempotent)
        finalize_res = await self.zm.timeline_finalize(self.run_id)
        timeline_path = (
            finalize_res.get("gif")
            or finalize_res.get("output_path")
            or getattr(self.zm, "last_output_for", lambda _rid: "")(
                self.run_id
            )
        )

        # Extract the best solution from ATS result
        best = result_ctx.get("final_solution") or {}
        context["final_prompt"] = best.get("code") or best.get("plan") or ""
        context["final_prompt_metric"] = best.get("metric")
        context["final_prompt_summary"] = best.get("summary")
        context["timeline_path"] = timeline_path

        # Accurate search stats for dashboards/logging
        context["prompt_search_stats"] = {
            "iterations": getattr(self.ats, "iteration", 0),
            "tree_size": len(getattr(self.ats, "tree", []) or []),
            "best_metric": getattr(self.ats, "best_metric", None),
        }

        return context

    # ---------------------------------------------------------------------
    # Sink 1/2 for EmitBroadcaster: safe logger
    # ---------------------------------------------------------------------
    def _emit_to_logger(self, event: str, payload: Dict[str, Any]) -> None:
        """
        Synchronous sink used by EmitBroadcaster to mirror ATS events into logs.
        Never raises; avoid blocking.
        """
        self.logger.log(f"PromptCompiler::{event}", payload)
