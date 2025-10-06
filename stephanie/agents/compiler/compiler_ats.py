# stephanie/agents/compiler/compiler_ats.py
from __future__ import annotations
from typing import Any, Dict, Optional, Awaitable, Callable

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.agentic_tree_search import AgenticTreeSearch, ExecutionResult
from stephanie.agents.compiler.ats_prompt_executor import PromptExecutor
from stephanie.agents.compiler.ats_prompt_verifier import CompilerVerifier

try:
    from stephanie.agents.pipeline.pipeline_runner import PipelineRunnerAgent  # type: ignore
except Exception:
    PipelineRunnerAgent = object  # type: ignore


class CompilerATSAgent(BaseAgent):
    """
    Compiler built on top of AgenticTreeSearch.

    - Treats a "plan" as a candidate prompt/strategy description
    - Uses PromptExecutor to score plans via your PipelineRunnerAgent
    - Uses CompilerVerifier to extract a single scalar metric in [0,1]

    Returns a context with `final_prompt`, `final_prompt_metric`, and `final_prompt_summary`.
    """

    def __init__(self, cfg, memory=None, container=None, logger=None, full_cfg=None):
        super().__init__(cfg, memory, container, logger)
        self.runner = PipelineRunnerAgent(cfg, memory=memory, logger=logger, full_cfg=full_cfg)  # type: ignore[call-arg]

        # Core search engine
        self.ats = AgenticTreeSearch(
            agent=self,                 # reuse BaseAgent.llm
            N_init=4,
            max_iterations=80,
            time_limit=60 * 20,         # 20 minutes
            no_improve_patience=25,
            H_debug=0.0,                # not running code → no debug branch
            H_greedy=0.5,
            metric_fn=lambda m: 0.0 if m is None else float(m),
            emit_cb=self._emit_to_logger,
        )

        # Swap out exec + verifier for prompt arena
        self.prompt_executor = PromptExecutor(self.runner)
        self.compiler_verifier = CompilerVerifier()

        async def _execute_code_override(code: str) -> ExecutionResult:
            # In this usage, `code` carries the plan/prompt text
            return await self.prompt_executor.execute(code, context={})

        def _verify_override(stdout: str, stderr: str, has_submission_file: bool):
            v = self.compiler_verifier.verify(stdout, has_submission_file)
            # Ensure merged_output exists for ATS logging
            v["merged_output"] = stdout if not stderr else (stdout + "\n--- STDERR ---\n" + stderr)
            return v

        # Monkey-patch the behavior on the instance
        self.ats.execute_code = _execute_code_override  # type: ignore[assignment]
        self.ats.verifier.verify = _verify_override     # type: ignore[assignment]

    # BaseAgent.llm is inherited; ATS will call `await self.llm(prompt)`

    async def _emit_to_logger(self, event: str, payload: dict):
        if getattr(self, "logger", None):
            try:
                self.logger.log(f"ATS::{event}", payload)
            except Exception:
                pass

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Expect context["goal"]["goal_text"] to describe what the prompt should achieve
        out = await self.ats.run(context)
        best = out.get("final_solution") or {}
        context["final_prompt"] = best.get("code") or best.get("plan") or ""
        context["final_prompt_metric"] = best.get("metric")
        context["final_prompt_summary"] = best.get("summary")
        return context
