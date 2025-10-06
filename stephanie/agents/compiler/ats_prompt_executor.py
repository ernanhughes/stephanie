# stephanie/agents/compiler/ats_prompt_executor.py
from __future__ import annotations
import json
from typing import Dict, Any

from stephanie.agents.agentic_tree_search import ExecutionResult
from stephanie.agents.pipeline.pipeline_runner import PipelineRunnerAgent


class PromptExecutor:
    """
    Bridges AgenticTreeSearch to your existing prompt/pipeline world.

    Given a PLAN (string), build/choose the final prompt (here we use the plan
    directly), run your PipelineRunnerAgent or judge, and return an ExecutionResult
    whose stdout contains a JSON payload with a numeric `score` in [0,1].

    You can adapt this to your schema by editing the `execute` method below.
    """

    def __init__(self, runner: PipelineRunnerAgent):
        self.runner = runner

    async def execute(self, plan_text: str, context: Dict[str, Any]) -> ExecutionResult:
        """Execute a plan/prompt via your pipeline/judge and return a score.

        Expected runner output shape (example):
            {
              "selected": {
                 "text": "... the evaluated output ...",
                 "score": 0.83   # fraction in [0,1]
              }
            }
        """
        # In simplest form, treat plan text as the prompt to evaluate.
        prompt = (plan_text or "").strip()

        # Build a minimal pipeline request that your runner understands.
        merged_ctx = {
            **(context or {}),
            "current_thought": prompt,
            # This is just a sketch; update with your actual pipeline stages if needed.
            "pipeline_stages": [
                {
                    "name": "score",
                    "type": "stephanie.agents.scoring.PipelineJudgeAgent",
                    "config": {"input_key": "current_thought", "output_key": "selected"},
                }
            ],
        }
        try:
            result = await self.runner.run(merged_ctx)  # type: ignore[attr-defined]
            selected = (result or {}).get("selected", {})
            payload = {
                "selected_text": selected.get("text"),
                "score": selected.get("score"),  # must be numeric in [0,1]
            }
            return ExecutionResult(stdout=json.dumps(payload), stderr="", returncode=0, has_submission_file=False)
        except Exception as e:
            return ExecutionResult(stdout="", stderr=str(e), returncode=1, has_submission_file=False)




