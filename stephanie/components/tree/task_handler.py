# stephanie/components/tree/task_handler.py
"""
Routes task types (code, prompt, summarization, agent-call)
to their appropriate execution handlers.
"""

from __future__ import annotations

from typing import Any, Dict
from stephanie.components.tree.task_executor import TaskExecutor
from stephanie.components.tree.output_verifier import OutputVerifier
from stephanie.components.tree.plan_generator import PlanGenerator


class TaskHandler:
    def __init__(self, agent, task_executor: TaskExecutor, verifier: OutputVerifier, plan_gen: PlanGenerator):
        self.agent = agent
        self.task_executor = task_executor
        self.verifier = verifier
        self.plan_gen = plan_gen

        self.handlers = {
            "prompt_improvement": self._handle_prompt_improvement,
            "code_compile": self._handle_code_compile,
            "summarize": self._handle_summarize,
            "call_agent": self._handle_call_agent,
        }

    # --------------------------------------------------------------- #
    async def handle(self, task_type: str, plan: str, context: dict) -> Dict[str, Any]:
        """Entry point: dispatch a task to its appropriate handler."""
        handler = self.handlers.get(task_type)
        if not handler:
            raise ValueError(f"Unknown task type: {task_type}")
        try:
            result = await handler(plan, context)
            result["task_type"] = task_type
            return result
        except Exception as e:
            return {"error": str(e), "metric": None, "summary": "Task failed."}

    # --------------------------------------------------------------- #
    async def _handle_prompt_improvement(self, plan: str, context: dict) -> Dict[str, Any]:
        """Refine a prompt using the LLM and verify the improvement."""
        result = await self.task_executor.execute_task(plan, context)
        return self.verifier.verify(result, "", False)

    async def _handle_code_compile(self, plan: str, context: dict) -> Dict[str, Any]:
        """Score the plan as text instead of running code."""
        result = await self.task_executor.execute_task(plan, context)
        result["is_bug"] = False
        return result

    async def _handle_summarize(self, plan: str, context: dict) -> Dict[str, Any]:
        """Summarize any given text content."""
        prompt = f"Summarize the following text clearly and concisely:\n\n{plan}"
        summary = await self.agent.async_call_llm(prompt, context=context)
        return {
            "metric": len(summary) / max(1, len(plan)),
            "summary": summary.strip(),
            "merged_output": summary.strip(),
            "is_bug": False,
        }

    async def _handle_call_agent(self, plan: str, context: dict) -> Dict[str, Any]:
        """Invoke another agent specified in context."""
        target = context.get("target_agent")
        if not target:
            raise ValueError("No target_agent specified in context.")
        subcontext = dict(context)
        subcontext["invoked_by"] = self.agent.__class__.__name__
        result = await target.async_call_llm(plan, context=subcontext)
        return self.verifier.verify(result, "", False)
