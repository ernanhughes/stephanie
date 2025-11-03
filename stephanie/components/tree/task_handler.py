# stephanie/components/tree/task_handler.py
"""

- _handle_call_agent: wrap string outputs into a structured dict before verify()
- Ensure every handler returns an 'is_bug' flag
"""

from __future__ import annotations

from typing import Any, Dict

from stephanie.components.tree.output_verifier import OutputVerifier
from stephanie.components.tree.plan_generator import PlanGenerator
from stephanie.components.tree.task_executor import TaskExecutor


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

    async def handle(self, task_type: str, plan: str, context: dict) -> Dict[str, Any]:
        handler = self.handlers.get(task_type)
        if not handler:
            return {"error": f"Unknown task type: {task_type}", "metric": 0.0, "summary": "Task failed.", "is_bug": True}
        try:
            result = await handler(plan, context)
            result["task_type"] = task_type
            result.setdefault("is_bug", False)
            return result
        except Exception as e:
            return {"error": str(e), "metric": 0.0, "summary": "Task failed.", "is_bug": True}

    async def _handle_prompt_improvement(self, plan: str, context: dict) -> Dict[str, Any]:
        result = await self.task_executor.execute_task(plan, context)
        out = self.verifier.verify(result, "", False)
        out.setdefault("is_bug", not out.get("is_verified", False))
        return out

    async def _handle_code_compile(self, plan: str, context: dict) -> Dict[str, Any]:
        result = await self.task_executor.execute_task(plan, context)
        result["is_bug"] = False
        return result

    async def _handle_summarize(self, plan: str, context: dict) -> Dict[str, Any]:
        prompt = f"Summarize the following text clearly and concisely:\n\n{plan}"
        summary = await self.agent.async_call_llm(prompt, context=context)
        return {
            "metric": len(summary) / max(1, len(plan)),
            "summary": summary.strip(),
            "merged_output": summary.strip(),
            "is_bug": False,
        }

    async def _handle_call_agent(self, plan: str, context: dict) -> Dict[str, Any]:
        target = context.get("target_agent")
        if not target:
            return {"error": "No target_agent specified in context.", "metric": 0.0, "summary": "Missing target", "is_bug": True}
        subcontext = dict(context)
        subcontext["invoked_by"] = self.agent.__class__.__name__
        raw = await target.async_call_llm(plan, context=subcontext)

        # wrap string outputs
        if isinstance(raw, str):
            result = {"metric": 0.0, "summary": raw[:400], "merged_output": raw, "vector": {}}
        elif isinstance(raw, dict):
            result = raw
        else:
            result = {"metric": 0.0, "summary": "Unsupported agent output", "merged_output": str(raw), "vector": {}}

        out = self.verifier.verify(result, "", False)
        out.setdefault("is_bug", not out.get("is_verified", False))
        return out
