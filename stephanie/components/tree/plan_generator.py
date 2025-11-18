# stephanie/components/tree/plan_generator.py
"""
Responsible for generating, improving, and debugging plans
based on a given task description and context.
"""

from __future__ import annotations

import logging

from stephanie.agents.base_agent import BaseAgent

log = logging.getLogger(__name__)

llm_dict = {
    "name": "ollama/qwen:0.5b",
    # "name": "ollama/qwen3",
    "api_base": "http://localhost:11434",
    "api_key": None,
}


class PlanGenerator:
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    # --------------------------------------------------------------- #
    # Generic Plan Lifecycle
    # --------------------------------------------------------------- #
    async def draft_plan(self, task_description: str, context: dict, llm_cfg: dict = llm_dict) -> str:
        """Generate an initial plan for the given task."""
        style = context.get("plan_style", "analytical")
        knowledge = context.get("knowledge", [])
        prompt = f"""
You are an expert assistant.
Create a detailed, step-by-step plan for this task in a {style} style.
Do not include any code unless explicitly required.
Task:
{task_description}

Relevant knowledge: {" ".join(knowledge) if knowledge else "None"}
Return only the plan text.
"""
        response = await self.agent.async_call_llm(
            prompt, context=context, llm_cfg=llm_cfg
        )
        log.info(f"Drafted plan: {response}")
        return response.strip()

    async def improve_plan(
        self, previous_plan: str, feedback: str, context: dict, llm_cfg: dict = llm_dict
    ) -> str:
        """Generate an improved version of a previous plan."""
        knowledge = context.get("knowledge", [])
        prompt = f"""
Improve the following plan, addressing feedback and improving clarity.

Previous Plan:
{previous_plan}

Feedback: {feedback}
Additional knowledge: {" ".join(knowledge) if knowledge else "None"}

Return only the improved plan.
"""
        response = await self.agent.async_call_llm(prompt, context=context, llm_cfg=llm_cfg)
        log.info(f"Improved plan: {response}")
        return response.strip()

    async def debug_plan(
        self, previous_plan: str, error_log: str, context: dict
    ) -> str:
        """Repair a failed plan given error logs."""
        prompt = f"""
The following plan failed or produced an error.
Please rewrite it to fix possible mistakes while keeping the original intent.

Plan:
{previous_plan}

Error Log:
{error_log}

Return only the corrected plan text.
"""
        response = await self.agent.async_call_llm(prompt, context=context)
        log.info(f"Debugged plan: {response}")
        return response.strip()
