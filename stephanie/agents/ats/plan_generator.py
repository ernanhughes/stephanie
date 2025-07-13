# stephanie/agents/ats/plan_generator.py

from typing import Any, Dict, Optional

from stephanie.agents.base_agent import BaseAgent


class PlanGenerator:
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    async def draft_plan(self, task_description: str, knowledge: list) -> str:
        prompt = f"""
        You are an expert machine learning engineer.
        Create a detailed solution plan for this task:
        {task_description}

        Some relevant tricks from past solutions:
        {" ".join(knowledge)}

        Output only the plan as natural language text.
        """
        response = await self.agent.llm(prompt)
        return response.strip()

    async def improve_plan(
        self, previous_plan: str, feedback: str, knowledge: list
    ) -> str:
        prompt = f"""
        Improve this ML solution plan:
        {previous_plan}

        Feedback: {feedback}
        Additional knowledge: {" ".join(knowledge)}

        Output only the improved plan.
        """
        response = await self.agent.llm(prompt)
        return response.strip()

    async def debug_plan(self, previous_plan: str, error_log: str) -> str:
        prompt = f"""
        Fix this buggy ML solution plan:
        {previous_plan}

        Error log: {error_log}
        Output only the corrected plan.
        """
        response = await self.agent.llm(prompt)
        return response.strip()
