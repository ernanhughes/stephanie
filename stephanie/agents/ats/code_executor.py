# stephanie/agents/ats/code_executor.py

from typing import Any, Dict, Optional

from stephanie.agents.base_agent import BaseAgent


class CodeExecutor:
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    async def score_complexity(self, task_description: str, plan: str) -> float:
        prompt = f"""
        Rate the complexity of this task and plan on a scale of 1–5.
        Task: {task_description}
        Plan: {plan}
        """
        response = await self.agent.llm(prompt)
        try:
            return float(response.strip())
        except:
            return 3.0

    async def one_pass_codegen(self, plan: str) -> str:
        prompt = f"""
        Generate Python code for the following plan:
        {plan}
        """
        response = await self.agent.llm(prompt)
        return response.strip()

    async def stepwise_codegen(self, plan: str) -> str:
        prompt = f"""
        Break the following plan into steps:
        {plan}
        """
        response = await self.agent.llm(prompt)
        steps = response.get("decomposed_steps", [])
        full_code = ""
        for step in steps:
            code = await self.one_pass_codegen(step["details"])
            full_code += code + "\n"
        return full_code