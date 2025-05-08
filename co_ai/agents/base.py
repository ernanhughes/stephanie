# co_ai/agents/base.py
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, memory=None, tools=None):
        self.memory = memory
        self.tools = tools or []

    @abstractmethod
    async def run(self, input_data: dict) -> dict:
        """
        All agents must accept input_data and return a dictionary with results.
        Example input_data: { 'goal': 'Identify AML treatments' }
        """
        pass

    def register_tool(self, tool):
        self.tools.append(tool)

    def log(self, message):
        print(f"[{self.__class__.__name__}] {message}")
