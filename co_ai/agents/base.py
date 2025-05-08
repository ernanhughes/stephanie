# co_ai/agents/base.py
from abc import ABC, abstractmethod

from co_ai.logs import JSONLogger


class BaseAgent(ABC):
    def __init__(self, memory=None, tools=None, logger=None):
        self.memory = memory
        self.tools = tools or []
        self.logger = logger or JSONLogger()

    @abstractmethod
    async def run(self, input_data: dict) -> dict:
        """
        All agents must accept input_data and return a dictionary with results.
        Example input_data: { 'goal': 'Identify AML treatments' }
        """
        pass

    def register_tool(self, tool):
        self.tools.append(tool)

    def log(self, message, structured=True):
        if structured:
            self.logger.log({
                "agent": self.__class__.__name__,
                "event": message if isinstance(message, str) else "log",
                "details": message if isinstance(message, dict) else None
            })
        else:
            print(f"[{self.__class__.__name__}] {message}")
