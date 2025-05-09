# co_ai/agents/base.py
import re
from abc import ABC, abstractmethod

import litellm

from co_ai.logs import JSONLogger


class BaseAgent(ABC):
    def __init__(self, cfg, memory=None, logger=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger or JSONLogger()
        self.model_config = cfg.models.get(self.__class__.__name__, {})
        self.llm = self.init_llm()

    def init_llm(self):
        if self.model_config:
            return {
                "model": self.model_config["name"],
                "api_base": self.model_config["api_base"],
                "api_key": self.model_config.get("api_key")
            }
        return None

    def call_llm(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = litellm.completion(
            model=self.llm["model"],
            messages=messages,
            api_base=self.llm["api_base"],
            api_key=self.llm.get("api_key")
        )
        return response['choices'][0]['message']['content']

    def extract_list_items(self, text: str) -> list[str]:
        return [
            match.strip()
            for match in re.findall(r"(?:^|\n)[\-\*\d]+\.\s*(.+)", text)
        ]

    def log(self, message, structured=True):
        if structured:
            self.logger.log({
                "agent": self.__class__.__name__,
                "event": message if isinstance(message, str) else "log",
                "details": message if isinstance(message, dict) else None
            })
        else:
            print(f"[{self.__class__.__name__}] {message}")



    @abstractmethod
    async def run(self, input_data: dict) -> dict:
        pass
