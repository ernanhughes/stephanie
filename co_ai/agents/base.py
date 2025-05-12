# co_ai/agents/base.py
import re
from abc import ABC, abstractmethod

import litellm
import yaml

from co_ai.logs import JSONLogger
from co_ai.utils import PromptLoader


def camel_to_snake(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class BaseAgent(ABC):
    def __init__(self, cfg, memory=None, logger=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger or JSONLogger()
        agent_key = self.__class__.__name__.replace("Agent", "").lower()
        self.log(
            f"📡 Initializing {agent_key} agent with config:\n{cfg}", structured=False
        )

        self.model_config = cfg.get("model", {})
        self.prompt_loader = PromptLoader(memory=self.memory, logger=self.logger)

        self.prompt_match_re = cfg.get("prompt_match_re", "")
        self.llm = self.init_llm()

    def log(self, message, structured=True):
        if structured:
            self.logger.log(
                self.__class__.__name__,
                {
                    "agent": self.__class__.__name__,
                    "event": message if isinstance(message, str) else "log",
                    "details": message if isinstance(message, dict) else None,
                },
            )
        else:
            print(f"[{self.__class__.__name__}] {message}")

    def init_llm(self):
        required_keys = ["name", "api_base"]
        for key in required_keys:
            if key not in self.model_config:
                raise ValueError(f"Missing required LLM config key: {key}")
        return {
            "model": self.model_config["name"],
            "api_base": self.model_config["api_base"],
            "api_key": self.model_config.get("api_key"),
        }

    def call_llm(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            response = litellm.completion(
                model=self.llm["model"],
                messages=messages,
                api_base=self.llm["api_base"],
                api_key=self.llm.get("api_key"),
            )
            output = response["choices"][0]["message"]["content"]
            if self.cfg.get("save_prompt", False) and self.memory:
                self.memory.store_prompt(
                    self.__class__.__name__,
                    self.cfg.get("prompt_path", self.cfg.get("prompt_", "default")),
                    prompt,
                    output,
                )
            return output
        except Exception as e:
            self.log(f"LLM call failed: {e}", structured=False)
            raise

    def extract_list_items(self, text: str) -> list[str]:
        return [
            match.strip() for match in re.findall(r"(?m)^\s*[\-\*\d]+\.\s+(.*)", text)
        ]

    @abstractmethod
    async def run(self, input_data: dict) -> dict:
        pass
