# co_ai/agents/base.py
import re
from abc import ABC, abstractmethod

import litellm
import yaml

from co_ai.logs import JSONLogger
from co_ai.utils import load_prompt_from_file


def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

class BaseAgent(ABC):
    def __init__(self, cfg, memory=None, logger=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger or JSONLogger()
        agent_key = self.__class__.__name__.replace("Agent", "").lower()
        print(f"📡 Initializing {agent_key} agent with config: \n{cfg}")
        self.model_config = cfg.get("model", {})
        # Load prompt
        self.prompt_template = self.get_prompt_template(cfg)
        self.prompt_match_re = cfg.get("prompt_match_re", "")
        self.use_prompt_refiner = cfg.get("use_prompt_refiner", False)
        self.llm = self.init_llm()

    def log(self, message, structured=True):
        if structured:
            self.logger.log(self.__class__.__name__,{
                "agent": self.__class__.__name__,
                "event": message if isinstance(message, str) else "log",
                "details": message if isinstance(message, dict) else None
            })
        else:
            print(f"[{self.__class__.__name__}] {message}")


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
        response = response['choices'][0]['message']['content']
        if self.cfg.get("save_prompt", False):
            self.memory.store_prompt(self.__class__.__name__, prompt, response)
        return response

    def extract_list_items(self, text: str) -> list[str]:
        return [
            match.strip()
            for match in re.findall(r"(?:^|\n)[\-\*\d]+\.\s*(.+)", text)
        ]

    def get_prompt_template(self, input_data: dict) -> str:
        prompt_mode = input_data.get("prompt_mode", "static")
        if prompt_mode == "static":
            prompt = input_data.get("prompt_template")
            if not prompt:
                raise ValueError("Prompt is required in static mode.")
            return prompt
        elif prompt_mode == "file":
            return load_prompt_from_file(input_data["prompt_path"])
        else:
            raise ValueError(f"Unknown prompt mode: {prompt_mode}")
        

    @abstractmethod
    async def run(self, input_data: dict) -> dict:
        pass
