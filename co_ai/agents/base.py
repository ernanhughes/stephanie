# co_ai/agents/base.py
import re
from abc import ABC, abstractmethod

import litellm

from co_ai.logs import JSONLogger
from co_ai.utils import PromptLoader

def remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

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
        self.save_context = cfg.get("save_context", False)
        self.output_keys = cfg.get("output_keys", cfg.name)

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
                    agent_name=self.__class__.__name__,
                    prompt_key=self.cfg.get("prompt_path", ""),
                    prompt_text=prompt,
                    response=output,
                    source=self.cfg.get("prompt_type", "file"),
                    strategy=self.cfg.get("strategy",  ""),
                    version=self.cfg.get("version", 1),
                    metadata={}
                )
            return remove_think_blocks(output)
        except Exception as e:
            self.log(f"LLM call failed: {e}", structured=False)
            raise

    def extract_list_items(self, text: str) -> list[str]:
        return [
            match.strip() for match in re.findall(r"(?m)^\s*[\-\*\d]+\.\s+(.*)", text)
        ]

    @abstractmethod
    async def run(self, context: dict) -> dict:
        pass

    def _save_context(self, context: dict):
        if self.memory and self.cfg.get("save_context", False):
            run_id = context.get("run_id")
            name = self.cfg.get("name", self.__class__.__name__)
            self.memory.save_context(run_id, name, context, self.cfg)

    def _get_completed(self, context: dict) -> dict | None :
        run_id = context.get("run_id")
        name = self.cfg.get("name", self.__class__.__name__)
        if self.memory.has_completed(run_id, name):
           return self.memory.load_context(run_id, name)
        return None
