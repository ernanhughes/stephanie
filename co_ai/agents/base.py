# co_ai/agents/base.py
import re
from abc import ABC, abstractmethod

import litellm

from co_ai.logs import JSONLogger
from co_ai.utils import PromptLoader
from co_ai.parsers import ResponseParser

from co_ai.constants import (
    API_BASE,
    MODEL,
    STRATEGY,
    EVENT,
    DETAILS,
    API_KEY,
    PROMPT_PATH,
    SAVE_PROMPT,
    SAVE_CONTEXT,
    OUTPUT_KEYS,
    AGENT,
    RUN_ID,
    NAME,
    PROMPT_MATCH_RE,
)


def remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class BaseAgent(ABC):
    def __init__(self, cfg, memory=None, logger=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger or JSONLogger()
        agent_key = self.__class__.__name__.replace(AGENT, "").lower()
        self.log(
            f"📡 Initializing {agent_key} agent with config:\n{cfg}", structured=False
        )

        self.model_config = cfg.get(MODEL, {})
        self.prompt_loader = PromptLoader(memory=self.memory, logger=self.logger)

        self.prompt_match_re = cfg.get(PROMPT_MATCH_RE, "")
        self.llm = self.init_llm()
        self.save_context = cfg.get(SAVE_CONTEXT, False)
        self.output_keys = cfg.get(
            OUTPUT_KEYS, cfg.get(NAME, self.__class__.__name__)
        )

    def log(self, message, structured=True):
        if structured:
            self.logger.log(
                self.__class__.__name__,
                {
                    AGENT: self.__class__.__name__,
                    EVENT: message if isinstance(message, str) else "log",
                    DETAILS: message if isinstance(message, dict) else None,
                },
            )
        else:
            print(f"[{self.__class__.__name__}] {message}")

    def init_llm(self):
        required_keys = [NAME, API_BASE]
        for key in required_keys:
            if key not in self.model_config:
                raise ValueError(f"Missing required LLM config key: {key}")
        return {
            MODEL: self.model_config[NAME],
            API_BASE: self.model_config[API_BASE],
            API_KEY: self.model_config.get(API_KEY),
        }

    def call_llm(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            response = litellm.completion(
                model=self.llm[MODEL],
                messages=messages,
                api_base=self.llm[API_BASE],
                api_key=self.llm.get(API_KEY),
            )
            output = response["choices"][0]["message"]["content"]
            if self.cfg.get(SAVE_PROMPT, False) and self.memory:
                self.memory.prompt.log(
                    agent_name=self.__class__.__name__,
                    prompt_key=self.cfg.get(PROMPT_PATH, ""),
                    prompt_text=prompt,
                    response=output,
                    # source=self.cfg.get("prompt_type", "file"),
                    strategy=self.cfg.get(STRATEGY, ""),
                    version=self.cfg.get("version", 1),
                    # metadata={}
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
