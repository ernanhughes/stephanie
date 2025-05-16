# co_ai/agents/base.py
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import litellm

from co_ai.logs import JSONLogger
from co_ai.utils import PromptLoader

from co_ai.constants import (
    API_BASE,
    MODEL,
    STRATEGY,
    INPUT_KEY,
    API_KEY,
    PROMPT_PATH,
    SAVE_PROMPT,
    SAVE_CONTEXT,
    OUTPUT_KEY,
    AGENT,
    NAME,
    PROMPT_MATCH_RE,
    HYPOTHESES,
)


def remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class BaseAgent(ABC):
    def __init__(self, cfg, memory=None, logger=None):
        self.cfg = cfg
        agent_key = self.__class__.__name__.replace(AGENT, "").lower()
        self.name = cfg.get(NAME, agent_key)
        self.memory = memory
        self.logger = logger or JSONLogger()
        self.model_config = cfg.get(MODEL, {})
        self.prompt_loader = PromptLoader(memory=self.memory, logger=self.logger)
        self.prompt_match_re = cfg.get(PROMPT_MATCH_RE, "")
        self.llm = self.init_llm()
        self.save_context = cfg.get(SAVE_CONTEXT, False)
        self.input_key = cfg.get(INPUT_KEY, HYPOTHESES)
        self.preferences = cfg.get("preferences", {})
        self.output_key = cfg.get(OUTPUT_KEY, self.name)
        self.logger.log(
            "AgentInitialized",
            {
                "agent_key": agent_key,
                "class": self.__class__.__name__,
                "config": self.cfg,
            },
        )

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

    def call_llm(self, prompt: str, context: dict) -> str:
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
                self.memory.prompt.save(
                    agent_name=self.__class__.__name__,
                    prompt_key=self.cfg.get(PROMPT_PATH, ""),
                    prompt_text=prompt,
                    response=output,
                    # source=self.cfg.get("prompt_type", "file"),
                    strategy=self.cfg.get(STRATEGY, ""),
                    version=self.cfg.get("version", 1),
                    # metadata={}
                )
            response = remove_think_blocks(output)
            if "prompt_history" not in context:
                context["prompt_history"] = {}
            context["prompt_history"][self.name] = {
                "prompt": prompt,
                "agent": self.name,
                "response": response,  # Adding think will confuse the refinement
                "preferences": self.preferences,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            return response
        except Exception as e:
            self.logger.log("LLMCallError", {"exception": e})
            raise

    @abstractmethod
    async def run(self, context: dict) -> dict:
        pass
