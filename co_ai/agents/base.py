# co_ai/agents/base.py
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import litellm

from co_ai.constants import (AGENT, API_BASE, API_KEY, BATCH_SIZE, CONTEXT,
                             GOAL, HYPOTHESES, INPUT_KEY, MODEL, NAME,
                             OUTPUT_KEY, PROMPT_MATCH_RE, PROMPT_PATH,
                             SAVE_CONTEXT, SAVE_PROMPT, SOURCE, STRATEGY)
from co_ai.logs import JSONLogger
from co_ai.prompts import PromptLoader


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
        self.source = cfg.get(SOURCE, CONTEXT)
        self.batch_size = cfg.get(BATCH_SIZE, 6)
        self.save_context = cfg.get(SAVE_CONTEXT, False)
        self.input_key = cfg.get(INPUT_KEY, HYPOTHESES)
        self.preferences = cfg.get("preferences", {})
        self.remove_think = cfg.get("remove_think", True)
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
            NAME: self.model_config[NAME],
            API_BASE: self.model_config[API_BASE],
            API_KEY: self.model_config.get(API_KEY),
        }

    def call_llm(self, prompt: str, context: dict, llm_cfg: dict = None) -> str:
        """Call the default or custom LLM, log the prompt, and handle output."""
        props = llm_cfg or self.llm  # Use passed-in config or default
        
        messages = [{"role": "user", "content": prompt}]
        try:
            response = litellm.completion(
                model=props[NAME],
                messages=messages,
                api_base=props[API_BASE],
                api_key=props.get(API_KEY, ""),
            )
            output = response["choices"][0]["message"]["content"]

            # Save prompt and response if enabled
            if self.cfg.get(SAVE_PROMPT, False) and self.memory:
                self.memory.prompt.save(
                    BaseAgent.extract_goal_text(context.get("goal")),
                    agent_name=self.name,
                    prompt_key=self.cfg.get(PROMPT_PATH, ""),
                    prompt_text=prompt,
                    response=output,
                    strategy=self.cfg.get(STRATEGY, ""),
                    version=self.cfg.get("version", 1),
                )

            # Remove [THINK] blocks if configured
            response_cleaned = remove_think_blocks(output) if self.remove_think else output

            # Optionally add to context history
            if self.cfg.get("add_prompt_to_history", True):
                self.add_to_prompt_history(context, prompt, {"response": response_cleaned})

            return response_cleaned

        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            self.logger.log("LLMCallError", {"exception": str(e)})
            raise


    @abstractmethod
    async def run(self, context: dict) -> dict:
        pass

    def add_to_prompt_history(
        self,
        context: dict,
        prompt: str,
        metadata: dict = None
    ):
        """
        Appends a prompt record to the context['prompt_history'] under the agent's name.

        Args:
            context (dict): The context dict to modify
            prompt (str): prompt to store
            metadata (dict): any extra info
        """
        if "prompt_history" not in context:
            context["prompt_history"] = {}
        if self.name not in context["prompt_history"]:
            context["prompt_history"][self.name] = []
        entry = {
            "prompt": prompt,
            "agent": self.name,
            "preferences": self.preferences,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            entry.update(metadata)
        context["prompt_history"][self.name].append(entry)

    def get_hypotheses(self, context: dict) -> list[str]:
        try:
            if self.source == "context":
                hypotheses = context.get(self.input_key, [])
                if not hypotheses:
                    self.logger.log("NoHypothesesInContext", {"agent": self.name})
                return hypotheses

            elif self.source == "database":
                goal = context.get(GOAL)
                hypotheses = self.get_hypotheses_from_db(goal.get("goal_text"))
                if not hypotheses:
                    self.logger.log("NoUnReflectedInDatabase", {"agent": self.name, "goal": goal})
                return hypotheses or []

            else:
                self.logger.log("InvalidSourceConfig", {
                    "agent": self.name,
                    "source": self.source
                })
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            self.logger.log(
                "HypothesisFetchError",
                {"agent": self.name, "source": self.source, "error": str(e)},
            )

        return []

    def get_hypotheses_from_db(self, goal_text:str):
        return self.memory.hypotheses.get_latest(goal_text, self.batch_size)
    
    @staticmethod
    def extract_goal_text(goal):
        return goal.get("goal_text") if isinstance(goal, dict) else goal