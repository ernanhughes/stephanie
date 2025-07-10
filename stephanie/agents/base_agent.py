# stephanie/agents/base.py
import random
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone

import litellm

from stephanie.constants import (AGENT, API_BASE, API_KEY, BATCH_SIZE, CONTEXT,
                                 GOAL, HYPOTHESES, INPUT_KEY, MODEL, NAME,
                                 OUTPUT_KEY, PIPELINE, PIPELINE_RUN_ID,
                                 PROMPT_MATCH_RE, PROMPT_PATH, SAVE_CONTEXT,
                                 SAVE_PROMPT, SOURCE, STRATEGY)
from stephanie.logs import JSONLogger
from stephanie.models import PromptORM
from stephanie.prompts import PromptLoader
from stephanie.rules import SymbolicRuleApplier


def remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class BaseAgent(ABC):
    def __init__(self, cfg, memory=None, logger=None):
        self.cfg = cfg
        agent_key = self.__class__.__name__.replace(AGENT, "").lower()
        self.name = cfg.get(NAME, agent_key)
        self.memory = memory
        self.logger = logger or JSONLogger()
        self.rule_applier = SymbolicRuleApplier(cfg, memory, logger)
        self.model_config = cfg.get(MODEL, {})
        self.prompt_loader = PromptLoader(memory=self.memory, logger=self.logger)
        self.prompt_match_re = cfg.get(PROMPT_MATCH_RE, "")
        self.llm = self.init_llm()  # TODO do we need to init here?
        self.strategy = cfg.get(STRATEGY, "default")
        self.model_name = self.llm.get(NAME, "")
        self.source = cfg.get(SOURCE, CONTEXT)
        self.batch_size = cfg.get(BATCH_SIZE, 6)
        self.save_context = cfg.get(SAVE_CONTEXT, False)
        self.input_key = cfg.get(INPUT_KEY, HYPOTHESES)
        self.preferences = cfg.get("preferences", {})
        self.remove_think = cfg.get("remove_think", True)
        self.output_key = cfg.get(OUTPUT_KEY, self.name)
        self._goal_id_cache = {}
        self._prompt_id_cache = {}
        self._hypothesis_id_cache = {}
        self.logger.log(
            "AgentInitialized",
            {
                "agent_key": agent_key,
                "class": self.__class__.__name__,
                "config": self.cfg,
            },
        )

    def init_llm(self, cfg=None):
        config = cfg or self.cfg
        model_cfg = config.get(MODEL, {})
        if not model_cfg:
            return {
                NAME: None,
                API_BASE: None,
                API_KEY: None,
            }
        required_keys = [NAME, API_BASE]
        for key in required_keys:
            if key not in model_cfg:
                self.logger.log(
                    "MissingLLMConfig", {"agent": self.name, "missing_key": key}
                )
        return {
            NAME: model_cfg.get(NAME, "ollama/qwen3"),
            API_BASE: model_cfg.get(API_BASE, "http://localhost:11434"),
            API_KEY: model_cfg.get(API_KEY),
        }

    def get_or_save_prompt(self, prompt_text: str, context: dict) -> PromptORM:
        prompt = self.memory.prompt.get_from_text(prompt_text)
        if prompt is None:
            self.memory.prompt.save(
                context.get("goal"),
                agent_name=self.name,
                prompt_key=self.cfg.get(PROMPT_PATH, ""),
                prompt_text=prompt_text,
                strategy=self.cfg.get(STRATEGY, ""),
                pipeline_run_id=context.get("pipeline_run_id"),
                version=self.cfg.get("version", 1),
            )
            prompt = self.memory.prompt.get_from_text(prompt_text)
        if prompt is None:
            raise ValueError(
                f"Please check this prompe: {prompt_text}. "
                "Ensure it is saved before use."
            )
        return prompt

    def call_llm(self, prompt: str, context: dict, llm_cfg: dict = None) -> str:
        updated_cfg = self.rule_applier.apply_to_prompt(self.cfg, context)
        if self.llm is None:
            # 🔁 Apply rules here (now that goal is known)
            updated_cfg = self.rule_applier.apply_to_agent(self.cfg, context)
            self.llm = self.init_llm(cfg=updated_cfg)  # initialize with updated config

        """Call the default or custom LLM, log the prompt, and handle output."""
        props = llm_cfg or self.llm  # Use passed-in config or default

        agent_name = self.name

        strategy = updated_cfg.get(STRATEGY, "")
        prompt_key = updated_cfg.get(PROMPT_PATH, "")
        use_memory_for_fast_prompts = updated_cfg.get(
            "use_memory_for_fast_prompts", False
        )

        # 🔁 Check cache
        if self.memory and use_memory_for_fast_prompts:
            previous = self.memory.prompt.find_similar_prompt(
                agent_name=agent_name,
                prompt_text=prompt,
                strategy=strategy,
                similarity_threshold=0.8,
            )
            if previous:
                chosen = random.choice(previous)
                cached_response = chosen.get("response_text")
                self.logger.log(
                    "LLMCacheHit",
                    {
                        "agent": agent_name,
                        "strategy": strategy,
                        "prompt_key": prompt_key,
                        "cached": True,
                        "count": len(previous),
                        "emoji": "📦🔁💬",
                    },
                )
                return cached_response

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
            if updated_cfg.get(SAVE_PROMPT, False) and self.memory:
                self.memory.prompt.save(
                    context.get("goal"),
                    agent_name=self.name,
                    prompt_key=updated_cfg.get(PROMPT_PATH, ""),
                    prompt_text=prompt,
                    response=output,
                    strategy=updated_cfg.get(STRATEGY, ""),
                    pipeline_run_id=context.get("pipeline_run_id"),
                    version=updated_cfg.get("version", 1),
                )

            # Remove [THINK] blocks if configured
            response_cleaned = (
                remove_think_blocks(output) if self.remove_think else output
            )

            # Optionally add to context history
            if updated_cfg.get("add_prompt_to_history", True):
                self.add_to_prompt_history(
                    context, prompt, {"response": response_cleaned}
                )

            return response_cleaned

        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            self.logger.log("LLMCallError", {"exception": str(e)})
            raise

    @abstractmethod
    async def run(self, context: dict) -> dict:
        pass

    def add_to_prompt_history(self, context: dict, prompt: str, metadata: dict = None):
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

    def get_hypotheses(self, context: dict) -> list[dict]:
        try:
            if self.source == "context":
                hypothesis_dicts = context.get(self.input_key, [])
                if not hypothesis_dicts:
                    self.logger.log("NoHypothesesInContext", {"agent": self.name})
                return hypothesis_dicts

            elif self.source == "database":
                goal = context.get(GOAL)
                hypotheses = self.get_hypotheses_from_db(goal.get("goal_text"))
                if not hypotheses:
                    self.logger.log(
                        "NoHypothesesInDatabase", {"agent": self.name, "goal": goal}
                    )
                return [h.to_dict() for h in hypotheses] if hypotheses else []

            else:
                self.logger.log(
                    "InvalidSourceConfig", {"agent": self.name, "source": self.source}
                )
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            self.logger.log(
                "HypothesisFetchError",
                {"agent": self.name, "source": self.source, "error": str(e)},
            )

        return []

    def get_hypotheses_from_db(self, goal_text: str):
        return self.memory.hypotheses.get_latest(goal_text, self.batch_size)

    @staticmethod
    def extract_goal_text(goal):
        return goal.get("goal_text") if isinstance(goal, dict) else goal

    def execute_prompt(self, merged_context: dict, prompt_file: str = None):
        if prompt_file:
            prompt = self.prompt_loader.from_file(prompt_file, self.cfg, merged_context)
        else:
            prompt = self.prompt_loader.load_prompt(self.cfg, merged_context)
        response = self.call_llm(prompt, context=merged_context)
        self.logger.log(
            "PromptExecuted",
            {
                "prompt_text": prompt[:200],
                "response_snippet": response[:200],
                "prompt_file": prompt_file,
            },
        )
        return response

    def get_goal_id(self, goal: dict):
        if not isinstance(goal, dict):
            raise ValueError(
                f"Expected goal to be a dict, got {type(goal).__name__}: {goal}"
            )
        goal_text = goal.get("goal_text", "")
        if goal_text in self._goal_id_cache:
            return self._goal_id_cache[goal_text][0]
        goal = self.memory.goals.get_from_text(goal_text)
        self._goal_id_cache[goal_text] = (goal.id, goal)
        return goal.id

    def get_hypothesis_id(self, hypothesis_dict: dict):
        if not isinstance(hypothesis_dict, dict):
            raise ValueError(
                f"Expected hypothesis_text to be a dict, got {type(hypothesis_dict).__name__}: {hypothesis_dict}"
            )
        text = hypothesis_dict.get("text")
        if text in self._hypothesis_id_cache:
            return self._hypothesis_id_cache[text][0]
        hypothesis = self.memory.hypotheses.get_from_text(text)
        self._hypothesis_id_cache[text] = (hypothesis.id, hypothesis)
        return hypothesis.id

    def _log_timing_diagram(self):
        """Log timing breakdown as Mermaid diagram"""
        timing_data = self.logger.get_logs_by_type("FunctionTiming")
        function_times = defaultdict(float)

        for log in timing_data:
            key = f"{log['class']}.{log['function']}"
            function_times[key] += log["duration_ms"]

        mermaid = ["```mermaid\ngraph TD"]
        total = sum(function_times.values())

        for func, duration in function_times.items():
            percent = (duration / total) * 100
            mermaid.append(f"A{func.replace('.', '_')}[{func} | {percent:.1f}%]")

        mermaid.append("```")
        self.logger.log("TimingBreakdown", {"diagram": "\n".join(mermaid)})

    def _analyze_performance(self):
        """Print performance summary"""
        from tabulate import tabulate

        timing_logs = self.logger.get_logs_by_type("FunctionTiming")
        function_times = defaultdict(list)
        for log in timing_logs:
            data = log["data"]
            key = f"{data['class']}.{data['function']}"
            function_times[key].append(data["duration_ms"])

        table = []
        for key, durations in function_times.items():
            table.append(
                [
                    key,
                    f"{sum(durations) / len(durations):.2f}ms",
                    len(durations),
                    f"{max(durations):.2f}ms",
                ]
            )

        print("\n⏱️ Performance Breakdown")
        print(tabulate(table, headers=["Function", "Avg Time", "Calls", "Max Time"]))

    def save_hypothesis(self, hypothesis_dict: dict, context: dict):
        """
        Central method to save hypotheses and track document section links.
        """
        from stephanie.models.hypothesis import HypothesisORM
        from stephanie.models.hypothesis_document_section import \
            HypothesisDocumentSectionORM

        # Ensure metadata is set if not already in the dict
        goal = context.get(GOAL, {})
        hypothesis_dict["goal_id"] = self.get_goal_id(goal)
        hypothesis_dict["pipeline_signature"] = context.get(PIPELINE)
        hypothesis_dict["pipeline_run_id"] = context.get(PIPELINE_RUN_ID)
        hypothesis_dict["source"] = self.name
        hypothesis_dict["strategy"] = self.strategy

        hypothesis = HypothesisORM(**hypothesis_dict)
        self.memory.session.add(hypothesis)
        self.memory.session.flush()  # ensures ID is available

        # Link to document sections if provided
        section_ids = context.get("used_document_section_ids", [])
        for section_id in section_ids:
            link = HypothesisDocumentSectionORM(
                hypothesis_id=hypothesis.id,
                document_section_id=section_id,
            )
            self.memory.session.add(link)

        self.memory.session.commit()
        return hypothesis
