# co_ai/utils/prompt_loader.py

import os
from pathlib import Path

from jinja2 import Template
from co_ai.constants import PROMPT_DIR, PROMPT_TYPE, NAME, STRATEGY, DEFAULT, FILE, PROMPT_FILE


def get_text_from_file(file_path: str) -> str:
    """Get text from a file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

class PromptLoader:
    def __init__(self, memory=None, logger=None):
        self.memory = memory
        self.logger = logger

    def load_prompt(self, config: dict, context: dict) -> str:
        """
        Load prompt based on configured strategy: file, template, tuning, or static.

        Args:
            config: Agent config with prompt_type, prompt_text, etc.
            context: Shared pipeline context

        Returns:
            str: The loaded prompt text
        """
        prompt_type = config.get(PROMPT_TYPE, FILE)
        prompts_dir = context.get(PROMPT_DIR, "prompts")

        if not os.path.isdir(prompts_dir):
            raise FileNotFoundError(f"Prompts Directory not found: {prompts_dir} please check your config.")
        try:
            merged = self._merge_context(config, context)

            if prompt_type == "static":
                prompt_text = config.get("prompt_text", None)
                if prompt_text is None:
                        raise ValueError(f"""Static prompt text is not provided in the config. 
                                           When you choose static you need to add the prompt text in a [prompt_text] node. 
                                            \n{config}""")
            elif prompt_type == "tuning":
                agent_name = config.get(NAME, "default")
                return self._load_best_version(agent_name, context.get("goal", ""), merged)

            return self._load_from_file(merged)

        except Exception as e:
            if self.logger:
                self.logger.log("PromptLoadFailed", {
                    "agent": config.get(NAME, DEFAULT),
                    "error": str(e),
                    "config_used": config
                })
            return self._fallback_prompt(context.get("goal", ""))

    def from_file(self, file_name: str, config: dict, context: dict) -> str:
        """Load prompt from a specific file"""
        path = self.get_file_name(file_name, config, context)
        prompt = get_text_from_file(path)
        merged = self._merge_context(config, context)
        return prompt.format(**merged)

    @staticmethod
    def get_file_name(file_name: str, cfg: dict, context: dict) -> str:
        """Constructs a file path for a prompt file"""
        prompts_dir = context.get(PROMPT_DIR, "prompts")
        if file_name.endswith(".txt"):
            path = os.path.join(prompts_dir, f"{cfg['name']}/{file_name}")
        else:
            path = os.path.join(prompts_dir, f"{cfg['name']}/{file_name}.txt")
        return path

    def _load_from_file(self, config: dict) -> str:
        """Load prompt from disk"""
        prompt_dir = config.get(PROMPT_DIR, "prompts/")
        prompt_type = config.get(PROMPT_TYPE, FILE)

        file_name = DEFAULT
        if prompt_type == STRATEGY:
            file_name = config.get(STRATEGY, DEFAULT)
        elif prompt_type == FILE:
            file_name = config.get(PROMPT_FILE, DEFAULT)
        if file_name.lower().endswith(".txt"):
            path = os.path.join(prompt_dir, f"{config[NAME]}/{file_name}")
        else:
            path =os.path.join(prompt_dir, f"{config[NAME]}/{file_name}.txt")
        if not os.path.exists(path):
            if self.logger:
                self.logger.log("PromptFileNotFound", {
                    "path": path,
                    "agent": config.get(NAME, DEFAULT)
                })
            raise FileNotFoundError(f"Prompt file not found: {path}")

        prompt = get_text_from_file(path)
        merged = self._merge_context(config, {})
        try: 
            return Template(prompt).render(**merged)
        except KeyError as ke:
            if self.logger:
                self.logger.log("PromptFormattingError", {
                    "missing_key": str(ke),
                    "path": path
                })
            return prompt  # Return unformatted version as fallback

    def _load_best_version(self, agent_name: str, goal: str, config: dict) -> str:
        """Load the most effective prompt version from memory"""
        best_prompt = self.memory.prompt.get_best_prompt_for_agent(
            agent_name=agent_name,
            strategy=config.get(STRATEGY, DEFAULT),
            goal=goal
        )

        if best_prompt:
            return best_prompt["prompt_text"]
        else:
            if self.logger:
                self.logger.log("UsingFallbackPrompt", {"reason": "no_tuned_prompt_found"})
            return self._load_from_file(config)

    def _fallback_prompt(self, goal: str = "") -> str:
        """Hardcoded fallback if everything else fails"""
        return f"Generate hypothesis for goal: {goal or '[unspecified goal]'}"

    @staticmethod
    def _merge_context(config: dict, context: dict) -> dict:
        """Merge context and config dictionaries"""
        return {**context, **config}
