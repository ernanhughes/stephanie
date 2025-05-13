# co_ai/utils/prompt_loader.py

import os
from pathlib import Path

from jinja2 import Template


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
        prompt_type = config.get("prompt_type", "file")
        prompts_dir = context.get("prompts_dir", "prompts")

        if not os.path.isdir(prompts_dir):
            raise FileNotFoundError(f"Prompts Directory not found: {prompts_dir}")

        try:
            merged = self._merge_context(config, context)

            if prompt_type == "static":
                return config.get("prompt_text", "")

            elif prompt_type == "file":
                return self._load_from_file(merged)

            elif prompt_type == "template":
                base_prompt = self._load_from_file(config)
                return Template(base_prompt).render(**merged)

            elif prompt_type == "tuning":
                agent_name = config.get("name", "default")
                return self._load_best_version(agent_name, context.get("goal", ""), merged)

            else:
                return self._fallback_prompt(context.get("goal", ""))

        except Exception as e:
            if self.logger:
                self.logger.log("PromptLoadFailed", {
                    "agent": config.get("name", "default"),
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
        prompts_dir = context.get("prompts_dir", "prompts")
        if file_name.endswith(".txt"):
            path = os.path.join(prompts_dir, f"{cfg['name']}/{file_name}")
        else:
            path = os.path.join(prompts_dir, f"{cfg['name']}/{file_name}.txt")
        return path

    def get_template_path(self, prompts_dir: str, config: dict) -> str:
        """Get the path to the template file"""
        template_name = config.get("template_name", "default_template.txt")
        return os.path.join(prompts_dir, template_name)

    def _load_from_file(self, config: dict) -> str:
        """Load prompt from disk"""
        prompt_dir = config.get("prompts_dir", "prompts/")
        strategy = config.get("strategy", "default")
        path = os.path.join(prompt_dir, f"{config['name']}/{strategy}.txt")

        prompt = get_text_from_file(path)
        merged = self._merge_context(config, {})
        try:
            return prompt.format(**merged)
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
            strategy=config.get("strategy", "default"),
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
