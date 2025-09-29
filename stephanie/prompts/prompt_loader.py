# stephanie/prompts/prompt_loader.py
import os

import logging
from jinja2 import Template

from stephanie.constants import (DEFAULT, FILE, NAME, PROMPT_DIR, PROMPT_FILE,
                                 PROMPT_MODE, STRATEGY)

_logger = logging.getLogger(__name__)

def get_text_from_file(file_path: str) -> str:
    """Reads and returns stripped text from a file."""
    _logger.info(f"Loading prompt from file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


class PromptLoader:
    def __init__(self, memory, logger):
        self.memory = memory
        self.logger = logger

    def load_prompt(self, config: dict, context: dict) -> str:
        """
        Load and render a prompt based on the strategy defined in the agent config.
        Supports: file, template, tuning, or static.
        """
        prompt_type = config.get(PROMPT_MODE, FILE)
        prompts_dir = context.get(PROMPT_DIR, "prompts")

        if not os.path.isdir(prompts_dir):
            raise FileNotFoundError(f"Prompt directory not found: {prompts_dir}")

        merged = self._merge_context(config, context)

        try:
            if prompt_type == "static":
                prompt_text = config.get("prompt_text")
                if not prompt_text:
                    raise ValueError("Missing 'prompt_text' in static config.")
                return Template(prompt_text).render(**merged)

            if prompt_type == "tuning":
                agent_name = config.get(NAME, "default")
                return self._load_best_version(
                    agent_name, context.get("goal", ""), merged
                )

            return self._load_from_file(merged)

        except Exception as e:
            _logger.error(f"❌ Exception:  {type(e).__name__}: {e}")
            return self._fallback_prompt(context.get("goal", ""))

    def from_file(self, file_name: str, config: dict, context: dict) -> str:
        """Manually load and render a prompt file."""
        path = self.get_file_path(file_name, config, context)
        prompt_text = get_text_from_file(path)
        merged = self._merge_context(config, context)
        try:
            return Template(prompt_text).render(**merged)
        except KeyError as e:
            _logger.error(f"❌ Exception:  {type(e).__name__}: {e}")

    def score_prompt(self, file_name: str, config: dict, context: dict) -> str:
        """Manually load and render a prompt file."""
        path = self.get_score_path(file_name, config, context)
        prompt_text = get_text_from_file(path)
        merged = self._merge_context(config, context)
        try:
            return Template(prompt_text).render(**merged)
        except KeyError as e:
            _logger.error(f"❌ Exception:  {type(e).__name__}: {e}")


    @staticmethod
    def get_file_path(file_name: str, cfg: dict, context: dict) -> str:
        """Builds full prompt file path."""
        prompts_dir = context.get(PROMPT_DIR, "prompts")
        filename = file_name if file_name.endswith(".txt") else f"{file_name}.txt"
        _logger.info(f"PromptLoader.get_file_path: prompts_dir={prompts_dir}, cfg={cfg}, filename={filename}")
        return os.path.join(prompts_dir, cfg.get("name", "default"), filename)

    @staticmethod
    def get_score_path(file_name: str, cfg: dict, context: dict) -> str:
        """Builds full prompt file path."""
        prompts_dir = context.get(PROMPT_DIR, "prompts")
        filename = file_name if file_name.endswith(".txt") else f"{file_name}.txt"
        _logger.info(f"PromptLoader.get_score_path: prompts_dir={prompts_dir}, cfg={cfg}, filename={filename}")
        return os.path.join(prompts_dir, "scoring", filename)


    def _load_from_file(self, config: dict) -> str:
        """Loads and renders a prompt file based on config."""
        prompt_dir = config.get(PROMPT_DIR, "prompts")
        file_key = config.get(PROMPT_FILE) or config.get(STRATEGY) or DEFAULT
        file_name = f"{file_key}.txt" if not file_key.endswith(".txt") else file_key
        path = os.path.join(prompt_dir, config.get(NAME, "default"), file_name)

        _logger.info(
            f"PromptFileLoading file_key={file_key}, resolved_file={file_name}, path={path}"
        )

        if not os.path.exists(path):
            _logger.error(f"❌ PromptFileNotFound: path={path}, agent={config.get(NAME, DEFAULT)}")
            raise FileNotFoundError(f"Prompt file not found: {path}")

        try:
            prompt_text = get_text_from_file(path)
            rendered = Template(prompt_text).render(**self._merge_context(config, {}))
            _logger.info(f"PromptFileLoaded: path={path}, rendered_preview={rendered[:100]}")
            return rendered
        except KeyError as e:
            _logger.error(f"❌ PromptFormattingError: missing_key={e}, path={path}")
            return prompt_text  # Fallback: return raw

    def _load_best_version(self, agent_name: str, goal: str, config: dict) -> str:
        """Load a tuned version of the prompt if available."""
        best_prompt = self.memory.prompts.get_best_prompt_for_agent(
            agent_name=agent_name, strategy=config.get(STRATEGY, DEFAULT), goal=goal
        )
        if best_prompt:
            return best_prompt["prompt_text"]

        _logger.info(f"PromptLoader.get_best_version: agent_name={agent_name}, goal={goal}, config={config}")   
        return self._load_from_file(config)

    def _fallback_prompt(self, goal: str = "") -> str:
        """Minimal backup prompt if nothing else works."""
        return f"Generate hypothesis for goal: {goal or '[unspecified goal]'}"

    @staticmethod
    def _merge_context(config: dict, context: dict) -> dict:
        """Merges agent config and pipeline context."""
        return {**context, **config}
