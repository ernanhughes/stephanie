# co_ai/utils/prompt_renderer.py
import os
from typing import Dict, Any
from jinja2 import Template
import logging

from co_ai.utils import load_prompt_from_file


class PromptRenderer:
    def __init__(self, cfg: dict, memory=None, logger=None):
        """
        A generic prompt renderer that dynamically injects context into templates.
        
        Args:
            prompt_dir (str): Directory containing prompt templates
            template_format (str): Currently supports 'jinja2' only
        """
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

    def render(self, context: dict) -> str:
        """
        Render a prompt by loading the appropriate template and injecting context variables.
        
        Args:
            context (dict): The shared pipeline context (e.g., goal, literature, hypotheses)

        Returns:
            str: Fully rendered prompt ready to send to LLM
        """
        # Load raw template
        if self.cfg.prompt_mode == "file":
            template_str = load_prompt_from_file(self.cfg.prompt_path)
        else:
            template_str = self.cfg.prompt_template
        if not template_str:
            raise ValueError(f"Failed to load prompt '{self.cfg}'")
        template = Template(template_str)
        merged = {**self.cfg, **context}
        return template.render(**merged).strip()


