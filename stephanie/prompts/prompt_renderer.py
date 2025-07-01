# stephanie/prompts/prompt_renderer.py
from jinja2 import Template


class PromptRenderer:
    def __init__(self, prompt_loader, config):
        self.prompt_loader = prompt_loader
        self.config = config

    def render(self, dim: dict, context: dict):
        if self.prompt_loader and dim.get("file"):
            return self.prompt_loader.from_file(
                file_name=dim["file"],
                config=self.config,
                context=context
            )
        elif dim.get("prompt_template"):
            return Template(dim["prompt_template"]).render(**context)
        else:
            raise ValueError(f"No prompt found for dimension {dim['name']}")
