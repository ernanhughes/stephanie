# co_ai/agents/base.py
import re
from abc import ABC, abstractmethod
from dspy import LM
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose

class BaseAgent(ABC):
    def __init__(self, memory=None, logger=None):
        self.memory = memory
        self.logger = logger
        self.model_config = self.load_model_config()
        self.lm = self.init_lm()

    def load_model_config(self):
        # Only initialize Hydra once globally
        if not GlobalHydra.instance().is_initialized():
            initialize(config_path="../configs", version_base=None)
        cfg = compose(config_name="pipeline")

        model_key = self.__class__.__name__
        return cfg.models.get(model_key, {})

    def init_lm(self):
        if self.model_config:
            return LM(
                self.model_config["name"],
                api_base=self.model_config["api_base"],
                api_key=self.model_config.get("api_key")
            )
        return None

    def extract_list_items(self, text: str) -> list[str]:
        return [
            match.strip()
            for match in re.findall(r"(?:^|\n)[\-\*\d]+\.\s*(.+)", text)
        ]

    @abstractmethod
    async def run(self, input_data: dict) -> dict:
        pass
