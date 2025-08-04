import abc
from typing import List

import torch

from stephanie.scoring.model_locator_mixin import ModelLocatorMixin
from stephanie.scoring.scorable import Scorable
from stephanie.data.score_bundle import ScoreBundle


class BaseScorer(ModelLocatorMixin, abc.ABC):
    def __init__(self, cfg: dict, memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        self.embedding_type = self.memory.embedding.type
        self.dim = self.memory.embedding.dim
        self.hdim = self.memory.embedding.hdim

        self.model_path = cfg.get("model_path", "models")
        self.target_type = cfg.get("target_type", "document")
        self.model_type = cfg.get("model_type", "svm")  # Override in subclass
        self.version = cfg.get("model_version", "v1")

        self.force_rescore = cfg.get("force_rescore", False)
        self.dimensions = cfg.get("dimensions", [])
        self.device = torch.device(cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        """Returns a canonical name for the scorer."""
        return f"{self.model_type}"

    def get_model_name(self) -> str:
        return f"{self.target_type}_{self.model_type}_{self.version}"

    @abc.abstractmethod
    def score(
        self,
        goal: dict,
        scorable: Scorable,
        dimensions: List[str],
    ) -> ScoreBundle:
        """
        Score a single item (Scorable) for a given goal and a set of dimensions.

        Returns:
            ScoreBundle containing ScoreResults for each dimension.
        """
        raise NotImplementedError("Subclasses must implement score()")

    def log_event(self, event: str, data: dict):
        if self.logger:
            self.logger.log(event, data)
