import abc
from typing import Any, Dict, List, Protocol

import torch

from stephanie.data.score_bundle import ScoreBundle
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorer.model_locator_mixin import ModelLocatorMixin
import logging

_logger = logging.getLogger(__name__)

class ScoringPlugin(Protocol):
    """Protocol for scoring plugins that can post-process model outputs."""
    def post_process(self, *, tap_output: Dict[str, Any]) -> Dict[str, float]:
        """Return extra metric key→value pairs (e.g., scm.*)."""


class BaseScorer(ModelLocatorMixin, abc.ABC):
    def __init__(self, cfg: dict, memory, container, logger, enable_plugins: bool = True):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self.enable_plugins = enable_plugins

        self.embedding_type = self.memory.embedding.name
        self.dim = self.memory.embedding.dim
        self.hdim = self.memory.embedding.hdim

        self.model_path = cfg.get("model_path", "models")
        self.target_type = cfg.get("target_type", "document")
        self.model_type = cfg.get("model_type", "svm")  # Override in subclass
        self.version = cfg.get("model_version", "v1")

        self.force_rescore = cfg.get("force_rescore", False)
        self.dimensions = cfg.get("dimensions", [])
        self.device = torch.device(cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")
        self._plugins: List[ScoringPlugin] = [] if enable_plugins else []


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


    def add_plugin(self, plugin: ScoringPlugin) -> None:
        self._plugins.append(plugin)

    def log_event(self, event: str, data: dict):
        self.logger.log(event, data)

