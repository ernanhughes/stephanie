# stephanie/scoring/metrics/features/base_feature.py
from __future__ import annotations

import abc
from typing import Dict, Any
from stephanie.scoring.scorable import Scorable

class BaseFeature(abc.ABC):
    """
    A Feature transforms (scorable, acc, context) → updated acc.

    Rules:
    - NEVER mutate the `scorable` object.
    - ALWAYS return the updated accumulator dict.
    - Each feature is fully self-contained.
    - Features may depend on external tools/services (passed in __init__).
    """

    name: str = "base_feature"

    def __init__(self, cfg: dict, memory, container, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger

    @abc.abstractmethod
    async def apply(
        self,
        scorable: Scorable,
        acc: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Given a Scorable + current accumulator + pipeline context,
        compute new attributes and return updated accumulator.
        """
        raise NotImplementedError
