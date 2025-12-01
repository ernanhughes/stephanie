# stephanie/components/nexus/vpm/logic_program.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from zeromodel.vpm.logic import vpm_and, vpm_not, vpm_or


@dataclass
class LogicProgram:
    """
    Compose attention map from base maps.
    Default: (quality AND NOT uncertainty) OR (novelty AND NOT uncertainty)
    """
    def run(self, *, quality: np.ndarray, novelty: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
        good = vpm_and(quality, vpm_not(uncertainty))
        explore = vpm_and(novelty, vpm_not(uncertainty))
        return vpm_or(good, explore)  # normalized float32
