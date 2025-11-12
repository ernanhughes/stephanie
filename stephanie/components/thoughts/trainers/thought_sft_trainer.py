from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

from stephanie.core.app_context import AppContext

_logger = logging.getLogger(__name__)

@dataclass
class SFTExample:
    goal: str
    thought: str
    label: float   # e.g., usefulness in [0,1]

class ThoughtSFTTrainer:
    """Tiny baseline SFT trainer that fine-tunes a small head to predict thought usefulness.
    Integrates with your existing SFT registry or wraps a simple linear/regression head.
    """
    def __init__(self, app: AppContext, model_name: str = "tiny/thought-head"):
        self.app = app
        self.model_name = model_name
        # Plug into your trainer registry if present; else fallback to local torch head.
        self.impl = app.try_import("stephanie.training.sft", "SFTTrainerRegistry")
        if self.impl:
            _logger.info("Using SFTTrainerRegistry for %s", model_name)
        else:
            _logger.warning("Falling back to local linear head (no registry found)")
            self.impl = None

    def fit(self, data: List[SFTExample], **kwargs) -> Dict[str, Any]:
        if self.impl:
            # Convert to registry format
            records = [{"goal": d.goal, "input": d.thought, "label": float(d.label)} for d in data]
            return self.impl.train(self.model_name, records, **kwargs)
        else:
            # Minimal no-op stub you can replace with your tiny torch head
            _logger.info("[stub] trained linear head on %d examples", len(data))
            return {"trained": True, "n": len(data)}
