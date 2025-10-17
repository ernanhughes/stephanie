# stephanie/components/gap/scm/plugin.py
from __future__ import annotations

from typing import Dict, Any
from stephanie.scoring.scorer.base_scorer import ScoringPlugin
from .head import UniversalSCMHead
import torch

class SCMPlugin(ScoringPlugin):
    """Drop-in plugin that runs the SCM head on tapped features and returns scm.* metrics."""
    def __init__(self, *, head: UniversalSCMHead):
        self.head = head.eval()  # sidecar by default

    def post_process(self, *, tap_output: Dict[str, Any]) -> Dict[str, float]:
        with torch.no_grad():
            return self.head.forward(tap_output)
