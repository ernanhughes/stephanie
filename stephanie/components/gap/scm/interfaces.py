# stephanie/components/gap/scm/interfaces.py
from __future__ import annotations

from typing import Protocol, Dict, Any

class FeatureTap(Protocol):
    """Model-specific adapter that exposes common features to the SCM head."""
    def collect_features(self, *, input_text: str, output_tokens: Any, internals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a dict that may include:
          - "final_logits":  [T, V]
          - "token_entropies": [T]
          - "attn_stats": { ... }           # per-layer summaries
          - "latent": tensor [D] or [L, D]  # pooled embeddings / state traces
          - "steps": List[Dict[str, Any]]   # recursive/iterative step snapshots (if any)
          - "len_effect": float             # optional, if you can estimate
          - "temp": float                   # sampling temp if used
          - "agree_hat": float              # optional agreement proxy
        """
        ...

class SCMHead(Protocol):
    def forward(self, tapped: Dict[str, Any]) -> Dict[str, float]:
        """Returns the SCM metrics dict (names above, values in [0,1] where applicable)."""
