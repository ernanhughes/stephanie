# stephanie/scoring/vpm_scorable.py

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from stephanie.scoring.scorable import Scorable

class VPMScorable(Scorable):
    """
    Image-based scorable for Visual Policy Maps with per-dimension weights & order.
    metadata may contain:
      - dimension_weights: Dict[str, float]
      - dimension_order: List[str] (most important -> least important)
      - resize_method: str ("bilinear", "nearest", ...)
    """
    def __init__(self, id: str, image_array: np.ndarray, text: str = "Visual Policy Map", metadata: Optional[Dict[str, Any]]=None):
        super().__init__(id=id, text=text, metadata=metadata or {})
        self._img = image_array

    def get_image_array(self) -> np.ndarray:
        return self._img

    # Optional: some pipelines expect this helper
    def get_image_tensor(self) -> torch.Tensor:
        x = torch.from_numpy(self._img).float()
        if x.ndim == 2:  # H,W -> add channel
            x = x.unsqueeze(-1)
        if x.max() > 1.0:
            x = x / 255.0
        if x.ndim == 3:  # H,W,C -> B,C,H,W
            x = x.permute(2, 0, 1).unsqueeze(0)
        return x
