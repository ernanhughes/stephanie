from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import torch
from stephanie.scoring.scorer.base_scorer import BaseScorer
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.model.vpm_thought_policy import VPMThoughtPolicy, VPMThoughtModelConfig
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult

class VPMThoughtPolicyScorer(BaseScorer):
    """
    Produces the next visual action proposal (+ value estimate) from a VPM frame,
    so SIS/DAIMON can visualize & gate execution.
    """
    def __init__(self, cfg: Dict[str, Any], memory, container, logger=None):
        super().__init__(cfg, memory, container, logger)
        mcfg = VPMThoughtModelConfig(**cfg.get("model", {}))
        self.model = VPMThoughtPolicy(mcfg).eval()
        self.device = torch.device(cfg.get("device", "cpu"))
        self.model.to(self.device)
        # Load weights if provided
        w = cfg.get("weights")
        if w:
            state = torch.load(w, map_location=self.device)
            self.model.load_state_dict(state["model_state"], strict=False)

    def _score_core(self, context: dict, scorable: Scorable, dimensions: List[str]) -> ScoreBundle:
        # Expect scorable to carry a VPM tensor or a path → load to [C,H,W] float32 in [0,1]
        vpm = scorable.get_image_array()
        if vpm.ndim == 2:
            vpm = vpm[None, ...]
        if vpm.max() > 1.0:
            vpm = vpm / 255.0
        vpm_t = torch.from_numpy(vpm).unsqueeze(0).to(self.device)

        # Map dims to a goal vector (keep 4-length like trainer)
        goal_weights = {
            "separability": scorable.meta.get("goal_separability", 1.0),
            "bridge_proxy": scorable.meta.get("goal_bridge_proxy", 0.0),
            "symmetry": scorable.meta.get("goal_symmetry", 0.0),
            "spectral_gap": scorable.meta.get("goal_spectral_gap", 0.0),
        }
        g = np.array([goal_weights["separability"], goal_weights["bridge_proxy"],
                      goal_weights["symmetry"], goal_weights["spectral_gap"]], dtype=np.float32)
        g_t = torch.from_numpy(g).unsqueeze(0).to(self.device)

        with torch.no_grad():
            op_logits, param_mean, _, value = self.model(vpm_t, g_t)

        op_idx = int(torch.argmax(op_logits, dim=1).item())
        params = param_mean[0].cpu().numpy().tolist()
        score = float(torch.sigmoid(value.squeeze()).item())

        # Return a compact bundle
        return ScoreBundle(results={
            "thought_value": ScoreResult("thought_value", score, "Predicted utility of the next visual step.", "vpm_thought"),
            "op_index":     ScoreResult("op_index", float(op_idx), "Argmax op type (0..K-1).", "vpm_thought",
                                        attributes={"params": params}),
        })
