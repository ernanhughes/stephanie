# stephanie/components/jitter/coupling/structural_coupling.py
from __future__ import annotations
from typing import Dict, Any

class StructuralCoupling:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.resonance_threshold = float(self.cfg.get("resonance_threshold", 0.7))

    def impact(self, perturb: Dict[str, Any], S: Dict[str, Any]) -> float:
        if "vpm_energy" in perturb:
            return float(perturb["vpm_energy"])  # already [0,1]
        return 0.5

    def adapt(self, perturb: Dict[str, Any], S: Dict[str, Any]) -> Dict[str, Any]:
        imp = self.impact(perturb, S)
        if imp > self.resonance_threshold:
            return dict(
                membrane = dict(thickness= +0.05*imp, permeability= -0.03*imp),
                cognitive = dict(attn={"reptilian": +0.10*imp, "primate": -0.10*imp}),
                metabolic = dict(conversion_bias= +0.05*imp),
            )
        return dict(cognitive=dict(attn={"primate": +0.01}))
