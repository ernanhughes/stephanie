# stephanie/components/jitter/production/closed_production.py
from __future__ import annotations
from typing import Dict, Any
import numpy as np

class ProductionNetwork:
    def __init__(self, cfg: Dict[str, Any] | None = None):
        self.cfg = cfg or {}
        self._rates = dict(membrane=0.10, metabolic_pathways=0.20,
                           cognitive_structures=0.15, regulatory_mechanisms=0.05)
        self._hist = []

    def update_rates(self, S: Dict[str, Any]) -> None:
        memE = float(S["energy"].get("metabolic", 50.0))
        integ = float(S["membrane"].get("integrity", 0.6))
        coh  = float(S["cognitive"].get("coherence", 0.5))
        self._rates["membrane"]              = 0.10 * (memE / 100.0)
        self._rates["cognitive_structures"]  = 0.15 * integ
        self._rates["regulatory_mechanisms"] = 0.05 * coh

    def produce(self, S: Dict[str, Any]) -> Dict[str, float]:
        self.update_rates(S)
        changes = dict(
            membrane = self._rates["membrane"] * float(S["metabolic"].get("pathways", 1.0)),
            metabolic_pathways = self._rates["metabolic_pathways"] * float(S["cognitive"].get("structures", 1.0)),
            cognitive_structures = self._rates["cognitive_structures"] * float(S["membrane"].get("integrity", 0.6)),
            regulatory_mechanisms = self._rates["regulatory_mechanisms"]
        )
        self._hist.append(dict(rates=self._rates.copy(), changes=changes))
        if len(self._hist) > 512: self._hist.pop(0)
        return changes

    def efficiency(self) -> float:
        if len(self._hist) < 3: return 0.5
        rs = [h["rates"] for h in self._hist]
        var = np.var([r["membrane"] for r in rs])
        bal = 1.0 - np.std(list(self._rates.values()))
        return float(0.6 * (1/(1+var)) + 0.4 * bal)
