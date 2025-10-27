# stephanie/components/jitter/cognition/sense_making.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple

class SenseMakingEngine:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self._sens: List[Dict[str, Any]] = []
        self._act:  List[Dict[str, Any]] = []
        self._H = int(self.cfg.get("max_buffer_size", 128))

    def _sim(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        # trivial similarity for now
        return 1.0 if a.get("type")==b.get("type") else 0.5

    def process(self, sensory: Dict[str, Any], action: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        sims = [self._sim(sensory, s) for s in self._sens[-8:]]
        fam = sum(sims)/max(1, len(sims))
        resonance = 0.9*fam + 0.1*float(outcome.get("quality", 0.5))
        mtype = "affordance" if outcome.get("quality", 0.5) > 0.6 and resonance>0.6 else \
                "constraint"  if outcome.get("quality", 0.5) < 0.4 and resonance>0.6 else "neutral"
        self._sens.append(sensory); self._act.append(action)
        if len(self._sens) > self._H: self._sens.pop(0); self._act.pop(0)
        return dict(meaning_type=mtype, resonance=resonance, familiarity=fam, outcome_quality=outcome.get("quality",0.5))
All right so I'm going to Quan for right now Let's get it gone