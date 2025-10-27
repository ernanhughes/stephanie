# stephanie/components/jitter/jaf_seeder.py
from __future__ import annotations
from typing import Any, Dict, Optional
import json
import os

class JAFSeeder:
    def __init__(self, path: Optional[str]=None):
        self.path = path or os.getenv("JAS_JAF_LAST", "logs/jas_last.jaf.json")

    def load(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.path): return None
        with open(self.path, "r", encoding="utf-8") as f: return json.load(f)

    def apply(self, triune, homeostasis) -> None:
        data = self.load()
        if not data: return
        w = data.get("triune_attn_weights")
        if w and hasattr(triune, "set_weights"): triune.set_weights(w)
        sp = data.get("homeostasis_setpoints", {})
        for name, val in sp.items():
            if hasattr(homeostasis, "set_setpoint"): homeostasis.set_setpoint(name, float(val))
