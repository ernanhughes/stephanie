# stephanie/components/ssp/util.py
from __future__ import annotations

import json
import sys
from dataclasses import dataclass


def _dummy_llm_response(prompt: str) -> str:
    return json.dumps({
        "query": "How does VPM evolution affect HRM improvements on long-horizon tasks?",
        "verification_approach": "Compare HRM/MARS deltas before/after VPM evolve on matched cases; bootstrap CI.",
        "difficulty": 0.5,
        "connections": ["VPM", "HRM", "MARS"]
    })

class DummyModel:
    def __call__(self, prompt: str) -> str:
        return _dummy_llm_response(prompt)

def get_model_safe(name: str):
    try:
        from stephanie.core.model_router import \
            get_model as _gm  # type: ignore
        return _gm(name)
    except Exception:
        return DummyModel()

class _NullTraceLogger:
    def log(self, trace):
        try:
            tid = getattr(trace, "trace_id", "trace")
            role = getattr(trace, "role", "?")
            status = getattr(trace, "status", "")
            print(f"[TRACE] {tid} role={role} status={status}", flush=True)
        except Exception:
            # swallow any stdout/stderr issues during finalization
            try:
                sys.stderr.write("[TRACE] <logging error suppressed>\\n")
            except Exception:
                pass

def get_trace_logger():
    try:
        from stephanie.utils.trace_logger import trace_logger  # type: ignore
        return trace_logger
    except Exception:
        return _NullTraceLogger()

@dataclass
class _PlanTrace:
    trace_id: str
    role: str
    goal: str
    status: str
    metadata: dict
    input: str
    output: str
    artifacts: dict

def PlanTrace_safe(**kwargs):
    try:
        from stephanie.traces.plan_trace import PlanTrace  # type: ignore
        return PlanTrace(**kwargs)
    except Exception:
        return _PlanTrace(**kwargs)

class VPMEvolverSafe:
    def __init__(self, cfg=None):
        self._tick = 0
    def get_current_state(self):
        import numpy as np
        self._tick += 1
        tensor = np.random.rand(1, 3, 64, 64).astype("float32")
        sig = np.random.rand(1, 1, 16, 16).astype("float32")
        return {"tensor": tensor, "significance_map": sig, "metadata": {"tick": self._tick}}
    def evolve_once(self, vpm):
        return vpm

class MemCubeSafe:
    def search_neighbors(self, query: str, k: int = 5):
        return [{"id": i, "score": 0.8 - i*0.05} for i in range(min(k, 3))]
