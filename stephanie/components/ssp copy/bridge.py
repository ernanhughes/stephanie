# stephanie/components/ssp/bridge.py
from __future__ import annotations

import time
from dataclasses import asdict
from stephanie.components.ssp.types import SensoryBundle
from stephanie.components.ssp.util import get_trace_logger, PlanTrace_safe

class Bridge:
    def __init__(self, ssp, cfg):
        self.ssp = ssp
        self.cfg = cfg.self_play.jitter
        self.trace_logger = get_trace_logger()

    def get_sensory(self) -> SensoryBundle:
        vpm = self.ssp.vpm.get_current_state()
        scm = {"coherence": 0.70, "novelty": 0.60, "complexity": 0.50}
        sr = self.ssp.trainer.success_history
        success_rate = sum(sr)/len(sr) if sr else 0.5
        epistemic = {
            "difficulty": self.ssp.proposer.difficulty,
            "success_rate": success_rate,
            "knowledge_growth": min(1.0, max(0.0, success_rate*0.8 + 0.1)),
            "verification_rate": success_rate
        }
        bundle = SensoryBundle(vpm=vpm, scm=scm, epistemic=epistemic,
                               meta={"tick_interval": self.cfg.get("tick_interval", 2.0),
                                     "timestamp": time.time(), "episode_count": self.ssp.episode_count})
        tr = PlanTrace_safe(
            trace_id=f"jitter-bridge-{int(time.time()*1000)%1000000}", role="jitter",
            goal="substrate feed", status="completed",
            metadata={"channels": ["vpm","scm","epistemic"]}, input="",
            output="sensory bundle", artifacts=asdict(bundle)
        )
        self.trace_logger.log(tr)
        return bundle
