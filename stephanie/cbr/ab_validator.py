# stephanie/cbr/ab_validator.py
from typing import Callable, Dict, Tuple


class DefaultABValidator:
    def __init__(self, cfg, memory, container, logger, ns, assessor):
        self.cfg, self.memory, self.logger = cfg, memory, container, logger
        self.ns, self.assessor = ns, assessor
        self.delta_eps = float(cfg.get("ab_validation",{}).get("delta_eps", 1e-6))

    async def run_two(self, ctx, run_cbr: Callable[[], dict], run_baseline: Callable[[], dict]) -> Tuple[str, Dict]:
        base = await run_baseline()
        cbr  = await run_cbr()
        q_base = float(base["metrics"]["quality"])
        q_cbr  = float(cbr["metrics"]["quality"])
        improved = q_cbr > (q_base + self.delta_eps)
        winner = "cbr" if improved else "baseline"
        return winner, {"q_base": q_base, "q_cbr": q_cbr, "improved": improved, "winner": winner}
