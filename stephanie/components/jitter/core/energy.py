# stephanie/components/jitter/core/energy.py
from __future__ import annotations


class EnergyPools:
    def __init__(self, cognitive: float = 50.0, metabolic: float = 50.0, reserve: float = 20.0):
        self.energy_pools = {
            "cognitive": float(cognitive),
            "metabolic": float(metabolic),
            "reserve": float(reserve),
        }
        self.max_reserve = 100.
        self.pathway_rate_factor: float = 1.0

    def level(self, name: str) -> float:
        return float(self.energy_pools.get(name, 0.0))

    def replenish(self, name: str, amount: float) -> None:
        self.energy_pools[name] = max(0.0, self.energy_pools.get(name, 0.0) + float(amount))

    def consume(self, name: str, amount: float) -> None:
        self.energy_pools[name] = max(0.0, self.energy_pools.get(name, 0.0) - float(amount))

    def transfer(self, src: str, dst: str, amount: float) -> float:
        amount = max(0.0, float(amount))
        avail = min(self.level(src), amount)
        self.consume(src, avail)
        self.replenish(dst, avail)
        return avail

    def alive(self) -> bool:
        # minimal: dies if both cognitive & metabolic exhausted
        return self.level("metabolic") > 0.5 or self.level("cognitive") > 0.5

    def adjust_pathway_rates(self, factor: float) -> None:
        """
        Adjust global pathway “throughput”. For simple pools this is stored as a
        scalar that other components may consult. It’s clamped to a safe range.
        """
        try:
            f = max(0.1, float(factor))
        except Exception:
            f = 1.0
        # multiplicative update, then clamp overall factor
        self.pathway_rate_factor = max(0.2, min(5.0, self.pathway_rate_factor * f))
