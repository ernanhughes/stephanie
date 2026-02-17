from dataclasses import dataclass
from typing import List, Any


@dataclass
class EpisodeLog:
    episode: int
    energy: float
    variance: float
    dominance: bool


class DynamicStabilityBenchmark:
    def __init__(self, system: Any):
        self.system = system
        self.logs: List[EpisodeLog] = []

    def run(self, queries: List[Any], episodes: int = 100):
        for ep in range(episodes):
            query = queries[ep % len(queries)]
            result = self.system.step(query)

            self.logs.append(
                EpisodeLog(
                    episode=ep,
                    energy=result.reward_vector.values.get("hallucination_energy", 0.0),
                    variance=0.0,
                    dominance=True,
                )
            )
