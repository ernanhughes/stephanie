from typing import Callable


class DynamicStabilityBenchmark:

    def __init__(self, container, steps: int = 1000):
        self.container = container
        self.steps = steps

    def run(self, input_generator: Callable):

        decisions = []
        energies = []

        for _ in range(self.steps):
            decision, result = self.container.run(input_generator())
            decisions.append(decision)
            energies.append(result.support_diagnostics.mean_energy)

        return {
            "decisions": decisions,
            "energies": energies,
        }
