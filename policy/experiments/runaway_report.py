# policy/experiments/runaway_report.py

from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

import scipy

from policy.experiments.runaway_experiment import RunawayDeclineExperiment


class RunawayReportHarness:
    """
    Full plotting + reporting harness.

    Compares:
        - Unbounded system
        - Policy-bounded system

    Produces:
        - time series plots
        - histogram
        - statistical report
    """

    def __init__(
        self,
        ai_callable,
        energy_function,
        policy_container=None,
        episodes: int = 1000,
        noise_scale: float = 0.05,
        out_dir: str = "data/policy_runs",
    ):
        self.ai_callable = ai_callable
        self.energy_function = energy_function
        self.policy_container = policy_container
        self.episodes = episodes
        self.noise_scale = noise_scale
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # RUN BOTH CONDITIONS
    # ============================================================

    def run(self):

        print("\nRunning baseline (no policy)...")

        baseline_exp = RunawayDeclineExperiment(
            ai_callable=self.ai_callable,
            energy_function=self.energy_function,
            policy=None,
            episodes=self.episodes,
            noise_scale=self.noise_scale,
        )

        baseline = baseline_exp.run()

        print("Running policy-bounded experiment...")

        policy_exp = RunawayDeclineExperiment(
            ai_callable=self.ai_callable,
            energy_function=self.energy_function,
            policy=self.policy_container,
            episodes=self.episodes,
            noise_scale=self.noise_scale,
        )

        bounded = policy_exp.run()


        from policy.experiments.plotting import (
            plot_trajectories,
            plot_rolling_variance,
            plot_rolling_mean,
            plot_histogram,
        )

        out_dir = self.out_dir / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_trajectories(baseline["energies"], bounded["energies"], out_dir)
        plot_rolling_variance(baseline["energies"], bounded["energies"], out_dir)
        plot_rolling_mean(baseline["energies"], bounded["energies"], out_dir)
        plot_histogram(baseline["energies"], bounded["energies"], out_dir)

        report = self._analyze(baseline, bounded)

        self._plot_timeseries(baseline["energies"], bounded["energies"])
        self._plot_histogram(baseline["energies"], bounded["energies"])

        with open(self.out_dir / "summary.json", "w") as f:
            json.dump(report, f, indent=2)

        print("\nReport saved to:", self.out_dir)

        return report

    # ============================================================
    # ANALYSIS
    # ============================================================

    def _analyze(self, baseline: Dict, bounded: Dict):

        def slope(y):
            x = np.arange(len(y))
            m = np.polyfit(x, y, 1)[0]
            return float(m)

        report = {
            "baseline": {
                "mean_energy": float(np.mean(baseline["energies"])),
                "max_energy": float(np.max(baseline["energies"])),
                "final_energy": float(baseline["energies"][-1]),
                "slope": slope(baseline["energies"]),
                "variance": float(np.var(baseline["energies"])),
                "energy_std": float(np.std(baseline["energies"])),
                "wasserstein_distance": float(scipy.stats.wasserstein_distance(baseline["energies"], bounded["energies"]))
            },
            "bounded": {
                "mean_energy": float(np.mean(bounded["energies"])),
                "max_energy": float(np.max(bounded["energies"])),
                "final_energy": float(bounded["energies"][-1]),
                "slope": slope(bounded["energies"]),
                "variance": float(np.var(bounded["energies"])),
                "accept_rate": bounded["accept_rate"],
                "energy_std": float(np.std(bounded["energies"])),
                "wasserstein_distance": float(scipy.stats.wasserstein_distance(bounded["energies"], baseline["energies"]))
            }
        }

        return report

    # ============================================================
    # PLOTS
    # ============================================================

    def _plot_timeseries(self, baseline, bounded):

        plt.figure(figsize=(10, 5))
        plt.plot(baseline, label="Unbounded", alpha=0.7)
        plt.plot(bounded, label="Policy-Bounded", alpha=0.9)
        plt.xlabel("Episode")
        plt.ylabel("Energy")
        plt.title("Energy Over Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.out_dir / "energy_timeseries.png")
        plt.close()

    def _plot_histogram(self, baseline, bounded):

        plt.figure(figsize=(8, 5))
        plt.hist(baseline, bins=40, alpha=0.5, label="Unbounded")
        plt.hist(bounded, bins=40, alpha=0.5, label="Bounded")
        plt.xlabel("Energy")
        plt.ylabel("Frequency")
        plt.title("Energy Distribution Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.out_dir / "energy_histogram.png")
        plt.close()
