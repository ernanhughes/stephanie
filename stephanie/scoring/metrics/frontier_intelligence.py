# stephanie/scoring/metrics/intelligent_frontier.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FrontierIntelligence:
    """
    Meta-controller for FrontierLens.

    Tracks how well individual metrics separate target vs baseline over time,
    and uses that (plus critic feedback) to choose a frontier metric.

    This is where learning *about* the metric space happens.
    """

    stability_window: int = 5
    alpha: float = 0.5     # weight on std (penalty)
    beta: float = 0.2      # weight on critic quality (bonus)

    # metric_name -> list of AUCs from past runs
    metric_history: Dict[str, List[float]] = field(default_factory=dict)
    # critic-wide quality scores (e.g., AUC over an eval set)
    critic_quality_history: List[float] = field(default_factory=list)

    # dynamically maintained list of core metrics
    core_metrics: List[str] = field(default_factory=list)

    def update_metric_stability(
        self,
        metric_importance: List[Dict[str, float]],
        critic_quality_score: Optional[float] = None,
    ) -> None:
        """
        Log AUC stability per metric and optionally log critic-wide quality
        for this run.

        metric_importance:
            [
              {"metric": "sicql.coverage.score", "auc_roc": 0.91, ...},
              ...
            ]
        critic_quality_score:
            e.g., TinyCritic AUC over held-out eval for this run.
        """
        # Per-metric AUC history
        for metric in metric_importance:
            name = metric["metric"]
            auc = float(metric["auc_roc"])

            hist = self.metric_history.setdefault(name, [])
            hist.append(auc)
            # Trim history to stability window
            if len(hist) > self.stability_window:
                hist.pop(0)

        # Critic quality history
        if critic_quality_score is not None:
            self.critic_quality_history.append(float(critic_quality_score))
            if len(self.critic_quality_history) > self.stability_window:
                self.critic_quality_history.pop(0)

        self._update_core_metrics()

    def _update_core_metrics(self) -> None:
        """
        Identify the most stable, useful metrics across runs.

        Score:
            stability_score = avg_auc
                              - alpha * std_auc
                              + beta  * avg_critic_quality
        """
        stable_metrics: List[Tuple[str, float, float, float]] = []

        avg_critic_quality = (
            float(np.mean(self.critic_quality_history))
            if self.critic_quality_history
            else 0.5
        )

        for metric, scores in self.metric_history.items():
            # skip special key
            if metric == "critic_quality":
                continue

            if len(scores) < max(2, self.stability_window // 2):
                continue

            scores_arr = np.asarray(scores, dtype=np.float64)
            stability = float(scores_arr.std())
            avg_auc = float(scores_arr.mean())

            stability_score = (
                avg_auc
                - self.alpha * stability
                + self.beta * avg_critic_quality
            )
            stable_metrics.append((metric, stability_score, avg_auc, stability))

        # Sort by stability_score descending
        stable_metrics.sort(key=lambda x: x[1], reverse=True)
        self.core_metrics = [m[0] for m in stable_metrics[: max(5, len(stable_metrics) // 3)]]

        if self.core_metrics:
            log.info(
                "FrontierIntelligence: updated core_metrics (%d): %s%s",
                len(self.core_metrics),
                ", ".join(self.core_metrics[:3]),
                "..." if len(self.core_metrics) > 3 else "",
            )
        else:
            log.warning(
                "FrontierIntelligence: no stable metrics yet "
                "(need more runs or labels)."
            )

    def select_frontier_metric(
        self,
        candidate_metrics: Sequence[str],
        fallback: Optional[str] = None,
    ) -> str:
        """
        Choose the best frontier metric from the candidate list.

        Preference order:
          1) First core metric that appears in candidate_metrics,
          2) explicit fallback,
          3) first candidate.
        """
        cand_set = set(candidate_metrics)

        # Prefer learned core metrics
        for m in self.core_metrics:
            if m in cand_set:
                log.info("FrontierIntelligence: selected frontier_metric=%r", m)
                return m

        # Fallbacks
        if fallback and fallback in cand_set:
            log.warning(
                "FrontierIntelligence: no core metrics in candidate set; "
                "using fallback=%r",
                fallback,
            )
            return fallback

        if candidate_metrics:
            log.warning(
                "FrontierIntelligence: no core metrics; using first candidate=%r",
                candidate_metrics[0],
            )
            return candidate_metrics[0]

        raise ValueError("FrontierIntelligence.select_frontier_metric: no candidate_metrics provided")


def compute_dynamic_bands(
    frontier_values: np.ndarray,
    target_good_ratio: float = 0.4,
    min_band_size: int = 5,
) -> Tuple[float, float, float]:
    """
    Dynamically adjust band boundaries based on the empirical distribution
    of the frontier metric.

    Args:
        frontier_values:
            1D array of frontier metric values for all examples
            (already normalized into [0,1] is ideal).
        target_good_ratio:
            Desired fraction of examples in the "good" band.
        min_band_size:
            Minimum #examples on each side (below and above the band).

    Returns:
        (low_threshold, high_threshold, actual_good_ratio)
    """
    vals = np.asarray(frontier_values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        # degenerate case: no finite data
        log.warning(
            "compute_dynamic_bands: no finite frontier_values; using default [0.25, 0.75]."
        )
        return 0.25, 0.75, 0.5

    vals.sort()
    n = vals.size

    # Ideal indices assuming symmetric tails for "bad"
    tail_fraction = (1.0 - float(target_good_ratio)) / 2.0
    low_idx = max(min_band_size, int(n * tail_fraction))
    high_idx = min(n - min_band_size, n - int(n * tail_fraction))

    # Safety clamps
    low_idx = max(0, min(low_idx, n - 1))
    high_idx = max(low_idx + 1, min(high_idx, n - 1))

    low_threshold = float(vals[low_idx])
    high_threshold = float(vals[high_idx])

    # Compute actual achieved good ratio
    mask_good = (vals >= low_threshold) & (vals <= high_threshold)
    actual_good_ratio = float(mask_good.mean())

    if high_threshold <= low_threshold:
        log.warning(
            "compute_dynamic_bands: degenerate thresholds (low=%.3f, high=%.3f); "
            "falling back to [0.25, 0.75].",
            low_threshold,
            high_threshold,
        )
        low_threshold, high_threshold = 0.25, 0.75
        actual_good_ratio = 0.5

    log.info(
        "FrontierIntelligence: dynamic bands [%.3f, %.3f] "
        "(target_good_ratio=%.2f, actual_good_ratio=%.2f, n=%d)",
        low_threshold,
        high_threshold,
        target_good_ratio,
        actual_good_ratio,
        n,
    )

    return low_threshold, high_threshold, actual_good_ratio
