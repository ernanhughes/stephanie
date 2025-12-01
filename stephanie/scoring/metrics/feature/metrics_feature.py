# stephanie/scoring/metrics/feature/metrics_feature.py
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List

import numpy as np

from stephanie.scoring.scorable import Scorable

from .base_feature import BaseFeature
from .feature_report import FeatureReport

log = logging.getLogger(__name__)


def _clip01(x: float) -> float:
    try:
        if not np.isfinite(x): return 0.0
        if x < 0.0: return 0.0
        if x > 1.0: return 1.0
        return float(x)
    except Exception:
        return 0.0


class _NoOpObserver:
    def observe(self, **kwargs):  # signature match; do nothing
        pass


class MetricsFeature(BaseFeature):
    """
    Computes the canonical metrics vector for each scorable by calling ScoringService.

    Populates:
      - acc["metrics_vector"]  : Dict[str, float]
      - acc["metrics_columns"] : List[str]
      - acc["metrics_values"]  : List[float]

    Also maintains diagnostics for .report().
    """

    name = "metrics"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # feature toggles
        self.enabled: bool = bool(self.cfg.get("enabled", True))
        self.attach_scores: bool = bool(self.cfg.get("attach_scores", True))
        self.persist: bool = bool(self.cfg.get("persist", False))

        # scoring scope
        self.scorers: List[str] = list(self.cfg.get("scorers", []))
        self.dimensions: List[str] = list(self.cfg.get("dimensions", []))

        # services
        self.scoring = container.get("scoring")  # <-- your ScoringService

        # diagnostics for report() I
        self._num_rows: int = 0
        self._col_minmax: Dict[str, tuple[float, float]] = {}
        self._nan_counts: Dict[str, int] = {}
        self._sum_sq: Dict[str, float] = {}

    # -------------------------------------------------------------
    async def apply(self, scorable, acc: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return acc

        # if already present (e.g., upstream tool), normalize + keep
        existing = dict(acc.get("metrics_vector") or {})
        if existing and self.attach_scores is False:
            cols = sorted(existing.keys())
            vals = [_clip01(float(existing[c])) for c in cols]
            acc["metrics_vector"]  = dict(zip(cols, vals))
            acc["metrics_columns"] = cols
            acc["metrics_values"]  = vals
            self._bump_diag(cols, vals)
            return acc

        # --- Primary path: use ScoringService ---
        vector: Dict[str, float] = {}
        if self.scoring and self.attach_scores and self.scorers:
            t0_scores = time.perf_counter()
            goal_text = Scorable.get_goal_text(scorable, context=context)
            run_id = context.get("pipeline_run_id")
            ctx = {"goal": {"goal_text": goal_text}, "pipeline_run_id": run_id}

            log.debug("[MetricsFeature] score start scorers=%s dims=%s", self.scorers, self.dimensions)

            for name in self.scorers:
                t0 = time.perf_counter()
                try:
                    api = self.scoring.score_and_persist if self.persist else self.scoring.score
                    bundle = api(
                        scorer_name=name,
                        scorable=scorable,
                        context=ctx,
                        dimensions=self.dimensions,
                    )
                    if asyncio.iscoroutine(bundle):
                        bundle = await bundle

                    # model alias + flat metrics
                    try:
                        model_alias = self.scoring.get_model_name(name) or name
                    except Exception:
                        model_alias = name

                    agg = float(bundle.aggregate()) if hasattr(bundle, "aggregate") else None
                    flat = bundle.flatten(
                        include_scores=True,
                        include_attributes=True,
                        numeric_only=True,
                    ) if hasattr(bundle, "flatten") else {}

                    for k, v in flat.items():
                        vector[f"{model_alias}.{k}"] = _clip01(float(v))
                    if agg is not None:
                        vector[f"{model_alias}.aggregate"] = _clip01(agg)

                    log.debug(
                        "[MetricsFeature] scorer=%s alias=%s agg=%s added=%d in %.1fms",
                        name, model_alias, f"{agg:.4f}" if agg is not None else "na",
                        (len(flat) + (1 if agg is not None else 0)),
                        (time.perf_counter() - t0) * 1000.0,
                    )
                except Exception as e:
                    log.warning("[MetricsFeature] scorer '%s' failed: %s", name, e)

                await asyncio.sleep(0)  # cooperative yield

            log.debug(
                "[MetricsFeature] score done total_keys=%d in %.1fms",
                len(vector), (time.perf_counter() - t0_scores) * 1000.0
            )

        # Fallback: heuristics if vector still empty (keeps pipeline alive)
        if not vector:
            vector = self._fallback_heuristics((scorable.text or ""))

        # Deterministic ordering + stash
        cols = sorted(vector.keys())
        vals = [float(vector[c]) for c in cols]

        acc["metrics_vector"]  = vector
        acc["metrics_columns"] = cols
        acc["metrics_values"]  = vals

        # Diagnostics & optional observe/persist
        self._bump_diag(cols, vals)

        # try:
        #     self.metric_observer.observe(
        #         metrics=dict(zip(cols, vals)),
        #         run_id=context.get("run_id", "unknown_run"),
        #         cohort=context.get("cohort", "default"),
        #         is_correct=(getattr(scorable, "meta", {}) or {}).get("is_correct"),
        #     )
        # except Exception:
        #     pass  # never block pipeline on telemetry

        return acc

    # -------------------------------------------------------------
    def report(self) -> FeatureReport:
        if self._num_rows == 0 or not self._col_minmax:
            return FeatureReport(
                name=self.name, kind="row", ok=True, quality=None,
                summary="no metrics collected yet (num_rows=0)",
            )

        total_cols = len(self._col_minmax)
        oob = {k: (lo, hi) for k, (lo, hi) in self._col_minmax.items() if (hi > 1.001) or (lo < -0.001)}
        nan_cols = {k: c for k, c in self._nan_counts.items() if c > 0}
        var_proxy = {k: (s / max(self._num_rows, 1)) for k, s in self._sum_sq.items()}
        var_top = dict(sorted(var_proxy.items(), key=lambda kv: -kv[1])[:20])

        bad_cols = len(oob) + len(nan_cols)
        quality = float(max(0.0, 1.0 - (bad_cols / max(total_cols, 1))))

        return FeatureReport(
            name=self.name,
            kind="row",
            ok=(bad_cols == 0),
            quality=quality,
            summary=f"{total_cols} metrics across {self._num_rows} rows; out_of_bounds={len(oob)}; NaN_cols={len(nan_cols)}",
            details={
                "out_of_bounds_examples": dict(list(oob.items())[:10]),
                "nan_columns": nan_cols,
                "variance_proxy_top": var_top,
            },
            warnings=(
                (["Found metrics outside [0,1]"] if oob else [])
                + (["Found NaNs in metrics"] if nan_cols else [])
            ),
        )

    # -------------------------------------------------------------
    # internals
    def _fallback_heuristics(self, text: str) -> Dict[str, float]:
        t = text.strip()
        n_chars = len(t)
        n_tokens = len(t.split())
        digit_ratio = (sum(c.isdigit() for c in t) / max(n_chars, 1))
        upper_ratio = (sum(c.isupper() for c in t) / max(n_chars, 1))
        punct_ratio = (sum(c in ".,?!;:()[]{}\"'" for c in t) / max(n_chars, 1))

        # crude entropy → squash to [0,1]
        counts = {}
        for c in t:
            counts[c] = counts.get(c, 0) + 1
        total = sum(counts.values()) or 1
        ent = -sum((c/total) * np.log((c/total) + 1e-12) for c in counts.values())
        ent_norm = float(np.tanh(ent / 5.0))

        len_norm = float(np.tanh(n_tokens / 200.0))

        return {
            "heuristics.length.score": _clip01(len_norm),
            "heuristics.entropy.score": _clip01(ent_norm),
            "heuristics.digit_ratio.score": _clip01(digit_ratio),
            "heuristics.upper_ratio.score": _clip01(upper_ratio),
            "heuristics.punct_ratio.score": _clip01(punct_ratio),
        }

    def _bump_diag(self, cols: List[str], vals: List[float]) -> None:
        if not cols or not vals:
            return
        self._num_rows += 1
        for k, v in zip(cols, vals):
            if v is None or not np.isfinite(v):
                self._nan_counts[k] = self._nan_counts.get(k, 0) + 1
                continue
            v = float(v)
            lo, hi = self._col_minmax.get(k, (v, v))
            if v < lo: lo = v
            if v > hi: hi = v
            self._col_minmax[k] = (lo, hi)
            self._sum_sq[k] = self._sum_sq.get(k, 0.0) + v * v
