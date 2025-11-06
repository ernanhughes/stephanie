# stephanie/scoring/calculations/mars_calculator.py
from __future__ import annotations

import json
import math
import os
import traceback
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

from stephanie.data.score_corpus import ScoreCorpus
from stephanie.scoring.calculations.base_calculator import BaseScoreCalculator
from stephanie.utils.json_sanitize import json_sanitize
from stephanie.utils.serialization import default_serializer


def _safe_scalar(x: Any) -> float | None:
    """Convert to float; return None if not finite or convertible."""
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _safe_mean(values: List[float | None]) -> float | None:
    vals = [_safe_scalar(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _safe_std(values: List[float | None]) -> float:
    vals = [_safe_scalar(v) for v in values]
    vals = [v for v in vals if v is not None]
    if len(vals) <= 1:
        return 0.0
    return float(np.std(vals))


def _safe_min(values: List[float | None]) -> float | None:
    vals = [_safe_scalar(v) for v in values]
    vals = [v for v in vals if v is not None]
    return float(np.min(vals)) if vals else None


def _safe_max(values: List[float | None]) -> float | None:
    vals = [_safe_scalar(v) for v in values]
    vals = [v for v in vals if v is not None]
    return float(np.max(vals)) if vals else None


def _to_python(value):
    """Make sure numpy types and non-finite floats become JSON-safe Python values."""
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_python(v) for k, v in value.items()}
    return value


class MARSCalculator(BaseScoreCalculator):
    """
    Model Agreement and Reasoning Signal (MARS) Calculator

    Operates over a ScoreCorpus to analyze agreement/divergence across scorers.
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        self.trust_reference = self.cfg.get("trust_reference", "llm")
        self.variance_threshold = self.cfg.get("variance_threshold", 0.15)
        self.metrics = self.cfg.get("metrics", ["score"])
        self.dimension_configs = self.cfg.get("dimension_configs", {}) or {}
        self.save_conflicts = self.cfg.get("save_conflicts", True)

        # Logging options
        self.log_enabled = self.cfg.get("log_enabled", True)
        self.log_path = self.cfg.get("log_path", "reports")
        self.include_full_data = self.cfg.get("include_full_data", True)

        if self.log_enabled and self.logger:
            self.logger.log(
                "MARSLoggerConfigured",
                {
                    "log_path": os.path.abspath(self.log_path),
                    "include_full_data": bool(self.include_full_data),
                    "enabled": bool(self.log_enabled),
                },
            )

    def _write_json_report(self, mars_results: Dict[str, Any], corpus: "ScoreCorpus"):
        """Write MARS results to JSON; always JSON-safe."""
        if not self.log_enabled:
            return
        try:
            os.makedirs(self.log_path, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.log_path, f"mars_report_{ts}.json")

            report_data = {
                "metadata": {
                    "timestamp": ts,
                    "document_count": len(corpus.bundles),
                    "scorers": list(corpus.scorers),
                    "metrics": list(corpus.metrics),
                },
                "results": json_sanitize(mars_results),
                "corpus_summary": json_sanitize(corpus.get_summary()),
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, default=default_serializer)

            if self.logger:
                self.logger.log(
                    "MARSReportSaved",
                    {
                        "filepath": filepath,
                        "document_count": len(corpus.bundles),
                        "scorers_count": len(corpus.scorers),
                    },
                )
        except Exception as e:
            if self.logger:
                self.logger.log(
                    "MARSReportError",
                    {"error": str(e), "traceback": traceback.format_exc()},
                )

    def calculate(self, corpus: "ScoreCorpus", context: dict) -> Dict[str, Any]:
        """
        Calculate per-dimension MARS and persist a JSONB 'result' per dimension.
        Returns: {dimension: result_dict}
        """
        mars_results: Dict[str, Any] = {}

        # Log a tiny, serializable preview
        try:
            first_dim = next(iter(corpus.dimensions), None)
            head_preview = None
            if first_dim:
                head_preview = str(corpus.get_dimension_matrix(first_dim).head())
            if self.logger:
                self.logger.log(
                    "MARSCalculationStarted",
                    {
                        "document_count": len(corpus.bundles),
                        "dimensions": list(corpus.dimensions),
                        "head_preview": head_preview,
                    },
                )
        except Exception:
            pass

        for dimension in corpus.dimensions:
            result = self._calculate_dimension_mars(corpus, dimension, context)
            # ensure JSON-safety before using result further
            safe_result = json_sanitize(result)
            mars_results[dimension] = safe_result

            # Persist one row per dimension as JSONB 'result'
            try:
                self.memory.mars_results.add(
                    pipeline_run_id=context.get("pipeline_run_id"),
                    plan_trace_id=context.get("plan_trace_id"),
                    result=safe_result,
                )
            except Exception as e:
                if self.logger:
                    self.logger.log(
                        "MARSResultPersistenceError",
                        {"error": str(e), "dimension": dimension},
                    )

        # Optional file report
        self._write_json_report(mars_results, corpus)
        return mars_results

    def _get_dimension_config(self, dimension: str) -> Dict:
        """Get dimension-specific config with fallbacks."""
        return self.dimension_configs.get(
            dimension,
            {
                "trust_reference": self.trust_reference,
                "variance_threshold": self.variance_threshold,
                "metrics": self.metrics,
            },
        )

    def _calculate_dimension_mars(
        self, corpus: "ScoreCorpus", dimension: str, context: dict
    ) -> Dict[str, Any]:
        dim_cfg = self._get_dimension_config(dimension)
        trust_ref = dim_cfg.get("trust_reference", self.trust_reference)
        metrics = list(dim_cfg.get("metrics", self.metrics) or ["score"])

        matrix = corpus.get_dimension_matrix(dimension)  # [docs x scorers]

        if matrix.empty or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return {
                "dimension": str(dimension),
                "agreement_score": 0.0,
                "std_dev": 0.0,
                "preferred_model": "none",
                "primary_conflict": ["none", "none"],
                "delta": 0.0,
                "high_disagreement": False,
                "explanation": "No data available for this dimension",
                "scorer_metrics": {},
                "metric_correlations": {},
                "source": "mars",
                "average_score": 0.0,
            }

        # Column means (per scorer) → then average across scorers
        col_means = matrix.mean(axis=0, skipna=True)
        avg_score = _safe_scalar(col_means.mean()) or 0.0

        # Column std across docs (per scorer), then mean std across scorers
        col_stds = matrix.std(axis=0, ddof=0, skipna=True)
        std_dev = _safe_scalar(col_stds.mean()) or 0.0

        # Agreement (1 - dispersion), clipped
        agreement_score = max(0.0, min(1.0, 1.0 - float(std_dev)))

        # Primary conflict = max difference among scorer means
        scorer_means = col_means.fillna(0.0)
        if len(scorer_means) >= 1:
            max_name = str(scorer_means.idxmax())
            min_name = str(scorer_means.idxmin())
            delta = float(scorer_means[max_name] - scorer_means[min_name])
            primary_conflict = [max_name, min_name]
        else:
            primary_conflict = ["none", "none"]
            delta = 0.0

        # Preferred model: closest to trust_ref by L1 distance to trust_ref’s scores
        if trust_ref in matrix.columns:
            trust_scores = matrix[trust_ref]
            closest = None
            min_diff = float("inf")
            for scorer in matrix.columns:
                if scorer == trust_ref:
                    continue
                diff = (matrix[scorer] - trust_scores).abs().mean(skipna=True)
                dval = _safe_scalar(diff)
                if dval is not None and dval < min_diff:
                    min_diff = dval
                    closest = scorer
            preferred_model = str(closest) if closest is not None else "unknown"
        else:
            # Fallback: median of scorer means
            ordered = scorer_means.sort_values()
            ix = int(len(ordered) / 2)
            preferred_model = str(ordered.index[ix]) if len(ordered) else "unknown"

        high_disagreement = float(std_dev) > float(dim_cfg.get("variance_threshold", 0.15))

        # Per-scorer metric summaries (never NaN)
        scorer_metrics = self._analyze_scorer_metrics(corpus, dimension, metrics)

        # Metric correlations (only when variance exists)
        metric_correlations = self._calculate_metric_correlations(corpus, dimension, metrics)

        # Human-readable explanation
        exp = [
            f"MARS agreement: {agreement_score:.3f} (std: {float(std_dev):.3f})",
            f"Most aligned with {trust_ref}: {preferred_model}" if preferred_model != "unknown" else "No trust reference alignment available",
            f"Primary conflict: {primary_conflict[0]} vs {primary_conflict[1]} (Δ={delta:.3f})",
        ]
        if high_disagreement:
            exp.append(f"⚠️ High disagreement (>{dim_cfg.get('variance_threshold')})")
        explanation = " | ".join(exp)

        result = {
            "dimension": str(dimension),
            "agreement_score": float(agreement_score),
            "std_dev": float(std_dev),
            "preferred_model": str(preferred_model),
            "primary_conflict": list(primary_conflict),
            "delta": float(delta),
            "high_disagreement": bool(high_disagreement),
            "explanation": explanation,
            "scorer_metrics": _to_python(scorer_metrics),
            "metric_correlations": _to_python(metric_correlations),
            "source": "mars",
            "average_score": float(avg_score),
        }

        # Optional: persist top conflict
        if (
            self.save_conflicts
            and result["primary_conflict"] != ["none", "none"]
        ):
            try:
                self.memory.mars_conflicts.add(
                    pipeline_run_id=context.get("pipeline_run_id"),
                    plan_trace_id=context.get("plan_trace_id"),
                    dimension=result["dimension"],
                    conflict=json_sanitize(result["primary_conflict"]),
                    delta=float(result["delta"]),
                    explanation=result["explanation"],
                    agreement_score=float(result["agreement_score"]),
                    preferred_model=result["preferred_model"],
                )
                if self.logger:
                    self.logger.log(
                        "MARSConflictStored",
                        {
                            "pipeline_run_id": context.get("pipeline_run_id"),
                            "plan_trace_id": context.get("plan_trace_id"),
                            "dimension": result["dimension"],
                            "conflict": result["primary_conflict"],
                            "delta": result["delta"],
                            "agreement_score": result["agreement_score"],
                            "preferred_model": result["preferred_model"],
                        },
                    )
            except Exception as e:
                if self.logger:
                    self.logger.log(
                        "MARSConflictStoreError",
                        {"error": str(e), "dimension": result["dimension"]},
                    )

        return result

    def _analyze_scorer_metrics(
        self, corpus: "ScoreCorpus", dimension: str, metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Summarize selected metrics for each scorer in this dimension.
        Returns: {scorer: {metric: {mean,std,min,max,count}}}
        """
        out: Dict[str, Dict[str, float]] = {}
        for scorer in corpus.scorers:
            metric_values = corpus.get_metric_values(dimension, scorer, metrics)
            stats_dict: Dict[str, Dict[str, float]] = {}
            for metric, vals in metric_values.items():
                if metric == "scorable_id":
                    continue
                # clean values → floats, finite only
                cleaned: List[float] = []
                for v in vals:
                    fv = _safe_scalar(v)
                    if fv is not None:
                        cleaned.append(fv)
                if not cleaned:
                    continue
                stats_dict[metric] = {
                    "mean": float(np.mean(cleaned)),
                    "std": float(np.std(cleaned)) if len(cleaned) > 1 else 0.0,
                    "min": float(np.min(cleaned)),
                    "max": float(np.max(cleaned)),
                    "count": int(len(cleaned)),
                }
            if stats_dict:
                out[scorer] = stats_dict
        return out

    def _calculate_metric_correlations(
        self, corpus: "ScoreCorpus", dimension: str, metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Correlations between metrics over scorables (only when both sides have variance).
        Returns: {metricA: {metricB: corr}}
        """
        if len(metrics) < 2:
            return {}

        # ScoreCorpus should now expose this; otherwise build locally
        try:
            metric_values = corpus.get_all_metric_values(dimension, metrics)
        except AttributeError:
            # Fallback: build across *all scorers* combined
            metric_values = {m: [] for m in metrics}
            for scorer in corpus.scorers:
                vals = corpus.get_metric_values(dimension, scorer, metrics)
                for m in metrics:
                    metric_values[m].extend(
                        [_safe_scalar(v) for v in vals.get(m, [])]
                    )

        correlations: Dict[str, Dict[str, float]] = {}
        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                m1, m2 = metrics[i], metrics[j]
                v1 = metric_values.get(m1, [])
                v2 = metric_values.get(m2, [])

                pairs: List[Tuple[float, float]] = []
                for a, b in zip(v1, v2):
                    fa, fb = _safe_scalar(a), _safe_scalar(b)
                    if fa is not None and fb is not None:
                        pairs.append((fa, fb))

                if len(pairs) <= 1:
                    continue

                a_vals, b_vals = zip(*pairs)
                # Require variance on both series
                if np.std(a_vals) == 0.0 or np.std(b_vals) == 0.0:
                    continue
                try:
                    corr, _ = stats.pearsonr(a_vals, b_vals)
                except Exception:
                    continue
                if not math.isfinite(corr):
                    continue

                correlations.setdefault(m1, {})[m2] = float(corr)

        return correlations

    def get_aggregate_score(self, mars_results: Dict[str, Dict]) -> float:
        total = 0.0
        weight_sum = 0.0
        for _dim, res in mars_results.items():
            w = _safe_scalar(res.get("agreement_score"))
            s = _safe_scalar(res.get("average_score"))
            if w is None or s is None:
                continue
            total += s * w
            weight_sum += w
        return round(total / weight_sum, 3) if weight_sum > 0 else 0.0

    def get_high_disagreement_documents(
        self, corpus: "ScoreCorpus", dimension: str, threshold: float | None = None
    ) -> List[str]:
        if threshold is None:
            threshold = self._get_dimension_config(dimension)["variance_threshold"]
        matrix = corpus.get_dimension_matrix(dimension)
        if matrix.empty:
            return []
        disagreement = matrix.std(axis=1, ddof=0)
        return disagreement[disagreement > float(threshold)].index.tolist()

    def get_scorer_reliability(
        self, corpus: "ScoreCorpus", dimension: str
    ) -> Dict[str, float]:
        dim_cfg = self._get_dimension_config(dimension)
        trust_ref = dim_cfg["trust_reference"]
        matrix = corpus.get_dimension_matrix(dimension)
        if matrix.empty:
            return {}

        reliability: Dict[str, float] = {}
        if trust_ref in matrix.columns:
            ref = matrix[trust_ref]
            for sc in matrix.columns:
                if sc == trust_ref:
                    reliability[sc] = 1.0
                    continue
                pair = matrix[[sc, trust_ref]].dropna()
                if len(pair) <= 1:
                    reliability[sc] = 0.0
                    continue
                try:
                    corr, _ = stats.pearsonr(pair[sc], pair[trust_ref])
                    reliability[sc] = float(corr) if math.isfinite(corr) else 0.0
                except Exception:
                    reliability[sc] = 0.0
        else:
            # Fallback: lower std across docs => higher reliability
            s_std = matrix.std(axis=0, ddof=0)
            m = _safe_scalar(s_std.max()) or 0.0
            for sc, v in s_std.items():
                vv = _safe_scalar(v) or 0.0
                reliability[sc] = float(1.0 - (vv / m)) if m > 0 else 1.0
        return reliability

    def generate_recommendations(self, mars_results: Dict[str, Dict]) -> List[str]:
        recs: List[str] = []
        for dimension, res in mars_results.items():
            if res.get("high_disagreement"):
                pc = res.get("primary_conflict", ["—", "—"])
                delta = _safe_scalar(res.get("delta")) or 0.0
                recs.append(
                    f"⚠️ High disagreement in {dimension}: {pc[0]} vs {pc[1]} "
                    f"(Δ={delta:.3f}). Consider human review."
                )

            # Example: look for metric variability flags
            sm = res.get("scorer_metrics") or {}
            if len(sm) > 2:
                for scorer, md in sm.items():
                    unc = md.get("uncertainty")
                    if unc and _safe_scalar(unc.get("std", 0.0)) and float(unc["std"]) > 0.2:
                        recs.append(
                            f"⚠️ {scorer} shows high uncertainty variability in {dimension}. Consider calibration."
                        )

            corrs = res.get("metric_correlations") or {}
            for m1, sub in corrs.items():
                for m2, c in (sub or {}).items():
                    cc = _safe_scalar(c)
                    if cc is not None and abs(cc) > 0.7:
                        recs.append(
                            f"💡 In {dimension}, {m1} and {m2} are strongly correlated ({cc:.2f})."
                        )

        # Overall system recommendation
        agreements = [_safe_scalar(r.get("agreement_score")) for r in mars_results.values()]
        agreements = [a for a in agreements if a is not None]
        if agreements and (sum(agreements) / len(agreements)) < 0.7:
            recs.append(
                "⚠️ Overall scoring agreement is low (<0.7). Consider human review for high-disagreement items."
            )
        return recs
