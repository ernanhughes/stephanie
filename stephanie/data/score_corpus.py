# stephanie/scoring/score_corpus.py
from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from stephanie.data.score_bundle import ScoreBundle


class ScoreCorpus:
    """
    Collection of ScoreBundles across multiple documents/scorables for tensor-based analysis.

    Implements a true 4D tensor structure:
        [scorables × dimensions × scorers × metrics]

    Enables:
    - Agreement/divergence analysis (MARS)
    - Metric slicing (energy, uncertainty, etc.)
    - Scorer bias/reliability checks
    - Conversion to DataFrame/tensor for ML integration
    """

    def __init__(self, bundles: Dict[str, ScoreBundle], meta: Dict[str, Any] = None):
        self.bundles = bundles
        self.meta = meta or {}
        self._dimensions = None
        self._scorers = None
        self._metrics = None
        self._dimension_matrix_cache = {}
        self._metric_matrix_cache = {}

        # Precompute dimension→scorer→[(scorable_id, result)]
        self._results_by_dimension_scorer = self._index_results()

    def _index_results(self) -> Dict[str, Dict[str, List[Tuple[str, Any]]]]:
        index = defaultdict(lambda: defaultdict(list))
        for scorable_id, bundle in self.bundles.items():
            for key, result in bundle.results.items():
                dim = getattr(result, "dimension", key)
                src = getattr(result, "source", "unknown")
                index[dim][src].append((scorable_id, result))
        return index

    @property
    def dimensions(self) -> List[str]:
        if self._dimensions is None:
            self._dimensions = sorted({
                getattr(result, "dimension", key)
                for bundle in self.bundles.values()
                for key, result in bundle.results.items()
            })
        return self._dimensions

    @property
    def scorers(self) -> List[str]:
        self._scorers = sorted({
            getattr(result, "source", "unknown")
            for bundle in self.bundles.values()
            for result in bundle.results.values()
        })
        return self._scorers

    # ---------------- Properties ----------------

    @property
    def metrics(self) -> Set[str]:
        if self._metrics is None:
            metrics = {"score"}
            for bundle in self.bundles.values():
                for result in bundle.results.values():
                    if result.attributes:
                        metrics.update(result.attributes.keys())
            self._metrics = metrics
        return self._metrics

    # ---------------- Matrix methods ----------------

    def get_dimension_matrix(self, dimension: str) -> pd.DataFrame:
        """
        Get scores as DataFrame [scorables × scorers].
        """
        if dimension in self._dimension_matrix_cache:
            return self._dimension_matrix_cache[dimension]

        if dimension not in self._results_by_dimension_scorer:
            return pd.DataFrame(index=list(self.bundles.keys()), columns=self.scorers, dtype=float)

        df = pd.DataFrame(index=list(self.bundles.keys()), columns=self.scorers, dtype=float)

        for scorer in self.scorers:
            for scorable_id, result in self._results_by_dimension_scorer[dimension][scorer]:
                try:
                    df.at[scorable_id, scorer] = float(result.score)
                except Exception:
                    df.at[scorable_id, scorer] = np.nan

        self._dimension_matrix_cache[dimension] = df
        return df

    def get_metric_matrix(self, dimension: str, metric: str) -> pd.DataFrame:
        """
        Get specific metric as DataFrame [scorables × scorers].
        """
        cache_key = (dimension, metric)
        if cache_key in self._metric_matrix_cache:
            return self._metric_matrix_cache[cache_key]

        if dimension not in self._results_by_dimension_scorer:
            return pd.DataFrame(index=list(self.bundles.keys()), columns=self.scorers, dtype=float)

        df = pd.DataFrame(index=list(self.bundles.keys()), columns=self.scorers, dtype=float)

        for scorer in self.scorers:
            for scorable_id, result in self._results_by_dimension_scorer[dimension][scorer]:
                if metric == "score":
                    value = result.score
                elif result.attributes and metric in result.attributes:
                    value = result.attributes[metric]
                else:
                    value = np.nan

                try:
                    df.at[scorable_id, scorer] = float(value)
                except Exception:
                    df.at[scorable_id, scorer] = np.nan

        self._metric_matrix_cache[cache_key] = df
        return df

    # ---------------- New aggregation APIs (MARS expects these) ----------------

    def _aggregate_rows(self, df: pd.DataFrame, agg: str = "mean") -> pd.Series:
        """
        Aggregate across scorers (columns) per scorable (row).
        NaN-safe for common reducers.
        """
        agg = (agg or "mean").lower()
        if df.empty:
            # Return a Series aligned to the expected index if possible
            return pd.Series([np.nan] * len(self.bundles), index=list(self.bundles.keys()), dtype=float)

        if agg == "mean":
            return df.mean(axis=1, skipna=True)
        if agg == "median":
            return df.median(axis=1, skipna=True)
        if agg == "max":
            return df.max(axis=1, skipna=True)
        if agg == "min":
            return df.min(axis=1, skipna=True)
        if agg == "sum":
            return df.sum(axis=1, skipna=True)
        if agg == "first_non_nan":
            return df.apply(lambda r: next((x for x in r.values if not (isinstance(x, float) and np.isnan(x))), np.nan), axis=1)

        warnings.warn(f"Unknown agg '{agg}', defaulting to mean.")
        return df.mean(axis=1, skipna=True)

    def get_values_for_metric(self, dimension: str, metric: str, agg: str = "mean") -> List[float]:
        """
        Returns a list of values for `metric` across all scorables, aggregating across scorers.
        Order matches insertion order of `bundles.keys()`.
        Missing/invalid values are NaN and will be handled by downstream code.
        """
        df = self.get_metric_matrix(dimension, metric)
        # Ensure row order matches bundles insertion order
        desired_index = list(self.bundles.keys())
        df = df.reindex(index=desired_index)
        series = self._aggregate_rows(df, agg=agg)
        # Ensure order and convert to Python floats
        series = series.reindex(desired_index)
        return [float(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else np.nan for x in series.tolist()]

    def get_all_metric_values(self, dimension: str, metrics: List[str], agg: str = "mean") -> Dict[str, List[float]]:
        """
        Returns { metric_name: [values_per_scorable_in_order] } for the given dimension,
        aggregating across scorers with the provided reducer (default: mean).
        """
        metrics = metrics or ["score"]
        return {m: self.get_values_for_metric(dimension, m, agg=agg) for m in metrics}

    # ---------------- Utility methods ----------------

    def get_metric_values(self, dimension: str, scorer: str, metrics: List[str]) -> Dict[str, List[Any]]:
        """
        Get values for specific metrics across scorables for one scorer.
        """
        results = {metric: [] for metric in metrics}
        scorable_ids = []

        for scorable_id, result in self._results_by_dimension_scorer[dimension][scorer]:
            scorable_ids.append(scorable_id)
            for metric in metrics:
                if metric == "score":
                    val = result.score
                elif result.attributes and metric in result.attributes:
                    val = result.attributes[metric]
                else:
                    val = np.nan
                try:
                    results[metric].append(float(val))
                except Exception:
                    results[metric].append(np.nan)

        results["scorable_id"] = scorable_ids
        return results

    def to_tensor(self, dimensions: List[str] = None, scorers: List[str] = None, metrics: List[str] = None) -> np.ndarray:
        """
        Convert to 4D tensor: [scorables × dimensions × scorers × metrics]
        """
        dimensions = dimensions or self.dimensions
        scorers = scorers or self.scorers
        metrics = metrics or list(self.metrics)

        tensor = np.full((len(self.bundles), len(dimensions), len(scorers), len(metrics)), np.nan)
        scorable_idx_map = {scorable_id: idx for idx, scorable_id in enumerate(self.bundles.keys())}

        for dim_idx, dimension in enumerate(dimensions):
            for scorer_idx, scorer in enumerate(scorers):
                for scorable_id, result in self._results_by_dimension_scorer.get(dimension, {}).get(scorer, []):
                    if scorable_id not in scorable_idx_map:
                        continue
                    s_idx = scorable_idx_map[scorable_id]
                    for metric_idx, metric in enumerate(metrics):
                        try:
                            if metric == "score":
                                tensor[s_idx, dim_idx, scorer_idx, metric_idx] = float(result.score)
                            elif result.attributes and metric in result.attributes:
                                tensor[s_idx, dim_idx, scorer_idx, metric_idx] = float(result.attributes[metric])
                        except Exception:
                            tensor[s_idx, dim_idx, scorer_idx, metric_idx] = np.nan
        return tensor

    def get_high_disagreement_scorables(self, dimension: str, threshold: float) -> List[str]:
        """
        Convenience wrapper used by plan_trace_scorer.
        Returns scorable IDs whose *across-scorers* std-dev for `dimension` exceeds `threshold`.
        """
        matrix = self.get_dimension_matrix(dimension)  # [scorable x scorer]
        if matrix.empty:
            return []
        disagreement = matrix.std(axis=1)              # per scorable row
        return disagreement[disagreement > threshold].index.tolist()

    def get_summary(self) -> dict:
        """
        Lightweight summary (MARSCalculator’s writer calls this).
        """
        return {
            "scorable_count": len(self.bundles),
            "dimensions": list(self.dimensions),
            "scorers": list(self.scorers),
            "metrics": list(self.metrics),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scorable_ids": list(self.bundles.keys()),
            "dimensions": self.dimensions,
            "scorers": self.scorers,
            "metrics": list(self.metrics),
            "meta": self.meta,
        }

    def __len__(self):
        return len(self.bundles)

    def __repr__(self):
        return (f"<ScoreCorpus(scorables={len(self.bundles)}, "
                f"dimensions={len(self.dimensions)}, "
                f"scorers={len(self.scorers)}, "
                f"metrics={len(self.metrics)})>")
