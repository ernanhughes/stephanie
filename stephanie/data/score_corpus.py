# stephanie/scoring/score_corpus.py
import warnings
from typing import Any, Dict, List, Set, Tuple
from collections import defaultdict

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
        """Index results by dimension and scorer."""
        index = defaultdict(lambda: defaultdict(list))
        for scorable_id, bundle in self.bundles.items():
            for dimension, result in bundle.results.items():
                index[dimension][result.source].append((scorable_id, result))
        return index

    # ---------------- Properties ----------------

    @property
    def dimensions(self) -> List[str]:
        if self._dimensions is None:
            self._dimensions = sorted({
                dim for bundle in self.bundles.values() for dim in bundle.results.keys()
            })
        return self._dimensions

    @property
    def scorers(self) -> List[str]:
        if self._scorers is None:
            self._scorers = sorted({
                result.source for bundle in self.bundles.values() for result in bundle.results.values()
            })
        return self._scorers

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
