# stephanie/scoring/score_corpus.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Set, Tuple
import warnings

from stephanie.data.score_bundle import ScoreBundle


class ScoreCorpus:
    """
    Collection of ScoreBundles across multiple documents/scorables for tensor-based analysis.
    
    This class implements the true 4D tensor structure [scorables × dimensions × scorers × metrics]
    that enables powerful slicing and analysis capabilities.
    
    Key features:
    - Convert to 4D tensor for ML integration
    - Slice by metric type (energy, uncertainty, etc.)
    - Analyze scoring agreement patterns
    - Identify systematic scorer biases
    - Support for MARS calculator integration
    """
    
    def __init__(self, bundles: Dict[str, ScoreBundle], meta: Dict[str, Any] = None):
        """
        Initialize a ScoreCorpus from a collection of ScoreBundles.
        
        Args:
            bundles: Dictionary mapping scorable IDs to ScoreBundles
            meta: Optional metadata about the corpus
        """
        self.bundles = bundles
        self.meta = meta or {}
        self._dimensions = None
        self._scorers = None
        self._metrics = None
        self._dimension_matrix_cache = {}
        self._metric_matrix_cache = {}
    
    @property
    def dimensions(self) -> List[str]:
        """Get all dimensions present across bundles"""
        if self._dimensions is None:
            self._dimensions = self._discover_dimensions()
        return self._dimensions
    
    @property
    def scorers(self) -> List[str]:
        """Get all scorers present across bundles"""
        if self._scorers is None:
            self._scorers = self._discover_scorers()
        return self._scorers
    
    @property
    def metrics(self) -> Set[str]:
        """Get all metrics present across bundles (including 'score')"""
        if self._metrics is None:
            self._metrics = self._discover_metrics()
        return self._metrics
    
    def _discover_dimensions(self) -> List[str]:
        """Discover all dimensions present in the corpus"""
        dimensions = set()
        for bundle in self.bundles.values():
            dimensions.update(bundle.results.keys())
        return sorted(list(dimensions))
    
    def _discover_scorers(self) -> List[str]:
        """Discover all scorers present in the corpus"""
        scorers = set()
        for bundle in self.bundles.values():
            for result in bundle.results.values():
                scorers.add(result.source)
        return sorted(list(scorers))
    
    def _discover_metrics(self) -> Set[str]:
        """Discover all metrics present in the corpus"""
        metrics = {"score"}  # Always include the core score
        for bundle in self.bundles.values():
            for result in bundle.results.values():
                if result.attributes:
                    metrics.update(result.attributes.keys())
        return metrics
    
    def get_dimension_matrix(self, dimension: str) -> pd.DataFrame:
        """
        Get scores as a DataFrame: [scorables × scorers]
        
        Args:
            dimension: The dimension to extract
            
        Returns:
            DataFrame where rows are scorables and columns are scorers
        """
        # Check cache first
        if dimension in self._dimension_matrix_cache:
            return self._dimension_matrix_cache[dimension]
        
        # Build matrix
        data = {}
        for scorable_id, bundle in self.bundles.items():
            if dimension in bundle.results:
                result = bundle.results[dimension]
                data[scorable_id] = {result.source: result.score}
        
        # Create DataFrame
        df = pd.DataFrame.from_dict(data, orient='index')
        
        # Ensure all scorers are present as columns
        for scorer in self.scorers:
            if scorer not in df.columns:
                df[scorer] = np.nan
        
        # Sort columns by scorers list
        df = df[self.scorers]
        
        # Cache result
        self._dimension_matrix_cache[dimension] = df
        
        return df
    
    def get_metric_matrix(self, dimension: str, metric: str) -> pd.DataFrame:
        """
        Get a specific metric as a DataFrame: [scorables × scorers]
        
        Args:
            dimension: The dimension to extract
            metric: The metric to extract (e.g., "uncertainty", "q_value")
            
        Returns:
            DataFrame where rows are scorables and columns are scorers
        """
        # Check cache first
        cache_key = (dimension, metric)
        if cache_key in self._metric_matrix_cache:
            return self._metric_matrix_cache[cache_key]
        
        # Build matrix
        data = {}
        for scorable_id, bundle in self.bundles.items():
            if dimension in bundle.results:
                result = bundle.results[dimension]
                value = result.attributes.get(metric, np.nan) if result.attributes else np.nan
                data[scorable_id] = {result.source: value}
        
        # Create DataFrame
        df = pd.DataFrame.from_dict(data, orient='index')
        
        # Ensure all scorers are present as columns
        for scorer in self.scorers:
            if scorer not in df.columns:
                df[scorer] = np.nan
        
        # Sort columns by scorers list
        df = df[self.scorers]
        
        # Cache result
        self._metric_matrix_cache[cache_key] = df
        
        return df
    
    def get_metric_values(self, dimension: str, scorer: str, metrics: List[str]) -> Dict[str, List[Any]]:
        """
        Get values for specific metrics across all scorables for a dimension and scorer.
        
        Args:
            dimension: The dimension to extract
            scorer: The scorer to extract
            metrics: List of metrics to extract
            
        Returns:
            Dictionary mapping metric names to lists of values
        """
        results = {metric: [] for metric in metrics}
        
        for bundle in self.bundles.values():
            if dimension in bundle.results:
                result = bundle.results[dimension]
                if result.source == scorer:
                    for metric in metrics:
                        if result.attributes and metric in result.attributes:
                            results[metric].append(result.attributes[metric])
                        else:
                            results[metric].append(None)
        
        return results
    
    def get_all_metric_values(self, dimension: str, metrics: List[str]) -> Dict[str, List[Any]]:
        """
        Get values for specific metrics across all scorables and scorers for a dimension.
        
        Args:
            dimension: The dimension to extract
            metrics: List of metrics to extract
            
        Returns:
            Dictionary mapping metric names to lists of values
        """
        results = {metric: [] for metric in metrics}
        
        for bundle in self.bundles.values():
            if dimension in bundle.results:
                result = bundle.results[dimension]
                for metric in metrics:
                    if result.attributes and metric in result.attributes:
                        results[metric].append(result.attributes[metric])
                    else:
                        results[metric].append(None)
        
        return results
    
    def to_tensor(self, dimensions: List[str] = None, 
                 scorers: List[str] = None, 
                 metrics: List[str] = None) -> np.ndarray:
        """
        Convert to 4D tensor: [scorables × dimensions × scorers × metrics]
        
        Args:
            dimensions: Optional list of dimensions to include (defaults to all)
            scorers: Optional list of scorers to include (defaults to all)
            metrics: Optional list of metrics to include (defaults to all)
            
        Returns:
            4D numpy array of shape (n_scorables, n_dimensions, n_scorers, n_metrics)
        """
        # Default to all dimensions/scorers/metrics if not specified
        dimensions = dimensions or self.dimensions
        scorers = scorers or self.scorers
        metrics = metrics or list(self.metrics)
        
        # Create tensor with zeros
        tensor = np.zeros((len(self.bundles), len(dimensions), len(scorers), len(metrics)))
        
        # Fill tensor with values
        for scorable_idx, (scorable_id, bundle) in enumerate(self.bundles.items()):
            for dim_idx, dimension in enumerate(dimensions):
                if dimension in bundle.results:
                    result = bundle.results[dimension]
                    scorer_idx = scorers.index(result.source)
                    
                    # Fill in metric values
                    for metric_idx, metric in enumerate(metrics):
                        if metric == "score":
                            tensor[scorable_idx, dim_idx, scorer_idx, metric_idx] = result.score
                        elif result.attributes and metric in result.attributes:
                            try:
                                tensor[scorable_idx, dim_idx, scorer_idx, metric_idx] = float(result.attributes[metric])
                            except (TypeError, ValueError):
                                tensor[scorable_idx, dim_idx, scorer_idx, metric_idx] = 0.0
                        # Otherwise leave as 0.0
        
        return tensor
    
    def to_dataframe(self, dimensions: List[str] = None, 
                    scorers: List[str] = None, 
                    metrics: List[str] = None) -> pd.DataFrame:
        """
        Convert to multi-index DataFrame for analysis.
        
        The DataFrame will have:
        - Index: scorable IDs
        - Columns: MultiIndex of (dimension, scorer, metric)
        
        Args:
            dimensions: Optional list of dimensions to include (defaults to all)
            scorers: Optional list of scorers to include (defaults to all)
            metrics: Optional list of metrics to include (defaults to all)
            
        Returns:
            Multi-index DataFrame
        """
        # Default to all dimensions/scorers/metrics if not specified
        dimensions = dimensions or self.dimensions
        scorers = scorers or self.scorers
        metrics = metrics or list(self.metrics)
        
        # Create column index
        column_tuples = [(dim, scorer, metric) 
                        for dim in dimensions 
                        for scorer in scorers 
                        for metric in metrics]
        columns = pd.MultiIndex.from_tuples(column_tuples, 
                                         names=['dimension', 'scorer', 'metric'])
        
        # Create DataFrame
        df = pd.DataFrame(index=list(self.bundles.keys()), columns=columns)
        
        # Fill DataFrame
        for scorable_id, bundle in self.bundles.items():
            for dim in dimensions:
                if dim in bundle.results:
                    result = bundle.results[dim]
                    for metric in metrics:
                        if metric == "score":
                            value = result.score
                        elif result.attributes and metric in result.attributes:
                            value = result.attributes[metric]
                        else:
                            value = None
                        
                        df.loc[scorable_id, (dim, result.source, metric)] = value
        
        return df
    
    def analyze_scorer_reliability(self, dimension: str, 
                                 trust_reference: str = "llm") -> Dict[str, float]:
        """
        Analyze which scorers are most reliable for a dimension.
        
        Args:
            dimension: The dimension to analyze
            trust_reference: The scorer to use as gold standard
            
        Returns:
            Dictionary mapping scorers to reliability scores (higher = more reliable)
        """
        if trust_reference not in self.scorers:
            warnings.warn(f"Trust reference '{trust_reference}' not found. Using median scorer instead.")
            return self._analyze_scorer_consistency(dimension)
        
        # Get the document × scorer matrix
        matrix = self.get_dimension_matrix(dimension)
        
        # Calculate correlation with trust reference
        reliability = {}
        trust_scores = matrix[trust_reference]
        
        for scorer in self.scorers:
            if scorer == trust_reference:
                reliability[scorer] = 1.0  # Perfect correlation with itself
                continue
            
            # Calculate correlation
            valid_pairs = matrix[[scorer, trust_reference]].dropna()
            if len(valid_pairs) > 1:
                try:
                    corr = valid_pairs[scorer].corr(valid_pairs[trust_reference])
                    reliability[scorer] = float(corr) if not pd.isna(corr) else 0.0
                except:
                    reliability[scorer] = 0.0
            else:
                reliability[scorer] = 0.0
        
        return reliability
    
    def _analyze_scorer_consistency(self, dimension: str) -> Dict[str, float]:
        """Analyze scorer consistency when no trust reference is available"""
        matrix = self.get_dimension_matrix(dimension)
        scorer_std = matrix.std()
        max_std = scorer_std.max()
        
        # Higher reliability for lower standard deviation
        return {scorer: 1.0 - (std / max_std) if max_std > 0 else 1.0 
                for scorer, std in scorer_std.items()}
    
    def get_high_disagreement_scorables(self, dimension: str, 
                                     threshold: float = 0.15) -> List[str]:
        """
        Get scorables with high disagreement across scorers for a dimension.
        
        Args:
            dimension: The dimension to analyze
            threshold: Threshold for disagreement (standard deviation)
            
        Returns:
            List of scorable IDs with high disagreement
        """
        # Get the document × scorer matrix
        matrix = self.get_dimension_matrix(dimension)
        
        # Calculate disagreement per document (standard deviation across scorers)
        disagreement = matrix.std(axis=1)
        
        # Return scorables with disagreement above threshold
        return disagreement[disagreement > threshold].index.tolist()
    
    def get_outlier_scorables(self, dimension: str, scorer: str, 
                            threshold: float = 2.0) -> List[str]:
        """
        Get scorables where a specific scorer significantly differs from consensus.
        
        Args:
            dimension: The dimension to analyze
            scorer: The scorer to check
            threshold: Threshold in standard deviations
            
        Returns:
            List of scorable IDs where the scorer is an outlier
        """
        # Get the document × scorer matrix
        matrix = self.get_dimension_matrix(dimension)
        if scorer not in matrix.columns:
            return []
        
        # Calculate consensus (mean excluding the scorer)
        consensus = matrix.drop(columns=[scorer]).mean(axis=1)
        
        # Calculate difference from consensus
        diff = (matrix[scorer] - consensus).abs()
        std_dev = diff.std()
        
        # Return scorables where difference is above threshold
        if std_dev > 0:
            return diff[diff > threshold * std_dev].index.tolist()
        return []
    
    def get_metric_correlations(self, dimension: str, 
                              metrics: List[str] = None) -> Dict[Tuple[str, str], float]:
        """
        Get correlations between different metrics for a dimension.
        
        Args:
            dimension: The dimension to analyze
            metrics: Optional list of metrics to analyze (defaults to all)
            
        Returns:
            Dictionary mapping (metric1, metric2) to correlation coefficient
        """
        metrics = metrics or list(self.metrics - {"score"})
        if len(metrics) < 2:
            return {}
        
        # Get all metric matrices
        metric_matrices = {
            metric: self.get_metric_matrix(dimension, metric)
            for metric in metrics
        }
        
        # Calculate correlations
        correlations = {}
        for i in range(len(metrics)):
            for j in range(i+1, len(metrics)):
                metric1, metric2 = metrics[i], metrics[j]
                
                # Stack values
                values1 = []
                values2 = []
                for scorable_id in self.bundles.keys():
                    val1 = metric_matrices[metric1].loc.get(scorable_id, np.nan)
                    val2 = metric_matrices[metric2].loc.get(scorable_id, np.nan)
                    
                    # Skip if either value is NaN
                    if not pd.isna(val1) and not pd.isna(val2):
                        values1.append(val1)
                        values2.append(val2)
                
                # Calculate correlation
                if len(values1) > 1:
                    try:
                        corr = pd.Series(values1).corr(pd.Series(values2))
                        if not pd.isna(corr):
                            correlations[(metric1, metric2)] = float(corr)
                    except:
                        pass
        
        return correlations
    
    def find_metric_outliers(self, dimension: str, metric: str, 
                           threshold: float = 2.0) -> List[Tuple[str, float]]:
        """
        Find scorables with outlier values for a specific metric.
        
        Args:
            dimension: The dimension to analyze
            metric: The metric to check
            threshold: Threshold in standard deviations
            
        Returns:
            List of (scorable_id, z_score) tuples
        """
        # Get the metric matrix
        matrix = self.get_metric_matrix(dimension, metric)
        
        # Stack all values
        all_values = []
        for scorer in self.scorers:
            values = matrix[scorer].dropna().values
            all_values.extend(values)
        
        if not all_values:
            return []
        
        # Calculate mean and std
        mean_val = np.mean(all_values)
        std_val = np.std(all_values)
        
        if std_val == 0:
            return []
        
        # Find outliers
        outliers = []
        for scorable_id in self.bundles.keys():
            for scorer in self.scorers:
                value = matrix.loc.get((scorable_id, scorer), np.nan)
                if not pd.isna(value):
                    z_score = (value - mean_val) / std_val
                    if abs(z_score) > threshold:
                        outliers.append((scorable_id, z_score))
        
        # Sort by absolute z-score
        outliers.sort(key=lambda x: abs(x[1]), reverse=True)
        return outliers
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "scorable_ids": list(self.bundles.keys()),
            "dimensions": self.dimensions,
            "scorers": self.scorers,
            "metrics": list(self.metrics),
            "meta": self.meta
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], 
                 bundles: Dict[str, ScoreBundle] = None) -> "ScoreCorpus":
        """Reconstruct from dictionary (with optional bundles)"""
        # If bundles are provided, filter to match scorable IDs
        if bundles:
            scorable_ids = data.get("scorable_ids", [])
            filtered_bundles = {k: v for k, v in bundles.items() if k in scorable_ids}
            return cls(bundles=filtered_bundles, meta=data.get("meta", {}))
        
        # Without bundles, just return empty corpus with metadata
        return cls(bundles={}, meta=data.get("meta", {}))
    
    def __len__(self) -> int:
        """Return number of scorables in the corpus"""
        return len(self.bundles)
    
    def __getitem__(self, scorable_id: str) -> ScoreBundle:
        """Get a specific ScoreBundle by scorable ID"""
        return self.bundles[scorable_id]
    
    def __iter__(self):
        """Iterate over scorables"""
        return iter(self.bundles.items())
    
    def __repr__(self):
        return (f"<ScoreCorpus(scorables={len(self.bundles)}, "
                f"dimensions={len(self.dimensions)}, "
                f"scorers={len(self.scorers)}, "
                f"metrics={len(self.metrics)})>")
    

    def get_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the ScoreCorpus.
        
        Returns:
            Dictionary containing key statistics and insights about the corpus.
        """
        summary = {
            "metadata": {
                "scorables_count": len(self.bundles),
                "dimensions": self.dimensions,
                "scorers": self.scorers,
                "metrics": list(self.metrics),
                "created_at": self.meta.get("created_at", ""),
                "source": self.meta.get("source", "unknown")
            },
            "statistics": {
                "average_scores": {},
                "score_ranges": {},
                "uncertainty_stats": {},
                "agreement_stats": {}
            },
            "insights": {
                "high_uncertainty_count": 0,
                "strong_agreement_count": 0,
                "scorer_reliability": {}
            }
        }
        
        # Calculate basic statistics
        for dimension in self.dimensions:
            scores = []
            for bundle in self.bundles.values():
                if dimension in bundle.results:
                    try:
                        scores.append(bundle.results[dimension].score)
                    except (AttributeError, TypeError):
                        continue
            
            if scores:
                summary["statistics"]["average_scores"][dimension] = float(np.mean(scores))
                summary["statistics"]["score_ranges"][dimension] = {
                    "min": float(min(scores)),
                    "max": float(max(scores)),
                    "std": float(np.std(scores))
                }
        
        # Calculate uncertainty statistics
        for dimension in self.dimensions:
            uncertainties = []
            for bundle in self.bundles.values():
                if dimension in bundle.results:
                    result = bundle.results[dimension]
                    try:
                        if hasattr(result, 'uncertainty') and result.uncertainty is not None:
                            uncertainties.append(result.uncertainty)
                    except (AttributeError, TypeError):
                        continue
            
            if uncertainties:
                high_uncertainty_count = sum(1 for u in uncertainties if u > 0.3)
                summary["statistics"]["uncertainty_stats"][dimension] = {
                    "average": float(np.mean(uncertainties)),
                    "high_uncertainty_count": high_uncertainty_count,
                    "total_count": len(uncertainties)
                }
                summary["insights"]["high_uncertainty_count"] += high_uncertainty_count
        
        # Analyze agreement patterns
        for dimension in self.dimensions:
            # Get scores from all scorers for this dimension
            scores_by_scorer = {}
            for bundle in self.bundles.values():
                if dimension in bundle.results:
                    result = bundle.results[dimension]
                    try:
                        scorer = result.source
                        if scorer not in scores_by_scorer:
                            scores_by_scorer[scorer] = []
                        scores_by_scorer[scorer].append(result.score)
                    except (AttributeError, TypeError):
                        continue
            
            # Calculate agreement across scorers
            all_scores = []
            for scores in scores_by_scorer.values():
                all_scores.extend(scores)
            
            if all_scores:
                std_dev = float(np.std(all_scores))
                summary["statistics"]["agreement_stats"][dimension] = {
                    "average_score": float(np.mean(all_scores)),
                    "std_deviation": std_dev,
                    "agreement_score": 1.0 - min(std_dev, 1.0)
                }
            
            # Count strong agreement items
            strong_agreement_count = 0
            for bundle in self.bundles.values():
                dimension_scores = []
                for dim, result in bundle.results.items():
                    if dim == dimension:
                        try:
                            dimension_scores.append(result.score)
                        except (AttributeError, TypeError):
                            continue
                
                if len(dimension_scores) > 1 and np.std(dimension_scores) < 0.1:
                    strong_agreement_count += 1
            
            summary["insights"]["strong_agreement_count"] += strong_agreement_count
        
        # Calculate scorer reliability
        for scorer in self.scorers:
            all_scores = []
            for bundle in self.bundles.values():
                for result in bundle.results.values():
                    try:
                        if result.source == scorer:
                            all_scores.append(result.score)
                    except (AttributeError, TypeError):
                        continue
            
            if all_scores:
                summary["insights"]["scorer_reliability"][scorer] = {
                    "average_score": float(np.mean(all_scores)),
                    "score_count": len(all_scores),
                    "std_deviation": float(np.std(all_scores))
                }
        
        return summary