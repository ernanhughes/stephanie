import numpy as np
import os
from statistics import mean
from typing import Dict, Any, List
from scipy import stats
import json
from datetime import datetime
import traceback

from stephanie.scoring.calculations.base_calculator import BaseScoreCalculator
from stephanie.data.score_corpus import ScoreCorpus

from stephanie.utils.serialization import default_serializer

class MARSCalculator(BaseScoreCalculator):
    """
    Model Agreement and Reasoning Signal (MARS) Calculator

    Analyzes agreement patterns across multiple scoring models/adapters to:
    - Quantify scoring consensus or divergence across documents
    - Identify which scorers disagree systematically
    - Determine which model aligns best with trust reference
    - Measure uncertainty in the overall assessment
    - Provide diagnostic insights for scoring system improvement

    Unlike traditional aggregators, MARS operates at the ScoreCorpus level (multiple documents)
    to detect reliability patterns rather than just computing an average score.
    """

    def __init__(self, config: Dict = None, logger=None):
        """
        Initialize MARS calculator with configuration

        Args:
            config: Optional configuration with:
                - trust_reference: Which scorer to use as gold standard (default: "llm")
                - variance_threshold: Threshold for flagging high disagreement (default: 0.15)
                - dimensions: Dimension-specific configurations
                - metrics: Which metrics to analyze (default: ["score"] for core score)
        """
        self.config = config or {}
        self.logger = logger
        self.trust_reference = self.config.get("trust_reference", "llm")
        self.variance_threshold = self.config.get("variance_threshold", 0.15)
        self.metrics = self.config.get(
            "metrics", ["score"]
        )  # Core score by default
        self.dimension_configs = self.config.get("dimensions", {})

        # Configure logging options
        self.log_enabled = self.config.get("log_enabled", True)
        self.log_path = self.config.get("log_path", "mars_reports")
        self.include_full_data = self.config.get("include_full_data", True)

        if self.log_enabled and self.logger:
            self.logger.log("MARSLoggerConfigured", {
                "log_path": os.path.abspath(self.log_path),
                "include_full_data": self.include_full_data,
                "enabled": self.log_enabled
            })


    def _write_json_report(self, mars_results: Dict[str, Any], corpus: "ScoreCorpus"):
        """Write MARS results to a JSON file with proper serialization"""
        if not self.log_enabled:
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.log_path, exist_ok=True)
            
            # Generate filename with timestamp and dimension
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mars_report_{timestamp}.json"
            filepath = os.path.join(self.log_path, filename)
            
            # Prepare report data
            report_data = {
                "metadata": {
                    "timestamp": timestamp,
                    "document_count": len(corpus.bundles),
                    "scorers": list(corpus.scorers),
                    "metrics": ["agreement", "uncertainty", "std_deviation", "outliers"]
                },
                "results": mars_results
            }
            
            # Include full data if configured
            if self.include_full_data:
                report_data["full_data"] = {
                    "corpus_summary": corpus.get_summary(),
                    "metric_matrices": {
                        metric: mars_results.get(metric, {}).get("matrix", {}).tolist() 
                        for metric in ["agreement", "uncertainty", "std_deviation"]
                        if metric in mars_results
                    }
                }
            
            # Write the report with proper serialization
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, default=default_serializer)
            
            if self.logger:
                self.logger.log("MARSReportSaved", {
                    "filepath": filepath,
                    "document_count": len(corpus.bundles),
                    "scorers_count": len(corpus.scorers)
                })
                
        except Exception as e:
            if self.logger:
                self.logger.log("MARSReportError", {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })


    def calculate(self, corpus: "ScoreCorpus") -> Dict[str, Any]:
        """
        Calculate MARS metrics across all scoring models in the corpus

        Args:
            corpus: ScoreCorpus containing results from multiple scorers across multiple documents

        Returns:
            Dictionary containing comprehensive MARS analysis metrics
        """
        # Calculate MARS metrics for each dimension
        mars_results = {}
        for dimension in corpus.dimensions:
            mars_results[dimension] = self._calculate_dimension_mars(
                corpus, dimension
            )

        if self.log_enabled:
            self._write_json_report(mars_results, corpus)
        return mars_results

    def _get_dimension_config(self, dimension: str) -> Dict:
        """Get dimension-specific configuration with fallbacks"""
        return self.dimension_configs.get(
            dimension,
            {
                "trust_reference": self.trust_reference,
                "variance_threshold": self.variance_threshold,
                "metrics": self.metrics,
            },
        )

    def _calculate_dimension_mars(
        self, corpus: "ScoreCorpus", dimension: str
    ) -> Dict[str, Any]:
        """
        Calculate MARS metrics for a specific dimension

        Args:
            corpus: ScoreCorpus containing evaluation results
            dimension: The dimension being analyzed

        Returns:
            Dictionary with MARS metrics for this dimension
        """
        # Get dimension-specific configuration
        dim_config = self._get_dimension_config(dimension)
        trust_ref = dim_config["trust_reference"]
        metrics = dim_config["metrics"]

        # Get the document × scorer matrix for this dimension
        matrix = corpus.get_dimension_matrix(dimension)

        # If no data for this dimension, return empty results
        if matrix.empty:
            return {
                "dimension": dimension,
                "agreement_score": 0.0,
                "std_dev": 0.0,
                "preferred_model": "none",
                "primary_conflict": ("none", "none"),
                "delta": 0.0,
                "high_disagreement": False,
                "explanation": "No data available for this dimension",
                "scorer_metrics": {},
                "metric_correlations": {},
            }

        # Calculate basic statistics
        avg_score = matrix.mean().mean()  # Overall average score
        std_dev = (
            matrix.std().mean()
        )  # Average standard deviation across documents

        # Calculate agreement score (1.0 = perfect agreement)
        agreement_score = 1.0 - min(std_dev, 1.0)

        # Identify primary conflict (largest average score difference)
        scorer_means = matrix.mean()
        max_valuer = scorer_means.idxmax()
        min_valuer = scorer_means.idxmin()
        delta = scorer_means[max_valuer] - scorer_means[min_valuer]
        primary_conflict = (max_valuer, min_valuer)

        # Determine which model aligns best with trust reference
        preferred_model = "unknown"
        if trust_ref in matrix.columns:
            trust_scores = matrix[trust_ref]
            closest = None
            min_diff = float("inf")

            for scorer in matrix.columns:
                if scorer == trust_ref:
                    continue

                # Calculate average absolute difference
                diff = (matrix[scorer] - trust_scores).abs().mean()
                if diff < min_diff:
                    min_diff = diff
                    closest = scorer

            preferred_model = closest if closest else "unknown"
        else:
            # If trust reference isn't available, use median scorer
            sorted_scorers = scorer_means.sort_values()
            median_idx = len(sorted_scorers) // 2
            preferred_model = sorted_scorers.index[median_idx]

        # Identify high-disagreement areas
        high_disagreement = std_dev > dim_config["variance_threshold"]

        # Analyze scorer metrics (q_value, uncertainty, etc.)
        scorer_metrics = self._analyze_scorer_metrics(
            corpus, dimension, metrics
        )

        # Calculate metric correlations
        metric_correlations = self._calculate_metric_correlations(
            corpus, dimension, metrics
        )

        # Generate explanation
        explanation_parts = [
            f"MARS agreement: {agreement_score:.3f} (std: {std_dev:.3f})"
        ]

        if high_disagreement:
            explanation_parts.append(
                f"⚠️ High disagreement detected (threshold: {dim_config['variance_threshold']})"
            )

        if preferred_model != "unknown":
            explanation_parts.append(
                f"Most aligned with {trust_ref}: {preferred_model}"
            )

        explanation_parts.append(
            f"Primary conflict: {primary_conflict[0]} vs {primary_conflict[1]} (Δ={delta:.3f})"
        )

        # Check for systematic bias
        above_mean = [
            scorer
            for scorer, mean_score in scorer_means.items()
            if mean_score > avg_score
        ]
        below_mean = [
            scorer
            for scorer, mean_score in scorer_means.items()
            if mean_score < avg_score
        ]

        if len(above_mean) == 1 or len(below_mean) == 1:
            outlier = above_mean[0] if len(above_mean) == 1 else below_mean[0]
            explanation_parts.append(f"⚠️ {outlier} appears to be an outlier")

        explanation = " | ".join(explanation_parts)

        return {
            "dimension": dimension,
            "agreement_score": round(agreement_score, 3),
            "std_dev": round(std_dev, 3),
            "preferred_model": preferred_model,
            "primary_conflict": primary_conflict,
            "delta": round(delta, 3),
            "high_disagreement": high_disagreement,
            "explanation": explanation,
            "scorer_metrics": scorer_metrics,
            "metric_correlations": metric_correlations,
            "source": "mars",
            "average_score": round(avg_score, 3),
        }

    def _analyze_scorer_metrics(
        self, corpus: "ScoreCorpus", dimension: str, metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze extended metrics for each scorer in this dimension
        """
        scorer_metrics = {}

        for scorer in corpus.scorers:
            # Get all attribute values for this scorer and dimension
            metric_values = corpus.get_metric_values(
                dimension, scorer, metrics
            )

            # Calculate statistics for each metric
            metrics_stats = {}
            for metric, values in metric_values.items():
                if not values:
                    continue

                # Filter out None/NaN values
                valid_values = [v for v in values if v is not None]
                if not valid_values:
                    continue

                metrics_stats[metric] = {
                    "mean": float(np.mean(valid_values)),
                    "std": float(np.std(valid_values)),
                    "min": float(min(valid_values)),
                    "max": float(max(valid_values)),
                    "count": len(valid_values),
                }

            if metrics_stats:
                scorer_metrics[scorer] = metrics_stats

        return scorer_metrics

    def _calculate_metric_correlations(
        self, corpus: "ScoreCorpus", dimension: str, metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlations between different metrics for this dimension
        """
        if len(metrics) < 2:
            return {}

        # Get all metric values for this dimension
        metric_values = corpus.get_all_metric_values(dimension, metrics)

        # Calculate correlations
        correlations = {}
        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                metric1, metric2 = metrics[i], metrics[j]

                # Get valid pairs of values
                pairs = [
                    (v1, v2)
                    for v1, v2 in zip(
                        metric_values[metric1], metric_values[metric2]
                    )
                    if v1 is not None and v2 is not None
                ]

                if len(pairs) > 1:
                    values1, values2 = zip(*pairs)
                    try:
                        corr, _ = stats.pearsonr(values1, values2)
                        if metric1 not in correlations:
                            correlations[metric1] = {}
                        correlations[metric1][metric2] = float(corr)
                    except:
                        pass

        return correlations

    def get_aggregate_score(self, mars_results: Dict[str, Dict]) -> float:
        """
        Get a single aggregate score from MARS analysis

        This provides a weighted average of dimension scores based on agreement reliability

        Args:
            mars_results: Results from calculate() method

        Returns:
            Weighted aggregate score where dimensions with higher agreement contribute more
        """
        total = 0
        weight_sum = 0

        for dimension, results in mars_results.items():
            # Weight by agreement score (higher agreement = more weight)
            weight = results["agreement_score"]
            total += results["average_score"] * weight
            weight_sum += weight

        return round(total / weight_sum, 3) if weight_sum > 0 else 0.0

    def get_high_disagreement_documents(
        self, corpus: "ScoreCorpus", dimension: str, threshold: float = None
    ) -> List[str]:
        """
        Identify documents with high scoring disagreement for this dimension

        Args:
            corpus: ScoreCorpus to analyze
            dimension: Dimension to check
            threshold: Custom disagreement threshold (uses config default if None)

        Returns:
            List of document IDs with high disagreement
        """
        if threshold is None:
            dim_config = self._get_dimension_config(dimension)
            threshold = dim_config["variance_threshold"]

        # Get the document × scorer matrix
        matrix = corpus.get_dimension_matrix(dimension)
        if matrix.empty:
            return []

        # Calculate disagreement per document (standard deviation across scorers)
        disagreement = matrix.std(axis=1)

        # Return documents with disagreement above threshold
        return disagreement[disagreement > threshold].index.tolist()

    def get_scorer_reliability(
        self, corpus: "ScoreCorpus", dimension: str
    ) -> Dict[str, float]:
        """
        Calculate reliability score for each scorer in this dimension

        Args:
            corpus: ScoreCorpus to analyze
            dimension: Dimension to check

        Returns:
            Dictionary mapping scorer names to reliability scores (higher = more reliable)
        """
        # Get dimension-specific configuration
        dim_config = self._get_dimension_config(dimension)
        trust_ref = dim_config["trust_reference"]

        # Get the document × scorer matrix
        matrix = corpus.get_dimension_matrix(dimension)
        if matrix.empty:
            return {}

        # Calculate reliability as correlation with trust reference
        reliability = {}
        if trust_ref in matrix.columns:
            trust_scores = matrix[trust_ref]

            for scorer in matrix.columns:
                if scorer == trust_ref:
                    reliability[scorer] = (
                        1.0  # Perfect correlation with itself
                    )
                    continue

                # Calculate correlation with trust reference
                valid_pairs = matrix[[scorer, trust_ref]].dropna()
                if len(valid_pairs) > 1:
                    try:
                        corr, _ = stats.pearsonr(
                            valid_pairs[scorer], valid_pairs[trust_ref]
                        )
                        reliability[scorer] = float(corr)
                    except:
                        reliability[scorer] = 0.0
                else:
                    reliability[scorer] = 0.0

        # If no trust reference, use consistency across documents
        else:
            scorer_std = matrix.std()
            max_std = scorer_std.max()
            for scorer, std in scorer_std.items():
                # Higher reliability for lower standard deviation
                reliability[scorer] = (
                    1.0 - (std / max_std) if max_std > 0 else 1.0
                )

        return reliability

    def generate_recommendations(
        self, mars_results: Dict[str, Dict]
    ) -> List[str]:
        """
        Generate actionable recommendations based on MARS analysis

        Args:
            mars_results: Results from calculate() method

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        for dimension, results in mars_results.items():
            # High disagreement recommendations
            if results["high_disagreement"]:
                primary_conflict = results["primary_conflict"]
                recommendations.append(
                    f"⚠️ High disagreement in {dimension}: {primary_conflict[0]} and {primary_conflict[1]} "
                    f"differ by {results['delta']:.3f}. Consider human review for ambiguous cases."
                )

            # Outlier scorer recommendations
            scorer_metrics = results["scorer_metrics"]
            if (
                len(scorer_metrics) > 2
            ):  # Need at least 3 scorers to identify outliers
                # Check for scorers with unusual metric patterns
                for scorer, metrics in scorer_metrics.items():
                    if (
                        "uncertainty" in metrics
                        and metrics["uncertainty"]["std"] > 0.2
                    ):
                        recommendations.append(
                            f"⚠️ {scorer} shows high uncertainty variability in {dimension}. "
                            "Consider retraining or adding calibration."
                        )

            # Correlation-based recommendations
            metric_correlations = results["metric_correlations"]
            for metric1, correlations in metric_correlations.items():
                for metric2, corr in correlations.items():
                    if abs(corr) > 0.7:  # Strong correlation
                        recommendations.append(
                            f"💡 In {dimension}, {metric1} and {metric2} are strongly correlated ({corr:.2f}). "
                            "Consider using one as a proxy for the other."
                        )

        # Overall system recommendations
        overall_agreement = mean(
            [r["agreement_score"] for r in mars_results.values()]
        )
        if overall_agreement < 0.7:
            recommendations.append(
                "⚠️ Overall scoring agreement is low (<0.7). Consider implementing human review "
                "for documents with high disagreement."
            )

        return recommendations
