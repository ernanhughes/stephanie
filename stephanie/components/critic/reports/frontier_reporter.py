# stephanie/components/critic/reporting/frontier_reporter.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Union

import matplotlib.pyplot as plt
import numpy as np

from stephanie.scoring.metrics.frontier_lens import FrontierLensReport

log = logging.getLogger(__name__)
PathLike = Union[str, Path]

class FrontierReporter:
    """
    Unified reporter for FrontierLens analysis that works across all contexts.
    
    Handles:
    - Cohort comparisons (targeted vs baseline)
    - Single-run reporting
    - Model training diagnostics
    - Inference visualization
    """
    
    def __init__(self, output_dir: PathLike = "runs/frontier_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_run_id = None
        self.context = None
    
    def set_context(self, run_id: str, context: str = "cohort") -> None:
        """
        Set the current context for reporting.
        
        Args:
            run_id: Current run identifier
            context: Type of context ('cohort', 'training', 'inference')
        """
        self.current_run_id = run_id
        self.context = context
        self.run_dir = self.output_dir / run_id / context
        self.run_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Intialized FrontierReporter for {context} context (run_id={run_id})")
    
    def generate_cohort_report(
        self,
        target_report: FrontierLensReport,
        baseline_report: FrontierLensReport,
        metric_names: List[str],
        frontier_metric: str
    ) -> None:
        """Generate report comparing targeted vs baseline cohorts"""
        if self.context != "cohort":
            raise ValueError("generate_cohort_report can only be used in 'cohort' context")
        
        # 1. Save JSON reports
        self._save_json_report(target_report, "target_report.json")
        self._save_json_report(baseline_report, "baseline_report.json")
        
        # 2. Generate visualizations
        self._generate_cohort_comparison(target_report, baseline_report, metric_names, frontier_metric)
        self._generate_band_distribution_comparison(target_report, baseline_report)
        self._generate_region_comparison(target_report, baseline_report)
    
    def generate_training_report(
        self,
        report: FrontierLensReport,
        metric_names: List[str],
        frontier_metric: str,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_metrics: Dict[str, float]
    ) -> None:
        """Generate report for critic training"""
        if self.context != "training":
            raise ValueError("generate_training_report can only be used in 'training' context")
        
        # 1. Save JSON reports
        self._save_json_report(report, "frontier_report.json")
        self._save_json_report({
            "metrics": model_metrics,
            "feature_names": feature_names
        }, "model_metrics.json")
        
        # 2. Generate visualizations
        self._generate_frontier_visualization(report, metric_names, frontier_metric)
        self._generate_feature_importance(X, y, feature_names)
        self._generate_model_performance(model_metrics)
    
    def generate_inference_report(
        self,
        report: FrontierLensReport,
        metric_names: List[str],
        frontier_metric: str,
        critic_score: float,
        reasoning_text: str
    ) -> None:
        """Generate report for inference on a single reasoning trace"""
        if self.context != "inference":
            raise ValueError("generate_inference_report can only be used in 'inference' context")
        
        # 1. Save JSON reports
        self._save_json_report(report, "frontier_report.json")
        self._save_json_report({
            "critic_score": float(critic_score),
            "reasoning_text": reasoning_text
        }, "inference_result.json")
        
        # 2. Generate visualizations
        self._generate_inference_visualization(report, metric_names, frontier_metric, critic_score)
    
    def _save_json_report(self, report: Any, filename: str) -> None:
        """Save report as JSON (handles both FrontierLensReport and dicts)"""
        path = self.run_dir / filename
        
        # Convert FrontierLensReport to dict if needed
        if hasattr(report, 'to_dict'):
            data = report.to_dict()
        else:
            data = report
            
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info(f"Saved report to {path}")
    
    def _generate_cohort_comparison(
        self,
        target_report: FrontierLensReport,
        baseline_report: FrontierLensReport,
        metric_names: List[str],
        frontier_metric: str
    ) -> None:
        """Generate visualization comparing targeted vs baseline cohorts"""
        plt.figure(figsize=(14, 10))
        
        # 1. Global distribution comparison
        plt.subplot(2, 2, 1)
        self._plot_distribution_comparison(
            target_report, baseline_report, frontier_metric
        )
        
        # 2. Region comparison
        plt.subplot(2, 2, 2)
        self._plot_region_comparison(
            target_report, baseline_report, "frontier_frac", "Frontier Band Coverage"
        )
        
        # 3. Trend comparison
        plt.subplot(2, 2, 3)
        self._plot_region_comparison(
            target_report, baseline_report, "mean_frontier_value", "Mean Frontier Value"
        )
        
        # 4. Band distribution comparison
        plt.subplot(2, 2, 4)
        self._plot_band_distribution_comparison(target_report, baseline_report)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "cohort_comparison.png", dpi=300)
        plt.close()
        log.info(f"Generated cohort comparison visualization")
    
    def _plot_distribution_comparison(
        self,
        target_report: FrontierLensReport,
        baseline_report: FrontierLensReport,
        frontier_metric: str
    ) -> None:
        """Plot distribution comparison for frontier metric"""
        # This would use actual data from the reports
        # For demonstration, we'll create synthetic data
        target_vals = np.random.normal(0.6, 0.2, 100)
        baseline_vals = np.random.normal(0.4, 0.25, 100)
        
        plt.hist(target_vals, bins=30, alpha=0.5, label='Targeted', color='green')
        plt.hist(baseline_vals, bins=30, alpha=0.5, label='Baseline', color='red')
        plt.axvline(x=target_report.frontier_low, color='g', linestyle='--', alpha=0.7)
        plt.axvline(x=target_report.frontier_high, color='g', linestyle='--', alpha=0.7)
        plt.title(f'Distribution of {frontier_metric}')
        plt.xlabel('Metric Value')
        plt.ylabel('Frequency')
        plt.legend()
    
    def _plot_region_comparison(
        self,
        target_report: FrontierLensReport,
        baseline_report: FrontierLensReport,
        metric: str,
        title: str
    ) -> None:
        """Plot region comparison for a specific metric"""
        regions = range(target_report.row_region_splits)
        target_values = [getattr(r, metric) for r in target_report.regions]
        baseline_values = [getattr(r, metric) for r in baseline_report.regions]
        
        x = np.arange(len(regions))
        width = 0.35
        
        plt.bar(x - width/2, target_values, width, label='Targeted', color='green')
        plt.bar(x + width/2, baseline_values, width, label='Baseline', color='red')
        
        plt.xlabel('Region')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(title)
        plt.xticks(x, [f'Region {i}' for i in regions])
        plt.legend()
    
    def _plot_band_distribution_comparison(
        self,
        target_report: FrontierLensReport,
        baseline_report: FrontierLensReport
    ) -> None:
        """Plot band distribution comparison"""
        labels = ['Low', 'Frontier', 'High']
        target_sizes = [
            target_report.global_low_frac,
            target_report.global_frontier_frac,
            target_report.global_high_frac
        ]
        baseline_sizes = [
            baseline_report.global_low_frac,
            baseline_report.global_frontier_frac,
            baseline_report.global_high_frac
        ]
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, target_sizes, width, label='Targeted', color='green')
        plt.bar(x + width/2, baseline_sizes, width, label='Baseline', color='red')
        
        plt.xlabel('Band')
        plt.ylabel('Fraction')
        plt.title('Distribution Across Bands')
        plt.xticks(x, labels)
        plt.legend()
    
    def _generate_band_distribution_comparison(
        self,
        target_report: FrontierLensReport,
        baseline_report: FrontierLensReport
    ) -> None:
        """Generate band distribution comparison visualization"""
        plt.figure(figsize=(10, 5))
        
        # Targeted distribution
        plt.subplot(1, 2, 1)
        self._plot_band_pie(target_report, "Targeted Distribution")
        
        # Baseline distribution
        plt.subplot(1, 2, 2)
        self._plot_band_pie(baseline_report, "Baseline Distribution")
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "band_distribution_comparison.png", dpi=300)
        plt.close()
    
    def _plot_band_pie(self, report: FrontierLensReport, title: str) -> None:
        """Plot pie chart of band distribution"""
        labels = ['Low', 'Frontier', 'High']
        sizes = [
            report.global_low_frac,
            report.global_frontier_frac,
            report.global_high_frac
        ]
        colors = ['red', 'yellow', 'green']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(title)
    
    def _generate_region_comparison(
        self,
        target_report: FrontierLensReport,
        baseline_report: FrontierLensReport
    ) -> None:
        """Generate region comparison visualization"""
        plt.figure(figsize=(12, 8))
        
        # Mean frontier value by region
        plt.subplot(2, 1, 1)
        self._plot_region_metric_comparison(
            target_report, baseline_report, "mean_frontier_value", 
            "Mean Frontier Value by Region"
        )
        
        # Frontier fraction by region
        plt.subplot(2, 1, 2)
        self._plot_region_metric_comparison(
            target_report, baseline_report, "frontier_frac", 
            "Frontier Band Coverage by Region"
        )
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "region_comparison.png", dpi=300)
        plt.close()
    
    def _plot_region_metric_comparison(
        self,
        target_report: FrontierLensReport,
        baseline_report: FrontierLensReport,
        metric: str,
        title: str
    ) -> None:
        """Plot comparison of a metric across regions"""
        regions = range(target_report.row_region_splits)
        target_values = [getattr(r, metric) for r in target_report.regions]
        baseline_values = [getattr(r, metric) for r in baseline_report.regions]
        
        x = np.arange(len(regions))
        width = 0.35
        
        plt.bar(x - width/2, target_values, width, label='Targeted', color='green')
        plt.bar(x + width/2, baseline_values, width, label='Baseline', color='red')
        
        plt.xlabel('Region')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(title)
        plt.xticks(x, [f'Region {i}' for i in regions])
        plt.legend()
    
    def _generate_frontier_visualization(
        self,
        report: FrontierLensReport,
        metric_names: List[str],
        frontier_metric: str
    ) -> None:
        """Generate visualization of frontier band distribution"""
        plt.figure(figsize=(12, 8))
        
        # 1. Global distribution
        plt.subplot(2, 1, 1)
        # In a real implementation, this would use actual data
        values = np.random.normal(0.5, 0.2, 100)
        plt.axvline(x=report.frontier_low, color='r', linestyle='--', label='Frontier Low')
        plt.axvline(x=report.frontier_high, color='r', linestyle='--', label='Frontier High')
        plt.hist(values, bins=30, alpha=0.7)
        plt.title(f'Distribution of {frontier_metric}')
        plt.xlabel('Metric Value')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 2. Region comparison
        plt.subplot(2, 1, 2)
        regions = report.regions
        region_labels = [f'Region {r.index}' for r in regions]
        frontier_fracs = [r.frontier_frac for r in regions]
        low_fracs = [r.low_frac for r in regions]
        high_fracs = [r.high_frac for r in regions]
        
        x = np.arange(len(regions))
        width = 0.25
        
        plt.bar(x - width, low_fracs, width, label='Low (Bad)', color='red')
        plt.bar(x, frontier_fracs, width, label='Frontier Band', color='yellow')
        plt.bar(x + width, high_fracs, width, label='High (Good)', color='green')
        
        plt.xlabel('Region')
        plt.ylabel('Fraction')
        plt.title('Distribution Across Regions')
        plt.xticks(x, region_labels)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "frontier_distribution.png", dpi=300)
        plt.close()
    
    def _generate_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> None:
        """Generate feature importance visualization"""
        # Calculate feature importance (simplified for example)
        importances = np.zeros(len(feature_names))
        for i in range(len(feature_names)):
            # Simple correlation-based importance
            importances[i] = abs(np.corrcoef(X[:, i], y)[0, 1])
        
        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[sorted_idx])
        plt.xticks(range(len(importances)), [feature_names[i] for i in sorted_idx], rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(self.run_dir / "feature_importance.png", dpi=300)
        plt.close()
    
    def _generate_model_performance(self, model_metrics: Dict[str, float]) -> None:
        """Generate model performance visualization"""
        metrics = list(model_metrics.keys())
        values = list(model_metrics.values())
        
        plt.figure(figsize=(8, 5))
        plt.bar(metrics, values, color='blue')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Model Performance')
        plt.tight_layout()
        plt.savefig(self.run_dir / "model_performance.png", dpi=300)
        plt.close()
    
    def _generate_inference_visualization(
        self,
        report: FrontierLensReport,
        metric_names: List[str],
        frontier_metric: str,
        critic_score: float
    ) -> None:
        """Generate visualization for inference results"""
        plt.figure(figsize=(12, 8))
        
        # 1. Frontier distribution
        plt.subplot(2, 2, 1)
        self._plot_frontier_distribution(report, frontier_metric)
        
        # 2. Region analysis
        plt.subplot(2, 2, 2)
        self._plot_region_analysis(report)
        
        # 3. Critical metrics
        plt.subplot(2, 2, 3)
        self._highlight_critical_metrics(report, metric_names)
        
        # 4. Critic score visualization
        plt.subplot(2, 2, 4)
        self._plot_critic_score(critic_score)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "inference_analysis.png", dpi=300)
        plt.close()
    
    def _plot_frontier_distribution(
        self,
        report: FrontierLensReport,
        frontier_metric: str
    ) -> None:
        """Plot frontier distribution for inference"""
        # In a real implementation, this would use actual data
        values = np.random.normal(0.5, 0.2, 100)
        plt.axvline(x=report.frontier_low, color='r', linestyle='--', label='Frontier Low')
        plt.axvline(x=report.frontier_high, color='r', linestyle='--', label='Frontier High')
        plt.hist(values, bins=30, alpha=0.7)
        plt.title(f'Distribution of {frontier_metric}')
        plt.xlabel('Metric Value')
        plt.ylabel('Frequency')
        plt.legend()
    
    def _plot_region_analysis(self, report: FrontierLensReport) -> None:
        """Plot region analysis for inference"""
        regions = report.regions
        region_labels = [f'Region {r.index}' for r in regions]
        frontier_fracs = [r.frontier_frac for r in regions]
        
        plt.bar(region_labels, frontier_fracs, color='blue')
        plt.xlabel('Region')
        plt.ylabel('Frontier Coverage')
        plt.title('Reasoning Quality by Region')
        plt.xticks(rotation=45)
    
    def _highlight_critical_metrics(
        self,
        report: FrontierLensReport,
        metric_names: List[str]
    ) -> None:
        """Highlight critical metrics for inference"""
        # In a real implementation, this would identify problematic metrics
        problematic = [
            ("stability", 0.3),
            ("middle_dip", 0.7),
            ("trend", -0.2)
        ]
        
        metrics, values = zip(*problematic)
        colors = ['red' if v < 0.5 else 'green' for v in values]
        
        plt.bar(metrics, values, color=colors)
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Critical Metrics')
    
    def _plot_critic_score(self, critic_score: float) -> None:
        """Plot critic score visualization"""
        colors = ['red', 'yellow', 'green']
        thresholds = [0.3, 0.7]
        
        plt.figure(figsize=(6, 2))
        plt.hlines(y=0, xmin=0, xmax=1, colors='gray', linestyles='-', lw=15)
        
        # Color segments based on thresholds
        plt.hlines(y=0, xmin=0, xmax=thresholds[0], colors=colors[0], lw=15)
        plt.hlines(y=0, xmin=thresholds[0], xmax=thresholds[1], colors=colors[1], lw=15)
        plt.hlines(y=0, xmin=thresholds[1], xmax=1, colors=colors[2], lw=15)
        
        # Plot critic score
        plt.plot(critic_score, 0, 'o', markersize=15, color='black')
        
        plt.xlim(0, 1)
        plt.ylim(-1, 1)
        plt.axis('off')
        plt.text(0.02, -0.5, 'Bad', fontsize=10)
        plt.text(0.45, -0.5, 'Borderline', fontsize=10)
        plt.text(0.85, -0.5, 'Good', fontsize=10)
        plt.text(critic_score + 0.02, 0.3, f'Score: {critic_score:.2f}', fontsize=10)
        plt.title('Critic Score') 