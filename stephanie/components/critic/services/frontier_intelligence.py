# stephanie/components/critic/services/frontier_intelligence.py
from __future__ import annotations
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import TheilSenRegressor
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)

class FrontierIntelligence:
    """
    Intelligent meta-cognitive system for critic self-monitoring and lifecycle management.
    
    This service provides:
    - Persistent tracking of critic performance across runs via MetricStore
    - Statistical trend analysis of key metrics
    - Actionable decision policy for critic management
    - Visual diagnostics for monitoring critic health
    - Safe model versioning and rollback
    
    Unlike a passive data store, this system actively understands its own performance
    and makes intelligent decisions about the critic's lifecycle.
    """
    
    def __init__(self, cfg, memory, container, logger, run_id: str = "unknown"):
        """
        Args:
            memory: Stephanie memory system containing the MetricStore
            run_id: Current run identifier for context
        """
        self.cfg = cfg
        self.memory = memory
        self.run_id = run_id
        self.metric_store = self.memory.metrics
        
        # Decision policy configuration
        self.policy_config = {
            "min_history": 4,
            "target_auc": 0.8,
            "improve_tol": 0.005,
            "degrade_tol": -0.01,
            "min_band_sep": 0.05,
            "stability_threshold": 0.05,
            "consistency_threshold": 0.1,
            "confidence_level": 0.95
        }
        
        # LLM advisory configuration
        self.llm_advisory = {
            "enabled": False,
            "temperature": 0.3,
            "max_tokens": 150
        }

        # In-process ephemeral state (not persisted in the DB)
        # Used for things like "last_selected_metric" during a run.
        self.state: Dict[str, Any] = {
            "last_selected_metric": None,
        }
        
        log.info(f"Intialized FrontierIntelligence with policy config: "
                f"min_history={self.policy_config['min_history']}, "
                f"target_auc={self.policy_config['target_auc']:.2f}")
    
    def update(
        self,
        metric_importance: List[Dict],
        critic_auc: float,
        run_id: str
    ) -> None:
        """
        Update with new metric importance data and critic performance
        
        Args:
            metric_importance: List of metrics with AUC scores
            critic_auc: AUC of the tiny critic on this run
            run_id: Current run identifier
        """
        # Update group with critic data
        self.metric_store.update_group_with_critic_data(
            run_id=run_id,
            auc_score=critic_auc
        )
        
        # Update core metrics in group meta
        core_metrics = self._extract_core_metrics(metric_importance)
        self._update_group_meta_with_core_metrics(run_id, core_metrics)
        
        log.info(f"FrontierIntelligence: Updated group {run_id} with critic data")
    
    def _extract_core_metrics(self, metric_importance: List[Dict]) -> List[str]:
        """Extract core metrics from metric importance data"""
        # Sort by AUC
        sorted_metrics = sorted(metric_importance, key=lambda x: x["auc_roc"], reverse=True)
        # Take top 3-5 metrics
        return [m["metric"] for m in sorted_metrics[:max(3, len(sorted_metrics) // 3)]]
    
    def _update_group_meta_with_core_metrics(self, run_id: str, core_metrics: List[str]) -> None:
        """Update the metric group meta with core metrics"""
        meta_patch = {
            "core_metrics": core_metrics,
            "frontier_intelligence": {
                "timestamp": datetime.now().isoformat(),
                "core_metrics_count": len(core_metrics)
            }
        }
        self.metric_store.upsert_group_meta(run_id, meta_patch)
    
    
    def _calculate_metric_stability(self, critic_runs: List[Dict]) -> Dict[str, float]:
        """Calculate stability scores for metrics based on recent critic runs"""
        # Collect AUC scores for each metric
        metric_scores = {}
        
        for run in critic_runs:
            # Get metric importance from the run's meta
            group = self.metric_store.get_group(run["run_id"])
            if not group or not group.meta:
                continue
                
            metric_importance = group.meta.get("metric_importance", [])
            for metric in metric_importance:
                name = metric["metric"]
                if name not in metric_scores:
                    metric_scores[name] = []
                metric_scores[name].append(metric["auc_roc"])
        
        # Calculate stability (higher = more stable)
        stability_scores = {}
        for metric, scores in metric_scores.items():
            if len(scores) >= 2:  # Need at least 2 points for stability
                stability = 1.0 / (np.std(scores) + 1e-6)  # Inverse of std dev
                stability_scores[metric] = stability
        
        return stability_scores
    
    def get_critic_quality_score(self) -> float:
        """Get a quality score for the current critic state"""
        # Get recent critic runs
        critic_runs = self.metric_store.get_critic_runs(limit=5)
        
        if not critic_runs:
            return 0.5
        
        # Calculate weighted average of recent AUCs
        aucs = [run["auc"] for run in critic_runs]
        weights = np.linspace(0.5, 1.5, len(aucs))
        
        return float(np.average(aucs, weights=weights))
    
    def get_band_separation_score(self) -> float:
        """Get the average band separation score from recent runs"""
        critic_runs = self.metric_store.get_critic_runs(limit=10)
        
        if not critic_runs:
            return 0.0
        
        # Filter out runs with no band separation data
        valid_runs = [run for run in critic_runs if run["band_separation"] is not None]
        
        if not valid_runs:
            return 0.0
        
        return float(np.mean([run["band_separation"] for run in valid_runs]))
    
    # ============================================================
    # META-COGNITIVE ENHANCEMENTS (THE "AI" LAYER)
    # ============================================================
    
    def record_run(
        self,
        *,
        run_id: str,
        model_version: str,
        auc: float,
        band_separation: float = 0.0,
        stability_score: float = 0.0,
        feature_consistency: float = 0.0,
        config_snapshot: Optional[Dict[str, Any]] = None,
        metrics_snapshot: Optional[Dict[str, float]] = None,
        promoted: bool = False,
        decision_action: str = "WAIT",
        decision_confidence: float = 0.0,
        decision_reason: str = "",
        decision_advice: str = ""
    ) -> None:
        """
        Log one critic run outcome into persistent state.
        
        Args:
            run_id: Unique identifier for this run
            model_version: Version identifier for the model
            auc: AUC score of the critic
            band_separation: Measure of separation between good/bad reasoning
            stability_score: Measure of metric stability across runs
            feature_consistency: Consistency of important features
            config_snapshot: Configuration used for this run
            metrics_snapshot: Additional metrics to track
            promoted: Whether this model was promoted to production
            decision_action: Action decided by the critic
            decision_confidence: Confidence in the decision
            decision_reason: Reason for the decision
            decision_advice: Additional advice
        """
        # Record in critic_runs table
        self.metric_store.record_critic_run(
            run_id=run_id,
            model_version=model_version,
            auc=auc,
            band_separation=band_separation,
            stability_score=stability_score,
            feature_consistency=feature_consistency,
            is_promoted=promoted,
            decision_action=decision_action,
            decision_confidence=decision_confidence,
            decision_reason=decision_reason,
            decision_advice=decision_advice
        )
        
        # Update metric group with critic data
        self.metric_store.update_group_with_critic_data(
            run_id=run_id,
            frontier_metric=self.get_last_selected_metric(),
            critic_status=self._get_critic_status(decision_action),
            critic_action=decision_action,
            critic_confidence=decision_confidence,
            is_best_model=promoted,  # Simplified for example
            model_version=model_version,
            auc_score=auc,
            band_separation=band_separation,
            stability_score=stability_score,
            feature_consistency=feature_consistency
        )
        
        # Update critic models
        self.metric_store.update_critic_model(
            model_version=model_version,
            run_id=run_id,
            auc=auc,
            band_separation=band_separation,
            stability_score=stability_score,
            feature_consistency=feature_consistency,
            is_active=promoted,
            is_best=promoted,  # Simplified for example
            promoted_at=datetime.now() if promoted else None
        )
        
        log.info(f"FrontierIntelligence: Recorded critic run {run_id} (model={model_version}, auc={auc:.4f})")
        
        # Generate visual diagnostics
        self._generate_diagnostics(run_id)
    
    def _get_critic_status(self, decision_action: str) -> str:
        """Map decision action to critic status"""
        status_map = {
            "IMPROVING": ["TRAIN_MORE"],
            "STABLE": ["KEEP"],
            "DEGRADING": ["RESET"],
            "STAGNANT": ["WAIT"]
        }
        
        for status, actions in status_map.items():
            if decision_action in actions:
                return status
        
        return "UNKNOWN"
    
    def _compute_trend(
        self,
        key: str,
        window: int = 5,
        min_points: int = 3
    ) -> Tuple[float, float, List[float], bool]:
        """
        Compute statistical trend over recent runs using Theil-Sen estimator.
        
        Args:
            key: The metric to analyze (e.g., "auc", "band_separation")
            window: Number of most recent runs to consider
            min_points: Minimum number of points needed for analysis
            
        Returns:
            (slope, p_value, values, significant)
            - slope: Trend slope (positive = improving)
            - p_value: Statistical significance
            - values: Values used in analysis
            - significant: Whether trend is statistically significant
        """
        # Get trend data from store
        trend_data = self.metric_store.get_critic_trend_data(window=window)
        
        if key not in trend_data or not trend_data[key]:
            return 0.0, 1.0, [], False
        
        # Extract values
        vals = [item["value"] for item in trend_data[key]]
        
        if len(vals) < min_points:
            return 0.0, 1.0, vals, False
        
        # Use Theil-Sen estimator (resistant to outliers)
        x = np.arange(len(vals)).reshape(-1, 1)
        y = np.array(vals)
        
        try:
            model = TheilSenRegressor()
            model.fit(x, y)
            slope = model.coef_[0]
            
            # Calculate p-value using permutation test
            _, p_value = stats.pearsonr(x.flatten(), y)
            
            # Determine significance
            significant = p_value < (1 - self.policy_config["confidence_level"])
            
            return float(slope), float(p_value), vals, significant
        except Exception as e:
            log.warning(f"Error computing trend for {key}: {str(e)}")
            return 0.0, 1.0, vals, False
    
    def get_progress_signal(self, window: int = 5) -> Dict[str, Any]:
        """
        Summarize how the critic is evolving over recent runs.
        
        Returns a comprehensive progress report including:
        - Trend analysis for key metrics
        - Statistical significance
        - Current status relative to targets
        """
        # Compute trends for key metrics
        auc_slope, auc_p, auc_vals, auc_sig = self._compute_trend("auc", window)
        sep_slope, sep_p, sep_vals, sep_sig = self._compute_trend("band_separation", window)
        stab_slope, stab_p, stab_vals, stab_sig = self._compute_trend("stability_score", window)
        cons_slope, cons_p, cons_vals, cons_sig = self._compute_trend("feature_consistency", window)
        
        # Get latest values
        latest_auc = auc_vals[-1] if auc_vals else 0.5
        latest_sep = sep_vals[-1] if sep_vals else 0.0
        latest_stab = stab_vals[-1] if stab_vals else 0.0
        latest_cons = cons_vals[-1] if cons_vals else 0.0
        
        # Determine overall progress status
        improving_metrics = sum([
            auc_sig and auc_slope > 0,
            sep_sig and sep_slope > 0,
            stab_sig and stab_slope > 0,
            cons_sig and cons_slope > 0
        ])
        
        degrading_metrics = sum([
            auc_sig and auc_slope < 0,
            sep_sig and sep_slope < 0,
            stab_sig and stab_slope < 0,
            cons_sig and cons_slope < 0
        ])
        
        status = "STABLE"
        if improving_metrics >= 3:
            status = "IMPROVING"
        elif degrading_metrics >= 2:
            status = "DEGRADING"
        elif improving_metrics == 0 and degrading_metrics == 0:
            status = "STAGNANT"
        
        return {
            "status": status,
            "auc": {
                "slope": auc_slope,
                "p_value": auc_p,
                "significant": auc_sig,
                "values": auc_vals,
                "latest": latest_auc
            },
            "band_separation": {
                "slope": sep_slope,
                "p_value": sep_p,
                "significant": sep_sig,
                "values": sep_vals,
                "latest": latest_sep
            },
            "stability": {
                "slope": stab_slope,
                "p_value": stab_p,
                "significant": stab_sig,
                "values": stab_vals,
                "latest": latest_stab
            },
            "feature_consistency": {
                "slope": cons_slope,
                "p_value": cons_p,
                "significant": cons_sig,
                "values": cons_vals,
                "latest": latest_cons
            },
            "n_points": len(auc_vals),
            "improving_metrics": improving_metrics,
            "degrading_metrics": degrading_metrics
        }
    
    def decide_action(
        self,
        policy_config: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Decide what to do next with the critic based on recent trends.
        
        Args:
            policy_config: Optional override for policy configuration
            
        Returns:
            {
                "action": "WAIT"|"KEEP"|"TRAIN_MORE"|"RESET"|"PROMOTE",
                "reason": str,
                "progress": {...progress_signal...},
                "best_model": {...} | None,
                "confidence": float,  # 0-1 confidence in this decision
                "advice": str          # Optional LLM-generated advice
            }
        """
        # Use provided policy config or default
        config = policy_config or self.policy_config
        min_history = config.get("min_history", 4)
        target_auc = config.get("target_auc", 0.8)
        improve_tol = config.get("improve_tol", 0.005)
        degrade_tol = config.get("degrade_tol", -0.01)
        min_band_sep = config.get("min_band_sep", 0.05)
        stability_threshold = config.get("stability_threshold", 0.05)
        consistency_threshold = config.get("consistency_threshold", 0.1)
        
        # Get progress signal
        prog = self.get_progress_signal()
        n = prog["n_points"]
        
        # Initialize decision
        decision = {
            "action": "WAIT",
            "reason": "",
            "progress": prog,
            "best_model": self._get_best_model_info(),
            "confidence": 0.0,
            "advice": ""
        }
        
        # Not enough history
        if n < min_history:
            decision["reason"] = f"only {n} runs, need >= {min_history}"
            return decision
        
        # Extract key metrics
        auc_slope = prog["auc"]["slope"]
        auc_sig = prog["auc"]["significant"]
        latest_auc = prog["auc"]["latest"]
        
        sep_slope = prog["band_separation"]["slope"]
        latest_sep = prog["band_separation"]["latest"]
        
        stab_slope = prog["stability"]["slope"]
        latest_stab = prog["stability"]["latest"]
        
        cons_slope = prog["feature_consistency"]["slope"]
        latest_cons = prog["feature_consistency"]["latest"]
        
        # 1) Degrading critic → suggest RESET
        if auc_slope < degrade_tol and latest_auc < target_auc:
            decision["action"] = "RESET"
            decision["reason"] = (
                f"critic degrading (auc_slope={auc_slope:.4f}, "
                f"latest_auc={latest_auc:.3f} < target {target_auc:.3f})"
            )
            decision["confidence"] = 0.9 if auc_sig else 0.7
        
        # 2) Improving but still below target → TRAIN_MORE
        elif auc_slope > improve_tol and latest_auc < target_auc:
            decision["action"] = "TRAIN_MORE"
            decision["reason"] = (
                f"critic improving (auc_slope={auc_slope:.4f}) "
                f"but still below target AUC {target_auc:.3f}"
            )
            decision["confidence"] = 0.85 if auc_sig else 0.65
        
        # 3) Good critic with stable or rising band separation → KEEP
        elif latest_auc >= target_auc and latest_sep >= min_band_sep:
            decision["action"] = "KEEP"
            decision["reason"] = (
                f"critic good and stable (auc={latest_auc:.3f}, "
                f"band_sep={latest_sep:.3f})"
            )
            decision["confidence"] = 0.95
        
        # 4) Critic is good but unstable → PROMOTE with caution
        elif latest_auc >= target_auc and latest_stab < stability_threshold:
            decision["action"] = "PROMOTE"
            decision["reason"] = (
                f"critic meets AUC target but needs stability ({latest_stab:.3f} < {stability_threshold:.3f})"
            )
            decision["confidence"] = 0.8
        
        # 5) Critic is good but inconsistent → TRAIN_MORE with focus on consistency
        elif latest_auc >= target_auc and latest_cons < consistency_threshold:
            decision["action"] = "TRAIN_MORE"
            decision["reason"] = (
                f"critic meets AUC target but needs feature consistency ({latest_cons:.3f} < {consistency_threshold:.3f})"
            )
            decision["confidence"] = 0.85
        
        # 6) Default: TRAIN_MORE but not urgent
        else:
            decision["action"] = "TRAIN_MORE"
            decision["reason"] = (
                f"critic neither clearly degrading nor clearly optimal "
                f"(auc={latest_auc:.3f}, slope={auc_slope:.4f})"
            )
            decision["confidence"] = 0.7
        
        # Generate LLM advice if enabled
        if self.llm_advisory["enabled"]:
            decision["advice"] = self._generate_llm_advice(decision)
        
        return decision
    
    def _get_best_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the best model from the store"""
        best_model = self.metric_store.get_best_critic_model()
        if not best_model:
            return None
        
        return {
            "model_version": best_model.model_version,
            "auc": best_model.auc,
            "band_separation": best_model.band_separation,
            "stability_score": best_model.stability_score,
            "feature_consistency": best_model.feature_consistency,
            "is_active": best_model.is_active,
            "is_best": best_model.is_best,
            "created_at": best_model.created_at.isoformat(),
            "promoted_at": best_model.promoted_at.isoformat() if best_model.promoted_at else None
        }
    
    def _generate_llm_advice(self, decision: Dict[str, Any]) -> str:
        """Generate natural language advice using LLM (placeholder implementation)"""
        # In a real implementation, this would call an LLM
        # For now, we'll use template-based generation
        
        action = decision["action"]
        reason = decision["reason"]
        progress = decision["progress"]
        
        if action == "RESET":
            return (
                "The critic is showing concerning degradation in performance. "
                "I recommend rolling back to the best-performing version and "
                "investigating recent changes. Consider reviewing the training data "
                "for potential quality issues."
            )
        elif action == "TRAIN_MORE":
            if "improving" in reason:
                return (
                    "The critic is making steady progress but hasn't reached "
                    "optimal performance yet. I recommend continuing training with "
                    "the current configuration while monitoring band separation."
                )
            else:
                return (
                    "The critic is stable but could be improved further. "
                    "Consider training with augmented data or adjusting the frontier metric."
                )
        elif action == "KEEP":
            return (
                "The critic is performing well across all metrics. "
                "No immediate action is needed - continue monitoring for stability."
            )
        elif action == "PROMOTE":
            return (
                "The critic meets quality thresholds but shows some instability. "
                "I recommend promoting with close monitoring of consistency metrics."
            )
        else:
            return (
                "The critic's performance is in a stable range. "
                "Consider periodic retraining to maintain performance."
            )
    
    def _generate_diagnostics(self, run_id: str) -> None:
        """Generate visual diagnostics of critic progress"""
        try:
            # Get progress data
            prog = self.get_progress_signal()
            n = prog["n_points"]
            if n < 2:
                return
            
            # Create figure
            plt.figure(figsize=(14, 10))
            
            # 1. AUC Trend
            plt.subplot(2, 2, 1)
            auc_vals = prog["auc"]["values"]
            x = range(len(auc_vals))
            plt.plot(x, auc_vals, 'b-o', label='AUC')
            plt.axhline(y=self.policy_config["target_auc"], color='r', linestyle='--', label='Target')
            
            # Add trend line if significant
            if prog["auc"]["significant"]:
                slope = prog["auc"]["slope"]
                intercept = auc_vals[0]
                trend_line = [intercept + slope * i for i in range(len(auc_vals))]
                plt.plot(x, trend_line, 'g--', label=f'Trend ({slope:.4f})')
            
            plt.title('Critic AUC Trend')
            plt.xlabel('Run')
            plt.ylabel('AUC')
            plt.legend()
            plt.grid(True)
            
            # 2. Band Separation
            plt.subplot(2, 2, 2)
            sep_vals = prog["band_separation"]["values"]
            plt.plot(x, sep_vals, 'b-o', label='Band Separation')
            plt.axhline(y=self.policy_config["min_band_sep"], color='r', linestyle='--', label='Minimum')
            
            # Add trend line if significant
            if prog["band_separation"]["significant"]:
                slope = prog["band_separation"]["slope"]
                intercept = sep_vals[0]
                trend_line = [intercept + slope * i for i in range(len(sep_vals))]
                plt.plot(x, trend_line, 'g--', label=f'Trend ({slope:.4f})')
            
            plt.title('Band Separation Trend')
            plt.xlabel('Run')
            plt.ylabel('Band Separation')
            plt.legend()
            plt.grid(True)
            
            # 3. Stability vs Consistency
            plt.subplot(2, 2, 3)
            stab_vals = prog["stability"]["values"]
            cons_vals = prog["feature_consistency"]["values"]
            
            plt.plot(x, stab_vals, 'b-o', label='Stability')
            plt.plot(x, cons_vals, 'r-s', label='Feature Consistency')
            plt.axhline(y=self.policy_config["stability_threshold"], color='b', linestyle=':', alpha=0.5)
            plt.axhline(y=self.policy_config["consistency_threshold"], color='r', linestyle=':', alpha=0.5)
            
            plt.title('Stability & Consistency')
            plt.xlabel('Run')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            
            # 4. Progress Status
            plt.subplot(2, 2, 4)
            status = prog["status"]
            
            # Create status visualization
            colors = {
                "IMPROVING": "green",
                "STABLE": "blue",
                "STAGNANT": "orange",
                "DEGRADING": "red"
            }
            
            plt.text(0.5, 0.7, status, 
                    fontsize=24, 
                    ha='center', 
                    color=colors.get(status, "black"),
                    weight='bold')
            
            plt.text(0.5, 0.4, f"Improving: {prog['improving_metrics']}/4\n"
                            f"Degrading: {prog['degrading_metrics']}/4",
                    fontsize=14,
                    ha='center')
            
            plt.axis('off')
            plt.title('Current Progress Status')
            
            plt.tight_layout()
            
            # Save figure to the run directory
            run_dir = Path(f"runs/{run_id}/frontier_intelligence")
            run_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(run_dir / "critic_progress.png", dpi=300)
            plt.close()
            
            log.info(f"Generated critic progress diagnostics for run {run_id}")
            
        except Exception as e:
            log.warning(f"Failed to generate critic diagnostics: {str(e)}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get a comprehensive status report of the critic system"""
        prog = self.get_progress_signal()
        decision = self.decide_action()
        
        return {
            "status": {
                "progress": prog,
                "decision": {
                    "action": decision["action"],
                    "reason": decision["reason"],
                    "confidence": decision["confidence"]
                },
                "best_model": self._get_best_model_info(),
                "core_metrics": self._get_core_metrics(),
                "last_selected_metric": self.get_last_selected_metric()
            },
            "config": {
                "policy": self.policy_config
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_core_metrics(self) -> List[str]:
        """Get the current core metrics from the store"""
        # Get the most recent run
        recent_runs = self.metric_store.get_critic_runs(limit=1)
        if not recent_runs:
            return []
        
        # Get metric group for the run
        group = self.metric_store.get_group(recent_runs[0]["run_id"])
        if not group or not group.meta:
            return []
        
        return group.meta.get("core_metrics", [])
    
    def get_last_selected_metric(self) -> Optional[str]:
        """Get the last selected frontier metric from the most recent run"""
        # Get the most recent run
        recent_runs = self.metric_store.get_critic_runs(limit=1)
        if not recent_runs:
            return None
        
        # Get metric group for the run
        group = self.metric_store.get_group(recent_runs[0]["run_id"])
        if not group:
            return None
        
        return group.frontier_metric
    
    def reset_to_best(self) -> Optional[str]:
        """
        Reset the critic to the best-performing version.
        
        Returns:
            The model version that was reset to, or None if no best model
        """
        best_model = self.metric_store.get_best_critic_model()
        if not best_model:
            log.warning("No best model available for reset")
            return None
        
        # Update the active model
        self.metric_store.update_critic_model(
            model_version=best_model.model_version,
            run_id=best_model.run_id,
            auc=best_model.auc,
            band_separation=best_model.band_separation,
            stability_score=best_model.stability_score,
            feature_consistency=best_model.feature_consistency,
            is_active=True,
            is_best=True,
            promoted_at=datetime.now()
        )
        
        log.info(f"Resetting critic to best model: {best_model.model_version}")
        return best_model.model_version
    
    def adjust_policy(self, updates: Dict[str, float]) -> None:
        """
        Adjust policy configuration parameters.
        
        Args:
            updates: Dictionary of policy parameters to update
        """
        for key, value in updates.items():
            if key in self.policy_config:
                old_value = self.policy_config[key]
                self.policy_config[key] = value
                log.info(f"Adjusted policy: {key} = {old_value:.4f} → {value:.4f}")
            else:
                log.warning(f"Policy parameter {key} not found")

    def select_frontier_metric(
        self,
        metric_matrix: np.ndarray,
        metric_names: List[str],
        y: np.ndarray,
        run_id: str,
        fallback: str = "Tiny.faithfulness.attr.scm.aggregate01"
    ) -> str:
        """
        Dynamically select the best frontier metric based on current data.
        
        Uses a bandit-style approach to balance exploration and exploitation.
        
        Args:
            metric_matrix: N×M matrix of metric values (N examples, M metrics)
            metric_names: Names of all metrics
            y: Binary labels (0=bad, 1=good)
            run_id: Current run ID for context
            fallback: Default metric if selection fails
        
        Returns:
            Selected frontier metric name
        """
        log.info("🔍 Dynamically selecting frontier metric...")
        
        # 1. First pass: Calculate metric quality scores
        metric_scores = []
        for i, metric_name in enumerate(metric_names):
            metric_values = metric_matrix[:, i]
            
            # Calculate separation quality
            auc = self._calculate_auc(metric_values, y)
            cohens_d = self._calculate_cohens_d(metric_values, y)
            stability = self._get_metric_stability(metric_name, run_id)
            
            # Composite score (higher = better)
            score = (0.5 * auc) + (0.3 * cohens_d) + (0.2 * stability)
            metric_scores.append((metric_name, score, auc, cohens_d, stability))
        
        # 2. Second pass: Apply bandit exploration strategy
        progress = self.get_progress_signal()
        
        # Strategy depends on current status
        if progress["status"] == "IMPROVING":
            # Exploit: Focus on highest-scoring metrics
            metric_scores.sort(key=lambda x: x[1], reverse=True)
            top_metrics = [m[0] for m in metric_scores[:3]]
            selected = top_metrics[0]
            log.info(f"   Exploitation mode: Selected {selected} (score={metric_scores[0][1]:.4f})")
        
        elif progress["status"] == "STAGNANT":
            # Explore: Try a new metric with high potential
            # Sort by potential (AUC * stability)
            metric_scores.sort(key=lambda x: x[2] * x[4], reverse=True)
            # Skip the current metric to force exploration
            current_metric = self.get_last_selected_metric()
            top_metrics = [m[0] for m in metric_scores[:5] if m[0] != current_metric]
            
            if top_metrics:
                selected = top_metrics[0]
                log.info(f"   Exploration mode: Selected {selected} (potential={metric_scores[0][2]*metric_scores[0][4]:.4f})")
            else:
                selected = metric_scores[0][0]
                log.info(f"   Exploration fallback: Selected {selected}")
        
        else:  # STABLE or DEGRADING
            # Balance: Use historical stability
            metric_scores.sort(key=lambda x: x[4], reverse=True)
            top_metrics = [m[0] for m in metric_scores[:3]]
            selected = top_metrics[0]
            log.info(f"   Stability mode: Selected {selected} (stability={metric_scores[0][4]:.4f})")
        
        # 3. Update state and store selection
        self.state["last_selected_metric"] = selected
        self._record_metric_selection(run_id, selected, metric_scores)
        
        return selected

    def _calculate_auc(self, metric_values: np.ndarray, y: np.ndarray) -> float:
        """Calculate AUC for a single metric"""
        from sklearn.metrics import roc_auc_score
        
        # Handle edge cases
        if len(np.unique(y)) < 2:
            return 0.5
        
        try:
            # Convert to probability-like values
            metric_values = (metric_values - metric_values.min()) / (metric_values.max() - metric_values.min() + 1e-8)
            return roc_auc_score(y, metric_values)
        except:
            return 0.5

    def _calculate_cohens_d(self, metric_values: np.ndarray, y: np.ndarray) -> float:
        """Calculate Cohen's d effect size for a metric"""
        good = metric_values[y == 1]
        bad = metric_values[y == 0]
        
        if len(good) == 0 or len(bad) == 0:
            return 0.0
        
        pooled_std = np.sqrt((np.var(good) + np.var(bad)) / 2)
        return (np.mean(good) - np.mean(bad)) / (pooled_std + 1e-8)

    def _get_metric_stability(self, metric_name: str, run_id: str) -> float:
        """Get historical stability score for a metric"""
        # Get recent critic runs
        critic_runs = self.metric_store.get_critic_runs(limit=5)
        
        # Collect AUC scores for this metric
        auc_scores = []
        for run in critic_runs:
            # Get metric importance from the run's meta
            group = self.metric_store.get_group(run["run_id"])
            if group and group.meta:
                for metric in group.meta.get("metric_importance", []):
                    if metric["metric"] == metric_name:
                        auc_scores.append(metric["auc_roc"])
        
        # Calculate stability (higher = more stable)
        if len(auc_scores) >= 2:
            return 1.0 / (np.std(auc_scores) + 1e-6)
        return 0.5  # Neutral value if not enough history

    def _record_metric_selection(self, run_id: str, selected_metric: str, metric_scores: List[Tuple]) -> None:
        """Record metric selection decision for meta-learning"""
        # Sort scores by composite score
        metric_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Record top metrics
        top_metrics = [{
            "metric": name,
            "score": score,
            "auc": auc,
            "cohens_d": cohens_d,
            "stability": stability
        } for name, score, auc, cohens_d, stability in metric_scores[:5]]
        
        # Update group meta
        meta_patch = {
            "frontier_selection": {
                "selected_metric": selected_metric,
                "top_metrics": top_metrics,
                "timestamp": datetime.now().isoformat()
            }
        }
        self.metric_store.upsert_group_meta(run_id, meta_patch)


    def run_svm_frontier_validation(
        self,
        metric_matrix: np.ndarray,
        y: np.ndarray,
        metric_names: Optional[List[str]] = None,
        *,
        C: float = 1.0,
        class_weight: str = "balanced",
        max_iter: int = 5000,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Validation-only helper:
        Train a linear SVM on per-example metric vectors and report how well
        the frontier space separates baseline vs targeted examples.

        Args:
            metric_matrix: (N, M) matrix of metric values
            y:            (N,) labels in {0, 1} (0=baseline, 1=target)
            metric_names: optional list of length M for feature introspection

        Returns:
            Dict with:
              - type: "LinearSVC"
              - n_samples, n_features
              - train_auc: AUC on the training set (decision_function vs y)
              - hinge_loss: mean hinge loss on training set
              - margin: {mean, std, min, max}
              - support_fraction: fraction of points inside margin (|margin| <= 1)
              - coef: list[float] (flattened weight vector)
              - intercept: float
              - top_features: List[{name, weight, abs_weight}] (if metric_names given)
        """
        # Defensive: need at least 2 classes
        metric_matrix = np.asarray(metric_matrix)
        y = np.asarray(y).astype(int)

        unique = np.unique(y)
        if unique.size < 2:
            log.warning(
                "[FrontierIntelligence] SVM validation skipped: only one class present "
                "(labels=%r)", unique.tolist()
            )
            return {
                "enabled": False,
                "reason": "single_class",
                "n_samples": int(metric_matrix.shape[0]),
                "n_features": int(metric_matrix.shape[1]),
            }

        # Train a simple linear SVM (no persistence)
        svm = LinearSVC(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            dual=False,              # usually better when n_samples > n_features
            random_state=random_state,
        )
        svm.fit(metric_matrix, y)

        # Decision values (larger → more likely class 1)
        decision = svm.decision_function(metric_matrix)

        # AUC on training set (just for sanity)
        try:
            train_auc = float(roc_auc_score(y, decision))
        except Exception:
            train_auc = float("nan")

        # Margins: y ∈ {-1, +1}, margin = y * decision
        y_signed = np.where(y == 1, 1.0, -1.0)
        margins = y_signed * decision

        margin_stats = {
            "mean": float(np.mean(margins)),
            "std": float(np.std(margins)),
            "min": float(np.min(margins)),
            "max": float(np.max(margins)),
        }

        # Hinge loss = max(0, 1 - margin)
        hinge_losses = np.maximum(0.0, 1.0 - margins)
        hinge_loss = float(np.mean(hinge_losses))

        # Approx "support fraction": fraction of points inside margin band
        support_fraction = float(np.mean(margins <= 1.0))

        coef = svm.coef_.ravel()
        intercept = float(svm.intercept_[0])

        top_features: List[Dict[str, Any]] = []
        if metric_names is not None and len(metric_names) == coef.shape[0]:
            abs_coef = np.abs(coef)
            order = np.argsort(-abs_coef)  # descending
            for idx in order[: min(20, coef.shape[0])]:
                top_features.append(
                    {
                        "metric": metric_names[idx],
                        "weight": float(coef[idx]),
                        "abs_weight": float(abs_coef[idx]),
                    }
                )

        result: Dict[str, Any] = {
            "enabled": True,
            "type": "LinearSVC",
            "n_samples": int(metric_matrix.shape[0]),
            "n_features": int(metric_matrix.shape[1]),
            "train_auc": train_auc,
            "hinge_loss": hinge_loss,
            "margin": margin_stats,
            "support_fraction": support_fraction,
            "coef": coef.tolist(),
           "intercept": intercept,
            "decision": decision.tolist(),       # shape (N,)
            "labels": y.tolist(),                # 0 = baseline, 1 = targeted
        }
        if top_features:
            result["top_features"] = top_features

        log.info(
            "[FrontierIntelligence] SVM frontier validation: "
            "n=%d d=%d AUC=%.4f hinge=%.4f margin_mean=%.4f",
            result["n_samples"],
            result["n_features"],
            train_auc,
            hinge_loss,
            margin_stats["mean"],
        )

        return result
