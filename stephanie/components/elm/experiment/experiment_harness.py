# experiment/experiment_harness.py
from typing import Any, Dict, List, Optional
import numpy as np
from stephanie.components.elm.core.thresholds import CalibratedThresholds
from stephanie.components.elm.governance.signal_extractor import GovernanceSignalExtractor
from stephanie.components.elm.tracking.retention_tracker import RetentionTracker
from stephanie.components.elm.tracking.collapse_detector import CollapseDetector
from stephanie.components.elm.experiment.perturbation_injector import PerturbationInjector
import logging

logger = logging.getLogger(__name__)

class ScoreBundleExperiment:
    """Experimental harness for governed self-improvement"""
    
    def __init__(
        self,
        system: Any,
        queries: List[Any],
        thresholds: CalibratedThresholds,
        extractor: GovernanceSignalExtractor,
        episodes: int = 1000,
        perturbation_episode: Optional[int] = None,
        perturbation_severity: str = "moderate",
        log_to_db: bool = True,
        seed: int = 42
    ):
        self.system = system
        self.queries = queries
        self.thresholds = thresholds
        self.extractor = extractor
        self.episodes = episodes
        self.perturbation_episode = perturbation_episode
        self.perturbation_severity = perturbation_severity
        self.log_to_db = log_to_db
        self.seed = seed
        
        # State tracking
        self.episode_history: List[Dict] = []
        self.retention_tracker = RetentionTracker()
        self.collapse_detector = CollapseDetector(thresholds)
        self.perturbation_injector = PerturbationInjector(system)
        
        np.random.seed(seed)
        logger.info(f"Experiment initialized: {episodes} episodes, seed={seed}")
    
    def run(self) -> Dict[str, Any]:
        """Execute full experiment"""
        
        for episode in range(self.episodes):
            # Check for perturbation injection
            if (self.perturbation_episode is not None and 
                episode == self.perturbation_episode):
                self.perturbation_injector.inject(self.perturbation_severity)
                logger.info(f"Perturbation injected at episode {episode}")
            
            # Sample query and evaluate
            query = np.random.choice(self.queries)
            bundle_before = self.system.evaluate(query)
            
            # Attempt improvement
            improved = self.system.attempt_improvement(query, bundle_before)
            
            if improved:
                bundle_after = improved["bundle"]
                reflection_trace = improved.get("reflection")
                
                # Check dominance
                dominance_achieved = self.extractor.compute_dominance(
                    bundle_before, bundle_after
                )
                
                if dominance_achieved:
                    # Commit improvement
                    if self.log_to_db:
                        self.system.commit_improvement(
                            query, bundle_after, reflection_trace
                        )
                    
                    # Extract metrics
                    metrics = self.extractor.extract_from_bundle(bundle_after)
                    delta_vector = self.extractor.compute_delta_vector(
                        bundle_before, bundle_after
                    )
                    
                    # Log episode
                    episode_data = {
                        "episode": episode,
                        "query_id": getattr(query, "id", None),
                        "dominance_achieved": dominance_achieved,
                        "metrics": metrics,
                        "delta_vector": delta_vector,
                    }
                    self.episode_history.append(episode_data)
                    
                    # Update retention tracking
                    self.retention_tracker.update(episode, metrics)
                    
                    # Check for collapse
                    failure = self.collapse_detector.check_failure(episode, metrics)
                    if failure and failure.severity == "critical":
                        logger.critical(f"COLLAPSE DETECTED: {failure}")
                        return self._build_failure_result(episode, failure)
            
            # Progress logging
            if episode % 100 == 0:
                self._log_progress(episode)
        
        return self._build_success_result()
    
    def _log_progress(self, episode: int):
        """Log progress summary"""
        if not self.episode_history:
            return
        
        recent = self.episode_history[-100:] if len(self.episode_history) >= 100 else self.episode_history
        energies = [ep["metrics"].get("energy_raw", 0) for ep in recent]
        dominances = [ep["dominance_achieved"] for ep in recent]
        
        logger.info(
            f"Episode {episode}/{self.episodes} | "
            f"Energy: {np.mean(energies):.3f} | "
            f"Dominance: {np.mean(dominances):.2%}"
        )
    
    def _check_failure(self, metrics: Dict[str, float]) -> Optional[str]:
        """Check governance metrics against thresholds"""
        energy = metrics.get("energy_raw", 0)
        if energy > self.thresholds.energy_max:
            return f"Energy exceeded: {energy:.2f} > {self.thresholds.energy_max:.2f}"
        
        hrm = metrics.get("hrm_alignment", 1.0)
        if hrm < self.thresholds.hrm_min:
            return f"HRM alignment collapsed: {hrm:.2f} < {self.thresholds.hrm_min:.2f}"
        
        margin = metrics.get("embedding_margin", 0.0)
        if margin < self.thresholds.margin_min:
            return f"Embedding margin collapsed: {margin:.2f} < {self.thresholds.margin_min:.2f}"
        
        return None
    
    def _build_success_result(self) -> Dict[str, Any]:
        """Build success result dictionary"""
        if not self.episode_history:
            return {"status": "failed", "reason": "no episodes completed"}
        
        energies = [ep["metrics"].get("energy_raw", 0) for ep in self.episode_history]
        hrms = [ep["metrics"].get("hrm_alignment", 0) for ep in self.episode_history]
        dominances = [ep["dominance_achieved"] for ep in self.episode_history]
        
        return {
            "status": "success",
            "episodes_completed": len(self.episode_history),
            "metrics_summary": {
                "energy": {
                    "mean": float(np.mean(energies)),
                    "std": float(np.std(energies)),
                    "min": float(np.min(energies)),
                    "max": float(np.max(energies)),
                },
                "hrm_alignment": {
                    "mean": float(np.mean(hrms)),
                    "std": float(np.std(hrms)),
                },
                "dominance_rate": float(np.mean(dominances)),
            },
            "retention_scores": self.retention_tracker.get_scores(),
            "failure_history": self.collapse_detector.get_failure_history(),
        }
    
    def _build_failure_result(self, episode: int, failure: Any) -> Dict[str, Any]:
        """Build failure result dictionary"""
        return {
            "status": "failed",
            "episode": episode,
            "failure": failure.to_dict() if hasattr(failure, "to_dict") else str(failure),
            "metrics_summary": self._build_success_result()["metrics_summary"],
        }