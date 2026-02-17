
from typing import Any, Dict, List, Optional
import numpy as np
from ..governance.signal_extractor import GovernanceSignalExtractor
from ..core.thresholds import CalibratedThresholds
from stephanie.components.elm.tracking.retention_tracker import RetentionTracker


class ScoreBundleExperiment:
    """
    Experimental harness that works WITH your ScoreBundle system.
    
    Leverages existing persistence, logging, and scoring infrastructure.
    """
    
    def __init__(
        self,
        system: Any,  # Your Stephanie engine with ScoreBundle output
        queries: List[Any],
        thresholds: "CalibratedThresholds",
        extractor: GovernanceSignalExtractor,
        episodes: int = 1000,
        perturbation_episode: Optional[int] = None,
        log_to_db: bool = True,  # Use your existing DB persistence
        seed: int = 42
    ):
        self.system = system
        self.queries = queries
        self.thresholds = thresholds
        self.extractor = extractor
        self.episodes = episodes
        self.perturbation_episode = perturbation_episode
        self.log_to_db = log_to_db
        self.seed = seed
        
        # State tracking
        self.episode_history: List[Dict] = []
        self.retention_tracker = RetentionTracker()
        self.collapse_detector = CollapseDetector(thresholds)
        
        np.random.seed(seed)
    
    def run(self) -> Dict[str, Any]:
        """Execute experiment using your ScoreBundle infrastructure"""
        
        for episode in range(self.episodes):
            query = np.random.choice(self.queries)
            
            # Your system returns ScoreBundle
            bundle_before = self.system.evaluate(query)
            
            # Attempt improvement (reflection, retry, etc.)
            improved = self.system.attempt_improvement(query, bundle_before)
            
            if improved:
                bundle_after = improved["bundle"]
                reflection_trace = improved.get("reflection")
                
                # Check dominance using ScoreBundle.diff()
                dominance_achieved = self.extractor.compute_dominance(
                    bundle_before, bundle_after
                )
                
                if dominance_achieved:
                    # Commit improvement (your existing persistence)
                    if self.log_to_db:
                        self.system.commit_improvement(
                            query, bundle_after, reflection_trace
                        )
                    
                    # Extract governance metrics
                    metrics = self.extractor.extract_from_bundle(bundle_after)
                    delta_vector = self.extractor.compute_delta_vector(
                        bundle_before, bundle_after
                    )
                    
                    episode_data = {
                        "episode": episode,
                        "query_id": query.id if hasattr(query, "id") else None,
                        "dominance_achieved": dominance_achieved,
                        "metrics": metrics,
                        "delta_vector": delta_vector,
                        "bundle_before": bundle_before.to_dict(),
                        "bundle_after": bundle_after.to_dict(),
                    }
                    
                    self.episode_history.append(episode_data)
                    self.retention_tracker.update(episode, metrics)
                    
                    # Real-time failure detection
                    failure = self._check_failure(metrics)
                    if failure:
                        return self._build_failure_result(episode, failure)
            
            # Progress logging
            if episode % 100 == 0:
                self._log_progress(episode)
        
        return self._build_success_result()
    
    def _check_failure(self, metrics: Dict[str, float]) -> Optional[str]:
        """Check governance metrics against calibrated thresholds"""
        
        # Energy check
        energy = metrics.get("energy_raw", 0)
        if energy > self.thresholds.energy_max:
            return f"Energy exceeded: {energy:.2f} > {self.thresholds.energy_max:.2f}"
        
        # HRM alignment check
        hrm = metrics.get("hrm_alignment", 1.0)
        if hrm < self.thresholds.hrm_min:
            return f"HRM alignment collapsed: {hrm:.2f} < {self.thresholds.hrm_min:.2f}"
        
        # Embedding margin check
        margin = metrics.get("embedding_margin", 0.0)
        if margin < self.thresholds.margin_min:
            return f"Embedding margin collapsed: {margin:.2f} < {self.thresholds.margin_min:.2f}"
        
        return None
    
    def _build_success_result(self) -> Dict[str, Any]:
        """Aggregate results across all episodes"""
        
        if not self.episode_history:
            return {"status": "failed", "reason": "no episodes completed"}
        
        # Extract metrics arrays
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
        }