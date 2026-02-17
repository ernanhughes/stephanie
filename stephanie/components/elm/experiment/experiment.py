# experiment/experiment.py
from typing import Any, Dict, List, Optional
import numpy as np
import logging
from stephanie.components.elm.core.thresholds import CalibratedThresholds
from stephanie.components.elm.governance.signal_extractor import GovernanceSignalExtractor
from stephanie.components.elm.tracking.retention_tracker import RetentionTracker
from stephanie.components.elm.tracking.collapse_detector import CollapseDetector

logger = logging.getLogger(__name__)

class ScoreBundleExperiment:
    def __init__(
        self,
        system: Any,
        queries: List[Any],
        thresholds: CalibratedThresholds,
        extractor: GovernanceSignalExtractor,
        episodes: int = 1000,
        perturbation_episode: Optional[int] = None,
        log_to_db: bool = True,
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
        logger.info(f"Experiment initialized: {episodes} episodes")

    def run(self) -> Dict[str, Any]:
        for episode in range(self.episodes):
            query = np.random.choice(self.queries)
            bundle_before = self.system.evaluate(query)
            
            improved = self.system.attempt_improvement(query, bundle_before)
            
            if improved:
                bundle_after = improved["bundle"]
                dominance_achieved = self.extractor.compute_dominance(
                    bundle_before, bundle_after
                )
                
                if dominance_achieved:
                    if self.log_to_db:
                        self.system.commit_improvement(query, bundle_after)
                    
                    metrics = self.extractor.extract_from_bundle(bundle_after)
                    self.episode_history.append({
                        "episode": episode,
                        "dominance_achieved": dominance_achieved,
                        "metrics": metrics
                    })
                    
                    self.retention_tracker.update(episode, metrics)
                    
                    failure = self._check_failure(metrics)
                    if failure:
                        return self._build_failure_result(episode, failure)
            
            if episode % 100 == 0:
                self._log_progress(episode)
        
        return self._build_success_result()
    
    def _check_failure(self, metrics: Dict[str, float]) -> Optional[str]:
        energy = metrics.get("energy_raw", 0)
        if energy > self.thresholds.energy_max:
            return f"Energy exceeded: {energy:.2f}"
        
        hrm = metrics.get("hrm_alignment", 1.0)
        if hrm < self.thresholds.hrm_min:
            return f"HRM alignment collapsed: {hrm:.2f}"
        
        margin = metrics.get("embedding_margin", 0.0)
        if margin < self.thresholds.margin_min:
            return f"Embedding margin collapsed: {margin:.2f}"
        
        return None
    
    def _log_progress(self, episode: int):
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
    
    def _build_success_result(self) -> Dict[str, Any]:
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
        }
    
    def _build_failure_result(self, episode: int, failure: str) -> Dict[str, Any]:
        return {
            "status": "failed",
            "episode": episode,
            "failure": failure,
            "metrics_summary": self._build_success_result()["metrics_summary"],
        }