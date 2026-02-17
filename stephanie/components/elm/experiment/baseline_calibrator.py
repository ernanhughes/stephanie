# experiment/baseline_calibrator.py
from typing import Any, List
import numpy as np
from stephanie.components.elm.core.thresholds import CalibratedThresholds
from stephanie.components.elm.governance.signal_extractor import GovernanceSignalExtractor
import logging

logger = logging.getLogger(__name__)

class BaselineCalibrator:
    """Calibrate thresholds using baseline system behavior"""
    
    def __init__(self, baseline_system: Any, extractor: GovernanceSignalExtractor):
        self.baseline = baseline_system
        self.extractor = extractor
    
    def calibrate(
        self,
        queries: List[Any],
        episodes: int = 200
    ) -> CalibratedThresholds:
        """Run baseline and compute statistical thresholds (μ ± 2σ)"""
        
        all_metrics = []
        
        for ep in range(episodes):
            query = np.random.choice(queries)
            bundle = self.baseline.evaluate(query)
            
            metrics = self.extractor.extract_from_bundle(bundle)
            all_metrics.append(metrics)
        
        # Compute statistics
        energies = [m.get("energy_raw", 0) for m in all_metrics]
        hrms = [m.get("hrm_alignment", 0) for m in all_metrics]
        margins = [m.get("embedding_margin", 0) for m in all_metrics]
        
        thresholds = CalibratedThresholds(
            energy_max=np.mean(energies) + 2 * np.std(energies),
            energy_warning=np.mean(energies) + 1 * np.std(energies),
            hrm_min=np.mean(hrms) - 2 * np.std(hrms),
            margin_min=np.mean(margins) - 2 * np.std(margins),
            variance_min=0.3,  # Fixed based on embedding geometry
            collapse_index_max=10.0,  # Fixed based on eigenvalue ratio
            drift_max=0.15,  # Fixed based on angular drift
            baseline_episodes=episodes,
            baseline_system=self.baseline.__class__.__name__,
            statistical_method="mean_plus_2std"
        )
        
        logger.info(f"Calibration complete: {thresholds}")
        return thresholds