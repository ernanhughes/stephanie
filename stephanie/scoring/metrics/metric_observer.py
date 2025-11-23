# stephanie/scoring/metrics/metric_observer.py
# stephanie/scoring/metrics/metric_observer.py
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

@dataclass
class MetricStats:
    name: str
    count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')
    
    def update(self, value: float) -> None:
        self.count += 1
        self.last_seen = time.time()
        if self.count == 1:
            self.first_seen = self.last_seen
            
        try:
            val = float(value)
            if val < self.min_val:
                self.min_val = val
            if val > self.max_val:
                self.max_val = val
        except (TypeError, ValueError):
            pass

class MetricObserver:
    """Dynamically observes metrics as they flow through the system"""
    
    def __init__(self, enabled: bool = True, snapshot_path: Optional[str] = None):
        """Initialize the metric observer.
        
        Args:
            enabled: Whether metric observation is active
            snapshot_path: Path to save metric snapshots (optional)
        """
        self.enabled = enabled
        self.snapshot_path = snapshot_path
        self._lock = threading.Lock()
        self._metrics: Dict[str, MetricStats] = {}
        self._runs: Dict[str, List[str]] = {}  # run_id -> list of metrics
    
    def observe(
        self,
        metrics: Dict[str, float],
        run_id: Optional[str] = None,
        cohort: Optional[str] = None,
        is_correct: Optional[bool] = None
    ) -> None:
        """Record metrics from a single scorable"""
        if not self.enabled:
            return
            
        with self._lock:
            for name, value in metrics.items():
                if name not in self._metrics:
                    self._metrics[name] = MetricStats(name=name)
                self._metrics[name].update(value)
                
                if run_id:
                    if run_id not in self._runs:
                        self._runs[run_id] = []
                    if name not in self._runs[run_id]:
                        self._runs[run_id].append(name)
    
    def get_stable_core(
        self,
        min_runs: int = 5,
        stability_threshold: float = 0.6
    ) -> List[Tuple[str, float]]:
        """Find metrics that consistently distinguish good from bad reasoning"""
        if not self.enabled or not self._metrics:
            return []
            
        stable_metrics = []
        
        for metric_name, stats in self._metrics.items():
            # Only consider metrics that appeared in enough runs
            if metric_name not in self._runs or len(self._runs[metric_name]) < min_runs:
                continue
                
            # Calculate stability
            correct_runs = 0
            total_runs = 0
            
            # In a real implementation, you'd use your visicalc_metric_importance logic here
            # This is a simplified version for illustration
            for run_id in self._runs[metric_name]:
                # You would have stored is_correct per run somewhere
                # This is just a placeholder
                total_runs += 1
                # correct_runs += 1 if is_correct else 0
            
            if total_runs > 0:
                stability = correct_runs / total_runs
                if stability >= stability_threshold:
                    stable_metrics.append((metric_name, stability))
        
        # Sort by stability
        return sorted(stable_metrics, key=lambda x: -x[1])
    
    def save_snapshot(self) -> None:
        """Save current metric universe to disk"""
        if not self.enabled or not self.snapshot_path:
            return
            
        try:
            snapshot = {
                "metrics": {name: asdict(stats) for name, stats in self._metrics.items()},
                "runs": self._runs
            }
            
            path = Path(self.snapshot_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "w") as f:
                json.dump(snapshot, f, indent=2)
                
            log.info(f"Saved metric observer snapshot to {path}")
        except Exception as e:
            log.error(f"Failed to save metric observer snapshot: {str(e)}")