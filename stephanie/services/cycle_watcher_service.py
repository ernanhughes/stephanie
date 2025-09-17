# stephanie/services/cycle_watcher.py

from __future__ import annotations

import time
import traceback
from collections import defaultdict, deque
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List, Optional

from stephanie.services.service_protocol import Service


class CycleWatcherService(Service):
    """
    Monitors goal/dimension performance over time to detect stagnation, oscillation, or unstable loops.
    
    As a service, it:
    - Tracks cycle history in memory and database
    - Provides health metrics and statistics
    - Supports persistence across restarts
    - Integrates with the service container
    """

    def __init__(self, cfg: Dict, memory: Any, logger: Any):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.enabled = cfg.get("cycle_watcher", {}).get("enabled", True)
        self._initialized = False
        
        # Configuration parameters
        self.plateau_delta = cfg.get("cycle_watcher", {}).get("plateau_delta", 0.01)
        self.plateau_window = cfg.get("cycle_watcher", {}).get("plateau_window", 5)
        self.oscillation_window = cfg.get("cycle_watcher", {}).get("oscillation_window", 4)
        self.max_history = cfg.get("cycle_watcher", {}).get("max_history", 20)
        
        # Memory storage
        self.history = defaultdict(lambda: deque(maxlen=self.max_history))
        
        # Statistics tracking
        self._stats = {
            "total_records": 0,
            "stuck_count": 0,
            "oscillating_count": 0,
            "stable_count": 0,
            "last_record_time": None,
            "detection_history": []
        }

    @property
    def name(self) -> str:
        return "cycle"

    def initialize(self, **kwargs) -> None:
        """Initialize cycle watcher with proper error handling and logging."""
        if self._initialized or not self.enabled:
            return
            
        start_time = time.time()
        try:
            # Load existing cycle history from database
            self._load_history_from_db()
            
            self._initialized = True
            self._stats["last_updated"] = datetime.now().isoformat()
            
            duration = time.time() - start_time
            self.logger.log("CycleWatcherInitialized", {
                "duration": duration,
                "history_size": len(self.history),
                "max_history": self.max_history
            })
            
        except Exception as e:
            self.logger.log("CycleWatcherInitError", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            raise

    def health_check(self) -> Dict[str, Any]:
        """Return comprehensive health metrics for monitoring."""
        return {
            "status": "healthy" if self._initialized and self.enabled else "disabled",
            "enabled": self.enabled,
            "initialized": self._initialized,
            "history_size": len(self.history),
            "stats": self._stats,
            "config": {
                "plateau_delta": self.plateau_delta,
                "plateau_window": self.plateau_window,
                "oscillation_window": self.oscillation_window,
                "max_history": self.max_history
            },
            "last_updated": self._stats["last_updated"]
        }

    def shutdown(self) -> None:
        """Cleanly shut down the service and save state."""
        try:
            # Save current history to database
            self._save_history_to_db()
            
            # Clear memory
            self.history.clear()
            
            self.logger.log("CycleWatcherShutdown", {
                "status": "complete",
                "total_records": self._stats["total_records"]
            })
        except Exception as e:
            self.logger.log("CycleWatcherShutdownError", {
                "error": str(e)
            })

    def record_agreement(self, goal: str, dimension: str, agreement: float):
        """
        Record an agreement score for a goal/dimension pair.
        
        Args:
            goal: Goal identifier
            dimension: Dimension being monitored
            agreement: Agreement score (0.0-1.0)
        """
        if not self.enabled:
            return
            
        if not self._initialized:
            self.initialize()
            
        key = (goal, dimension)
        
        try:
            # Add to memory history
            self.history[key].append(agreement)
            
            # Save to database
            self._save_record_to_db(goal, dimension, agreement)
            
            # Update statistics
            self._stats["total_records"] += 1
            self._stats["last_record_time"] = datetime.now().isoformat()
            
            # Log the record
            self.logger.log(
                "CycleAgreementTracked",
                {
                    "goal": goal,
                    "dimension": dimension,
                    "agreement": agreement,
                    "history_length": len(self.history[key])
                }
            )
            
        except Exception as e:
            self.logger.log("CycleAgreementError", {
                "goal": goal,
                "dimension": dimension,
                "agreement": agreement,
                "error": str(e)
            })

    def is_stuck(self, goal: str, dimension: str) -> bool:
        """
        Detects a plateau — no net improvement over last N rounds.
        
        Args:
            goal: Goal identifier
            dimension: Dimension to check
            
        Returns:
            True if the goal/dimension is stuck in a plateau
        """
        if not self.enabled or not self._initialized:
            return False
            
        key = (goal, dimension)
        scores = list(self.history[key])[-self.plateau_window:]
        if len(scores) < self.plateau_window:
            return False
            
        result = max(scores) - min(scores) < self.plateau_delta
        if result:
            self._stats["stuck_count"] += 1
            self._stats["detection_history"].append({
                "type": "stuck",
                "goal": goal,
                "dimension": dimension,
                "timestamp": datetime.now().isoformat()
            })
        return result

    def is_oscillating(self, goal: str, dimension: str) -> bool:
        """
        Detects frequent back-and-forth confidence shifts.
        
        Args:
            goal: Goal identifier
            dimension: Dimension to check
            
        Returns:
            True if the goal/dimension is oscillating
        """
        if not self.enabled or not self._initialized:
            return False
            
        key = (goal, dimension)
        scores = list(self.history[key])[-self.oscillation_window:]
        if len(scores) < self.oscillation_window:
            return False
            
        # Count sign changes in delta
        deltas = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
        sign_changes = sum(
            1 for i in range(len(deltas) - 1) if deltas[i] * deltas[i + 1] < 0
        )
        
        result = sign_changes >= 2  # e.g. up-down-up or down-up-down
        if result:
            self._stats["oscillating_count"] += 1
            self._stats["detection_history"].append({
                "type": "oscillating",
                "goal": goal,
                "dimension": dimension,
                "timestamp": datetime.now().isoformat()
            })
        return result

    def status(self, goal: str, dimension: str) -> str:
        """
        Returns current state for a goal+dimension: 'stable', 'stuck', or 'oscillating'
        
        Args:
            goal: Goal identifier
            dimension: Dimension to check
            
        Returns:
            Current status as a string
        """
        if self.is_stuck(goal, dimension):
            return "stuck"
        if self.is_oscillating(goal, dimension):
            return "oscillating"
        self._stats["stable_count"] += 1
        return "stable"

    def get_history(self, goal: Optional[str] = None, dimension: Optional[str] = None) -> List[Dict]:
        """
        Get historical cycle data.
        
        Args:
            goal: Optional goal filter
            dimension: Optional dimension filter
            
        Returns:
            List of historical records
        """
        if not self._initialized:
            self.initialize()
            
        return self.memory.cycle_watcher.get_history(goal, dimension)

    def clear_history(self, goal: Optional[str] = None, dimension: Optional[str] = None):
        """
        Clear cycle history, optionally filtered by goal or dimension.
        
        Args:
            goal: Optional goal to filter
            dimension: Optional dimension to filter
        """
        if goal and dimension:
            key = (goal, dimension)
            if key in self.history:
                del self.history[key]
                self.memory.cycle_watcher.delete_records(goal, dimension)
        elif goal:
            keys_to_remove = [k for k in self.history.keys() if k[0] == goal]
            for key in keys_to_remove:
                del self.history[key]
            self.memory.cycle_watcher.delete_records(goal=goal)
        else:
            self.history.clear()
            self.memory.cycle_watcher.delete_all_records()
            
        self.logger.log("CycleHistoryCleared", {
            "goal": goal,
            "dimension": dimension,
            "records_cleared": len(keys_to_remove) if goal and dimension else "all"
        })

    def _load_history_from_db(self):
        """Load cycle history from database on initialization."""
        try:
            history = self.memory.cycle_watcher.get_all_records()
            for record in history:
                goal = record["goal"]
                dimension = record["dimension"]
                agreement = record["agreement"]
                key = (goal, dimension)
                self.history[key].append(agreement)
                
            self.logger.log("CycleHistoryLoaded", {
                "record_count": len(history),
                "unique_pairs": len(set((r["goal"], r["dimension"]) for r in history))
            })
        except Exception as e:
            self.logger.log("CycleHistoryLoadError", {
                "error": str(e)
            })

    def _save_history_to_db(self):
        """Save current cycle history to database."""
        try:
            # This could be optimized for bulk inserts
            for (goal, dimension), scores in self.history.items():
                for agreement in scores:
                    self._save_record_to_db(goal, dimension, agreement)
                    
            self.logger.log("CycleHistorySaved", {
                "record_count": self._stats["total_records"]
            })
        except Exception as e:
            self.logger.log("CycleHistorySaveError", {
                "error": str(e)
            })

    def _save_record_to_db(self, goal: str, dimension: str, agreement: float):
        """Save a single cycle record to database."""
        try:
            record = {
                "goal": goal,
                "dimension": dimension,
                "agreement": agreement,
                "timestamp": datetime.now().isoformat()
            }
            self.memory.cycle_watcher.insert(record)
        except Exception as e:
            self.logger.log("CycleRecordSaveError", {
                "goal": goal,
                "dimension": dimension,
                "agreement": agreement,
                "error": str(e)
            })

    def get_detection_history(self, limit: int = 100) -> List[Dict]:
        """Get recent cycle detection events (stuck/oscillating)."""
        return self._stats["detection_history"][-limit:]

    def get_cycle_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about cycle patterns."""
        if not self._initialized:
            self.initialize()
            
        total = self._stats["stuck_count"] + self._stats["oscillating_count"] + self._stats["stable_count"]
        
        return {
            "total_records": self._stats["total_records"],
            "stuck_count": self._stats["stuck_count"],
            "oscillating_count": self._stats["oscillating_count"],
            "stable_count": self._stats["stable_count"],
            "stuck_ratio": self._stats["stuck_count"] / max(1, total),
            "oscillating_ratio": self._stats["oscillating_count"] / max(1, total),
            "average_history_length": mean(len(h) for h in self.history.values()) if self.history else 0,
            "unique_goal_dimension_pairs": len(self.history)
        }