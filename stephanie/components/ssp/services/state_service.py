# stephanie/components/ssp/services/state_service.py
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional, List

from stephanie.services.service_protocol import Service


class StateService(Service):
    """
    Centralized state management service for the Self-Play System (SSP).

    Tracks:
    - Active episodes (proposal → solve → verify lifecycle)
    - Curriculum difficulty & success history
    - Per-dimension performance metrics
    - Component status and metadata
    - Ring buffers for recent activity

    Provides:
    - Real-time observability into SSP progress
    - Safe access to shared state for actors (Proposer, Solver, Verifier)
    - Historical data for adaptive control (e.g., Q-Max logic)
    - Integration points for UIs, dashboards, and logging.
    """

    def __init__(self, cfg: Dict[str, Any], memory, logger, container=None):
        self.cfg = cfg or {}
        self.memory = memory
        self.logger = logger
        self.container = container
        self._initialized = False

        # --- Core State ---
        # General configuration and component status
        self.state: Dict[str, Any] = {
            "episode_count": 0,
            "current_difficulty": 0.3,
            "success_rate": 0.0,
            "last_episode_id": None,
            "status": "idle",  # 'idle', 'running', 'paused'
            "start_time": None,
            "last_update": time.time(),
        }

        # Ring buffer of recent episode outcomes (1=success, 0=failure) for curriculum
        window_size = int(self.cfg.get("state", {}).get("competence_window", 100))
        self.success_history: deque[int] = deque(maxlen=window_size)

        # Track active proposals/solutions per goal or session
        self.active_episodes: Dict[str, Dict[str, Any]] = {}

        # Per-dimension scoring trends (simple moving average)
        self.dimension_scores: Dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=50)
        )

        # Recent queries proposed (for novelty checks)
        self.recent_queries: deque[str] = deque(maxlen=200)

        # --- Configuration ---
        c = self.cfg.get("state", {})
        self._buffer_recent_traces = bool(c.get("enable_trace_buffer", True))
        self._max_trace_buffer = int(c.get("trace_buffer_size", 1000))
        self._log_state_updates = bool(c.get("log_state_changes", True))

        # Trace/event ring buffer if enabled
        self._recent_events: Optional[deque[Dict[str, Any]]] = (
            deque(maxlen=self._max_trace_buffer)
            if self._buffer_recent_traces
            else None
        )

    # === Service Protocol ===

    def initialize(self, **kwargs) -> None:
        """Initialize the state service."""
        if self._initialized:
            return

        self._initialized = True
        self.state["start_time"] = time.time()
        self.state["status"] = "idle"

        if self.logger:
            self.logger.log(
                "StateServiceInit",
                {"status": "initialized", "config": self.cfg.get("state", {})}
            )

        # Optionally persist initial state
        if self.memory:
            try:
                self.memory.save("state_service", {"service_state": dict(self.state)})
            except Exception as e:
                if self.logger:
                    self.logger.log(
                        "StateServiceWarning",
                        {"event": "init_persist_failed", "error": str(e)}
                    )

    def health_check(self) -> Dict[str, Any]:
        """Return service health and summary of tracked state."""
        status = "healthy" if self._initialized else "unhealthy"
        now = time.time()

        return {
            "status": status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "metrics": {
                "episode_count": self.state["episode_count"],
                "current_difficulty": self.state["current_difficulty"],
                "recent_success_rate": self.state["success_rate"],
                "active_episodes": len(self.active_episodes),
                "tracked_dimensions": len(self.dimension_scores),
                "recent_query_count": len(self.recent_queries),
            },
            "dependencies": {
                "memory": self.memory is not None,
                "logger": self.logger is not None,
                "container": self.container is not None,
            },
            "last_update": self.state["last_update"],
        }

    def shutdown(self) -> None:
        """Clear all state and mark as uninitialized."""
        self.state.clear()
        self.success_history.clear()
        self.active_episodes.clear()
        self.dimension_scores.clear()
        self.recent_queries.clear()
        if self._recent_events:
            self._recent_events.clear()

        self._initialized = False

        if self.logger:
            self.logger.log("StateServiceShutdown", {"event": "shutdown_complete"})

    @property
    def name(self) -> str:
        return "ssp-state-service"

    # === Domain Logic - Episode Management ===

    def start_episode(self, episode_id: str, proposal: Dict[str, Any]) -> None:
        """Record the start of a new SSP episode."""
        now = time.time()
        self.active_episodes[episode_id] = {
            "episode_id": episode_id,
            "proposal": proposal,
            "started_at": now,
            "status": "proposed",
            "difficulty": proposal.get("difficulty", 0.3),
        }
        self.state["last_episode_id"] = episode_id
        self.state["episode_count"] += 1
        self.state["last_update"] = now

        self._log_event("episode_started", {"episode_id": episode_id})
        self._persist_state_snapshot()

    def update_episode_solution(self, episode_id: str, solution: Dict[str, Any]) -> None:
        """Update an episode with its solution."""
        if episode_id not in self.active_episodes:
            return

        self.active_episodes[episode_id].update({
            "solution": solution,
            "solved_at": time.time(),
            "status": "solved"
        })
        self.state["last_update"] = time.time()

        # Extract and record query for novelty checks
        query = solution.get("reasoning_path", [{}])[0].get("description", "")
        if query.strip():
            self.add_recent_query(query.strip())

        self._log_event("episode_solved", {"episode_id": episode_id})
        self._persist_state_snapshot()

    def complete_episode(
        self, episode_id: str, verification: Dict[str, Any], metrics: Dict[str, float]
    ) -> None:
        """Finalize an episode with verification results."""
        if episode_id not in self.active_episodes:
            return

        ep = self.active_episodes[episode_id]
        now = time.time()
        success = bool(verification.get("is_valid", False))

        ep.update({
            "verification": verification,
            "metrics": metrics,
            "completed_at": now,
            "success": success,
            "status": "verified" if success else "failed"
        })

        # Update global success history and rate
        self.success_history.append(1 if success else 0)
        self.state["success_rate"] = (
            sum(self.success_history) / len(self.success_history)
            if self.success_history else 0.0
        )

        # Update dimension scores
        dim_scores = verification.get("dimension_scores", {})
        for dim, score in dim_scores.items():
            self.dimension_scores[dim].append(float(score))

        self.state["current_difficulty"] = metrics.get("difficulty", self.state["current_difficulty"])
        self.state["last_update"] = now

        self._log_event("episode_completed", {
            "episode_id": episode_id,
            "success": success,
            "difficulty": self.state["current_difficulty"],
            "success_rate": self.state["success_rate"]
        })
        self._persist_state_snapshot()

    def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the full state of a specific episode."""
        return self.active_episodes.get(episode_id)

    def get_recent_episodes(self, limit: int = 10, success: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get a list of recently completed episodes, optionally filtered by success."""
        completed = [
            ep for ep in self.active_episodes.values()
            if ep.get("status") in ("verified", "failed")
        ]
        # Sort by completion time (assuming 'completed_at' exists)
        completed.sort(key=lambda x: x.get("completed_at", 0), reverse=True)
        
        if success is not None:
            completed = [ep for ep in completed if ep.get("success") == success]

        return completed[:limit]

    # === Domain Logic - Curriculum & Metrics ===

    def get_curriculum_state(self) -> Dict[str, Any]:
        """Get current curriculum parameters."""
        return {
            "current_difficulty": self.state["current_difficulty"],
            "success_rate": self.state["success_rate"],
            "success_history": list(self.success_history),
            "competence_window": self.success_history.maxlen,
            "target_success_rate": 0.7,  # Could be config-driven
        }

    def update_curriculum(self, new_difficulty: float, success: bool) -> None:
        """Update the curriculum state from trainer feedback."""
        self.state["current_difficulty"] = float(new_difficulty)
        self.state["last_update"] = time.time()
        # The Trainer already updates success_history; we just reflect it
        self.state["success_rate"] = (
            sum(self.success_history) / len(self.success_history)
            if self.success_history else 0.0
        )
        self._log_event("curriculum_updated", {
            "new_difficulty": new_difficulty,
            "success_rate": self.state["success_rate"]
        })

    def get_dimension_trend(self, dimension: str) -> Dict[str, Any]:
        """Get the recent score history for a specific dimension."""
        scores = list(self.dimension_scores[dimension])
        return {
            "dimension": dimension,
            "recent_scores": scores,
            "average_score": sum(scores) / len(scores) if scores else 0.5,
            "count": len(scores),
            "trend": "increasing" if len(scores) > 1 and scores[-1] > scores[0] else "stable"
        }

    # === Domain Logic - Query & Context Management ===

    def add_recent_query(self, query: str) -> None:
        """Add a query to the recent history for novelty checks."""
        if query.strip():
            self.recent_queries.append(query.strip())
            self._log_event("query_added", {"query_length": len(query)})

    def get_ring_buffer(self, key: str, maxlen: int = 100) -> deque:
        """
        Provide access to internal ring buffers.
        Used by actors (e.g., Proposer) for lightweight state sharing.
        """
        if key == "ssp.recent_queries":
            # Ensure the buffer has the requested maxlen
            if len(self.recent_queries) > maxlen:
                # Trim from the left if necessary
                while len(self.recent_queries) > maxlen:
                    self.recent_queries.popleft()
            return self.recent_queries
        # Add other buffers here as needed
        return deque(maxlen=maxlen)

    # === Internal Helpers ===

    def _log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Log a state change event if configured."""
        if not self._log_state_updates or not self.logger:
            return

        full_payload = {
            "event": event_type,
            "timestamp": time.time(),
            "service": self.name,
            **payload
        }
        self.logger.log(f"StateServiceEvent", extra=full_payload)

        # Add to in-memory event ring buffer
        if self._recent_events is not None:
            self._recent_events.append(full_payload)

    def _persist_state_snapshot(self) -> None:
        """Persist the current state snapshot to memory if available."""
        if not self.memory or not hasattr(self.memory, "save"):
            return

        try:
            snapshot = {
                "state": dict(self.state),
                "success_history": list(self.success_history),
                "active_episodes_count": len(self.active_episodes),
                "last_update": self.state["last_update"]
            }
            # Use a stable key
            self.memory.save("ssp_global_state", snapshot)
        except Exception as e:
            if self.logger:
                self.logger.log(
                    "StateServiceWarning",
                    {"event": "persist_snapshot_failed", "error": str(e)}
                )

    # === Debugging ===

    def __repr__(self):
        active_count = len(self.active_episodes)
        recent_success = f"{self.state['success_rate']:.2%}"
        return (
            f"<StateService: status={self.state['status']} "
            f"episodes={self.state['episode_count']} "
            f"active={active_count} "
            f"difficulty={self.state['current_difficulty']:.2f} "
            f"success_rate={recent_success}>"
        )