# stephanie/agents/paper_improver/bandit_router.py

# bandit_router.py — UCB1/Thompson Sampling for exemplar selection based on historical uplift.
# Logs plays, rewards, saves state. Routes by spec type or claim density.

import json
import math
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

class ExemplarBandit:
    """
    Multi-armed bandit for routing to best exemplar pack.
    Tracks uplift in target metric (e.g., pass_rate, coverage).
    Supports UCB1 and Thompson Sampling.
    Persists state to disk.
    """

    def __init__(
        self,
        save_path: str = "./bandit_state.json",
        strategy: str = "ucb1",  # or "thompson"
        reward_metric: str = "pass_rate",  # or "coverage", etc.
        default_reward: float = 0.5,
        smoothing: float = 1.0  # for Thompson prior
    ):
        self.save_path = Path(save_path)
        self.strategy = strategy
        self.reward_metric = reward_metric
        self.default_reward = default_reward
        self.smoothing = smoothing
        self.arms: Dict[str, Dict[str, Union[int, float]]] = {}
        self.total_plays = 0
        self._load_state()

    def choose(self, candidate_ids: List[str], context: Optional[Dict[str, Any]] = None) -> str:
        """
        Select best exemplar_id given candidates.
        Context can be spec type, claim count, etc. — ignored for now (future: contextual bandit).
        """
        if not candidate_ids:
            raise ValueError("No candidate exemplar IDs provided.")

        # Initialize unseen arms
        for eid in candidate_ids:
            if eid not in self.arms:
                self.arms[eid] = {
                    "plays": 0,
                    "reward_sum": 0.0,
                    "reward_avg": self.default_reward
                }

        if self.strategy == "ucb1":
            return self._choose_ucb1(candidate_ids)
        elif self.strategy == "thompson":
            return self._choose_thompson(candidate_ids)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _choose_ucb1(self, candidate_ids: List[str]) -> str:
        """UCB1: balance exploration + exploitation."""
        total_plays = max(1, sum(self.arms[eid]["plays"] for eid in candidate_ids))
        scores = []
        for eid in candidate_ids:
            arm = self.arms[eid]
            if arm["plays"] == 0:
                scores.append((float('inf'), eid))
            else:
                avg = arm["reward_avg"]
                confidence = math.sqrt(2 * math.log(total_plays) / arm["plays"])
                ucb = avg + confidence
                scores.append((ucb, eid))
        return max(scores, key=lambda x: x[0])[1]

    def _choose_thompson(self, candidate_ids: List[str]) -> str:
        """Thompson Sampling: sample from Beta posterior."""
        samples = []
        for eid in candidate_ids:
            arm = self.arms[eid]
            # Beta(alpha, beta) ~ Beta(successes + smoothing, failures + smoothing)
            successes = arm["reward_sum"] * arm["plays"]  # approximate
            failures = arm["plays"] - successes
            alpha = successes + self.smoothing
            beta = failures + self.smoothing
            sample = random.betavariate(alpha, beta)
            samples.append((sample, eid))
        return max(samples, key=lambda x: x[0])[1]

    def update(self, exemplar_id: str, reward: float):
        """
        Update arm with observed reward (e.g., pass_rate delta, coverage delta).
        Reward should be 0.0 – 1.0.
        """
        if exemplar_id not in self.arms:
            self.arms[exemplar_id] = {
                "plays": 0,
                "reward_sum": 0.0,
                "reward_avg": self.default_reward
            }

        arm = self.arms[exemplar_id]
        arm["plays"] += 1
        arm["reward_sum"] += reward
        arm["reward_avg"] = arm["reward_sum"] / arm["plays"]

        self.total_plays += 1
        self._save_state()

    def get_stats(self, exemplar_id: str) -> Dict[str, Any]:
        """Get current stats for an arm."""
        return self.arms.get(exemplar_id, {
            "plays": 0,
            "reward_avg": self.default_reward,
            "reward_sum": 0.0
        })

    def get_leaderboard(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return top-k arms by reward_avg."""
        scored = [
            {"exemplar_id": eid, **stats}
            for eid, stats in self.arms.items()
        ]
        return sorted(scored, key=lambda x: x["reward_avg"], reverse=True)[:top_k]

    def _load_state(self):
        """Load bandit state from disk."""
        if self.save_path.exists():
            try:
                data = json.loads(self.save_path.read_text())
                self.arms = data.get("arms", {})
                self.total_plays = data.get("total_plays", 0)
                print(f"📊 Bandit state loaded: {len(self.arms)} arms, {self.total_plays} total plays.")
            except Exception as e:
                print(f"⚠️ Failed to load bandit state: {e}")

    def _save_state(self):
        """Persist bandit state to disk."""
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "arms": self.arms,
                "total_plays": self.total_plays,
                "strategy": self.strategy,
                "reward_metric": self.reward_metric
            }
            self.save_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"⚠️ Failed to save bandit state: {e}")

    def reset(self):
        """Reset bandit state."""
        self.arms = {}
        self.total_plays = 0
        if self.save_path.exists():
            self.save_path.unlink()
        print("🔄 Bandit state reset.")

    def __repr__(self):
        return f"<ExemplarBandit strategy={self.strategy} arms={len(self.arms)} plays={self.total_plays}>"