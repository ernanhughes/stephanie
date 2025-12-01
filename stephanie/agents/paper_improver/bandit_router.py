# stephanie/agents/paper_improver/bandit_router.py
# Bandit for exemplar selection: UCB1 / Thompson / epsilon-greedy, with decay, context, and robust state.

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------- data models ----------

@dataclass
class ArmStats:
    plays: int = 0
    reward_sum: float = 0.0  # sum of rewards in [0,1]
    reward_avg: float = 0.5
    last_updated: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ---------- bandit ----------

class ExemplarBandit:
    """
    Multi-armed bandit for routing to best exemplar pack.
    Strategies: "ucb1", "thompson", "epsilon".
    Features: decay, contextual keys, atomic persistence, debug scores.
    """

    def __init__(
        self,
        save_path: str = "./bandit_state.json",
        strategy: str = "ucb1",          # "ucb1" | "thompson" | "epsilon"
        reward_metric: str = "pass_rate",# used by compute_reward helper (optional)
        default_reward: float = 0.5,
        smoothing: float = 1.0,          # Thompson prior α=β=smoothing
        epsilon: float = 0.05,           # for epsilon-greedy
        decay_gamma: float = 1.0,        # 1.0 => no decay; else EMA on updates
        seed: int = 0
    ):
        self.save_path = Path(save_path)
        self.strategy = strategy
        self.reward_metric = reward_metric
        self.default_reward = float(default_reward)
        self.smoothing = float(smoothing)
        self.epsilon = float(epsilon)
        self.decay_gamma = float(decay_gamma)
        self.total_plays = 0
        self.arms: Dict[str, ArmStats] = {}
        random.seed(seed)
        self._load_state()

    # ---------- selection ----------

    def choose(self, candidate_ids: List[str], context: Optional[Dict[str, Any]] = None) -> str:
        """Return best arm ID under current strategy."""
        if not candidate_ids:
            raise ValueError("No candidate exemplar IDs provided.")

        keys = [self._key(eid, context) for eid in candidate_ids]
        for k in keys:
            if k not in self.arms:
                self.arms[k] = ArmStats(reward_avg=self.default_reward)

        if self.strategy == "ucb1":
            return self._choose_ucb1(keys)
        if self.strategy == "thompson":
            return self._choose_thompson(keys)
        if self.strategy == "epsilon":
            return self._choose_epsilon(keys)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def choose_with_scores(self, candidate_ids: List[str], context: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, float]]:
        """Debug helper: also return scores per candidate."""
        if not candidate_ids:
            raise ValueError("No candidate exemplar IDs provided.")
        keys = [self._key(eid, context) for eid in candidate_ids]
        for k in keys:
            if k not in self.arms:
                self.arms[k] = ArmStats(reward_avg=self.default_reward)

        scores: Dict[str, float] = {}
        if self.strategy == "ucb1":
            total_plays = max(1, sum(self.arms[k].plays for k in keys))
            for k in keys:
                a = self.arms[k]
                score = float("inf") if a.plays == 0 else a.reward_avg + math.sqrt(2 * math.log(total_plays) / a.plays)
                scores[k] = score
        elif self.strategy == "thompson":
            for k in keys:
                a = self.arms[k]
                # Fractional successes; Beta(α,β) with α = successes + smoothing, β = failures + smoothing
                successes = a.reward_sum                      # sum of rewards
                failures  = max(0.0, a.plays - successes)     # assumes reward ∈ [0,1]
                alpha = successes + self.smoothing
                beta  = failures  + self.smoothing
                # Expected value of Beta(α,β)
                scores[k] = alpha / (alpha + beta)
        else:  # epsilon
            for k in keys:
                scores[k] = self.arms[k].reward_avg

        best = max(scores.items(), key=lambda kv: kv[1])[0]
        return best, scores

    # ---------- updates ----------

    def update(self, exemplar_id: str, reward: float, context: Optional[Dict[str, Any]] = None):
        """
        Record the outcome for an arm. Reward is clamped to [0,1].
        Decay (if γ<1) applies EMA toward new reward: reward_sum, reward_avg.
        """
        k = self._key(exemplar_id, context)
        if k not in self.arms:
            self.arms[k] = ArmStats(reward_avg=self.default_reward)

        r = max(0.0, min(1.0, float(reward)))
        a = self.arms[k]

        # Decayed update for non-stationarity: EMA on avg; sum kept for Thompson
        if self.decay_gamma < 1.0 and a.plays > 0:
            a.reward_avg = (1 - self.decay_gamma) * r + self.decay_gamma * a.reward_avg
        else:
            # standard incremental avg
            a.reward_avg = (a.reward_sum + r) / (a.plays + 1)

        a.reward_sum += r  # keep exact sum for Thompson
        a.plays += 1
        a.last_updated = time.time()
        self.total_plays += 1
        self._save_state()

    # ---------- helpers ----------

    def get_stats(self, exemplar_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        k = self._key(exemplar_id, context)
        a = self.arms.get(k, ArmStats(reward_avg=self.default_reward))
        return {
            "exemplar_id": exemplar_id,
            "key": k,
            **a.as_dict()
        }

    def get_leaderboard(self, top_k: int = 10) -> List[Dict[str, Any]]:
        entries = [{"key": k, **a.as_dict()} for k, a in self.arms.items()]
        entries.sort(key=lambda e: e["reward_avg"], reverse=True)
        return entries[:max(0, top_k)]

    def reset(self):
        self.arms.clear()
        self.total_plays = 0
        if self.save_path.exists():
            self.save_path.unlink()

    # Compute a reward from two VPM rows (before/after) on the selected metric.
    # Reward is Δmetric, shifted to [0,1] by mapping [-1, +1] → [0,1] (clamped).
    def compute_reward(self, before: Dict[str, float], after: Dict[str, float], metric: Optional[str] = None) -> float:
        m = metric or self.reward_metric
        b = float(before.get(m, 0.0))
        a = float(after.get(m, 0.0))
        delta = max(-1.0, min(1.0, a - b))
        return 0.5 + 0.5 * delta

    # ---------- internal selection impls ----------

    def _choose_ucb1(self, keys: List[str]) -> str:
        total = max(1, sum(self.arms[k].plays for k in keys))
        best_key, best_score = None, -float("inf")
        for k in keys:
            a = self.arms[k]
            if a.plays == 0:
                return k  # explore cold-start immediately
            conf = math.sqrt(2 * math.log(total) / a.plays)
            score = a.reward_avg + conf
            if score > best_score:
                best_key, best_score = k, score
        return best_key  # type: ignore[return-value]

    def _choose_thompson(self, keys: List[str]) -> str:
        best_key, best_draw = None, -1.0
        for k in keys:
            a = self.arms[k]
            successes = a.reward_sum                      # sum of rewards ∈ [0, plays]
            failures  = max(0.0, a.plays - successes)
            alpha = successes + self.smoothing
            beta  = failures  + self.smoothing
            draw = random.betavariate(alpha, beta)
            if draw > best_draw:
                best_key, best_draw = k, draw
        return best_key  # type: ignore[return-value]

    def _choose_epsilon(self, keys: List[str]) -> str:
        if random.random() < self.epsilon:
            return random.choice(keys)
        return max(keys, key=lambda k: self.arms[k].reward_avg)

    # ---------- persistence ----------

    def _load_state(self):
        if not self.save_path.exists():
            return
        try:
            data = json.loads(self.save_path.read_text())
            self.total_plays = int(data.get("total_plays", 0))
            self.strategy = data.get("strategy", self.strategy)
            self.reward_metric = data.get("reward_metric", self.reward_metric)
            arms_in = data.get("arms", {})
            self.arms = {k: ArmStats(**v) for k, v in arms_in.items()}
        except Exception as e:
            print(f"⚠️ Failed to load bandit state: {e}")

    def _save_state(self):
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.save_path.with_suffix(".tmp")
            payload = {
                "total_plays": self.total_plays,
                "strategy": self.strategy,
                "reward_metric": self.reward_metric,
                "arms": {k: a.as_dict() for k, a in self.arms.items()}
            }
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, self.save_path)  # atomic on POSIX
        except Exception as e:
            print(f"⚠️ Failed to save bandit state: {e}")

    # ---------- keying ----------

    def _key(self, exemplar_id: str, context: Optional[Dict[str, Any]]) -> str:
        """Namespace the arm by context (e.g., 'lm=small|task=code' → 'exid@lm=small|task=code')."""
        if not context:
            return exemplar_id
        ctx = "|".join(f"{k}={context[k]}" for k in sorted(context.keys()))
        return f"{exemplar_id}@{ctx}"

    def __repr__(self) -> str:
        return f"<ExemplarBandit strategy={self.strategy} arms={len(self.arms)} plays={self.total_plays}>"
