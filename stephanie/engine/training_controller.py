# stephanie/engine/training_controller.py
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class RetrainPolicy:
    # Hard gates before training is even considered
    min_pairs: int = 8                # require at least N pairwise examples in this call
    min_total_evals: int = 50         # require at least N historical evals for this (goal,dimension)

    # Targets / thresholds
    target_confidence: float = 0.75   # aim for ≥ this moving confidence
    margin: float = 0.05              # hysteresis margin (don’t flap around the target)
    slope_threshold: float = -0.01    # if trend is falling faster than this, consider retrain
    min_coverage: float = 0.50        # at least this fraction of pairs must be evaluable

    # Cooldowns / patience
    cooldown_steps: int = 5           # how many controller calls to wait after training
    patience_bad_streak: int = 3      # retrain if this many consecutive bad evals

@dataclass
class ValidationResult:
    confidence: float                 # [0,1], eg. % of pairwise wins
    coverage: float                   # [0,1], fraction of pairs used
    regret: float                     # [0,∞), lower is better (1 - confidence works too)
    sample_size: int                  # #pairs consumed for this validation
    extras: Dict = field(default_factory=dict)  # any extra diagnostics

class TrainingController:
    """
    Decides when to retrain based on validation feedback & tracker state.
    Owns cooldowns, thresholds, and calls into your 'trainer_fn'.
    """

    def __init__(
        self,
        cfg,
        memory,
        logger,
        validator,                 # SelfValidationEngine-like: validate_batch(goal, pairs, dimension) -> ValidationResult|dict
        tracker,                   # MetaConfidenceTracker-like: update/should_retrain/get_confidence/get_state
        trainer_fn: Callable,      # Callable(goal, dimension, **kwargs) -> None
        policy: Optional[RetrainPolicy] = None,
    ):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.validator = validator
        self.tracker = tracker
        self.trainer_fn = trainer_fn

        # Prefer explicit policy object; fall back to cfg; then defaults
        self.policy = policy or RetrainPolicy(
            min_pairs = getattr(cfg, "retrain_min_pairs", 8),
            min_total_evals = getattr(cfg, "retrain_min_total_evals", 50),
            target_confidence = getattr(cfg, "retrain_target_confidence", 0.75),
            margin = getattr(cfg, "retrain_margin", 0.05),
            slope_threshold = getattr(cfg, "retrain_slope_threshold", -0.01),
            min_coverage = getattr(cfg, "retrain_min_coverage", 0.50),
            cooldown_steps = getattr(cfg, "retrain_cooldown_steps", 5),
            patience_bad_streak = getattr(cfg, "retrain_patience_bad_streak", 3),
        )

        # cooldowns keyed by (goal, dimension)
        self.cooldowns: Dict[Tuple[str, str], int] = {}

    def _log(self, event: str, payload: Dict):
        if self.logger:
            # Consistent with your other code: logger.log("EventName", {...})
            self.logger.log(event, payload)

    def _on_cooldown(self, key: Tuple[str, str]) -> bool:
        c = self.cooldowns.get(key, 0)
        if c > 0:
            self.cooldowns[key] = c - 1
            return True
        return False

    def _arm_cooldown(self, key: Tuple[str, str]):
        self.cooldowns[key] = int(self.policy.cooldown_steps)

    def maybe_train(self, goal: str, dimension: str, pairs: List[dict]):
        """
        1) Validate current model on 'pairs'
        2) Update tracker (rolling stats)
        3) If not on cooldown and tracker says 'retrain', call trainer
        """
        key = (goal, dimension)

        # Guard: need a minimum batch to be meaningful
        if len(pairs) < self.policy.min_pairs:
            self._log("RetrainSkippedSmallBatch", {
                "goal": goal, "dimension": dimension, "pairs": len(pairs),
                "needed": self.policy.min_pairs
            })
            return

        # Step 1: validation
        vres = self.validator.validate_batch(goal, pairs, dimension)
        # normalize
        if isinstance(vres, dict):
            v = ValidationResult(
                confidence=float(vres.get("confidence", 0.0)),
                coverage=float(vres.get("coverage", 0.0)),
                regret=float(vres.get("regret", 1.0 - vres.get("confidence", 0.0))),
                sample_size=int(vres.get("sample_size", len(pairs))),
                extras={k: vres[k] for k in vres.keys() if k not in {"confidence","coverage","regret","sample_size"}},
            )
        else:
            v = vres  # already a ValidationResult

        # Step 2: tracker update
        snapshot = self.tracker.update(goal, dimension, v.confidence, coverage=v.coverage, regret=v.regret)
        total_evals = snapshot.get("total_evals", 0)
        bad_streak  = snapshot.get("bad_streak", 0)
        ema_conf    = snapshot.get("ema_confidence", 0.0)
        slope_conf  = snapshot.get("slope_confidence", 0.0)

        self._log("ValidationResult", {
            "goal": goal, "dimension": dimension,
            "confidence": v.confidence, "coverage": v.coverage,
            "regret": v.regret, "sample_size": v.sample_size,
            "ema_confidence": ema_conf, "slope_confidence": slope_conf,
            "bad_streak": bad_streak, "total_evals": total_evals
        })

        # Basic sanity gates
        if v.coverage < self.policy.min_coverage:
            self._log("RetrainSkippedLowCoverage", {
                "goal": goal, "dimension": dimension,
                "coverage": v.coverage, "min_coverage": self.policy.min_coverage
            })
            return
        if total_evals < self.policy.min_total_evals:
            self._log("RetrainWarmup", {
                "goal": goal, "dimension": dimension,
                "total_evals": total_evals, "needed": self.policy.min_total_evals
            })
            return

        # Cooldown gate
        if self._on_cooldown(key):
            self._log("RetrainOnCooldown", {
                "goal": goal, "dimension": dimension,
                "cooldown_left": self.cooldowns.get(key, 0)
            })
            return

        # Step 3: decide
        target = self.policy.target_confidence
        margin = self.policy.margin
        below_target = (ema_conf < (target - margin))
        trending_down = (slope_conf <= self.policy.slope_threshold)
        long_bad_streak = (bad_streak >= self.policy.patience_bad_streak)

        should = self.tracker.should_retrain(
            goal, dimension,
            below_target=below_target,
            trending_down=trending_down,
            long_bad_streak=long_bad_streak
        )

        if should:
            self._log("TriggeringRetraining", {
                "goal": goal, "dimension": dimension,
                "ema_confidence": ema_conf, "target": target,
                "slope_confidence": slope_conf, "bad_streak": bad_streak
            })
            # Call into your training launcher (can consume buffers, etc.)
            self.trainer_fn(goal=goal, dimension=dimension, stats=snapshot, validator_result=v)
            self._arm_cooldown(key)
            self.tracker.mark_retrained(goal, dimension, when=time.time())
        else:
            self._log("RetrainingSkipped", {
                "goal": goal, "dimension": dimension,
                "ema_confidence": ema_conf, "target": target,
                "slope_confidence": slope_conf, "bad_streak": bad_streak
            })
