# stephanie/services/training_service.py
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from stephanie.evaluator.mrq_trainer import MRQTrainer
from stephanie.scoring.training.base_trainer import BaseTrainer
from stephanie.scoring.training.sicql_trainer import SICQLTrainer
from stephanie.services.service_protocol import Service


@dataclass
class RetrainPolicy:
    # Hard gates
    min_pairs: int = 8
    min_total_evals: int = 50

    # Targets / thresholds
    target_confidence: float = 0.75
    margin: float = 0.05
    slope_threshold: float = -0.01
    min_coverage: float = 0.50

    # Cooldowns / patience
    cooldown_steps: int = 5
    patience_bad_streak: int = 3


@dataclass
class ValidationResult:
    confidence: float
    coverage: float
    regret: float
    sample_size: int
    extras: Dict = field(default_factory=dict)


class TrainingService(Service):
    """
    Decides when to retrain based on validation feedback & tracker state.
    If validator/tracker/trainer_fn are not provided, resolves them from the container.
    """

    def __init__(
        self,
        cfg,
        memory,
        logger,
        validator: Optional[Any] = None,          # SelfValidationService-like
        tracker: Optional[Any] = None,            # MetaConfidenceService-like
        trainer_fn: Optional[Callable] = None,    # Callable(goal, dimension, **kwargs)
        policy: Optional[RetrainPolicy] = None,
        container: Optional[Any] = None,
    ):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.container = container

        self.validator = validator
        self.tracker = tracker
        self.trainer_fn = trainer_fn

        self.policy = policy or RetrainPolicy(
            min_pairs=getattr(cfg, "retrain_min_pairs", 8),
            min_total_evals=getattr(cfg, "retrain_min_total_evals", 50),
            target_confidence=getattr(cfg, "retrain_target_confidence", 0.75),
            margin=getattr(cfg, "retrain_margin", 0.05),
            slope_threshold=getattr(cfg, "retrain_slope_threshold", -0.01),
            min_coverage=getattr(cfg, "retrain_min_coverage", 0.50),
            cooldown_steps=getattr(cfg, "retrain_cooldown_steps", 5),
            patience_bad_streak=getattr(cfg, "retrain_patience_bad_streak", 3),
        )

        self.cooldowns: Dict[Tuple[str, str], int] = {}
        self._initialized = False
        self.trainers: Dict[str, BaseTrainer] = {}  # e.g., {"sicql": ..., "mrq": ...}

    # === Service Protocol ===
    def initialize(self, **kwargs) -> None:
        # Lazy-resolve dependencies if not injected
        if self.validator is None:
            if not self.container:
                raise RuntimeError("TrainingService needs 'validator' or a 'container' to resolve it.")
            self.validator = self.container.get("validation")

        if self.tracker is None:
            if not self.container:
                raise RuntimeError("TrainingService needs 'tracker' or a 'container' to resolve it.")
            self.tracker = self.container.get("confidence")

        if self.trainer_fn is None:
            def _default_trainer(*, goal, dimension, stats=None, validator_result=None, **_):
                if self.logger:
                    self.logger.info(
                        f"[TRAIN] (default) Training RM for goal={goal!r}, "
                        f"dimension={dimension!r} | statsKeys={list((stats or {}).keys())}"
                    )
            self.trainer_fn = _default_trainer

        # Optional trainer registry (kept as-is)
        self.trainers["sicql"] = SICQLTrainer(self.cfg, self.memory, self.logger)
        self.trainers["mrq"] = MRQTrainer(self.cfg, self.memory, self.logger)

        self._initialized = True
        if self.logger:
            self.logger.log("TrainingServiceInit", {"status": "initialized"})

    def health_check(self) -> Dict[str, Any]:
        """Return active cooldowns and tracked metrics."""
        return {
            "status": "healthy" if self._initialized else "unhealthy",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {
                "active_cooldowns": len(self.cooldowns),
                "policy": self.policy.__dict__,
            },
            "dependencies": {
                "validator": "attached" if self.validator else "missing",
                "tracker": "attached" if self.tracker else "missing",
                "trainer_fn": "attached" if self.trainer_fn else "missing",
            },
        }

    def shutdown(self) -> None:
        """Clear cooldowns and mark uninitialized."""
        self.cooldowns.clear()
        self._initialized = False
        if self.logger:
            self.logger.log("TrainingServiceShutdown", {})

    @property
    def name(self) -> str:
        return "training-service-v1"

    # === Domain Logic ===
    def _log(self, event: str, payload: Dict):
        if self.logger:
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
        2) Update tracker
        3) If not on cooldown and tracker says 'retrain', call trainer_fn
        """
        key = (goal, dimension)

        if len(pairs) < self.policy.min_pairs:
            self._log("RetrainSkippedSmallBatch", {
                "goal": goal, "dimension": dimension, "pairs": len(pairs),
                "needed": self.policy.min_pairs
            })
            return

        # Step 1: validation
        vres = self.validator.validate_batch(goal, pairs, dimension)
        v = ValidationResult(
            confidence=float(vres.get("confidence", 0.0)),
            coverage=float(vres.get("coverage", 0.0)),
            regret=float(vres.get("regret", 1.0 - vres.get("confidence", 0.0))),
            sample_size=int(vres.get("sample_size", len(pairs))),
            extras={k: vres[k] for k in vres.keys() if k not in {"confidence","coverage","regret","sample_size"}},
        )

        # Step 2: tracker update
        snapshot = self.tracker.update(
            goal, dimension, v.confidence,
            coverage=v.coverage, regret=v.regret
        )
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

        # Step 2.1: sanity gates
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

        # Step 2.2: cooldown gate
        if self._on_cooldown(key):
            self._log("RetrainOnCooldown", {
                "goal": goal, "dimension": dimension,
                "cooldown_left": self.cooldowns.get(key, 0)
            })
            return

        # Step 3: retrain decision
        below_target = ema_conf < (self.policy.target_confidence - self.policy.margin)
        trending_down = slope_conf <= self.policy.slope_threshold
        long_bad_streak = bad_streak >= self.policy.patience_bad_streak

        should = self.tracker.should_retrain(
            goal, dimension,
            below_target=below_target,
            trending_down=trending_down,
            long_bad_streak=long_bad_streak
        )

        if should:
            self._log("TriggeringRetraining", {
                "goal": goal, "dimension": dimension,
                "ema_confidence": ema_conf, "target": self.policy.target_confidence,
                "slope_confidence": slope_conf, "bad_streak": bad_streak
            })
            # Pass useful context to the trainer
            self.trainer_fn(goal=goal, dimension=dimension, stats=snapshot, validator_result=v)
            self._arm_cooldown(key)
            self.tracker.mark_retrained(goal, dimension, when=time.time())
        else:
            self._log("RetrainingSkipped", {
                "goal": goal, "dimension": dimension,
                "ema_confidence": ema_conf, "target": self.policy.target_confidence,
                "slope_confidence": slope_conf, "bad_streak": bad_streak
            })
