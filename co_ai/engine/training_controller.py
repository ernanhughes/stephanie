# co_ai/engine/training_controller.py

from typing import Callable, List


class TrainingController:
    """
    Orchestrates when and how to retrain reward models based on validation metrics.
    """

    def __init__(
        self,
        cfg,
        memory,
        logger,
        validator,       # SelfValidationEngine
        tracker,         # MetaConfidenceTracker
        trainer_fn: Callable  # Function to launch training run
    ):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.validator = validator
        self.tracker = tracker
        self.trainer_fn = trainer_fn

        self.retrain_cooldown = getattr(cfg, "retrain_cooldown_steps", 5)
        self.retrain_state = {}  # (goal, dimension) → cooldown counter

    def maybe_train(
        self,
        goal: str,
        dimension: str,
        pairs: List[dict]
    ):
        """
        Validates current model, updates trust score, and triggers training if needed.
        """

        # Step 1: Run validation pass
        result = self.validator.validate_batch(goal, pairs, dimension)
        self.tracker.update(goal, dimension, result)

        # Step 2: Track cooldown status
        key = (goal, dimension)
        cooldown = self.retrain_state.get(key, 0)
        if cooldown > 0:
            self.retrain_state[key] = cooldown - 1
            if self.logger:
                self.logger.info("RetrainOnCooldown", extra={"goal": goal, "dimension": dimension, "cooldown_left": cooldown})
            return

        # Step 3: Trigger training if needed
        if self.tracker.should_retrain(goal, dimension):
            if self.logger:
                self.logger.info("TriggeringRetraining", extra={"goal": goal, "dimension": dimension})
            self.trainer_fn(goal=goal, dimension=dimension)
            self.retrain_state[key] = self.retrain_cooldown  # reset cooldown

        else:
            if self.logger:
                self.logger.info("RetrainingSkipped", extra={
                    "goal": goal,
                    "dimension": dimension,
                    "confidence": self.tracker.get_confidence(goal, dimension)
                })
