# stephanie/engine/self_validation.py

import random


class SelfValidationEngine:
    """
    Validates the predictions of a reward model using fallback LLM evaluations.
    Used to measure model correctness, trigger retraining, and prevent drift.
    """

    def __init__(
        self,
        cfg,
        memory,
        logger,
        reward_model: callable,      # (goal, doc_a, doc_b) -> "a" or "b"
        llm_judge: callable          # (goal, doc_a, doc_b) -> "a" or "b"
    ):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.reward_model = reward_model
        self.llm_judge = llm_judge
        self.validation_sample_rate = getattr(cfg, "validation_sample_rate", 0.05)

    def validate_batch(
        self,
        goal: str,
        pairs: list[dict],
        dimension: str = None
    ) -> dict:
        """
        Validates a batch of scored document pairs against LLM judgment.
        Returns agreement statistics and logs validation outcomes.
        """

        if not pairs:
            return {"validated": 0, "agreement": 0.0}

        sample = [
            pair for pair in pairs
            if random.random() < self.validation_sample_rate
        ]

        validated = 0
        matches = 0
        logs = []

        for pair in sample:
            doc_a = pair["text_a"]
            doc_b = pair["text_b"]

            # Model preference
            model_pref = self.reward_model(goal, doc_a, doc_b)

            # LLM judgment
            llm_pref = self.llm_judge(goal, doc_a, doc_b)

            is_match = model_pref == llm_pref
            validated += 1
            matches += int(is_match)

            log_entry = {
                "goal": goal,
                "dimension": dimension,
                "model_pref": model_pref,
                "llm_pref": llm_pref,
                "match": is_match,
                "text_a": doc_a[:300],  # truncate for logs
                "text_b": doc_b[:300]
            }
            logs.append(log_entry)

            if self.logger:
                self.logger.info("SelfValidationResult", extra=log_entry)

        agreement = matches / validated if validated else 0.0

        # Optional: persist validation results to memory
        if self.memory:
            self.memory.save("self_validation", {
                "goal": goal,
                "dimension": dimension,
                "agreement": agreement,
                "logs": logs
            })

        return {
            "validated": validated,
            "matches": matches,
            "agreement": round(agreement, 3),
            "logs": logs
        }
