# stephanie/evaluator/agreement_checker.py
# evaluators/agreement_checker.py


class EvaluatorAgreementChecker:
    def __init__(self, logger):
        self.logger = logger

    def check(self, mrq_scores: dict, llm_judgement: str, context: dict = None):
        """
        Compares MR.Q and LLM preferred outputs and logs agreement.
        `llm_judgement` should be 'a' or 'b' (or full judgement string if parsed).
        """
        mrq_preference = "a" if mrq_scores["value_a"] >= mrq_scores["value_b"] else "b"

        agreement = mrq_preference == llm_judgement

        self.logger.log(
            "JudgementAgreement",
            {
                "mrq_preference": mrq_preference,
                "llm_preference": llm_judgement,
                "agreement": agreement,
                "value_a": round(mrq_scores["value_a"], 4),
                "value_b": round(mrq_scores["value_b"], 4),
                "context": context or {},
            },
        )

        return agreement
