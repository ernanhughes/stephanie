class MRQScoresToTraining:
    def __init__(self, memory, logger, min_score_diff=0.1):
        self.memory = memory
        self.logger = logger
        self.min_score_diff = min_score_diff

    def get_training_pairs(self, context: dict = None) -> list:
        goal_groups = {}
        scores = self.memory.evaluations.get_all()

        for score in scores:
            hypothesis = self.memory.hypotheses.get_by_id(score.get("hypothesis_id"))
            if not hypothesis or not hypothesis.prompt or not hypothesis.text:
                continue

            goal_id = hypothesis.goal_id
            goal_groups.setdefault(goal_id, []).append({
                "prompt": hypothesis.prompt.prompt_text,
                "response": hypothesis.text,
                "score": score.get("score"),
                "hypothesis_id": hypothesis.id,
            })

        training_pairs = []
        for goal_id, items in goal_groups.items():
            sorted_items = sorted(items, key=lambda x: x["score"], reverse=True)
            for i in range(len(sorted_items)):
                for j in range(i + 1, len(sorted_items)):
                    hi = sorted_items[i]
                    hj = sorted_items[j]
                    if abs(hi["score"] - hj["score"]) >= self.min_score_diff:
                        training_pairs.append({
                            "prompt": hi["prompt"],
                            "output_a": hi["response"],
                            "output_b": hj["response"],
                            "preferred": "a",
                            "goal_id": goal_id,
                            "hypothesis_a_id": hi["hypothesis_id"],
                            "hypothesis_b_id": hj["hypothesis_id"],
                        })

        if self.logger:
            self.logger.log("MRQPairsGenerated", {"count": len(training_pairs)})

        if context is not None:
            context["mrq_training_pairs"] = training_pairs

        return training_pairs
