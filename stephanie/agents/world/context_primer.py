# stephanie/agents/world/context_primer.py


class ContextPrimerAgent:
    def __init__(self, memory, embedding_model, logger=None):
        self.memory = memory
        self.embedding_model = embedding_model
        self.logger = logger

    def generate_hints(self, goal_text: str, top_k=5) -> list[str]:
        goal_emb = self.embedding_model.encode(goal_text)
        similar_beliefs = self.memory.beliefs.find_similar(goal_emb, top_k=top_k)

        hints = []
        for belief in similar_beliefs:
            if belief.usefulness_score > 0.6:  # configurable
                hints.append(belief.brief_summary or belief.title)

        if self.logger:
            self.logger.log(
                "ContextPrimingGenerated",
                {
                    "goal": goal_text,
                    "num_hints": len(hints),
                },
            )

        return hints
