from co_ai.models import ScoreORM


class HypothesisEvaluator:
    def __init__(self, session):
        self.session = session

    def get_hypothesis_scores(self, hypothesis_id):
        scores = (
            self.session.query(ScoreORM)
            .filter(ScoreORM.hypothesis_id == hypothesis_id)
            .all()
        )
        return {
            score.score_type: score.score
            for score in scores
        }

    def get_composite_score(self, hypothesis_id):
        scores = self.get_hypothesis_scores(hypothesis_id)
        # Example: weighted average
        weights = {
            "review": 0.3,
            "reflection": 0.2,
            "llm_judge": 0.4,
            "elo": 0.1
        }
        total = sum(scores.get(k, 0) * v for k, v in weights.items())
        return round(total, 2)