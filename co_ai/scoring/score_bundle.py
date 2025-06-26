# co_ai/scoring/score_bundle.py
import json

from co_ai.scoring.score_result import ScoreResult


class ScoreBundle:
    def __init__(self, results: dict[str, ScoreResult]):
        from co_ai.scoring.calculations.weighted_average import \
            WeightedAverageCalculator
        self.results = results
        self.calculator = WeightedAverageCalculator()

    def aggregate(self):
        result = self.calculator.calculate(self) 
        print(f"ScoreBundle: Aggregated score: {result}")
        return result

    def to_dict(self) -> dict:
        return {k: v.to_dict() for k, v in self.results.items()}

    def to_json(self, stage: str):
        final_score = self.aggregate()
        return {
            "stage": stage,
            "dimensions": self.to_dict(),
            "final_score": final_score,
        }

    def to_orm(self, evaluation_id: int):
        from co_ai.models.score import ScoreORM
        return [
            ScoreORM(
                evaluation_id=evaluation_id,
                dimension=r.dimension,
                score=r.score,
                weight=r.weight,
                rationale=r.rationale,
                source=r.source,
            )
            for r in self.results.values()
        ] 

    def __repr__(self):
        summary = ", ".join(
            f"{dim}: {res.score}" for dim, res in self.results.items()
        )
        return f"<ScoreBundle({summary})>"

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)