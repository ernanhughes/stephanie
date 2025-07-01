# stephanie/scoring/score_bundle.py
import json

from stephanie.scoring.score_result import ScoreResult


class ScoreBundle:
    def __init__(self, results: dict[str, ScoreResult]):
        from stephanie.scoring.calculations.weighted_average import \
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
        from stephanie.models.score import ScoreORM
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

    @classmethod
    def from_dict(cls, data: dict) -> "ScoreBundle":
        """
        Reconstruct a ScoreBundle from a dict where each value is a ScoreResult-like dict.
        """
        from stephanie.scoring.score_result import ScoreResult

        results = {
            dim: ScoreResult(
                dimension=dim,
                score=entry.get("score"),
                weight=entry.get("weight", 1.0),
                rationale=entry.get("rationale", ""),
                source=entry.get("source", "from_dict"),
            )
            for dim, entry in data.items()
            if isinstance(entry, dict)  # Defensive: skip bad formats
        }

        return cls(results)
