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
                target_type=r.target_type,
                prompt_hash=r.prompt_hash,

            )
            for r in self.results.values()
        ]

    def __repr__(self):
        summary = ", ".join(f"{dim}: {res.score}" for dim, res in self.results.items())
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
                target_type=entry.get("target_type", "unknown"),
                prompt_hash=entry.get("prompt_hash", ""),
                
            )
            for dim, entry in data.items()
            if isinstance(entry, dict)  # Defensive: skip bad formats
        }

        return cls(results)

    def to_report(self, title: str = "Score Report") -> str:
        lines = [f"## {title}", ""]
        for dim, result in self.results.items():
            lines.append(f"### Dimension: `{dim}`")
            lines.append(f"- **Score**: `{result.score:.4f}`")
            lines.append(f"- **Weight**: `{result.weight:.2f}`")
            lines.append(f"- **Source**: `{result.source}`")
            lines.append(f"- **Target Type**: `{result.target_type}`")
            lines.append(f"- **Prompt Hash**: `{result.prompt_hash}`")
            if result.rationale:
                lines.append(f"- **Rationale**: {result.rationale}")

            # SICQL-specific fields
            if result.energy is not None:
                lines.append(f"- **Energy**: `{result.energy:.4f}`")
            if result.q_value is not None:
                lines.append(f"- **Q-Value**: `{result.q_value:.4f}`")
            if result.state_value is not None:
                lines.append(f"- **State Value**: `{result.state_value:.4f}`")
            if result.policy_logits is not None:
                logits_str = ", ".join(f"{x:.4f}" for x in result.policy_logits)
                lines.append(f"- **Policy Logits**: [{logits_str}]")
            if result.uncertainty is not None:
                lines.append(f"- **Uncertainty**: `{result.uncertainty:.4f}`")
            if result.entropy is not None:
                lines.append(f"- **Entropy**: `{result.entropy:.4f}`")
            if result.advantage is not None:
                lines.append(f"- **Advantage**: `{result.advantage:.4f}`")

            lines.append("")  # Empty line between dimensions

        lines.append(f"**Aggregate Score:** `{self.aggregate():.4f}`")
        return "\n".join(lines)
