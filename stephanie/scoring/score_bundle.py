# stephanie/scoring/score_bundle.py
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from stephanie.scoring.score_result import ScoreResult


@dataclass
class ScoreBundle:
    results: dict[str, ScoreResult] = field(default_factory=dict)

    def __init__(self, results: dict[str, ScoreResult]):
        from stephanie.scoring.calculations.weighted_average import \
            WeightedAverageCalculator

        self.results = results
        self.calculator = WeightedAverageCalculator()

    def aggregate(self):
        result = self.calculator.calculate(self)
        return result

    def get(self, dimension: str) -> Optional[ScoreResult]:
        return self.results.get(dimension)

    def to_dict(self) -> Dict[str, Any]:
        return {dim: result.to_dict() for dim, result in self.results.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoreBundle":
        """
        Reconstruct a ScoreBundle from a dictionary where each value is a ScoreResult-like dict.
        """
        results = {
            dim: ScoreResult.from_dict(score_data)
            for dim, score_data in data.items()
            if isinstance(score_data, dict)
        }
        return cls(results=results)

    def merge(self, other: "ScoreBundle") -> "ScoreBundle":
        """
        Merge two bundles, preferring `self` values but including all from both.
        If a dimension exists in both, the value from `self` is kept.
        """
        merged = dict(self.results)
        for dim, result in other.results.items():
            if dim not in merged:
                merged[dim] = result
        return ScoreBundle(merged)

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
                energy=r.energy,
                q_value=r.q_value,
                state_value=r.state_value,
                policy_logits=r.policy_logits,
                uncertainty=r.uncertainty,
                entropy=r.entropy,
                advantage=r.advantage,
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
                logits_str = ", ".join(
                    f"{x:.4f}" for x in result.policy_logits
                )
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
