# stephanie/scoring/score_bundle.py
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple
from statistics import stdev

from stephanie.scoring.calculations.weighted_average import (
    WeightedAverageCalculator,
)
import numpy as np

@dataclass
class ScoreBundle:
    """Represents all scores for a single Scorable across dimensions and scorers

    Key features:
    - Contains ScoreResults with flexible attributes dictionary
    - Supports tensor operations for analysis
    - Works with MARS calculator for agreement analysis
    - Maintains compatibility with ORM layer
    """
    from stephanie.data.score_result import ScoreResult

    results: Dict[str, ScoreResult] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self, results: Dict[str, ScoreResult], meta: Dict[str, Any] = None
    ):
        self.results = results
        self.meta = meta or {}
        self.calculator = WeightedAverageCalculator()

    def aggregate(self) -> float:
        """Calculate weighted average score across dimensions"""
        return self.calculator.calculate(self)

    def get(self, dimension: str) -> Optional[ScoreResult]:
        return self.results.get(dimension)

    def to_dict(self, include_attributes: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for serialization and storage"""
        bundle_dict = {}
        for dim, result in self.results.items():
            result_dict = result.to_dict()
            if not include_attributes:
                # Remove attributes to keep the dictionary lean
                result_dict.pop("attributes", None)
            bundle_dict[dim] = result_dict

        # Add meta information if present
        if self.meta:
            bundle_dict["_meta"] = self.meta

        return bundle_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoreBundle":
        """Reconstruct from dictionary"""
        from stephanie.data.score_result import ScoreResult
        # Extract meta if present
        meta = data.pop("_meta", None)

        results = {
            dim: ScoreResult.from_dict(score_data)
            for dim, score_data in data.items()
            if isinstance(score_data, dict)
        }
        return cls(results=results, meta=meta)

    def merge(self, other: "ScoreBundle") -> "ScoreBundle":
        """
        Merge two bundles, preferring `self` values but including all from both.
        If a dimension exists in both, the value from `self` is kept.
        """
        merged = dict(self.results)
        for dim, result in other.results.items():
            if dim not in merged:
                merged[dim] = result
        return ScoreBundle(merged, meta={**self.meta, **other.meta})

    def to_json(
        self, stage: str, include_attributes: bool = False
    ) -> Dict[str, Any]:
        """Convert to JSON structure for reporting"""
        final_score = self.aggregate()
        return {
            "stage": stage,
            "dimensions": self.to_dict(include_attributes=include_attributes),
            "final_score": final_score,
            "meta": self.meta,
        }

    def to_orm(self, evaluation_id: int) -> List[Dict[str, Any]]:
        """Convert to ORM-compatible dictionaries for database storage"""
        orm_dicts = []
        for result in self.results.values():
            # Core score data
            orm_dict = {
                "evaluation_id": evaluation_id,
                "dimension": result.dimension,
                "score": result.score,
                "weight": result.weight,
                "rationale": result.rationale,
                "source": result.source,
                "target_type": result.target_type,
                "prompt_hash": result.prompt_hash,
            }

            # Include attribute references
            if result.attributes:
                orm_dict["attributes"] = result.attributes

            orm_dicts.append(orm_dict)
        return orm_dicts

    def to_tensor(
        self, dimensions: List[str], scorers: List[str], metrics: List[str]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Convert to 4D tensor: [1 × dimensions × scorers × metrics]
        For a single ScoreBundle (1 scorable)

        Returns:
            tensor: numpy array of shape (1, n_dimensions, n_scorers, n_metrics)
            metric_metadata: dictionary with metric information
        """
        import numpy as np

        tensor = np.zeros((1, len(dimensions), len(scorers), len(metrics)))
        metric_metadata = {
            "dimensions": dimensions,
            "scorers": scorers,
            "metrics": metrics,
        }

        for dim_idx, dimension in enumerate(dimensions):
            if dimension in self.results:
                result = self.results[dimension]
                scorer_idx = scorers.index(result.source)

                # Fill in metric values
                for metric_idx, metric in enumerate(metrics):
                    if metric in result.attributes:
                        try:
                            tensor[0, dim_idx, scorer_idx, metric_idx] = float(
                                result.attributes[metric]
                            )
                        except (TypeError, ValueError):
                            tensor[0, dim_idx, scorer_idx, metric_idx] = 0.0
                    else:
                        tensor[0, dim_idx, scorer_idx, metric_idx] = 0.0

        return tensor, metric_metadata

    def get_metric_values(self, metric: str) -> Dict[str, float]:
        """Get values for a specific metric across dimensions"""
        return {
            dim: result.attributes.get(metric, None)
            for dim, result in self.results.items()
        }

    def __repr__(self):
        summary = ", ".join(
            f"{dim}: {res.score:.2f}" for dim, res in self.results.items()
        )
        return f"<ScoreBundle({summary})>"

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)

    def to_report(self, title: str = "Score Report") -> str:
        """Generate a comprehensive report including tensor analysis capabilities"""
        lines = [f"## {title}", ""]

        # Add dimension scores
        for dim, result in self.results.items():
            lines.append(f"### Dimension: `{dim}`")
            lines.append(f"- **Score**: `{result.score:.4f}`")
            lines.append(f"- **Weight**: `{result.weight:.2f}`")
            lines.append(f"- **Source**: `{result.source}`")
            if result.rationale:
                lines.append(f"- **Rationale**: {result.rationale}")

            # Add attributes section if present
            if result.attributes:
                lines.append("\n**Extended Metrics:**")
                for key, value in result.attributes.items():
                    # Format value based on type
                    if isinstance(value, (int, float)):
                        formatted = f"{value:.4f}"
                    elif isinstance(value, (list, tuple)):
                        if len(value) > 5:
                            formatted = f"[{', '.join([f'{x:.4f}' for x in value[:5]])}, ...]"
                        else:
                            formatted = (
                                f"[{', '.join([f'{x:.4f}' for x in value])}]"
                            )
                    else:
                        formatted = str(value)
                    lines.append(f"- `{key}`: `{formatted}`")
            lines.append("")  # Empty line between dimensions

        # Add aggregate score
        lines.append(f"**Aggregate Score:** `{self.aggregate():.4f}`")

        return "\n".join(lines)

    def analyze_agreement(self, dimension: str = None) -> Dict[str, Any]:
        """Analyze agreement patterns across scorers for this bundle"""
        if dimension:
            # Analyze specific dimension
            if dimension not in self.results:
                return {"error": "dimension_not_found", "dimension": dimension}

            # For single dimension, there's only one scorer in this bundle
            # So agreement analysis doesn't apply at bundle level
            return {
                "dimension": dimension,
                "agreement_score": 1.0,  # Only one scorer for this dimension
                "explanation": "Single-scorer bundle - no agreement analysis needed",
            }

        # For cross-dimension analysis (not typically meaningful)
        scores = [r.score for r in self.results.values()]
        if len(scores) < 2:
            return {
                "agreement_score": 1.0,
                "explanation": "Only one dimension scored - no agreement analysis needed",
            }

        std_dev = stdev(scores)
        agreement_score = 1.0 - min(std_dev, 1.0)

        return {
            "agreement_score": round(agreement_score, 3),
            "std_dev": round(std_dev, 3),
            "dimension_count": len(scores),
            "explanation": f"Cross-dimension agreement score: {agreement_score:.3f}",
        }
