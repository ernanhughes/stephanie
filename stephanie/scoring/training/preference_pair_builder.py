# stephanie/scoring/mrq/preference_pair_builder.py
from __future__ import annotations

from typing import Dict, List, Any

from stephanie.scoring.scorable import ScorableType


class PreferencePairBuilder:
    """
    Builds preference training pairs: (high-score sample, low-score sample) per dimension.

    Usage:
        builder = PreferencePairBuilder(memory)
        pairs = builder.get_training_pairs_by_dimension(
            limit=1000,
            dim=["reasoning", "clarity"],
            target_type="conversation_turn"
        )

    Output:
        Dict[dim] → List[{
            "title": str,
            "output_a": str (better),
            "output_b": str (worse),
            "value_a": float,
            "value_b": float
        }]
    """

    def __init__(self, memory, logger=None):
        self.memory = memory
        self.logger = logger

    def get_training_pairs_by_dimension(
        self,
        dimension: str = "knowledge",
        target_type: str = ScorableType.CONVERSATION_TURN,
        limit: int = 1000,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build top-vs-bottom preference pairs from reasoning samples.

        Args:
            limit: Approximate max number of total pairs across all dimensions.
            dim: Optional filter to specific dimensions (e.g., ['reasoning']).
            target_type: Filter by scorable_type (e.g., 'conversation_turn').

        Returns:
            {dimension: [pair_dicts]}
        """
        # 1. Fetch raw samples
        try:
            samples = self.memory.reasoning_samples.get_eval_pairs_by_dimension(
                target_type=target_type,
                dimension=dimension,
                limit=limit  # Overfetch slightly to ensure good pairing
            )
            return samples
        except Exception as e:
            if self.logger:
                self.logger.log("PreferencePairFetchFailed", {"error": str(e)})
            return {}
