import random
from typing import Dict, List, Optional

from sqlalchemy import String, cast

from stephanie.dataloaders.casebook_to_rlvr import RLVRItem
from stephanie.models.casebook import CaseBookORM, CaseORM, CaseScorableORM
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.score import ScoreORM


class ScoredRLVRDataset:
    """
    Builds a scored RLVR dataset from Stephanie's database for PACS training.

    This implementation:
    - Properly queries scored case_scorables
    - Converts continuous scores to binary rewards using thresholding
    - Handles the correct RLVRItem structure expected by PACS
    - Includes proper metadata for tracking
    """

    def __init__(
        self,
        memory,
        dimensions: Optional[List[str]] = None,
        reward_threshold: float = 0.7,
        min_samples_per_dimension: int = 10,
    ):
        """
        Args:
            memory: Stephanie's memory system
            dimensions: Scoring dimensions to use (default: ["alignment"])
            reward_threshold: Threshold for converting scores to binary rewards
            min_samples_per_dimension: Minimum samples needed per dimension
        """
        self.memory = memory
        self.dimensions = dimensions or ["alignment"]
        self.reward_threshold = reward_threshold
        self.min_samples_per_dimension = min_samples_per_dimension

    def build(
        self,
        limit: int = 1000,
        domain: Optional[str] = None,
        casebook_id: Optional[str] = None,
    ) -> List[RLVRItem]:
        """
        Build RLVR dataset from scored case_scorables.

        Args:
            limit: Maximum number of items to return
            domain: Optional domain filter
            casebook_id: Optional casebook filter

        Returns:
            List of RLVRItem objects for PACS training
        """
        session = self.memory.session

        # --- 1. Query scored case_scorables ---
        query = (
            session.query(
                CaseScorableORM, CaseORM, CaseBookORM, EvaluationORM, ScoreORM
            )
            .join(CaseORM, CaseScorableORM.case_id == CaseORM.id)
            .join(CaseBookORM, CaseORM.casebook_id == CaseBookORM.id)
            .join(
                EvaluationORM,
                EvaluationORM.scorable_id == cast(CaseScorableORM.id, String),
            )
            .join(ScoreORM, ScoreORM.evaluation_id == EvaluationORM.id)
            .filter(ScoreORM.dimension.in_(self.dimensions))
        )

        # Apply filters
        if domain:
            query = query.filter(CaseBookORM.domain == domain)
        if casebook_id:
            query = query.filter(CaseBookORM.id == casebook_id)

        # Order and limit
        query = query.order_by(ScoreORM.id.desc()).limit(limit)

        # Execute query
        results = query.all()

        # --- 2. Group by dimension for balanced sampling ---
        items_by_dimension = {dim: [] for dim in self.dimensions}

        for scorable, case, casebook, evaluation, score in results:
            # Convert continuous score to binary reward
            is_correct = 1 if score.score >= self.reward_threshold else 0

            domain_list = self.memory.scorable_domains.get_by_scorable(
                str(scorable.id), scorable.scorable_type
            )

            # Build metadata
            meta = {
                "case_id": case.id,
                "casebook_id": casebook.id,
                "scorable_id": str(scorable.id),
                "score_id": score.id,
                "dimension": score.dimension,
                "score_value": float(score.score),
                "is_correct": is_correct,
                "domain": domain_list[0].domain if domain_list else None,
                "created_at": str(evaluation.created_at),
                "casebook_name": casebook.name,
                "goal_text": getattr(casebook, "goal_text", ""),
            }

            items_by_dimension[score.dimension].append(
                (case.prompt_text, meta)
            )

        # --- 3. Balance samples across dimensions ---
        balanced_items = []
        valid_dims = [
            len(items)
            for items in items_by_dimension.values()
            if len(items) >= self.min_samples_per_dimension
        ]
        if not valid_dims:
            self.memory.logger.log(
                "RLVRDatasetWarning",
                {
                    "message": "Not enough samples per dimension to balance dataset",
                    "dimensions": {
                        dim: len(items)
                        for dim, items in items_by_dimension.items()
                    },
                },
            )
            return []

        min_per_dim = min(valid_dims)

        for dim, items in items_by_dimension.items():
            if len(items) >= self.min_samples_per_dimension:
                sample_size = min(min_per_dim, len(items))
                sampled = random.sample(items, sample_size)  # <-- FIXED
                balanced_items.extend(sampled)

        # --- 4. Build RLVR dataset ---
        dataset = []
        for prompt, meta in balanced_items:
            # Pick the scorable text (ensure you added this earlier in meta)
            response = meta.get("scorable_text", "")
            reward = (
                1.0 if meta["is_correct"] else 0.0
            )  # binary reward for PACS

            dataset.append(
                RLVRItem(
                    prompt=prompt, response=response, reward=reward, meta=meta
                )
            )

        # Log dataset statistics
        self.memory.logger.log(
            "RLVRDatasetBuilt",
            {
                "total_items": len(dataset),
                "dimensions": self.dimensions,
                "reward_threshold": self.reward_threshold,
                "items_per_dimension": {
                    dim: len(items)
                    for dim, items in items_by_dimension.items()
                },
            },
        )

        return dataset

    def get_reward_distribution(
        self, dataset: List[RLVRItem]
    ) -> Dict[str, float]:
        """
        Analyze reward distribution in the dataset.

        Args:
            dataset: RLVR dataset

        Returns:
            Dictionary with reward statistics per dimension
        """
        stats = {}
        for dim in self.dimensions:
            items = [item for item in dataset if item.meta["dimension"] == dim]
            if items:
                correct_count = sum(
                    1 for item in items if item.meta["is_correct"] == 1
                )
                stats[dim] = {
                    "total": len(items),
                    "correct": correct_count,
                    "accuracy": correct_count / len(items),
                    "threshold": self.reward_threshold,
                }

        return stats
