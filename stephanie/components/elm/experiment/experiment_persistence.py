from typing import Any, Dict, List
from stephanie.data.score_bundle import ScoreBundle

import logging
logger = logging.getLogger(__name__)

class ExperimentPersistence:
    """
    Leverage your existing ScoreBundle persistence infrastructure.
    
    No new database schema needed.
    """
    
    def __init__(self, memory_container: Any):
        self.memory = memory_container
    
    def log_experiment_episode(
        self,
        episode: int,
        query: Any,
        bundle_before: "ScoreBundle",
        bundle_after: "ScoreBundle",
        dominance_achieved: bool,
        experiment_metadata: Dict[str, Any]
    ):
        """
        Log experiment episode using your existing EvaluationORM.
        
        Adds experiment-specific metadata to the bundle's meta field.
        """
        # Add experiment metadata to bundle
        bundle_after.meta.update({
            "experiment_episode": episode,
            "dominance_achieved": dominance_achieved,
            "experiment_metadata": experiment_metadata,
        })
        
        # Use your existing save_bundle method
        self.memory.evaluations.save_bundle(
            bundle_after,
            scorable=query,
            context={"experiment_episode": episode},
            cfg={},  # Empty or experiment-specific config
            source="experiment_governed",
            embedding_type="experiment",
            evaluator_name="GovernedSelfImprovement",
        )
    
    def query_experiment_results(
        self,
        experiment_episodes: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Query experiment results from your database.
        
        Uses your existing ORM layer.
        """
        from sqlalchemy import select
        from stephanie.data.orm import EvaluationORM
        
        with self.memory.session() as s:
            stmt = (
                select(EvaluationORM)
                .where(EvaluationORM.source == "experiment_governed")
                .where(EvaluationORM.meta["experiment_episode"].astext.in_(
                    [str(ep) for ep in experiment_episodes]
                ))
                .order_by(EvaluationORM.created_at)
            )
            
            results = s.execute(stmt).scalars().all()
            
            return [
                {
                    "episode": eval.meta.get("experiment_episode"),
                    "bundle": eval.scores,  # Your stored bundle dict
                    "dominance": eval.meta.get("dominance_achieved"),
                }
                for eval in results
            ]