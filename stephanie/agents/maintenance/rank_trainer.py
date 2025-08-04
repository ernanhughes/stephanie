from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.mrq.preference_pair_builder import PreferencePairBuilder
from stephanie.scoring.training.rank_trainer import ContrastiveRankerTrainer


class ContrastiveRankerTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.pair_builder = PreferencePairBuilder(memory.session, logger)
        self.model_type = "contrastive_ranker"
        self.min_pairs = cfg.get("min_pairs", 100)
        self.dimensions = cfg.get("dimensions", [])
        self.trainer = ContrastiveRankerTrainer(cfg, memory=memory, logger=logger)
        self.logger.log("AgentInitialized", {
            "agent": "ContrastiveRankerTrainerAgent",
            "dimensions": self.trainer.dimensions,
            "min_pairs": self.trainer.min_pairs
        })

    async def run(self, context: dict) -> dict:
        """
        Agent entry point to train Contrastive Ranker models for all configured dimensions.
        
        This agent:
        1. Fetches pairwise preference data for each dimension
        2. Trains a contrastive learning model that learns from preferences
        3. Calibrates the model to produce absolute scores using baseline comparison
        4. Returns training statistics for monitoring
        
        Expected context:
        {
            "goal": {
                "id": "goal-123",
                "goal_text": "Improve response helpfulness"
            }
        }
        
        Returns:
        {
            "training_stats": {
                "helpfulness": {
                    "dimension": "helpfulness",
                    "model_type": "contrastive_ranker",
                    "training_pairs": 120,
                    "calibration_samples": 85
                },
                ...
            }
        }
        """
        goal = context.get("goal", {})
        goal_text = goal.get("goal_text", "")
        
        self.logger.log("ContrastiveRankerTrainingStarted", {
            "goal_text": goal_text,
            "dimensions": self.trainer.dimensions
        })
        
        results = {}
        for dimension in self.trainer.dimensions:
            self.logger.log("DimensionTrainingStart", {
                "dimension": dimension,
                "goal_text": goal_text
            })
            
            # Get pairwise preference data for this dimension
            pairs_by_dim = self.pair_builder.get_training_pairs_by_dimension(
                dim=[dimension],
                goal=goal_text,
                limit=self.cfg.get("pair_limit", 200)
            )
            
            samples = pairs_by_dim.get(dimension, [])
            samples = [{**s, "preferred": "A"} for s in samples]

            if not samples:
                self.logger.log("NoPreferencePairsFound", {
                    "dimension": dimension,
                    "goal_text": goal_text,
                    "min_required": self.trainer.min_pairs
                })
                continue
                
            # Train the contrastive ranker model
            stats = self.trainer.train(samples, dimension)
            
            if "error" in stats:
                self.logger.log("TrainingFailed", {
                    "dimension": dimension,
                    "error": stats["error"],
                    "found_pairs": len(samples),
                    "required_pairs": self.trainer.min_pairs
                })
                continue
                
            self.logger.log("DimensionTrainingComplete", {
                **stats,
                "dimension": dimension
            })
            results[dimension] = stats

        # Update context with results
        context["training_stats"] = results
        context["model_type"] = "contrastive_ranker"
        
        self.logger.log("ContrastiveRankerTrainingComplete", {
            "trained_dimensions": list(results.keys()),
            "total_pairs": sum(stats.get("training_pairs", 0) for stats in results.values())
        })
        
        return context