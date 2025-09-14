# stephanie/agents/maintenance/hrm_trainer_agent.py
from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.scorer.sicql_scorer import SICQLScorer
from stephanie.scoring.training.hrm_trainer import HRMTrainer


class HRMTrainerAgent(BaseAgent):
    """
    Agent to train the Hierarchical Reasoning Model (HRM) for multiple dimensions.
    Uses SICQL Q-values as training targets for each goal/document pair.
    """
    def __init__(self, cfg, memory, container, logger, full_cfg):
        super().__init__(cfg, memory, container, logger)
        self.dimensions = cfg.get("dimensions", [])  # e.g., ["alignment", "relevance"]

        self.trainer = HRMTrainer(full_cfg.scorer.hrm, memory, container=container, logger=logger)
        self.scorer = SICQLScorer(full_cfg.scorer.sicql, memory, container=container, logger=logger)
        self.use_context_for_training = cfg.get("use_context_for_training", False)
        self.max_documents = cfg.get("max_documents", 500)

    async def run(self, context: dict) -> dict:
        goal = context.get("goal", {})
        goal_text = goal.get("goal_text", "")
        if self.use_context_for_training:
            documents = context.get(self.input_key, [])
            if documents:
                self.logger.log("HRMTrainingAgentInfo", {
                    "message": "Using context for training.",
                    "input_key": self.input_key,
                    "num_documents": len(documents)
                })
        else:
            documents = self.memory.documents.get_all(limit=self.max_documents)

        if not documents:
            self.logger.log("HRMTrainingAgentError", {
                "message": "No documents provided for training.",
                "input_key": self.input_key
            })

        documents = [d.to_dict() for d in self.memory.documents.get_all(limit=self.max_documents)]
        self.logger.log("HRMTrainingAgentInfo", {
            "message": "Retrieved documents for training.",
            "input_key": self.input_key,
            "num_documents": len(documents)
        })

        dimensional_training_samples = {dim: [] for dim in self.dimensions}

        for doc in tqdm(documents, desc="HRM Training: Processing documents", unit="doc"):
            try:
                scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)
                score_bundle = self.scorer.score(
                    context=context,
                    scorable=scorable,
                    dimensions=self.dimensions
                )

                for dimension in self.dimensions:
                    score_result = score_bundle.results.get(dimension)
                    if not score_result:
                        self.logger.log("HRMTrainingAgentWarning", {
                            "message": f"Missing q_value for dimension '{dimension}'",
                            "doc_id": scorable.id
                        })
                        continue

                    dimensional_training_samples[dimension].append({
                        "goal_text": goal_text,
                        "scorable_text": scorable.text,
                        "target_score": score_result.attributes.get("q_value")
                    })

            except Exception as e:
                self.logger.log("HRMTrainingAgentDataError", {
                    "message": "Error processing document.",
                    "doc_id": doc.get("id", "unknown"),
                    "error": str(e)
                })

        # Log how many samples were prepared
        for dim, samples in dimensional_training_samples.items():
            self.logger.log("HRMTrainingDataPrepared", {
                "dimension": dim,
                "num_samples": len(samples)
            })

        # Train the HRM per dimension
        training_results = {}
        try:
            for dimension, samples in dimensional_training_samples.items():
                if not samples:
                    training_results[dimension] = {"status": "skipped", "reason": "no samples"}
                    continue

                result = self.trainer.train(samples=samples, dimension=dimension)
                training_results[dimension] = result

                self.logger.log("HRMTrainingAgentCompleted", {
                    "dimension": dimension,
                    "result": result
                })

            # Update context with structured results
            context[self.output_key] = {
                "status": "completed",
                "dimensions": self.dimensions,
                "results": training_results,
            }

        except Exception as e:
            self.logger.log("HRMTrainingAgentError", {
                "message": "Error during HRM training execution.",
                "error": str(e)
            })
            context[self.output_key] = {
                "status": "failed",
                "message": str(e)
            }

        return context
