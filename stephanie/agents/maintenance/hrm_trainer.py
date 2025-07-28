# stephanie/agents/maintenance/hrm_trainer_agent.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.sicql_scorer import SICQLScorer
from stephanie.scoring.training.hrm_trainer import HRMTrainer # Adjust path/name as needed
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType

class HRMTrainerAgent(BaseAgent):
    """
    Agent to train the Hierarchical Reasoning Model (HRM) for a specific task,
    e.g., predicting SICQL Q-values based on goal/document embeddings.
    """
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_type = "hrm"
        # The specific task/dimension name for this HRM instance (e.g., "sicql_aligned", "process_quality")
        self.hrm_dimension = cfg.get("hrm_dimension", "sicql_aligned") 
        # The scorer dimension to use as the training target (e.g., "alignment", "relevance")
        self.target_scorer_dimension = cfg.get("target_scorer_dimension", "alignment") 

        self.trainer = HRMTrainer(cfg.get("hrm", {}), memory, logger) 
        self.scorer = SICQLScorer(cfg.get("sicql", {}), memory, logger) 

    async def run(self, context: dict) -> dict:
        goal = context.get("goal", {})
        goal_text = goal.get("goal_text", "")
        
        self.logger.log("HRMTrainingAgentStarted", {
            "hrm_dimension": self.hrm_dimension,
            "target_dimension": self.target_scorer_dimension,
            "goal_id": goal.get("id", "unknown")
        })
        
        documents = context.get(self.input_key, [])
        training_samples = []


        for doc in documents:
            try:
                scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)
                
                # Score the document using SICQL to get the target value
                score_bundle = self.scorer.score(
                    goal=goal,
                    scorable=scorable,
                    dimensions=[self.target_scorer_dimension] # Score only the target dimension
                )
                
                score_result = score_bundle.results.get(self.target_scorer_dimension)
                if not score_result:
                    self.logger.log("HRMTrainingAgentWarning", {
                        "message": f"SICQL did not return score for dimension '{self.target_scorer_dimension}'.",
                        "doc_id": scorable.id
                    })
                    continue
                
                target_q_value = score_result.q_value # Use SICQL's Q-value as the target
                if target_q_value is None:
                    self.logger.log("HRMTrainingAgentWarning", {
                        "message": f"SICQL ScoreResult for '{self.target_scorer_dimension}' missing q_value.",
                        "doc_id": scorable.id
                    })
                    continue

                # Prepare the sample for HRM training
                training_samples.append({
                    "context_text": goal_text, # Goal text
                    "document_text": scorable.text, # Document text
                    "target_score": target_q_value # SICQL Q-value as target
                    # Could also log the source score for debugging
                    # "source_score_details": score_result.to_dict() 
                })
                
            except Exception as e:
                 self.logger.log("HRMTrainingAgentDataError", {
                     "message": "Error processing document for HRM training.",
                     "doc_id": doc.get("id", "unknown"),
                     "error": str(e)
                 })
                 # Continue with other documents
                 continue

        if not training_samples:
            self.logger.log("HRMTrainingAgentError", {"message": "No valid training samples prepared after scoring."})
            context[self.output_key] = {"status": "failed", "message": "No valid training samples prepared."}
            return context

        self.logger.log("HRMTrainingDataPrepared", {"num_samples": len(training_samples)})

        try:
            # Execute the core training logic
            # This should return a result dict like {"status": "trained", "final_loss": ...}
            training_result = self.trainer.train(samples=training_samples, dimension=self.hrm_dimension) 
            
            self.logger.log("HRMTrainingAgentCompleted", {
                "hrm_dimension": self.hrm_dimension,
                "result": training_result
            })
            
            # --- 3. Update Context ---
            context[self.output_key] = {
                "status": training_result.get("status", "unknown"),
                "training_result": training_result,
                "hrm_dimension": self.hrm_dimension,
                "target_dimension": self.target_scorer_dimension
            }
            
        except Exception as e:
            self.logger.log("HRMTrainingAgentError", {
                "message": "Error during HRM core training execution.",
                "hrm_dimension": self.hrm_dimension,
                "error": str(e)
            })
            context[self.output_key] = {
                "status": "failed",
                "message": f"HRM training failed: {e}",
                "hrm_dimension": self.hrm_dimension
            }
            
        return context
