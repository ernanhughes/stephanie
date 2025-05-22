from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class Score:
    goal: str
    hypothesis: str
    agent_name: str
    model_name: str
    evaluator_name: str
    score_type: str
    score: Optional[float] = None
    score_text: Optional[str] = None
    strategy: Optional[str] = None
    reasoning_strategy: Optional[str] = None
    rationale: Optional[str] = None
    reflection: Optional[str] = None       # NEW
    review: Optional[str] = None           # NEW
    meta_review: Optional[str] = None      # NEW
    run_id: Optional[str] = None
    metadata: Optional[dict] = None
    created_at: Optional[datetime] = None


    def set_score(self, value):
        self.score, self.score_text = self.to_float_or_text(value)

    def store(self, memory, logger=None):
        """
        Saves this score object to the database using the memory object.
        Resolves goal_id and hypothesis_id using goal/hypothesis text.
        """
        try:
            goal_id = memory.hypotheses.get_or_create_goal(self.goal)
            hypothesis_id = memory.hypotheses.get_id_by_text(self.hypothesis)

            if goal_id is None or hypothesis_id is None:
                raise ValueError("Missing goal_id or hypothesis_id")

            memory.hypotheses.insert_score(
                {
                    "goal_id": goal_id,
                    "hypothesis_id": hypothesis_id,
                    "agent_name": self.agent_name,
                    "model_name": self.model_name,
                    "evaluator_name": self.evaluator_name,
                    "score_type": self.score_type,
                    "score": self.score,
                    "score_text": self.score_text,
                    "strategy": self.strategy,
                    "reasoning_strategy": self.reasoning_strategy,
                    "rationale": self.rationale,
                    "reflection": self.reflection,
                    "review": self.review,
                    "meta_review": self.meta_review,
                    "run_id": self.run_id,
                    "metadata": self.metadata or {},
                }
            )

            if logger:
                logger.log(
                    "ScoreStored",
                    {
                        "goal_id": goal_id,
                        "hypothesis_id": hypothesis_id,
                        "score": self.score,
                        "strategy": self.strategy,
                        "score_type": self.score_type,
                    },
                )

        except Exception as e:
            if logger:
                logger.log("ScoreStorageError", {"error": str(e)})
            raise

    @staticmethod
    def to_float_or_text(value):
        try:
            return float(value), None
        except (ValueError, TypeError):
            return None, str(value)

    @staticmethod
    def build(goal: str, hypothesis: str, cfg: dict):
        model_name = cfg.get("model", {}).get("name")
        evaluator_name = cfg.get("judge", "mrq")
        score_type = cfg.get("name", "undefined")
        agent_name = cfg.get("name", "undefined")
        return Score(goal=goal, hypothesis=hypothesis,model_name=model_name, score_type=score_type, agent_name=agent_name, evaluator_name=evaluator_name)
