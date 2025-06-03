from co_ai.models import ScoreORM
from co_ai.scoring.proximity import ProximityScore
from co_ai.constants import PROXIMITY


class ProximityScoringMixin:
    """
    A mixin that provides proximity scoring functionality to any agent.
    Can be used in ProximityAgent, MetaReviewAgent, SharpeningAgent, etc.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proximity_scorer = None  # Will be initialized on first use

    def get_proximity_scorer(self) -> ProximityScore:
        """
        Lazily initialize the ProximityScore instance.
        """
        if not self.proximity_scorer:
            self.proximity_scorer = ProximityScore(self.cfg, memory=self.memory, logger=self.logger)
        return self.proximity_scorer

    def score_hypothesis_with_proximity(self, hyp: dict, context: dict) -> dict:
        """
        Score a hypothesis using proximity analysis.

        Args:
            hyp (dict): Hypothesis dictionary containing "text" and optionally "id".
            context (dict): Execution context including goal, pipeline_run_id, etc.

        Returns:
            float: The computed proximity score.
        """
        hyp_text = hyp.get("text")
        hyp_id = self.get_hypothesis_id(hyp)

        # Get goal from context
        goal = context.get("goal")

        # Build prompt context for summary generation
        prompt_context = {
            "goal": goal,
            "goal_text": goal.get("goal_text"),
            "hypothesis": hyp_text
        }

        # Load prompt and generate proximity summary
        summary_prompt = self.prompt_loader.load_prompt(self.cfg, prompt_context)
        summary_output = self.call_llm(summary_prompt, prompt_context)

        # Use ProximityScore to evaluate the summary
        scorer = self.get_proximity_scorer()
        score = scorer.get_score(hyp, context)

        # Log the scoring event
        if self.logger:
            self.logger.log("ProximityScoreComputed", {
                "hypothesis_id": hyp_id,
                "score": score,
                "analysis_snippet": summary_output[:300]
            })

        # Save score to DB
        self._save_proximity_score(hyp, context, score)
        return {"id": hyp_id, "score": score, PROXIMITY: summary_output, "scores": scorer.scores}


    def _save_proximity_score(self, hyp: dict, context: dict, score: float):
        """
        Save the proximity score to the database.
        """
        goal = context.get("goal")
        score_obj = ScoreORM(
            agent_name=self.name,
            model_name=self.model_name,
            goal_id=goal.get("id"),
            hypothesis_id=hyp.get("id"),
            score_type="proximity",
            evaluator_name=self.name,
            score=score,
            extra_data={"score": score},
            pipeline_run_id=context.get("pipeline_run_id"),
        )
        self.memory.scores.insert(score_obj)

    def _store_score(self, hypothesis: dict, context: dict, dimension: str, data: dict):
        score_obj = ScoreORM(
            goal_id=hypothesis.get("goal_id"),
            hypothesis_id=hypothesis.get("id"),
            agent_name=self.agent_name,
            model_name=self.model_name,
            evaluator_name="ProximityScore",
            score_type=dimension,
            score=data["score"],
            rationale=data.get("rationale", ""),
            pipeline_run_id=context.get("pipeline_run_id"),
            metadata={"source": "structured_reflection"},
        )
        self.memory.scores.insert(score_obj)

