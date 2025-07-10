from typing import Any, Dict

from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType


class ResearchEnv:
    """
    A simulated environment for testing hypotheses in a research context.
    Returns feedback based on internal knowledge and hypothesis scoring.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self):
        """
        Load static knowledge used for validating hypotheses.
        Can be extended to use vector DB or embedding search.
        """
        return {
            "Arthur’s Magazine": {
                "started": 1846,
                "merged_into": "Godey’s Lady’s Book"
            },
            "First for Women": {
                "started": 1989
            }
        }

    def reset(self, goal: str) -> str:
        """
        Start a new research task.
        """
        self.current_goal = goal
        return f"Goal: {goal}"

    def step(self, hypothesis: str) -> Dict[str, Any]:
        """
        Take a hypothesis as action, simulate feedback using scoring.
        Returns:
            dict: {
                "text": <natural language feedback>,
                "reward": <numerical score>,
                "success": <bool>
            }
        """
        # Simulate a hypothesis ORM object
        from stephanie.agents.mixins.scoring_mixin import ScoringMixin
        class DummyAgent(ScoringMixin): pass

        dummy_agent = DummyAgent({})
        scorable = Scorable(text=hypothesis, target_type=TargetType.HYPOTHESIS)
        score_result = dummy_agent.score_item(
            scorable, {"goal": {"goal_text": self.current_goal}}, metrics="reason"
        )

        # Build simulated feedback
        success = score_result["score"] > 70
        feedback = {
            "text": f"Hypothesis scored {score_result['score']} across dimensions.",
            "score": score_result,
            "dimensions": score_result["scores"],
            "reward": score_result["score"] / 100,
            "success": success
        }

        return feedback