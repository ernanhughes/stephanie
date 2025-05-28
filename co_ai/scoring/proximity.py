from co_ai.models import HypothesisORM
from co_ai.scoring.base_score import BaseScore
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ProximityScore(BaseScore):
    name = "proximity"
    default_value = 0.0

    def compute(self, hypothesis:HypothesisORM, context:dict) -> float:
        emb_hyp = self.memory.embedding.get_or_create(hypothesis.text)
        goal = self.memory.goals.get_by_id(hypothesis.goal_id)
        emb_goal = self.memory.embedding.get_or_create(goal.goal_text)
        h_vec = np.array(emb_hyp).reshape(1, -1)
        g_vec = np.array(emb_goal).reshape(1, -1)
        score = cosine_similarity(h_vec, g_vec)[0][0]
        return round((score + 1) / 2, 4)
