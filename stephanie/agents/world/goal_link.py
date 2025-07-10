from sklearn.metrics.pairwise import cosine_similarity

from stephanie.models.goal import GoalORM
from stephanie.models.world_view import WorldviewORM


class GoalLinkingAgent:
    def __init__(self, embedding_manager, memory, logger):
        self.embedding = embedding_manager
        self.memory = memory
        self.logger = logger

    def find_or_link_worldview(self, goal: dict, threshold=0.88) -> WorldviewORM:
        """Given a new goal, find the best matching worldview (or create a new one)."""
        goal_embedding = self.embedding.get_or_create(goal["description"])
        candidates = WorldviewORM.load_all()
        
        best_match = None
        best_score = 0.0
        for candidate in candidates:
            sim = cosine_similarity(goal_embedding, candidate.embedding_vector)
            if sim > best_score:
                best_score = sim
                best_match = candidate

        if best_score >= threshold:
            self.logger.log("GoalLinkedToWorldview", {
                "goal": goal["description"],
                "worldview_id": best_match.id,
                "similarity": best_score
            })
            return best_match
        else:
            # Create new worldview
            new_view = WorldviewORM.create_from_goal(goal)
            self.logger.log("NewWorldviewCreatedFromGoal", {
                "goal": goal["description"],
                "worldview_id": new_view.id
            })
            return new_view

    def relate_to_belief_systems(self, goal: dict):
        """Suggest belief systems relevant to this goal"""
        goal_embedding = self.embedding.get_or_create(goal["description"])
        belief_docs = self.memory.belief.get_all()

        scored_beliefs = []
        for belief in belief_docs:
            belief_emb = self.embedding.get_or_create(belief["summary"])
            score = cosine_similarity(goal_embedding, belief_emb)
            scored_beliefs.append((score, belief))

        scored_beliefs.sort(reverse=True)
        return [b for score, b in scored_beliefs if score > 0.75]
