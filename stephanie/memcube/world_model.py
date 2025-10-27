import torch


class WorldModel:
    def __init__(self, beliefs: list):
        self.beliefs = beliefs

    def get_belief(self, belief_id: str):
        for belief in self.beliefs:
            if belief.id == belief_id:
                return belief
        return None

    def add_belief(self, belief):
        self.beliefs.append(belief)

    def remove_belief(self, belief_id: str):
        self.beliefs = [b for b in self.beliefs if b.id != belief_id]

    def to_dict(self):
        return {
            "beliefs": [b.to_dict() for b in self.beliefs]
        }
    
    # In your pipeline
    def build_world_model(self, context: dict):
        world_model = WorldModel(goal=context["goal"]["goal_text"])
        
        # Add Scorables to world model
        for scorable in context["scorables"]:
            cube = MemCubeFactory.from_scordable(scorable)
            world_model.ingest(cube)
        
        return world_model
    
def generate_hypotheses(self, world_model: WorldModel):
    hypothesis_engine = HypothesisEngine(world_model)
    hypotheses = hypothesis_engine.generate_hypotheses(world_model.goal)
    
    # Score hypotheses
    scored = []
    for h in hypotheses:
        score = self.ebt.get_energy(world_model.goal, h.text)
        score = torch.sigmoid(torch.tensor(score)).item()
        h.relevance = score
        scored.append(h)
    
    # Select top hypothesis
    best = max(scored, key=lambda x: x.relevance * x.strength)
    return best

class HypothesisEngine:
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    def generate_hypotheses(self, goal: str) -> list:
        # Generate hypotheses based on the current world model and goal
        pass
