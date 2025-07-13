from stephanie.memcubes.belief import Belief
from stephanie.memcubes.memcube import MemCube, MemCubeFactory
from stephanie.memcubes.scorable import Scorable, TargetType

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
    
    # In your self-improvement loop
def generate_hypotheses(self, world_model: WorldModel):
    hypothesis_engine = HypothesisEngine(world_model)
    hypotheses = hypothesis_engine.generate_hypotheses(world_model.goal)
    
    # Score hypotheses
    scored = []
    for h in hypotheses:
        score = self.ebt.get_energy(world_model.goal, h.content)
        score = torch.sigmoid(torch.tensor(score)).item()
        h.relevance = score
        scored.append(h)
    
    # Select top hypothesis
    best = max(scored, key=lambda x: x.relevance * x.strength)
    return best