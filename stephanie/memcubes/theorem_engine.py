from stephanie.memcubes.theorem_validator import TheoremValidator
from stephanie.memcubes.theorem import Theorem


class TheoremEngine:
    def __init__(self, memory, belief_graph):
        self.memory = memory
        self.belief_graph = belief_graph
        self.validator = TheoremValidator(self.memory.ebt)

    def _find_root(self):
        return next(iter(self.belief_graph.nodes))

    def _find_goal_node(self):
        return next(iter(self.belief_graph.nodes))

    def _build_theorem(self, path):
        # Logic to build a theorem from the path
        pass

    def _is_valid_theorem(self, theorem):
        return self.validator.validate(theorem)

    # In theorem_engine.py
    def extract_theorems(self, belief_graph: nx.DiGraph):
        """Extract validated theorems from belief graph"""
        theorems = []
        for path in nx.all_simple_paths(belief_graph, source=self._find_root(), target=self._find_goal_node()):
            theorem = self._build_theorem(path)
            if self._is_valid_theorem(theorem):
                theorems.append(theorem)
        
        # Save to database
        for theorem in theorems:
            self.memory.db.execute(
                "INSERT INTO theorems VALUES (...)",
                theorem.to_dict()
            )
        return theorems
    


    
    def apply_theorem(self, theorem: Theorem, context: str):
        # Use theorem to transform context
        result = context
        for premise in theorem.premises:
            if self._is_valid_premise(premise):
                result = self.world_model.apply_premise(result, premise)
        
        # Validate final result
        energy = self.world_model.ebt.get_energy(result, theorem.conclusion)
        if energy > 0.8:
            theorem.strength = max(0.0, theorem.strength - 0.1)
        
        return result
    
    def _is_valid_premise(self, premise: str) -> bool:
        # Use EBT to check premise validity
        energy = self.world_model.ebt.get_energy(self.world_model.goal, premise)
        return energy < 0.5  # Valid if energy low