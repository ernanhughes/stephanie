
from stephanie.agents.mixins.ebt_mixin import EBTMixin
from stephanie.memcubes.theorem import Theorem


class TheoremValidator:
    def __init__(self, ebt: EBTMixin):
        self.ebt = ebt
    
    def validate(self, theorem: Theorem) -> float:
        """Validate theorem against goal"""
        # Check premise-conclusion compatibility
        energy = 0.0
        for premise in theorem.premises:
            energy += self.ebt.get_energy(theorem.conclusion, premise)
        
        # Normalize
        return 1 - (energy / len(theorem.premises))