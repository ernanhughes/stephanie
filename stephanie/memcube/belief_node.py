# stephanie/orm/memcube/belief_node.py
from __future__ import annotations

from stephanie.memcube.memcube import MemCube
from stephanie.memcube.memcube_factory import MemCubeFactory


class BeliefNode:
    def __init__(self, memcube: MemCube):
        self.memcube = memcube
        self.children = []
        self.parent = None
        self.score = None
        self.version_history = [memcube.version]
    
    def refine(self, goal: str):
        """Refine belief using EBT"""
        refined = self.ebt.optimize(goal, self.memcube.scorable.text)
        refined_memcube = MemCubeFactory.from_scorable(
            self.memcube.scorable,
            version=f"v{int(self.memcube.version[1:]) + 1}"
        )
        refined_memcube.scorable.text = refined["refined_text"]
        refined_memcube.metadata["refinement_trace"] = refined["energy_trace"]
        
        # Update node with new MemCube
        self.memcube = refined_memcube
        self.version_history.append(refined_memcube.version)
        
        return self.memcube