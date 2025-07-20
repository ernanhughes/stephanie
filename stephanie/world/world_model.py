# stephanie/worldmodel/world_model.py
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import networkx as nx
import torch

from stephanie.agents.inference.ebt_inference import DocumentEBTInferenceAgent
from stephanie.memcubes.belief import Belief
from stephanie.memcubes.memcube import MemCube
from stephanie.memcubes.theorem import Theorem
from stephanie.scoring.scorable import Scorable
from stephanie.utils.file_utils import load_json, save_json
from stephanie.utils.model_utils import get_model_path


class WorldModel:
    """
    Structured knowledge base that evolves based on goal-directed reasoning
    Integrates belief graph, contradiction resolution, and theorem extraction
    """
    
    def __init__(self, goal: str, ebt_scorer: DocumentEBTInferenceAgent):
        """
        Args:
            goal: Primary objective for knowledge organization
            ebt_scorer: EBT agent for belief validation
        """
        self.goal = goal
        self.graph = nx.DiGraph()  # Belief graph with strength/relevance
        self.memory = []          # Short-term belief store
        self.ebt = ebt_scorer
        self.version = "v1"
        self.embedding_type = "default"
        
        self.created_at = datetime.utcnow()
        self.last_modified = self.created_at
        self.max_memory_size = 1000  # Max beliefs in short-term memory
        self.contradiction_threshold = 0.7  # Energy threshold for contradiction
        self.logger = logging.getLogger(__name__)
        
        # Load existing graph if available
        self._load_from_disk()
    
    def _load_from_disk(self):
        """Load belief graph from persistent storage if exists"""
        model_path = get_model_path("world_models", self.goal)
        graph_file = os.path.join(model_path, "belief_graph.pkl")
        
        if os.path.exists(graph_file):
            try:
                with open(graph_file, "rb") as f:
                    self.graph = pickle.load(f)
                self.logger.log("WorldModelLoaded", {
                    "goal": self.goal,
                    "nodes": len(self.graph.nodes),
                    "edges": len(self.graph.edges)
                })
            except Exception as e:
                self.logger.log("WorldModelLoadError", {
                    "goal": self.goal,
                    "error": str(e)
                })
    
    def save_to_disk(self):
        """Save belief graph to persistent storage"""
        model_path = get_model_path("world_models", self.goal)
        os.makedirs(model_path, exist_ok=True)
        
        # Save belief graph
        graph_file = os.path.join(model_path, "belief_graph.pkl")
        with open(graph_file, "wb") as f:
            pickle.dump(self.graph, f)
        
        # Save metadata
        meta_file = os.path.join(model_path, "meta.json")
        meta = {
            "goal": self.goal,
            "version": self.version,
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
            "last_modified": self.last_modified.isoformat(),
            "contradictions": self._count_contradictions()
        }
        save_json(meta, meta_file)
    
    def _count_contradictions(self) -> int:
        """Count contradictions in current belief graph"""
        contradictions = 0
        for u, v, data in self.graph.edges(data=True):
            if data.get("contradiction", False):
                contradictions += 1
        return contradictions
    
    def ingest(self, memcube: MemCube):
        """Add verified knowledge to world model"""
        # Skip low-reliability inputs
        reliability = memcube.scorable.scores.get("reliability", 0.5)
        if reliability < 0.7:
            return False
        
        # Create belief from MemCube
        belief = self._extract_belief(memcube)
        self.memory.append(belief)
        
        # Prune old memories
        if len(self.memory) > self.max_memory_size:
            self._prune_memory()
        
        # Add to belief graph
        self.graph.add_node(belief.id, data=belief)
        
        # Update connections
        self._update_connections(belief)
        
        # Track modifications
        self.last_modified = datetime.utcnow()
        self.save_to_disk()
        return belief
    
    def _extract_belief(self, memcube: MemCube) -> Belief:
        """Convert MemCube content to structured belief"""
        return Belief(
            id=f"{hash(memcube.scorable.text)}_{memcube.version}",
            content=memcube.scorable.text,
            strength=memcube.scorable.scores.get("novelty", 0.5),
            relevance=self._calculate_relevance(memcube),
            source=memcube.id,
            created_at=datetime.utcnow()
        )
    
    def _calculate_relevance(self, memcube: MemCube) -> float:
        """Calculate belief relevance to current goal"""
        # Use EBT to score goal alignment
        goal_emb = torch.tensor(self.ebt.memory.embedding.get_or_create(self.goal)).to(self.ebt.device)
        text_emb = torch.tensor(self.ebt.memory.embedding.get_or_create(memcube.scorable.text)).to(self.ebt.device)
        
        with torch.no_grad():
            energy = self.ebt.models["alignment"](goal_emb, text_emb).item()
        
        # Convert energy to relevance score
        return torch.sigmoid(torch.tensor(energy)).item()
    
    def _update_connections(self, new_belief: Belief):
        """Link new belief to similar existing beliefs"""
        # Find similar beliefs
        similar = []
        for belief in self.memory[-100:]:
            if belief.id == new_belief.id:
                continue
            # Use EBT to check compatibility
            energy = self.ebt.get_energy(belief.content, new_belief.content, "relevance")
            if energy < self.contradiction_threshold:
                similar.append((belief.id, energy))
        
        # Add edges to belief graph
        for belief_id, energy in similar:
            self.graph.add_edge(new_belief.id, belief_id, energy=energy)
            self.graph.add_edge(belief_id, new_belief.id, energy=energy)
    
    def _prune_memory(self):
        """Remove oldest beliefs when memory exceeds capacity"""
        # Sort by relevance * strength
        self.memory.sort(key=lambda b: b.relevance * b.strength)
        # Keep top 80%
        self.memory = self.memory[int(len(self.memory) * 0.2):]
        
        # Remove pruned beliefs from graph
        all_ids = set(b.id for b in self.memory)
        for node in list(self.graph.nodes):
            if node not in all_ids:
                self.graph.remove_node(node)
    
    def _find_root(self) -> str:
        """Find most foundational belief (highest out-degree)"""
        degrees = dict(self.graph.out_degree())
        return max(degrees, key=degrees.get) if degrees else None
    
    def _find_goal_node(self) -> str:
        """Find belief closest to goal (highest relevance)"""
        if not self.memory:
            return None
        return max(self.memory, key=lambda b: b.relevance).id
    
    def query(self, question: str) -> List[Belief]:
        """Answer questions using belief graph"""
        # Use EBT to find relevant beliefs
        relevant = []
        for belief in self.memory:
            energy = self.ebt.get_energy(question, belief.content, "relevance")
            relevance = torch.sigmoid(torch.tensor(energy)).item()
            if relevance > 0.6:
                relevant.append(belief)
        
        # Sort by relevance * strength
        return sorted(relevant, key=lambda b: b.relevance * b.strength, reverse=True)
    
    def detect_contradictions(self) -> List[Dict]:
        """Find contradictory beliefs in the graph"""
        contradictions = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("contradiction", False):
                contradictions.append({
                    "belief_a": u,
                    "belief_b": v,
                    "dimension": data["dimension"],
                    "score": data["score"]
                })
        return contradictions
    
    def resolve_contradictions(self):
        """Resolve detected contradictions using EBT verification"""
        contradictions = self.detect_contradictions()
        for c in contradictions:
            belief_a = self.graph.nodes[c["belief_a"]]["data"]
            belief_b = self.graph.nodes[c["belief_b"]]["data"]
            
            # Use EBT to verify which belief is better
            energy_a = self.ebt.get_energy(self.goal, belief_a.content)
            energy_b = self.ebt.get_energy(self.goal, belief_b.content)
            
            # Keep the belief with lower energy (better alignment)
            if energy_a < energy_b:
                self.graph.nodes[belief_b.id]["data"].strength -= 0.2
                if self.graph.nodes[belief_b.id]["data"].strength < 0.3:
                    self._remove_belief(belief_b.id)
            else:
                self.graph.nodes[belief_a.id]["data"].strength -= 0.2
                if self.graph.nodes[belief_a.id]["data"].strength < 0.3:
                    self._remove_belief(belief_a.id)
    
    def _remove_belief(self, belief_id: str):
        """Remove belief and its connections"""
        if belief_id in self.graph:
            self.graph.remove_node(belief_id)
        self.memory = [b for b in self.memory if b.id != belief_id]
        self.logger.log("BeliefRemoved", {
            "belief_id": belief_id,
            "reason": "low_strength"
        })
    
    def extract_theorems(self, min_premises=2) -> List[Theorem]:
        """Extract validated reasoning patterns from belief graph"""
        theorems = []
        for path in nx.all_simple_paths(self.graph, source=self._find_root(), target=self._find_goal_node()):
            if len(path) >= min_premises + 1:  # Premises + conclusion
                theorem = self._build_theorem(path)
                if self._validate_theorem(theorem):
                    theorems.append(theorem)
        return theorems
    
    def _build_theorem(self, path: List) -> Theorem:
        """Convert belief path into theorem"""
        premises = [self.graph.nodes[b]["data"].content for b in path[:-1]]
        conclusion = self.graph.nodes[path[-1]]["data"].content
        
        return Theorem(
            id=hash("".join(premises + [conclusion])),
            premises=premises,
            conclusion=conclusion,
            score=self._calculate_theorem_score(path),
            source="belief_graph",
            version=self.version
        )
    
    def _calculate_theorem_score(self, path: List) -> float:
        """Score theorem based on belief strength and relevance"""
        strength = sum(self.graph.nodes[b]["data"].strength for b in path)
        relevance = sum(self.graph.nodes[b]["data"].relevance for b in path)
        return (strength / len(path)) * (relevance / len(path))
    
    def _validate_theorem(self, theorem: Theorem) -> bool:
        """Use EBT to validate theorem premises → conclusion"""
        total_energy = 0.0
        for premise in theorem.premises:
            # Get energy between premise and conclusion
            energy = self.ebt.get_energy(premise, theorem.conclusion, "alignment")
            total_energy += energy
        
        avg_energy = total_energy / len(theorem.premises)
        theorem.score = torch.sigmoid(torch.tensor(avg_energy)).item()
        return theorem.score > self.ebt.contrastive_threshold
    
    def strengthen_beliefs(self, reward: float = 0.1):
        """Reinforce beliefs based on usage"""
        for belief in self.memory:
            belief.strength = min(1.0, belief.strength + reward * belief.relevance)
            self.graph.nodes[belief.id]["data"] = belief
    
    def decay_beliefs(self, decay_rate: float = 0.01):
        """Decay belief strength over time"""
        for belief in self.memory:
            belief.strength = max(0.0, belief.strength - decay_rate)
            self.graph.nodes[belief.id]["data"] = belief
            if belief.strength < 0.4:
                self._remove_belief(belief.id)
    
    def find_hypothesis(self, uncertainty_threshold: float = 0.7) -> Belief:
        """Find uncertain but relevant beliefs for hypothesis generation"""
        candidates = []
        for belief in self.memory:
            if belief.relevance > 0.6 and belief.strength < uncertainty_threshold:
                candidates.append(belief)
        
        if not candidates:
            return None
            
        # Select weakest but most relevant belief
        return max(candidates, key=lambda b: b.relevance * (1 - b.strength))
    
    def generate_hypotheses(self, count: int = 3) -> List[Belief]:
        """Generate hypotheses to improve weak beliefs"""
        candidates = []
        for _ in range(count):
            hypothesis = self.find_hypothesis()
            if not hypothesis