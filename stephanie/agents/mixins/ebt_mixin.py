# stephanie/agents/mixins/ebt_mixin.py
import os
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.nn.functional import sigmoid

from stephanie.scoring.model.ebt_model import EBTModel
from stephanie.utils.file_utils import load_json
from stephanie.utils.model_utils import (discover_saved_dimensions,
                                         get_model_path)


class EBTMixin:
    """
    Mixin that provides EBT-based verification, refinement, and scoring.
    Can be added to any class that has:
    - self.cfg (configuration)
    - self.memory (with embedding store)
    - self.logger (optional)
    """
    
    def __init__(self, ebt_cfg: Dict):
        """
        Initialize EBT functionality from config
        
        Args:
            ebt_cfg: Configuration dict with:
                - model_path: Base path for saved models
                - model_type: Type of EBT model (default: "ebt")
                - target_type: Type of target (default: "document")
                - model_version: Model version (default: "v1")
                - dimensions: List of dimensions to load
                - device: Computation device ("cuda" or "cpu")
                - uncertainty_threshold: Energy threshold for uncertainty
        """
        # Configuration
        self.ebt_cfg = ebt_cfg
        self.model_path = ebt_cfg.get("model_path", "models")
        self.model_type = ebt_cfg.get("model_type", "ebt")
        self.target_type = ebt_cfg.get("target_type", "document")
        self.model_version = ebt_cfg.get("model_version", "v1")
        self.embedding_type = self.memory.embedding.type
        self.dimensions = ebt_cfg.get("dimensions", [])
        self.uncertainty_threshold = ebt_cfg.get("uncertainty_threshold", 0.75)
        
        # Device
        self.device = torch.device(
            ebt_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Model storage
        self.ebt_models: Dict[str, nn.Module] = {}
        self.ebt_meta: Dict[str, Dict] = {}
        
        # Initialize models
        self._initialize_ebt_models()
    
    def _initialize_ebt_models(self):
        """Load EBT models and metadata"""
        if not self.dimensions:
            self.dimensions = discover_saved_dimensions(
                model_type=self.model_type,
                target_type=self.target_type,
                base_path=self.model_path
            )
        
        for dim in self.dimensions:
            model_path = get_model_path(
                self.model_path,
                self.model_type,
                self.target_type,
                dim,
                self.model_version,
                self.embedding_type
            )
            infer_path = os.path.join(model_path, f"{dim}.pt")
            meta_path = os.path.join(model_path, f"{dim}.meta.json")
            
            try:
                # Load model
                model = self._load_ebt_model(infer_path)
                self.ebt_models[dim] = model
                
                # Load metadata
                if os.path.exists(meta_path):
                    self.ebt_meta[dim] = load_json(meta_path)
                else:
                    self.ebt_meta[dim] = {"min": 40, "max": 100}
                    
            except Exception as e:
                if self.ebt_cfg.get("strict_load", True):
                    raise RuntimeError(f"Failed to load EBT model for {dim}: {e}")
                else:
                    print(f"Skipping EBT model for {dim} due to error: {e}")
    
    def _load_ebt_model(self, path: str) -> nn.Module:
        """Load EBT model from disk"""
        model = EBTModel().to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model
    
    def get_energy(self, goal: str, text: str, dimension: Optional[str] = None) -> float:
        """Get raw energy value for a document-goal pair"""
        target_dim = dimension or next(iter(self.ebt_models.keys()), None)
        if not target_dim:
            raise ValueError("No EBT models available")
            
        ctx_emb = torch.tensor(self.memory.embedding.get_or_create(goal)).to(self.device)
        doc_emb = torch.tensor(self.memory.embedding.get_or_create(text)).to(self.device)
        
        with torch.no_grad():
            energy = self.ebt_models[target_dim](ctx_emb, doc_emb).squeeze().cpu().item()
        
        return energy
    
    def optimize(self, goal: str, text: str, dimension: Optional[str] = None,
                 steps: int = 10, step_size: float = 0.05) -> Dict:
        """
        Optimize document text by minimizing energy through gradient descent
        
        Returns:
            dict: {
                "refined_text": str,
                "final_energy": float,
                "energy_trace": List[float],
                "converged": bool
            }
        """
        target_dim = dimension or next(iter(self.ebt_models.keys()), None)
        if not target_dim:
            raise ValueError("No EBT models available")
            
        model = self.ebt_models[target_dim]
        ctx_emb = torch.tensor(self.memory.embedding.get_or_create(goal)).to(self.device)
        doc_emb = torch.tensor(self.memory.embedding.get_or_create(text)).to(self.device)
        
        # Make document embedding differentiable
        doc_tensor = doc_emb.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([doc_tensor], lr=step_size)
        
        energy_trace = []
        for step in range(steps):
            optimizer.zero_grad()
            energy = model(ctx_emb, doc_tensor)
            energy.backward()
            optimizer.step()
            energy_trace.append(energy.item())
        
        refined_emb = doc_tensor.detach()
        refined_text = self._embedding_to_text(refined_emb, goal, text)
        
        return {
            "refined_text": refined_text,
            "final_energy": energy_trace[-1],
            "energy_trace": [round(e, 4) for e in energy_trace],
            "converged": abs(energy_trace[-1] - energy_trace[0]) < 0.05,
            "dimension": target_dim
        }
    
    def is_unstable(self, goal: str, text: str, dimension: Optional[str] = None) -> bool:
        """Check if prediction is uncertain using energy + gradient magnitude"""
        energy = self.get_energy(goal, text, dimension)
        ctx_emb = torch.tensor(self.memory.embedding.get_or_create(goal)).to(self.device)
        doc_emb = torch.tensor(self.memory.embedding.get_or_create(text)).to(self.device)
        
        doc_tensor = doc_emb.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            energy = self.ebt_models[dimension or next(iter(self.ebt_models))](ctx_emb, doc_tensor)
            grad = torch.autograd.grad(energy, doc_tensor, retain_graph=False)[0]
        
        grad_norm = torch.norm(grad).item()
        uncertainty_score = abs(energy.item()) + grad_norm
        
        return uncertainty_score > self.uncertainty_threshold
    
    def score_document(self, goal: str, text: str, dimension: Optional[str] = None) -> float:
        """Get final scaled score from EBT model"""
        energy = self.get_energy(goal, text, dimension)
        meta = self.ebt_meta.get(dimension or next(iter(self.ebt_models)), {"min": 40, "max": 100})
        
        normalized = sigmoid(torch.tensor(energy)).item()
        final_score = normalized * (meta["max"] - meta["min"]) + meta["min"]
        return round(final_score, 4)
    
    def _embedding_to_text(self, embedding, goal, original_text):
        """Convert refined embedding back to text"""
        if self.ebt_cfg.get("use_llm_refinement", False):
            from stephanie.agents.llm import LLMGenerator
            llm = LLMGenerator(self.ebt_cfg.get("llm_config", {}))
            prompt = f"""Improve the following text to better align with this goal:

Goal: {goal}
Original Text: {original_text}

Refine it while preserving content."""
            return llm.generate(prompt)
        
        # Fallback to nearest neighbor search
        return self.memory.embedding.find_closest(embedding, k=1)[0].text
    
    def get_all_scores(self, goal: str, text: str) -> Dict[str, float]:
        """Get scores across all dimensions"""
        scores = {}
        for dim in self.dimensions:
            scores[dim] = self.score_document(goal, text, dim)
        return scores