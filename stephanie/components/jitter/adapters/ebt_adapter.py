from __future__ import annotations
import math
import random

import torch


class EBTAdapter:
    """Adapter for integrating with Stephanie's EBT scoring system"""
    
    def __init__(self, ebt_system):
        self.ebt = ebt_system
        self.core_identity = self._initialize_core_identity()
    
    def _initialize_core_identity(self) -> torch.Tensor:
        """Create or load the core identity vector"""
        try:
            return self.ebt.load_vector("core_identity")
        except:
            # Generate random core identity if none exists
            core = torch.randn(1024)
            core = core / (core.norm() + 1e-8)
            self.ebt.save_vector("core_identity", core)
            return core
    
    def score_compatibility(self, emb: torch.Tensor) -> float:
        """Score compatibility with core identity (0-1, 1=perfect match)"""
        emb = emb.view(-1)
        emb = emb / (emb.norm() + 1e-8)
        cos = float(torch.clamp(torch.dot(self.core_identity, emb), -1, 1))
        # Map cosine similarity to compatibility score
        return 0.5 * (1 + cos)