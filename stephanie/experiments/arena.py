# stephanie/experiments/arena.py
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)

class Arena:
    """
    Reusable arena tournament framework for comparing and improving candidates.
    
    Usage:
        arena = Arena(
            score_fn=score_candidate,
            improve_fn=improve_candidate,
            max_rounds=5,
            beam_width=8,
            marginal_threshold=0.02
        )
        
        winner = arena.run(initial_candidates)
    """
    
    def __init__(
        self,
        score_fn: Callable[[Any], Dict[str, float]],
        improve_fn: Callable[[Any, Dict[str, Any]], Any],
        max_rounds: int = 5,
        beam_width: int = 8,
        marginal_threshold: float = 0.02,
        plateau_rounds: int = 2
    ):
        self.score_fn = score_fn
        self.improve_fn = improve_fn
        self.max_rounds = max_rounds
        self.beam_width = beam_width
        self.marginal_threshold = marginal_threshold
        self.plateau_rounds = plateau_rounds
    
    def run(self, initial_candidates: List[Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the arena tournament"""
        context = context or {}
        current_beam = self._initialize_beam(initial_candidates)
        iterations = []
        plateau_count = 0
        best_overall = 0.0
        
        for round_idx in range(self.max_rounds):
            # Score current beam
            scored = self._score_beam(current_beam)
            
            # Track best overall
            current_best = max(scored, key=lambda x: x["score"]["overall"])
            current_overall = current_best["score"]["overall"]
            marginal = current_overall - best_overall
            
            # Check stopping conditions
            if current_overall >= 0.95:  # Perfect score
                reason = "threshold_met"
                break
                
            if marginal < self.marginal_threshold and round_idx > 0:
                plateau_count += 1
                if plateau_count >= self.plateau_rounds:
                    reason = "plateau"
                    break
            else:
                plateau_count = 0
                best_overall = current_overall
            
            # Record iteration
            iterations.append({
                "round": round_idx,
                "beam": scored,
                "best_overall": current_overall,
                "marginal": marginal,
                "timestamp": time.time()
            })
            
            # Generate next beam
            if round_idx < self.max_rounds - 1:
                next_candidates = self._generate_next_candidates(scored, context)
                current_beam = self._prune_beam(next_candidates)
        
        # Return results
        return {
            "winner": current_best,
            "initial_pool": initial_candidates,
            "beam": current_beam,
            "iterations": iterations,
            "rounds_run": round_idx + 1,
            "reason": reason if "reason" in locals() else "max_rounds"
        }
    
    def _initialize_beam(self, candidates: List[Any]) -> List[Dict[str, Any]]:
        """Initialize the beam with scored candidates"""
        return [{
            "text": c,
            "origin": "seed",
            "variant": f"seed_{i}",
            "score": self.score_fn(c)
        } for i, c in enumerate(candidates)]
    
    def _score_beam(self, beam: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score all items in the beam"""
        return [{
            **item,
            "score": self.score_fn(item["text"])
        } for item in beam]
    
    def _generate_next_candidates(self, scored: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate next round candidates by improving top performers"""
        # Take top performers
        top_candidates = sorted(scored, key=lambda x: x["score"]["overall"], reverse=True)[:self.beam_width]
        
        # Generate improvements
        next_candidates = []
        for i, candidate in enumerate(top_candidates):
            improved = self.improve_fn(candidate["text"], {
                **context,
                "candidate": candidate,
                "round": len(context.get("iterations", []))
            })
            next_candidates.append({
                "text": improved,
                "origin": "improved",
                "variant": f"improved_{i}",
                "score": self.score_fn(improved)
            })
        
        # Include some diversity (keep some lower-scoring candidates)
        diversity_count = max(1, int(len(scored) * 0.2))
        diverse_candidates = sorted(
            scored, 
            key=lambda x: x["score"]["overall"]
        )[-diversity_count:]
        
        return next_candidates + diverse_candidates
    
    def _prune_beam(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prune to beam width based on score"""
        return sorted(
            candidates, 
            key=lambda x: x["score"]["overall"], 
            reverse=True
        )[:self.beam_width]