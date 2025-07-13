# stephanie/scoring/energy_tuned_mrq.py
import logging
from typing import Dict, List, Optional, Union

import torch
from torch.nn.functional import sigmoid


class EnergyTunedMRQ:
    """
    Combines MRQ and EBT models for energy-based refinement and scoring
    Uses EBT to verify and refine MRQ predictions via energy minimization
    """
    
    def __init__(self, ebt, mrq, config=None):
        """
        Args:
            ebt: EBTInferenceAgent instance
            mrq: MRQScorer instance
            config: Optional config dict with:
                - refine_threshold: Energy threshold for refinement
                - fallback_threshold: Energy threshold for LLM fallback
                - max_steps: Max optimization steps for EBT
                - step_size: Learning rate for EBT refinement
        """
        self.ebt = ebt
        self.mrq = mrq
        self.config = config or {}
        
        # Configuration
        self.refine_threshold = self.config.get("refine_threshold", 0.75)
        self.fallback_threshold = self.config.get("fallback_threshold", 0.9)
        self.max_steps = self.config.get("max_steps", 10)
        self.step_size = self.config.get("step_size", 0.05)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    def score(self, context: str, text: str, dimension: str = None) -> Dict:
        """
        Score document with EBT-tuned MRQ refinement
        
        Returns:
            {
                "score": float,
                "source": str,  # "ebt" or "mrq" or "llm"
                "refined": bool,
                "converged": bool,
                "energy": float,
                "uncertainty": float
            }
        """
        # Initial MRQ score
        mrq_score = self.mrq.score(context, text, dimension)
        raw_score = mrq_score["score"]
        
        # Get energy from EBT
        energy = self.ebt.get_energy(context, text, dimension)
        uncertainty = sigmoid(torch.tensor(energy)).item()
        
        self.logger.debug("InitialScore", {
            "dimension": dimension,
            "mrq_score": raw_score,
            "energy": energy,
            "uncertainty": uncertainty
        })
        
        # Track refinement state
        refined = False
        source = "mrq"
        final_score = raw_score
        refinement_steps = 0
        energy_trace = []
        
        # Step 1: Refinement if uncertain
        if uncertainty > self.refine_threshold:
            refinement_result = self._refine_document(context, text, dimension)
            refined_text = refinement_result["refined_text"]
            refinement_steps = refinement_result["steps_used"]
            energy_trace = refinement_result["energy_trace"]
            
            # Score refined document
            refined_score = self.mrq.score(context, refined_text, dimension)
            final_score = refined_score["score"]
            refined = True
            source = "ebt"
            
            self.logger.info("DocumentRefined", {
                "dimension": dimension,
                "steps_used": refinement_steps,
                "energy_trace": energy_trace
            })
        
        # Step 2: Fallback if still uncertain
        if refined and energy_trace and energy_trace[-1] > self.fallback_threshold:
            from stephanie.agents.llm import LLMScorer
            llm_scorer = LLMScorer(self.config.get("llm", {}))
            llm_score = llm_scorer.score(context, refined_text if refined else text, dimension)
            final_score = llm_score["score"]
            source = "llm"
            self.logger.warning("LLMFallbackUsed", {
                "dimension": dimension,
                "refined": refined,
                "final_energy": energy_trace[-1] if energy_trace else energy
            })
        
        return {
            "score": final_score,
            "source": source,
            "refined": refined,
            "converged": refinement_steps < self.max_steps if refinement_steps else True,
            "energy": energy,
            "uncertainty": uncertainty,
            "refinement_steps": refinement_steps,
            "energy_trace": energy_trace
        }
    
    def _refine_document(self, context: str, text: str, dimension: str) -> Dict:
        """Refine document using EBT optimization"""
        # Get embeddings
        ctx_emb = torch.tensor(self.ebt.memory.embedding.get_or_create(context)).to(self.device)
        doc_emb = torch.tensor(self.ebt.memory.embedding.get_or_create(text)).to(self.device)
        
        # Make differentiable
        doc_tensor = doc_emb.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([doc_tensor], lr=self.step_size)
        
        energy_trace = []
        for step in range(self.max_steps):
            optimizer.zero_grad()
            energy = self.ebt.models[dimension](ctx_emb, doc_tensor)
            energy.backward()
            optimizer.step()
            energy_trace.append(energy.item())
            
            # Early stopping
            if len(energy_trace) > 1:
                delta = abs(energy_trace[-1] - energy_trace[-2])
                if delta < self.config.get("min_delta", 0.01):
                    break
        
        # Convert refined embedding to text
        refined_emb = doc_tensor.detach()
        refined_text = self.ebt._embedding_to_text(refined_emb, context, text)
        
        return {
            "refined_text": refined_text,
            "final_energy": energy_trace[-1],
            "energy_trace": [round(e, 4) for e in energy_trace],
            "steps_used": len(energy_trace),
            "converged": len(energy_trace) < self.max_steps
        }
    
    def train_from_refinement(self, examples: List[Dict]):
        """
        Retrain MRQ using EBT-refined examples
        
        Args:
            examples: List of {
                "context": str,
                "original": str,
                "refined": str,
                "dimension": str
            }
        """
        # Convert refinement examples to MRQ training data
        training_pairs = []
        for example in examples:
            # Original vs refined
            original_score = self.mrq.score(
                example["context"], 
                example["original"], 
                example["dimension"]
            )["score"]
            
            refined_score = self.mrq.score(
                example["context"], 
                example["refined"], 
                example["dimension"]
            )["score"]
            
            # Create preference pair
            if refined_score > original_score:
                training_pairs.append({
                    "title": example["context"],
                    "output_a": example["refined"],
                    "output_b": example["original"],
                    "value_a": refined_score,
                    "value_b": original_score,
                    "dimension": example["dimension"]
                })
        
        # Train MRQ using preference pairs
        if training_pairs:
            self.mrq.train_multidimensional_model(training_pairs)
            self.logger.info("MRQRetrained", {
                "examples_used": len(training_pairs),
                "dimensions_updated": set(e["dimension"] for e in training_pairs)
            })
    
    def tune(self, context: str, candidate: str, dimension: str = None) -> Dict:
        """
        Tune MRQ using EBT energy feedback
        
        Args:
            context: Goal or prompt text
            candidate: Document or output to evaluate
            dimension: Optional scoring dimension
            
        Returns:
            {
                "improvement": float,
                "before_score": float,
                "after_score": float,
                "dimension": str,
                "refined": bool
            }
        """
        # Get current score
        before_score = self.mrq.score(context, candidate, dimension)["score"]
        
        # Refine using EBT
        refinement = self._refine_document(context, candidate, dimension)
        refined_text = refinement["refined_text"]
        
        # Score refined version
        after_score = self.mrq.score(context, refined_text, dimension)["score"]
        
        # Update MRQ if improvement
        if after_score > before_score:
            self.mrq.update_model(
                context, candidate, refined_text, 
                before_score, after_score
            )
            improvement = after_score - before_score
            self.logger.info("MRQTuned", {
                "dimension": dimension,
                "improvement": improvement,
                "before": before_score,
                "after": after_score
            })
            return {
                "improvement": improvement,
                "before_score": before_score,
                "after_score": after_score,
                "dimension": dimension,
                "refined": True
            }
        
        return {
            "improvement": 0.0,
            "before_score": before_score,
            "after_score": before_score,
            "dimension": dimension,
            "refined": False
        }
    
    def is_uncertain(self, context: str, text: str, dimension: str = None) -> bool:
        """Check if prediction is uncertain using EBT energy"""
        energy = self.ebt.get_energy(context, text, dimension)
        return abs(energy) > self.fallback_threshold
    
    def get_refinement_diff(self, original: str, refined: str) -> str:
        """Return text diff between original and refined versions"""
        from difflib import Differ
        return "\n".join(
            line for line in Differ().compare(original.split(), refined.split())
        )