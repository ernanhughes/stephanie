# stephanie/agents/inference/scoring_memcube.py

import torch

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.inference.ebt_inference import EBTInferenceAgent
from stephanie.agents.inference.llm_inference import LLMInferenceAgent
from stephanie.agents.inference.mrq_inference import MRQInferenceAgent
from stephanie.memcubes.memcube_factory import MemCubeFactory
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType


class ScoringMemcubeAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.ebt_refine_threshold = cfg.get("ebt_refine_threshold", 0.7)
        self.llm_fallback_threshold = cfg.get("llm_fallback_threshold", 0.9)
        self.steps = cfg.get("optimization_steps", 10)
        self.step_size = cfg.get("step_size", 0.05)
        
        # Initialize versioned scorers
        self.ebt = EBTInferenceAgent(cfg.get("ebt"), self.memory, self.logger)
        self.mrq = MRQInferenceAgent(cfg.get("mrq"), self.memory, self.logger)
        self.llm = LLMInferenceAgent(cfg.get("llm"), self.memory, self.logger)

    async def run(self, context: dict) -> dict:
        goal_text = context["goal"]["goal_text"]
        docs = context[self.input_key]
        results = []
        
        for doc in docs:
            # Convert to MemCube
            scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)
            memcube = MemCubeFactory.from_scorable(scorable, version="auto")
            
            # Initial MRQ score
            mrq_scores = self.mrq.score(context, memcube.scorable)
            
            # Estimate uncertainty using EBT
            ebt_energy = self.ebt.get_energy(goal_text, memcube.scorable.text)
            uncertainty_by_dim = {
                dim: torch.sigmoid(torch.tensor(raw)).item()
                for dim, raw in ebt_energy.items()
            }
            
            # Refinement if uncertain
            if any(u > self.ebt_refine_threshold for u in uncertainty_by_dim.values()):
                refinement_result = self.ebt.optimize(goal_text, memcube.scorable.text)

                refined_scorable = ScorableFactory.from_dict({
                    "id":memcube.scorable.id,
                    "text":refinement_result["refined_text"],
                    "target_type":TargetType.REFINEMENT}
                )

                refined_memcube = MemCubeFactory.from_scorable(
                    refined_scorable,
                    version="auto"
                )

                refined_memcube.extra_data.update({
                    "refinement_trace": refinement_result["energy_trace"],
                    "original_memcube_id": memcube.id
                })

                # Now switch to using refined_memcube
                memcube = refined_memcube
            # LLM fallback if still uncertain
            if any(u > self.llm_fallback_threshold for u in uncertainty_by_dim.values()):
                llm_scores = self.llm.score(context, memcube.scorable)
                final_scores = llm_scores.to_dict()
                source = "llm"
            else:
                final_scores = mrq_scores
                source = "mrq"
            
            # Log MemCube with scores
            result = {
                "memcube": memcube.to_dict(),
                "scores": final_scores,
                "source": source,
                "uncertainty_by_dimension": uncertainty_by_dim,
            }
            
            # Save to memory
            self.memory.memcube.save_memcube(memcube)
            results.append(result)
        
        context[self.output_key] = results
        return context
    
    