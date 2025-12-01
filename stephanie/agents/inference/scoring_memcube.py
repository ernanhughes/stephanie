# stephanie/agents/inference/scoring_memcube.py
from __future__ import annotations

import torch
from stephanie.agents.inference.llm_inference import LLMInferenceAgent

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.inference.ebt_inference import EBTInferenceAgent
from stephanie.agents.inference.mrq_inference import MRQInferenceAgent
from stephanie.memcube.memcube_factory import MemCubeFactory
from stephanie.scoring.scorable import ScorableFactory, ScorableType


class ScoringMemcubeAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.ebt_refine_threshold = cfg.get("ebt_refine_threshold", 0.7)
        self.llm_fallback_threshold = cfg.get("llm_fallback_threshold", 0.9)
        self.steps = cfg.get("optimization_steps", 10)
        self.step_size = cfg.get("step_size", 0.05)

        self.ebt = EBTInferenceAgent(cfg.get("ebt"), self.memory, self.container, self.logger)
        self.mrq = MRQInferenceAgent(cfg.get("mrq"), self.memory, self.container, self.logger)
        self.llm = LLMInferenceAgent(cfg.get("llm"), self.memory, self.container, self.logger)

    async def run(self, context: dict) -> dict:
        goal_text = context["goal"]["goal_text"]
        docs = context[self.input_key]
        results = []

        for doc in docs:
            scorable = ScorableFactory.from_dict(doc, ScorableType.DOCUMENT)
            memcube = MemCubeFactory.from_scorable(scorable, version="auto")

            # Initial MRQ score (returns ScoreBundle)
            mrq_bundle = self.mrq.score(context, memcube.scorable)

            # Estimate uncertainty using EBT (returns raw energy dict)
            ebt_energy = self.ebt.get_energy(goal_text, memcube.scorable.text)
            uncertainty_by_dim = {
                dim: torch.sigmoid(torch.tensor(raw)).item()
                for dim, raw in ebt_energy.items()
            }

            # Refinement step if EBT uncertainty is high
            if any(u > self.ebt_refine_threshold for u in uncertainty_by_dim.values()):
                refinement_result = self.ebt.optimize(goal_text, memcube.scorable.text)
                refined_scorable = ScorableFactory.from_dict({
                    "id": memcube.scorable.id,
                    "text": refinement_result["refined_text"],
                    "target_type": ScorableType.REFINEMENT
                })

                refined_memcube = MemCubeFactory.from_scorable(refined_scorable, version="auto")
                refined_memcube.extra_data.update({
                    "refinement_trace": refinement_result["energy_trace"],
                    "original_memcube_id": memcube.id
                })
                memcube = refined_memcube

            # Fallback to LLM if uncertainty still high
            if any(u > self.llm_fallback_threshold for u in uncertainty_by_dim.values()):
                final_bundle = self.llm.score(context, memcube.scorable)
                source = "llm"
            else:
                final_bundle = mrq_bundle
                source = "mrq"

                # Save to memory
            self.memory.evaluations.save_bundle(
                bundle=final_bundle,
                scorable=scorable,
                context=context,
                cfg=self.cfg,
                source=self.name,
                agent_name=self.name,
                embedding_type=self.memory.embedding.name,
                model_name="ensemble",
                evaluator=str(self.enabled_scorers),
                container=self.container,
            )

            # Save MemCube to DB
            self.memory.memcubes.save_memcube(memcube)

            results.append({
                "memcube": memcube.to_dict(),
                "scores": final_bundle.to_dict(),
                "source": source,
                "uncertainty_by_dimension": uncertainty_by_dim,
            })

        context[self.output_key] = results
        return context
