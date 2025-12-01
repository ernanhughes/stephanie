# stephanie/pipeline/stages/skill_validation.py
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL, PIPELINE
from stephanie.zero.casebook_residual_extractor import \
    CaseBookResidualExtractor


class SkillValidationAgent(BaseAgent):
    """Validates skill filter alignment in both weight-space and VPM-space."""

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.validation_cases = cfg.get("validation_cases", 100)
        self.output_dir = cfg.get("output_dir", "skill_filters")

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        goal = context.get(GOAL, {})
        pipeline_stage = context.get(PIPELINE, "skill_validation")

        skill_id = context.get("skill_id")
        casebook_name = context.get("casebook_name")

        if not skill_id:
            self.logger.log("SkillValidationSkipped", {
                "stage": pipeline_stage, "reason": "no_skill_id"
            })
            return context

        # --- Retrieve skill filter ---
        skill = self.memory.skill_filters.get_by_id(skill_id)
        if not skill:
            self.logger.log("SkillNotFound", {
                "stage": pipeline_stage, "skill_id": skill_id
            })
            return context

        # --- Retrieve casebook ---
        cb = None
        if casebook_name:
            cb = self.memory.casebooks.get_by_name(casebook_name)
        if not cb:
            self.logger.log("CaseBookNotFound", {
                "stage": pipeline_stage, "casebook_name": casebook_name
            })
            return context

        # --- Load base model + tokenizer ---
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base_model = AutoModelForCausalLM.from_pretrained(skill.trained_model_path)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

        extractor = CaseBookResidualExtractor(
            session=self.memory.session,
            output_dir=self.output_dir,
        )

        weight_delta_path = skill.weight_delta_path or f"{self.output_dir}/{skill_id}.pt"

        # --- Run weight-space validation ---
        try:
            alignment_score = extractor._validate_skill_alignment(
                base_model.state_dict(),
                weight_delta_path,
                cb,
                tokenizer,
                self.validation_cases,
            )
        except Exception as e:
            self.logger.log("WeightValidationFailed", {
                "stage": pipeline_stage, "error": str(e)
            })
            alignment_score = None

        # --- Run VPM-space validation ---
        vpm_score = None
        if skill.vpm_residual_path:
            try:
                vpm_residual = np.load(skill.vpm_residual_path)
                vpm_score = float(np.clip(vpm_residual.mean(), -1.0, 1.0))  # placeholder metric
            except Exception as e:
                self.logger.log("VPMValidationFailed", {
                    "stage": pipeline_stage, "error": str(e)
                })

        # --- Aggregate stability metric ---
        stability_score = None
        if alignment_score is not None and vpm_score is not None:
            stability_score = 1.0 - abs(alignment_score - vpm_score)

        # --- Update skill filter record ---
        updates = {}
        if alignment_score is not None:
            updates["alignment_score"] = alignment_score
        if vpm_score is not None:
            updates["improvement_score"] = vpm_score
        if stability_score is not None:
            updates["stability_score"] = stability_score

        if updates:
            self.memory.skill_filters.update_filter(skill_id, updates)
            self.memory.commit()

        # --- Store in context ---
        context["skill_validation"] = {
            "skill_id": skill_id,
            "alignment_score": alignment_score,
            "vpm_score": vpm_score,
            "stability_score": stability_score,
        }

        self.logger.log("SkillValidated", {
            "stage": pipeline_stage,
            "skill_id": skill_id,
            "updates": updates,
        })

        return context
