# stephanie/agents/filter_bank_agent.py

from __future__ import annotations

from collections import OrderedDict
from typing import List

import numpy as np
import torch

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL, PIPELINE
from stephanie.models.skill_filter import SkillFilterORM


class FilterBankAgent(BaseAgent):
    """
    Stephanie agent for applying Skill Filters to models or VPMs.
    Wraps filter application logic in the standard agent run() pattern.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.mode = cfg.get("mode", "visual")  # "visual" | "weight"
        self.alpha = cfg.get("alpha", 1.0)
        self.filter_ids = cfg.get("filters", [])  # list of filter IDs
        self.case_metadata = cfg.get("case_metadata")
        self.domain_weights = cfg.get("domain_weights")

    async def run(self, context: dict) -> dict:
        """
        Main entrypoint for FilterBankAgent.

        Context should include either:
        - "vpm": a numpy array (for visual mode)
        - "model_sd": a state_dict (for weight mode)
        """
        goal = context.get(GOAL, {})
        pipeline_stage = context.get(PIPELINE, "filtering")

        filters = self._load_filters(self.filter_ids)
        if not filters:
            self.logger.log("NoFiltersProvided", {"stage": pipeline_stage})
            return context

        if self.mode == "visual":
            vpm = context.get("vpm")
            if vpm is None:
                self.logger.log("NoVPMProvided", {"stage": pipeline_stage})
                return context
            enhanced_vpm = self._apply_visual_filters(vpm, filters)
            context["enhanced_vpm"] = enhanced_vpm

        elif self.mode == "weight":
            model_sd = context.get("model_sd")
            if model_sd is None:
                self.logger.log("NoModelProvided", {"stage": pipeline_stage})
                return context
            enhanced_sd = self._apply_weight_filters(model_sd, filters)
            context["enhanced_model_sd"] = enhanced_sd

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        context["filters_applied"] = [f.id for f in filters]
        return context

    # -------- Retrieval --------
    def _load_filters(self, filter_ids: List[str]) -> List[SkillFilterORM]:
        filters = []
        for fid in filter_ids:
            f = self.memory.skill_filters.get_by_id(fid)
            if f:
                filters.append(f)
            else:
                self.logger.log("FilterNotFound", {"id": fid})
        return filters

    # -------- Weight-space helpers --------
    def _apply_weight_filters(
        self, model_sd: OrderedDict, filters: List[SkillFilterORM]
    ) -> OrderedDict:
        combined = model_sd.copy()
        for f in filters:
            combined = self._apply_weight_filter(combined, f, self.alpha)
        return combined

    def _apply_weight_filter(
        self,
        model_sd: OrderedDict,
        filter_obj: SkillFilterORM,
        alpha: float,
        safety_threshold: float = 0.15,
    ) -> OrderedDict:
        if not filter_obj.weight_delta_path:
            return model_sd

        v = torch.load(filter_obj.weight_delta_path, map_location="cpu")
        new_sd = OrderedDict()
        unstable = []

        for k, w in model_sd.items():
            if k not in v:
                new_sd[k] = w
                continue

            delta = alpha * v[k]
            abs_mean = torch.abs(w).mean().item()
            max_change = safety_threshold * (abs_mean + 1e-8)
            actual_change = torch.abs(delta).mean().item()

            if actual_change > max_change:
                unstable.append((k, actual_change / max_change))
                safe_alpha = max_change / (actual_change + 1e-8)
                delta = delta * safe_alpha

            new_sd[k] = w + delta.to(w.device).to(w.dtype)

        if unstable:
            self.logger.log(
                "WeightFilterScaled",
                {"filter": filter_obj.id, "layers_scaled": len(unstable)},
            )

        return new_sd

    # -------- Visual helpers --------
    def _apply_visual_filters(
        self, vpm: np.ndarray, filters: List[SkillFilterORM]
    ) -> np.ndarray:
        enhanced = vpm.copy()
        for f in filters:
            enhanced = self._apply_visual_filter(enhanced, f, self.alpha)
        return enhanced

    def _apply_visual_filter(
        self,
        vpm: np.ndarray,
        filter_obj: SkillFilterORM,
        alpha: float,
        clip: tuple[float, float] = (0.0, 1.0),
    ) -> np.ndarray:
        if not filter_obj.vpm_residual_path:
            return vpm

        residual = np.load(filter_obj.vpm_residual_path).astype(np.float32)
        if residual.shape != vpm.shape:
            residual = self._resize_residual(residual, vpm.shape)

        out = vpm + alpha * residual
        out = np.clip(out, clip[0], clip[1])
        return out

    def _resize_residual(self, residual: np.ndarray, target_shape: tuple) -> np.ndarray:
        from scipy.ndimage import zoom
        if residual.shape == target_shape:
            return residual
        scales = tuple(ts / s for ts, s in zip(target_shape, residual.shape))
        return zoom(residual, scales, order=1)
