# stephanie/zeromodel/casebook_residual_extractor.py
from __future__ import annotations

import os
import uuid
from collections import OrderedDict

import numpy as np
import torch

from stephanie.db import Session
from stephanie.orm.casebook import CaseBookORM
from stephanie.orm.skill_filter import SkillFilterORM
from stephanie.zeromodel.vpm_builder import CaseBookVPMBuilder


def diff_state_dict(sd_after: OrderedDict, sd_before: OrderedDict) -> OrderedDict:
    v = OrderedDict()
    for k, w in sd_after.items():
        if k in sd_before and w.shape == sd_before[k].shape:
            v[k] = (w - sd_before[k]).cpu()
    return v

class CaseBookResidualExtractor:
    def __init__(self, session: Session, tokenizer, logger=None):
        self.session = session
        self.tokenizer = tokenizer
        self.logger = logger or (lambda m: print(m))

    def extract_and_store(
        self,
        casebook_name: str,
        model_before,
        model_after,
        weight_sd_before: OrderedDict,
        weight_sd_after: OrderedDict,
        domain: str = "general",
        out_dir: str = "outputs/skills",
        description: str | None = None,
        validate: bool = True,
        num_test_cases: int = 100,
    ) -> SkillFilterORM:
        os.makedirs(out_dir, exist_ok=True)

        cb: CaseBookORM = (
            self.session.query(CaseBookORM).filter_by(name=casebook_name).one()
        )
        # 1) Weight delta
        v_weight = diff_state_dict(weight_sd_after, weight_sd_before)
        weight_path = os.path.join(out_dir, f"{casebook_name}_delta_{uuid.uuid4().hex}.pt")
        torch.save(v_weight, weight_path)
        weight_size_mb = os.path.getsize(weight_path) / (1024**2)

        # 2) VPM before/after
        builder = CaseBookVPMBuilder(self.tokenizer, metrics=["sicql", "ebt", "llm"])
        vpm_before = builder.build(cb, model_before)
        vpm_after  = builder.build(cb, model_after)

        residual = (vpm_after.astype(np.float32) - vpm_before.astype(np.float32))
        # Normalize residual for storage/preview
        res_norm = (residual - residual.min()) / (residual.ptp() + 1e-9)

        res_npy = os.path.join(out_dir, f"{casebook_name}_residual_{uuid.uuid4().hex}.npy")
        np.save(res_npy, res_norm)
        res_png = os.path.join(out_dir, f"{casebook_name}_residual_{uuid.uuid4().hex}.png")
        builder.save_image(res_norm, res_png, title=f"Residual {casebook_name}")

        # 3) Validation (alignment)
        alignment_score = None
        if validate:
            alignment_score = self._validate_skill_alignment(
                v_weight, res_norm, cb, model_before, builder, num_test_cases
            )

        # 4) Store SkillFilter
        sf = SkillFilterORM(
            id=uuid.uuid4().hex[:32],
            casebook_id=cb.id,
            domain=domain,
            description=description or f"Skill filter extracted from {casebook_name}",
            weight_delta_path=weight_path,
            weight_size_mb=weight_size_mb,
            vpm_residual_path=res_npy,
            vpm_preview_path=res_png,
            alignment_score=alignment_score,
            improvement_score=None,
            stability_score=None,
            compatible_domains=None,
            negative_interactions=None,
        )
        self.session.add(sf)
        self.session.commit()
        self.logger(f"Saved SkillFilter {sf.id} for casebook {casebook_name}")
        return sf

    # --- Validation: weight delta reproduces VPM residual on subset ---
    def _validate_skill_alignment(
        self,
        v_weight: OrderedDict,
        residual_vpm: np.ndarray,
        casebook: CaseBookORM,
        base_model,
        vpm_builder: CaseBookVPMBuilder,
        num_cases: int = 100,
    ) -> float:
        # 1) apply weight delta to cloned model
        test_model = self._apply_weight_delta(base_model, v_weight)

        # 2) subset of cases
        subs = casebook.cases[:num_cases] if len(casebook.cases) > num_cases else casebook.cases
        vpm_test = vpm_builder.build_subset(subs, test_model)
        vpm_base = vpm_builder.build_subset(subs, base_model)

        actual = vpm_test - vpm_base
        # normalize both
        actual_norm = (actual - actual.min()) / (actual.ptp() + 1e-9)

        # match shapes if needed
        if actual_norm.shape != residual_vpm.shape:
            residual_vpm = self._resize_to(residual_vpm, actual_norm.shape)

        expected_norm = (residual_vpm - residual_vpm.min()) / (residual_vpm.ptp() + 1e-9)
        alignment = 1.0 - float(np.mean(np.abs(actual_norm - expected_norm)))
        return max(0.0, min(1.0, alignment))

    def _apply_weight_delta(self, model, v_weight: OrderedDict, alpha: float = 1.0):
        import copy
        cloned = copy.deepcopy(model)
        params = dict(cloned.named_parameters())
        for name, delta in v_weight.items():
            if name in params:
                params[name].data = params[name].data + alpha * delta.to(params[name].device).to(params[name].dtype)
        return cloned

    def _resize_to(self, arr: np.ndarray, target_shape: tuple) -> np.ndarray:
        from scipy.ndimage import zoom
        if arr.shape == target_shape:
            return arr
        if arr.ndim == 1:
            scale = target_shape[0] / arr.shape[0]
            return zoom(arr, scale, order=1)
        scales = tuple(ts / s for ts, s in zip(target_shape, arr.shape))
        return zoom(arr, scales, order=1)
