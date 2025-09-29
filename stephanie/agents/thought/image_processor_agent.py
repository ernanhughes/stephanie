# stephanie/agents/thought/image_processor_agent.py

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

try:
    from PIL import Image, ImageEnhance, ImageFilter
except Exception:
    Image = None  # type: ignore

from stephanie.agents.base_agent import BaseAgent
from stephanie.services.image_profile_service import ImageProfileService
from stephanie.services.image_quality_metrics import ImageQualityMetrics


class ImageProcessorAgent(BaseAgent):
    """
    Applies a small set of deterministic augmenters, scores each step,
    emits VPM-like rows, and saves results to CaseBook/DynamicScorables.
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.vpm_rows: List[Dict[str, Any]] = []
        self.image_prof = ImageProfileService(memory, logger)
        self.reward = container.get("lfl_reward")  # your LFLRewardService instance
        self.augmenters = [
            ("contrast+sharpen", self._aug_contrast_sharpen),
            ("color_balance", self._aug_color_balance),
            ("denoise", self._aug_denoise),
            ("detail", self._aug_detail)
        ]

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        image_path = context.get("image_path")
        prompt = context.get("goal", {}).get("goal_text") or context.get("prompt") or ""
        casebook = context.get("casebook_name")

        if not image_path or Image is None:
            self.logger.log("ImageProcessorStart", {"ok": False, "reason": "no_image_or_pil"})
            return context

        # load base
        base_img = Image.open(image_path).convert("RGB")

        # step 0 metrics
        qim = ImageQualityMetrics()
        m0 = qim.get_metrics(base_img)
        vs = self.image_prof.score_image(image_path, prompt=prompt)
        vpm0 = self._emit_vpm("initial", m0, vs)
        lfl0 = self._to_lfl(vs, m0)

        best = {"step": "initial", "image": base_img, "path": image_path, "metrics": m0, "vs": vs, "lfl": lfl0}
        curr = base_img

        # iterate simple pipeline
        for name, fn in self.augmenters:
            try:
                nxt = fn(curr)
                m = qim.get_metrics(nxt)
                # fill relevance/style via profile
                # save tmp file to score with profile (CLIP uses file only for logging; we can pass in-memory if modified)
                tmp_path = self._save_tmp(nxt, suffix=name)
                vs_m = self.image_prof.score_image(tmp_path, prompt=prompt)
                vpm = self._emit_vpm(name, m, vs_m)
                lfl = self._to_lfl(vs_m, m)
                if lfl > best["lfl"]:
                    best = {"step": name, "image": nxt, "path": tmp_path, "metrics": m, "vs": vs_m, "lfl": lfl}
                curr = nxt
            except Exception as e:
                self.logger.log("ImageAugError", {"augmentation": name, "error": str(e)})

        # persist best result to casebook
        self._save_case(casebook, prompt, best, context)
        context.setdefault(self.output_key, {})
        context[self.output_key]["image_processing"] = {
            "best_step": best["step"],
            "best_lfl": best["lfl"],
            "vpm_rows": self.vpm_rows,
            "best_meta": {
                "metrics": best["metrics"],
                "vs": best["vs"],
                "path": best["path"]
            }
        }
        return context

    # ---------- augmenters ----------
    def _aug_contrast_sharpen(self, img: "Image.Image") -> "Image.Image":
        c = ImageEnhance.Contrast(img).enhance(1.15)
        s = c.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))
        return s

    def _aug_color_balance(self, img: "Image.Image") -> "Image.Image":
        # slight saturation + brightness
        s = ImageEnhance.Color(img).enhance(1.10)
        b = ImageEnhance.Brightness(s).enhance(1.03)
        return b

    def _aug_denoise(self, img: "Image.Image") -> "Image.Image":
        return img.filter(ImageFilter.MedianFilter(size=3))

    def _aug_detail(self, img: "Image.Image") -> "Image.Image":
        return img.filter(ImageFilter.DETAIL)

    # ---------- helpers ----------
    def _emit_vpm(self, step: str, m: Dict[str, float], vs: Dict[str, float]) -> Dict[str, Any]:
        row = {
            "unit": f"image:{step}",
            "kind": "image",
            "timestamp": time.time(),
            "dims": {
                # RS_img-ish
                "sharpness": m["sharpness"],
                "color_diversity": m["color_diversity"],
                "composition": m["composition"],
                "aesthetic_score": m["aesthetic_score"],
                "relevance": vs.get("relevance", 0.5),
                "noise_level": m["noise_level"],
                "contrast": m["contrast"],
                "color_balance": m["color_balance"],
                # VS_img-ish
                "style_clip": vs.get("style_clip", 0.5),
                "palette_match": vs.get("palette_match", 0.5),
                "composition_match": vs.get("composition_match", 0.5),
            },
            "meta": {"step": step}
        }
        self.vpm_rows.append(row)
        self.logger.log("VPMRow", {"step": step, "dims": row["dims"]})
        return row

    def _to_lfl(self, vs: Dict[str, float], m: Dict[str, float]) -> float:
        # adapt to your LFLRewardService: VS components + RS components passed through your existing API
        vs_metrics = {
            "VS1_embed": vs.get("style_clip", 0.5),   # style similarity
            "VS2_style": vs.get("palette_match", 0.5),
            "VS3_moves": vs.get("composition_match", 0.5)
        }
        rs_metrics = {
            "claim_coverage": m.get("color_diversity", 0.5),   # proxy (for images: diversity as coverage)
            "hallucination_rate": 1.0 - m.get("relevance", 0.5),  # low relevance ~ hallucination
            "structure": m.get("composition", 0.5),
            "faithfulness": m.get("relevance", 0.5),
            "HRM_norm": m.get("aesthetic_score", 0.5)  # stand-in if you want
        }
        try:
            return float(self.reward.calculate_lfl(vs_metrics, rs_metrics))
        except Exception:
            return 0.5

    def _save_tmp(self, img: "Image.Image", suffix: str) -> str:
        # deterministic-ish temp path under working dir
        out = f".image_tmp_{int(time.time()*1000)}_{suffix}.png"
        img.save(out, format="PNG")
        return out

    def _save_case(self, casebook_name: Optional[str], prompt: str, best: Dict[str, Any], context: Dict[str, Any]):
        try:
            # Ensure casebook (use your CaseBookStore)
            cb = self.memory.casebooks.ensure_casebook(casebook_name or "images::default", 
                                                       pipeline_run_id=context.get("pipeline_run_id"),
                                                       tags=["images"])
            # Create case and store best image path + metrics
            meta = {
                "type": "image_result",
                "step": best["step"],
                "metrics": best["metrics"],
                "vs": best["vs"],
                "lfl": best["lfl"],
                "source_path": best["path"],
                "prompt": prompt
            }
            case = self.memory.casebooks.add_case(
                casebook_id=cb.id,
                goal_id=context.get("goal", {}).get("id"),
                agent_name=self.name,
                prompt_text=prompt,
                meta=meta,
                scorables=[{
                    "role": "output",
                    "target_type": "image",
                    "meta": {"text": "", "image_path": best["path"], "metrics": best["metrics"], "vs": best["vs"], "lfl": best["lfl"]}
                }]
            )
            self.logger.log("ImageCaseSaved", {"case_id": case.id, "casebook_id": cb.id})
        except Exception as e:
            self.logger.log("ImageCaseSaveError", {"error": str(e)})
