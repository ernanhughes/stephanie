# stephanie/components/jitter/apoptosis.py
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

log = logging.getLogger(__file__)


class ApoptosisSystem:
    """
    Programmed cell-death guard:
      - decides when to initiate apoptosis
      - can (optionally) score a VPM snapshot for telemetry/auditing

    Notes:
      • All thresholds are configurable and hysteresis-aware to prevent flapping.
      • VPM scoring is optional; uses container services if present.
    """

    # sensible defaults
    DEFAULT_VPM_DIMS = [
        "clarity", "novelty", "confidence", "contradiction",
        "coherence", "complexity", "alignment",
    ]

    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        container=None,
        memory=None,
        logger: Optional[logging.Logger] = None,
        agent_name: str = "jas",
        vpm_scoring: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = dict(cfg or {})
        self.container = container
        self.memory = memory
        self.logger = logger or log
        self.agent_name = agent_name

        # crisis logic (with hysteresis)
        self.crisis_threshold = float(self.cfg.get("crisis_threshold", 0.70))
        self.crisis_clear = float(self.cfg.get("crisis_clear", self.crisis_threshold * 0.85))
        self.max_crisis_ticks = int(self.cfg.get("max_crisis_ticks", 50))
        self.crisis_counter = 0

        # boundary + energy thresholds
        self.boundary_min = float(self.cfg.get("boundary_min", 0.10))
        self.energy_min = float(self.cfg.get("energy_min", 1.0))  # both metabolic & cognitive

        # optional VPM scoring (fully guarded)
        vs = dict(vpm_scoring or self.cfg.get("vpm_scoring") or {})
        self.vpm_enabled = bool(vs.get("enabled", False))
        self.vpm_dims = list(vs.get("dimensions", self.DEFAULT_VPM_DIMS))
        self.vpm_dim_weights = dict(vs.get("dimension_weights", {}))
        self.vpm_resize_method = str(vs.get("resize_method", "bilinear"))
        self.vpm_force_rescore = bool(vs.get("force_rescore", False))
        self.vpm_save_results = bool(vs.get("save_results", False))
        self._tick = 0  # updated by caller via set_tick()

    # --------------------------------------------------------------------- #
    # Decision logic
    # --------------------------------------------------------------------- #
    def should_initiate(self, core, homeostasis) -> bool:
        """Return True if apoptosis must start."""
        try:
            # Energy depletion
            m = float(core.energy.level("metabolic"))
            c = float(core.energy.level("cognitive"))
            if (m < self.energy_min) and (c < self.energy_min):
                return True

            # Boundary failure
            boundary = float(getattr(core.membrane, "integrity", 0.0))
            if boundary < self.boundary_min:
                return True

            # Prolonged crisis (with hysteresis)
            telem = homeostasis.get_telemetry() or {}
            crisis = float(telem.get("crisis_level", 0.0))

            if crisis > self.crisis_threshold:
                self.crisis_counter += 1
            elif crisis < self.crisis_clear:
                # decay counter when safely below clear threshold
                self.crisis_counter = max(0, self.crisis_counter - 1)

            if self.crisis_counter > self.max_crisis_ticks:
                return True

            return False

        except Exception as e:
            self.logger.warning(f"Apoptosis check failed; defaulting to safe=False: {e}")
            return False

    def get_reason(self, core, homeostasis) -> str:
        """Explain why apoptosis was (or would be) initiated."""
        try:
            m = float(core.energy.level("metabolic"))
            c = float(core.energy.level("cognitive"))
            if (m < self.energy_min) and (c < self.energy_min):
                return "energy_depletion"

            boundary = float(getattr(core.membrane, "integrity", 0.0))
            if boundary < self.boundary_min:
                return "boundary_failure"

            telem = homeostasis.get_telemetry() or {}
            crisis = float(telem.get("crisis_level", 0.0))
            if crisis > self.crisis_threshold and self.crisis_counter > self.max_crisis_ticks:
                return "prolonged_crisis"
        except Exception:
            pass
        return "unknown"

    def set_tick(self, tick: int) -> None:
        """Let the host update the current tick for logging/IDs."""
        self._tick = int(tick)

    # --------------------------------------------------------------------- #
    # Optional: VPM scoring helper (non-blocking facade)
    # --------------------------------------------------------------------- #
    async def score_vpm_if_enabled(
        self,
        sensory_input: torch.Tensor,
        *,
        core=None,
        generation: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Async, safe facade: returns None if disabled or anything is missing.
        """
        if not self.vpm_enabled:
            return None
        return await asyncio.to_thread(self._score_vpm_blocking, sensory_input, core, generation)

    # --------------------------------------------------------------------- #
    # Internals (scoring)
    # --------------------------------------------------------------------- #
    def _score_vpm_blocking(
        self,
        sensory_input: torch.Tensor,
        core=None,
        generation: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        - builds a VPM image from ZeroModelService (preferred) or memory fallback
        - scores it via container 'scoring' → 'vpm_transformer'
        - optionally persists to memory evaluation store
        """
        try:
            img = self._render_vpm_image(sensory_input)
            if img is None:
                return None

            # normalize to HxWxC float32 [0,1]
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            img = np.asarray(img)
            if img.ndim == 2:
                img = img[..., None]
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0

            # import only when needed; stay decoupled if model code is absent
            try:
                from stephanie.scoring.model.vpm_model import \
                    AttentionMap  # noqa: F401 (type side-effect)
                from stephanie.scoring.scorer.vpm_transformer_scorer import \
                    VPMScorable  # existing class
            except Exception:
                # fallback: tiny scorable shim
                class VPMScorable:  # type: ignore
                    def __init__(self, id: str, image_array: np.ndarray, metadata: Dict[str, Any] | None = None):
                        self.id = id; self.image_array = image_array; self.metadata = metadata or {}
                    def get_image_array(self) -> np.ndarray: return self.image_array
                    def get_image_tensor(self) -> torch.Tensor:
                        t = torch.from_numpy(self.image_array).float()
                        if t.dim() == 3: t = t.unsqueeze(0)
                        return t

            scorable_id = f"vpm_tick_{self._tick}"
            scorable = VPMScorable(
                id=scorable_id,
                image_array=img,
                metadata={
                    "dimension_weights": self.vpm_dim_weights,
                    "dimension_order": self.vpm_dims,
                    "resize_method": self.vpm_resize_method,
                    "source": "apoptosis_guard",
                    "tick": self._tick,
                },
            )

            # skip if already scored and allowed
            if not self.vpm_force_rescore and hasattr(self.memory, "scores"):
                try:
                    existing = self.memory.scores.get_scores_for_target(
                        target_id=scorable_id,
                        target_type="vpm",
                        dimensions=self.vpm_dims,
                    )
                    if existing:
                        return {
                            "tick": self._tick,
                            "scorable_id": scorable_id,
                            "status": "skipped_already_scored",
                            "dimensions": list(existing.keys()),
                        }
                except Exception:
                    pass

            # score via generic scoring service
            scoring = None
            try:
                scoring = self.container.get("scoring")
            except Exception:
                pass
            if scoring is None:
                return None

            bundle = scoring.score(
                "vpm_transformer",
                context=self._scoring_context(core=core, generation=generation),
                scorable=scorable,
                dimensions=self.vpm_dims,
            )

            eval_id = None
            if self.vpm_save_results and hasattr(self.memory, "evaluations"):
                try:
                    eval_id = self.memory.evaluations.save_bundle(
                        bundle=bundle,
                        scorable=scorable,
                        context=self._scoring_context(core=core, generation=generation),
                        cfg={"vpm_scoring": True, "dims": self.vpm_dims},
                        agent_name=self.agent_name,
                        source="vpm_transformer",
                        model_name="vpm_transformer",
                        evaluator_name=self.agent_name,
                    )
                except Exception as e:
                    self.logger.warning(f"VPM evaluation persist failed: {e}")

            # flatten results
            scores = {}
            for dim, res in (bundle.results or {}).items():
                # res has .score, .source, .rationale, ...
                try:
                    weight = self.vpm_dim_weights.get(dim)
                    scores[dim] = {"score": float(res.score), "source": res.source, **({"weight": weight} if weight is not None else {})}
                except Exception:
                    pass

            return {
                "tick": self._tick,
                "scorable_id": scorable_id,
                "evaluation_id": eval_id,
                "dimensions": list(scores.keys()),
                "scores": scores,
                "order": list(self.vpm_dims),
            }

        except Exception as e:
            self.logger.warning(f"VPM scoring failed: {e}")
            return None

    def _render_vpm_image(self, sensory_input: torch.Tensor):
        """Try ZeroModelService first; fallback to memory VPM manager."""
        # 1) ZeroModel: single-row render via internal pipeline (best-effort)
        try:
            svc = self.container.get("zeromodel-service-v2")
            vec = self._to_row(sensory_input)
            # access internal pipeline safely (no GIF, just one frame)
            if getattr(svc, "_pipeline", None) is not None:
                vpm_out, _ = svc._pipeline.run(vec, {"enable_gif": False})
                if vpm_out is not None:
                    return vpm_out
        except Exception:
            pass

        # 2) memory-backed fallback
        try:
            vm = getattr(self.memory, "vpm_manager", None)
            if vm is not None:
                if hasattr(vm, "from_embedding"):
                    img = vm.from_embedding(sensory_input)
                    if img is not None:
                        return img
                if hasattr(vm, "get_latest_vpm"):
                    v = vm.get_latest_vpm()
                    if v is not None and hasattr(v, "image"):
                        return v.image
        except Exception:
            pass

        return None

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #
    def _scoring_context(self, *, core=None, generation: Optional[int] = None) -> Dict[str, Any]:
        try:
            gen = generation if generation is not None else (getattr(core, "generation", None) if core is not None else None)
        except Exception:
            gen = None
        return {
            "agent": self.agent_name,
            "tick": self._tick,
            "jas_id": getattr(core, "id", None) if core is not None else None,
            "generation": gen,
            "timestamp": time.time(),
        }

    @staticmethod
    def _to_row(x: torch.Tensor) -> np.ndarray:
        """Ensure a (1, D) float32 row for ZeroModel pipeline."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim > 2:
            x = x.reshape(1, -1)
        return x
