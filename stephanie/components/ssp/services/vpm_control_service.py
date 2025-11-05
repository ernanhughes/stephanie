# stephanie/components/ssp/services/vpm_control_service.py
"""
VPM Control Service - Manages Vectorized Performance Map decision-making for SSP components
(Updated: adds episode VPM generation + bus events; keeps PHOS/decision logic)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image  # optional
except Exception:  # pragma: no cover
    Image = None

from stephanie.memory.memory_tool import MemoryTool
from stephanie.services.service_protocol import Service
from stephanie.utils.json_sanitize import (dumps_safe,  # ← reuse your utils
                                           sanitize)
# Optional ZeroModel/PHOS imports (kept as in your existing file)
from stephanie.zeromodel.vpm_controller import Decision, Policy, Thresholds
from stephanie.zeromodel.vpm_controller import \
    VPMController as CoreVPMController
from stephanie.zeromodel.vpm_controller import VPMRow
from stephanie.zeromodel.vpm_phos import build_vpm_phos_artifacts


class VPMControlService(Service):
    """
    Service for managing VPM-based control decisions within the SSP framework.
    
    This service:
    - Creates and manages VPMControllers for individual processing units
    - Makes decisions based on multi-dimensional performance metrics
    - Generates VPM visualization artifacts for monitoring
    - Integrates with bandit systems for adaptive exemplar selection
    - Provides goal-aware stopping conditions
    
    The service follows a stateful pattern where each processing unit (e.g., question, episode)
    has its own controller instance that tracks its performance history.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        memory: MemoryTool,
        container,
        logger,
        run_id: int
    ):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger
        self.run_id = run_id

        self._initialized = False
        self._controllers: Dict[str, CoreVPMController] = {}
        self._bus = getattr(memory, "bus", None)

        # Config + viz setup
        self._setup_configuration()
        self._setup_visualization_paths()

        # Episode VPM output directory (configurable; defaults into viz dir)
        vpm_dir = (
            (self.cfg.get("artifacts", {}) or {}).get("vpm_dir")
            or os.path.join(self._viz_dir, "episodes")
        )
        self._vpm_dir = Path(vpm_dir)
        self._vpm_dir.mkdir(parents=True, exist_ok=True)

        # Bandit hooks
        self._bandit_choose: Optional[Callable[[List[str]], str]] = None
        self._bandit_update: Optional[Callable[[str, float], None]] = None

        # Metrics history for PHOS viz
        self._metrics_history: Dict[str, List[Dict[str, float]]] = {}

    # ---------------- Configuration ----------------

    def _setup_configuration(self) -> None:
        c = (self.cfg.get("vpm_control") or {})

        self._thr_code = Thresholds(
            mins=c.get(
                "mins_code",
                {
                    "tests_pass_rate": 1.0,
                    "coverage": 0.70,
                    "type_safe": 1.0,
                    "lint_clean": 1.0,
                    "complexity_ok": 0.8,
                },
            ),
            stop_margin=float(c.get("stop_margin_code", 0.0)),
            edit_margin=float(c.get("edit_margin_code", 0.0)),
        )

        self._thr_text = Thresholds(
            mins=c.get(
                "mins_text",
                {
                    "coverage": 0.80,
                    "correctness": 0.75,
                    "coherence": 0.75,
                    "citation_support": 0.65,
                    "entity_consistency": 0.80,
                },
            ),
            stop_margin=float(c.get("stop_margin_text", 0.02)),
            edit_margin=float(c.get("edit_margin_text", 0.01)),
        )

        self._policy = Policy(
            window=int(c.get("window", 5)),
            ema_alpha=float(c.get("ema_alpha", 0.4)),
            edit_margin=float(c.get("edit_margin", 0.05)),
            patience=int(c.get("patience", 3)),
            escalate_after=int(c.get("escalate_after", 2)),
            oscillation_window=int(c.get("oscillation_window", 6)),
            oscillation_threshold=int(c.get("oscillation_threshold", 3)),
            cooldown_steps=int(c.get("cooldown_steps", 1)),
            spinoff_dim=str(c.get("spinoff_dim", "novelty")),
            stickiness_dim=str(c.get("stickiness_dim", "stickiness")),
            spinoff_gate=tuple(c.get("spinoff_gate", (0.75, 0.45))),
            max_regressions=int(c.get("max_regressions", 2)),
            zscore_clip_dims=list(
                c.get(
                    "zscore_clip_dims",
                    ["coverage", "coherence", "correctness", "tests_pass_rate"],
                )
            ),
            zscore_clip_sigma=float(c.get("zscore_clip_sigma", 3.5)),
            local_gap_dims=list(
                c.get(
                    "local_gap_dims",
                    ["citation_support", "entity_consistency", "lint_clean", "type_safe"],
                )
            ),
            max_steps=int(c.get("max_steps", 50)),
            goal_kind=c.get("goal_kind"),
            goal_name=c.get("goal_name"),
            goal_min_score=float(c.get("goal_min_score", 0.75)),
            goal_allow_unmet=int(c.get("goal_allow_unmet", 0)),
        )

    def _setup_visualization_paths(self) -> None:
        c = (self.cfg.get("vpm_control") or {})
        base = Path(c.get("viz_dir", "./runs/vpm_visualizations"))
        self._viz_dir: Path = self._ensure_dir(base / self.run_id)

        self._raw_viz_dir = self._ensure_dir(self._viz_dir / "raw")
        self._phos_viz_dir = self._ensure_dir(self._viz_dir / "phos")
        self._compare_viz_dir = self._ensure_dir(self._viz_dir / "comparison")

        self._tl_fracs = c.get("tl_fracs", [0.25, 0.16, 0.36, 0.09])
        self._delta = c.get("delta", 0.02)

        self._dimensions = c.get(
            "dimensions",
            ["coverage", "correctness", "coherence", "citation_support", "entity_consistency"],
        )

    # ---------------- Service Protocol ----------------

    def initialize(self, **kwargs) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._log("VPMControlServiceInit", {
            "viz_dir": self._viz_dir,
            "vpm_dir": str(self._vpm_dir),
            "tl_fracs": self._tl_fracs,
            "dimensions": self._dimensions,
        })

    def shutdown(self) -> None:
        # persist controller state
        for unit, controller in self._controllers.items():
            try:
                controller._persist_state()
            except Exception as e:
                self._log("VPMControlServiceWarning", {
                    "event": "state_persist_failed",
                    "unit": unit,
                    "error": str(e),
                })
        self._controllers.clear()
        self._initialized = False
        self._log("VPMControlServiceShutdown", {})

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "units": len(self._controllers),
            "timestamp": time.time(),
            "viz_dir": self._viz_dir,
            "active_dimensions": self._dimensions,
        }

    @property
    def name(self) -> str:
        return "vpm-control-service"

    # ---------------- Episode VPM (NEW) ----------------

    async def generate_for_episode(
        self,
        episode: Any,
        *,
        size: int = 256,
        save_sidecar: bool = True,
        emit_bus: bool = True,
    ) -> Dict[str, Any]:
        """
        Encode a single EpisodeTrace into a grayscale VPM PNG + sidecar JSON.
        Emits 'ssp.vpm.generated' bus event with features + file path.
        """
        names, vals = self._episode_features(episode)
        img = self._encode_simple_vpm(vals, size=size)

        ep_id = getattr(episode, "episode_id", f"ssp-{int(time.time()*1000)}")
        png_path = self._vpm_dir / f"vpm_{ep_id}.png"
        json_path = self._vpm_dir / f"vpm_{ep_id}.json"

        self._save_png(img, png_path)

        meta = {
            "episode_id": ep_id,
            "features": {k: float(v) for k, v in zip(names, vals)},
            "shape": list(img.shape),
            "file": str(png_path),
        }
        if save_sidecar:
            json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if emit_bus:
            await self._publish_bus_event(
                "ssp.vpm.generated",
                {
                    "episode_id": ep_id,
                    "png": str(png_path),
                    "features": meta["features"],
                },
            )

        self._log("VPMGenerated", meta)
        return meta

    def _ensure_dir(self, p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _episode_features(self, episode: Any) -> Tuple[List[str], List[float]]:
        """
        Prefer EpisodeTrace.to_vpm_features(); otherwise produce a compact fallback.
        All features are in [0,1].
        """
        if hasattr(episode, "to_vpm_features"):
            return episode.to_vpm_features()

        # Fallback: minimal trio
        vs = float(getattr(episode, "reward", 0.0) or 0.0)
        verified = 1.0 if getattr(episode, "verified", False) else 0.0
        diff = float(getattr(episode, "difficulty", 0.0) or 0.0)
        return ["reward", "verified", "difficulty"], [vs, verified, diff]

    def _encode_simple_vpm(self, values: List[float], size: int = 256) -> np.ndarray:
        """
        Deterministic, dependency-light encoder:
          - Start with 1×K vector, tile to bands, upsample to size×size
          - Map [0,1]→[0,255] (uint8)
        """
        vec = np.asarray(values, dtype=np.float32).reshape(1, -1)
        band = np.repeat(vec, repeats=32, axis=0)               # 32×K
        factor_w = int(np.ceil(size / band.shape[1]))
        band = np.repeat(band, repeats=factor_w, axis=1)[:, :size]  # 32×size
        factor_h = int(np.ceil(size / band.shape[0]))
        img = np.repeat(band, repeats=factor_h, axis=0)[:size, :]   # size×size
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)

    def _save_png(self, img: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if Image is not None:
            Image.fromarray(img, mode="L").save(path)  # grayscale
        else:
            # Minimal PGM fallback
            with open(path.with_suffix(".pgm"), "wb") as f:
                f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode("ascii"))
                f.write(img.tobytes())

    async def _publish_bus_event(self, subject: str, payload: Dict[str, Any]) -> None:
        """
        Publish via memory.bus (await if coroutine) and log to memory.bus_events.
        Matches your preferred pattern.
        """
        try:
            # Local event log first (non-blocking)
            if hasattr(self.memory, "bus_events"):
                self.memory.bus_events.insert(subject, payload)

            # Publish over bus (supports sync/async impls)
            if self._bus and hasattr(self._bus, "publish"):
                res = self._bus.publish(subject, {"subject": subject, "payload": payload})
                if asyncio.iscoroutine(res):
                    await res
        except Exception as e:
            self._log("VPMControlServiceWarning", {
                "event": "publish_failed",
                "subject": subject,
                "error": str(e),
            })

    # ---------------- Decisions (existing) ----------------

    def decide(
        self,
        unit: str,
        *,
        kind: str,
        dims: Dict[str, float],
        step_idx: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
        candidate_exemplars: Optional[List[str]] = None,
    ) -> Decision:
        ctrl = self._get_controller(unit)
        m = dict(meta or {})
        if candidate_exemplars:
            m["candidate_exemplars"] = candidate_exemplars

        row = VPMRow(
            unit=unit,
            kind=("code" if kind == "code" else "text"),
            timestamp=time.time(),
            step_idx=step_idx,
            dims={k: float(v) for k, v in dims.items()},
            meta=m,
        )
        self._track_metrics(unit, dims, step_idx)

        dec = ctrl.add(row, candidate_exemplars=candidate_exemplars)
        self._publish_decision(unit, dec)

        # Optionally drop periodic PHOS viz
        if step_idx is not None and step_idx % 5 == 0:
            self.generate_visualization(unit, step_idx)

        return dec

    def decide_many(self, unit: str, frames: List[dict]) -> List[Decision]:
        return [
            self.decide(
                unit,
                kind=f.get("kind", "text"),
                dims=f.get("dims", {}),
                step_idx=f.get("step_idx"),
                meta=f.get("meta"),
                candidate_exemplars=f.get("candidate_exemplars"),
            )
            for f in frames
        ]

    def reset_unit(self, unit: str) -> None:
        if unit in self._controllers:
            del self._controllers[unit]
            self._log("VPMControlResetUnit", {"unit": unit})

    def get_unit_state(self, unit: str) -> Dict[str, Any]:
        ctrl = self._controllers.get(unit)
        if not ctrl:
            return {}
        return {
            "resample_counts": dict(ctrl.resample_counts),
            "cooldown_until_step": dict(ctrl.cooldown_until_step),
            "last_signal": {k: v.name for k, v in ctrl.last_signal.items()},
        }

    def set_goal_gate(
        self,
        *,
        goal_kind: Optional[str],
        goal_name: Optional[str],
        min_score: float = 0.75,
        allow_unmet: int = 0,
    ) -> None:
        self._policy.goal_kind = goal_kind
        self._policy.goal_name = goal_name
        self._policy.goal_min_score = float(min_score)
        self._policy.goal_allow_unmet = int(allow_unmet)
        for ctrl in self._controllers.values():
            ctrl.p = self._policy
        self._log("VPMControlGoalGateUpdated", {
            "goal_kind": goal_kind,
            "goal_name": goal_name,
            "min_score": min_score,
            "allow_unmet": allow_unmet,
        })

    def attach_bandit(
        self,
        choose_fn: Callable[[List[str]], str],
        update_fn: Callable[[str, float], None],
    ) -> None:
        self._bandit_choose = choose_fn
        self._bandit_update = update_fn
        for ctrl in self._controllers.values():
            ctrl.bandit_choose = choose_fn
            ctrl.bandit_update = update_fn
        self._log("VPMControlBanditAttached", {})

    def generate_visualization(
        self, unit: str, step_idx: Optional[int] = None, output_path: Optional[str] = None
    ) -> Dict[str, str]:
        metrics_history = self._metrics_history.get(unit, [])
        if not metrics_history:
            return {}

        df = self._convert_to_dataframe(metrics_history)
        if output_path is None:
            output_path = os.path.join(self._phos_viz_dir, f"{unit.replace(':', '_')}")

        artifacts = build_vpm_phos_artifacts(
            df,
            model="ssp",
            dimensions=self._dimensions,
            out_prefix=output_path,
            tl_frac=0.25,
            interleave=False,
            weights=None,
        )
        return {
            "raw": artifacts["paths"]["raw"],
            "phos": artifacts["paths"]["phos"],
            "metrics": json.dumps(artifacts["metrics"]),
        }

    def generate_comparison_visualization(
        self, unit: str, model_a: str, model_b: str, output_path: Optional[str] = None
    ) -> Dict[str, str]:
        if output_path is None:
            output_path = os.path.join(
                self._compare_viz_dir,
                f"{unit.replace(':', '_')}_{model_a}_vs_{model_b}",
            )
        return {"message": "Comparison visualization not yet implemented for this service"}

    # ---------------- Internal Helpers ----------------

    def _get_controller(self, unit: str) -> CoreVPMController:
        if unit not in self._controllers:
            state_path = os.path.join(self._viz_dir, f"vpm_state_{unit.replace(':', '_')}.json")
            ctrl = CoreVPMController(
                thresholds_code=self._thr_code,
                thresholds_text=self._thr_text,
                policy=self._policy,
                bandit_choose=self._bandit_choose,
                bandit_update=self._bandit_update,
                logger=lambda ev, d: self._log(ev, d),
                state_path=state_path,
            )
            self._controllers[unit] = ctrl
        return self._controllers[unit]

    def _publish_decision(self, unit: str, dec: Decision) -> None:
        try:
            raw_payload = {
                "unit": unit,
                **asdict(dec),             # may include Enums, numpy, etc.
            }
            sig = getattr(dec, "signal", None)
            if sig is not None:
                raw_payload["signal"] = getattr(sig, "name", str(sig))

            payload = sanitize(raw_payload)  # ← JSON-safe dict

            res = self._bus.publish("vpm.control.decision", payload)
                # tolerate async or sync
            if asyncio.iscoroutine(res):
                asyncio.create_task(res)
            if hasattr(self.memory, "bus_events"):
                self.memory.bus_events.insert("vpm.control.decision", payload)
        except Exception as e:
            self._log("VPMControlServiceWarning", {
                "event": "publish_failed",
                "unit": unit,
                "error": str(e),
            })

    def _persist_trace(self, unit: str, row: VPMRow, dec: Decision) -> None:
        # Optional: keep disabled unless you want DB traces
        pass

    def _track_metrics(self, unit: str, dims: Dict[str, float], step_idx: Optional[int]) -> None:
        if unit not in self._metrics_history:
            self._metrics_history[unit] = []
        record = {"step_idx": (step_idx if step_idx is not None else len(self._metrics_history[unit]))}
        record.update({k: float(v) for k, v in dims.items()})
        self._metrics_history[unit].append(record)

        max_history = self.cfg.get("vpm_control", {}).get("max_metrics_history", 100)
        if len(self._metrics_history[unit]) > max_history:
            self._metrics_history[unit] = self._metrics_history[unit][-max_history:]

    def _convert_to_dataframe(self, metrics_history: List[Dict]) -> Any:
        try:
            import pandas as pd
            df_data = []
            for record in metrics_history:
                step_idx = record["step_idx"]
                for dim, value in record.items():
                    if dim == "step_idx":
                        continue
                    df_data.append({"node_id": f"{step_idx}", "ssp": value, "dimension": dim})
            df = pd.DataFrame(df_data)
            df = df.pivot(index="node_id", columns="dimension", values="ssp").reset_index()
            return df
        except Exception:  # pragma: no cover
            # Minimal fallback structure
            return {"node_id": [str(i) for i in range(len(metrics_history))]}

    def _log(self, event: str, payload: Dict[str, Any]) -> None:
        try:
            if self.logger:
                self.logger.log(event, payload)
        except Exception:
            pass

    def __repr__(self):
        active_count = len(self._controllers)
        return f"<VPMControlService: status={'initialized' if self._initialized else 'uninitialized'}  units={active_count}  dimensions={len(self._dimensions)}>"
