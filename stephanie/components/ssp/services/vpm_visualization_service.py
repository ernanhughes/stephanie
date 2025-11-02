# stephanie/components/ssp/services/vpm_visualization_service.py
"""
VPM Visualization Service - Generates Vectorized Performance Map images for SSP episodes

All filesystem paths are centralized in `_setup_visualization_paths()`.
Every method writes ONLY under those paths. No ad-hoc dirs elsewhere.

Structure:
  _viz_dir/
    raw/
    phos/
    comparison/
    episode_data/
    progress/
      <unit>/
        frames...
        <unit>_progress.gif
        <unit>_filmstrip.png
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import PIL.Image as Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stephanie.logging.json_logger import JSONLogger
from stephanie.memory.memory_tool import MemoryTool
from stephanie.services.service_protocol import Service
from stephanie.components.ssp.utils.trace import EpisodeTrace

from stephanie.zeromodel.vpm_phos import (
    build_vpm_phos_artifacts,
    build_compare_guarded,
    robust01,
    save_img,
    vpm_vector_from_df,
    to_square,
    phos_sort_pack,
)
from stephanie.zeromodel.vpm_controller import VPMRow

_logger = logging.getLogger(__name__)

# Back-compat mapping: old -> new canonical names
_KEY_MAP = {
    "verifier_f1": "verifier_score",
    "steps_norm": "solver_steps",
    "evidence_cnt": "evidence_count",
}

# Canonical dimensions we snapshot during search
_CANON_DIMS = [
    "verifier_score",
    "verified",
    "difficulty",
    "question_len",
    "answer_len",
    "evidence_count",
    "solver_steps",
    "score",
    "best_score",
    "improvement",
    "depth",
    "novelty",
]


class VPMVisualizationService(Service):
    """
    Service for generating VPM visualization artifacts from SSP episode data.

    NOTE: All paths are defined in `_setup_visualization_paths`. Do not write elsewhere.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        memory: MemoryTool,
        logger: JSONLogger,
        container: Optional[Any],
        run_id: Optional[str],
    ):
        self.cfg = cfg or {}
        self.memory = memory
        self.logger = logger
        self.container = container
        self.run_id = run_id or f"run_{int(time.time())}"

        self._initialized = False

        # state
        self._feature_order: Dict[str, List[str]] = {}
        self._metrics_history: Dict[str, List[Dict[str, float]]] = {}
        self._episode_traces: Dict[str, EpisodeTrace] = {}
        self._progress_frames: Dict[str, List[Path]] = {}  # unit -> list of frame paths (Path)

        # dirs + metrics/vpm params
        self._setup_visualization_paths()
        self._setup_metrics()
        self._setup_vpm_parameters()

    # ----------------------------- paths ---------------------------------

    def _ensure_dir(self, p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _sanitize_unit(self, unit: str) -> str:
        # Keep filenames clean & cross-platform
        return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(unit))

    def _progress_unit_dir(self, unit: str) -> Path:
        """Per-unit folder under progress/ (auto-created)."""
        return self._ensure_dir(self._progress_root / self._sanitize_unit(unit))

    def _setup_visualization_paths(self) -> None:
        """Configure base and subdirectories (Path objects)."""
        c = self.cfg.get("vpm_viz") or {}
        base = Path(c.get("output_dir", "./runs/vpm_visualizations"))
        self._viz_dir: Path = self._ensure_dir(base / self.run_id)

        # subdirs
        self._raw_viz_dir: Path = self._ensure_dir(self._viz_dir / "raw")
        self._phos_viz_dir: Path = self._ensure_dir(self._viz_dir / "phos")
        self._compare_viz_dir: Path = self._ensure_dir(self._viz_dir / "comparison")
        self._episode_data_dir: Path = self._ensure_dir(self._viz_dir / "episode_data")
        self._progress_root: Path = self._ensure_dir(self._viz_dir / "progress")

        # sweep config
        self._tl_fracs = c.get("tl_fracs", [0.25, 0.16, 0.36, 0.09])
        self._delta = c.get("delta", 0.02)

        # dimensions
        dims_cfg = c.get("dimensions")
        self._dimensions = [_KEY_MAP.get(d, d) for d in dims_cfg] if dims_cfg else list(_CANON_DIMS)

        _logger.info(
            "VPMVisualizationService paths initialized",
            extra={
                "viz_dir": str(self._viz_dir),
                "raw_dir": str(self._raw_viz_dir),
                "phos_dir": str(self._phos_viz_dir),
                "compare_dir": str(self._compare_viz_dir),
                "episode_data_dir": str(self._episode_data_dir),
                "progress_root": str(self._progress_root),
                "dimensions": self._dimensions,
            },
        )

    # ----------------------------- metrics/vpm params ---------------------

    def _setup_metrics(self) -> None:
        self._metric_ranges = {
            "verifier_f1": (0.0, 1.0),
            "difficulty": (0.0, 1.0),
            "steps_norm": (0.0, 1.0),
            "evidence_cnt": (0.0, 1.0),
            "coverage": (0.0, 1.0),
            "correctness": (0.0, 1.0),
            "coherence": (0.0, 1.0),
            "citation_support": (0.0, 1.0),
            "entity_consistency": (0.0, 1.0),
        }
        metric_cfg = self.cfg.get("vpm_viz", {}).get("metric_ranges", {})
        for metric, rng in metric_cfg.items():
            try:
                self._metric_ranges[metric] = (float(rng[0]), float(rng[1]))
            except Exception:
                pass

    def _setup_vpm_parameters(self) -> None:
        c = self.cfg.get("vpm_viz") or {}
        self._phos_tl_frac = float(c.get("phos_tl_frac", 0.25))
        self._phos_interleave = bool(c.get("phos_interleave", False))
        self._phos_weights = c.get("phos_weights", None)

        self._raw_vpm_interleave = bool(c.get("raw_vpm_interleave", False))
        self._raw_vpm_weights = c.get("raw_vpm_weights", None)

        self._compare_models = c.get("compare_models", ["current", "baseline"])
        self._compare_tl_fracs = c.get("compare_tl_fracs", self._tl_fracs)
        self._compare_delta = float(c.get("compare_delta", self._delta))

    # ---------------- Service Protocol Implementation ----------------

    def initialize(self, **kwargs) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.logger.log(
            "VPMVisualizationServiceInit",
            {
                "viz_dir": str(self._viz_dir),
                "tl_fracs": self._tl_fracs,
                "delta": self._delta,
                "dimensions": self._dimensions,
                "metric_ranges": self._metric_ranges,
            },
        )

    def shutdown(self) -> None:
        self._metrics_history.clear()
        self._episode_traces.clear()
        self._progress_frames.clear()
        self._initialized = False
        self.logger.log("VPMVisualizationServiceShutdown", {})

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._initialized else "uninitialized",
            "episode_count": len(self._episode_traces),
            "traced_dimensions": len(self._metrics_history),
            "timestamp": time.time(),
            "viz_dir": str(self._viz_dir),
            "active_dimensions": self._dimensions,
        }

    @property
    def name(self) -> str:
        return "ssp-vpm-visualization"

    # ------------------------- Public API -------------------------------

    def finalize_progress(self, unit: str, *, gif_name: Optional[str] = None, fps: int = 2) -> str:
        """Build an animated GIF from collected frames for `unit`."""
        frames = self._progress_frames.get(unit, [])
        if not frames:
            return ""
        unit_dir = self._progress_unit_dir(unit)
        gif_path = unit_dir / (gif_name or f"{self._sanitize_unit(unit)}_progress.gif")

        imgs = [imageio.imread(str(p)) for p in frames]
        # duration per frame (s)
        duration = 1.0 / max(1, int(fps))
        imageio.mimsave(str(gif_path), imgs, duration=duration)

        self.logger.log(
            "VPMProgressGIFSaved",
            {"unit": unit, "gif": str(gif_path), "frames": len(frames)},
        )
        return str(gif_path)

    def generate_episode_visualization(self, unit: str, episode: EpisodeTrace) -> str:
        """Legacy 1xF bar render for a single EpisodeTrace (saved under raw/)."""
        names, vals = episode.to_vpm_features()
        arr = np.clip(np.asarray(vals, dtype=np.float32), 0.0, 1.0)
        if float(arr.max() - arr.min()) < 1e-6:
            arr[:] = 0.5

        bar = (arr * 255.0).astype(np.uint8)[None, :]  # (1, F)
        bar = np.repeat(bar, 16, axis=0)               # thicken

        out_path = self._raw_viz_dir / f"{self._sanitize_unit(unit)}.png"
        Image.fromarray(bar, mode="L").save(out_path)
        try:
            self.logger.info("VPM saved", extra={"unit": unit, "features": dict(zip(names, vals))})
        except Exception:
            pass
        return str(out_path)

    def generate_visualization(
        self,
        unit: str,
        step_idx: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """PHOS + RAW montages for the given unit from accumulated metrics."""
        metrics_history = self._metrics_history.get(unit, [])
        if not metrics_history:
            return {}

        df = self._convert_to_dataframe(metrics_history)
        df, _dims_pref = self._with_model_prefixed_columns(df, "ssp", self._dimensions)

        base_name = self._sanitize_unit(unit)
        out_prefix_phos = Path(output_path) if output_path else (self._phos_viz_dir / base_name)
        out_prefix_raw = self._raw_viz_dir / base_name

        artifacts = build_vpm_phos_artifacts(
            df,
            model="ssp",
            dimensions=self._dimensions,
            out_prefix=str(out_prefix_phos),
            tl_frac=self._phos_tl_frac,
            interleave=self._phos_interleave,
            weights=self._phos_weights,
        )
        raw_artifacts = build_vpm_phos_artifacts(
            df,
            model="ssp",
            dimensions=self._dimensions,
            out_prefix=str(out_prefix_raw),
            tl_frac=0.0,
            interleave=self._raw_vpm_interleave,
            weights=self._raw_vpm_weights,
        )

        self._save_episode_data(unit, metrics_history)

        return {
            "raw": raw_artifacts["paths"]["raw"],
            "phos": artifacts["paths"]["phos"],
            "metrics": json.dumps(artifacts["metrics"]),
            "episode_data": str(self._episode_data_dir / f"{base_name}.json"),
        }

    def generate_comparison_visualization(
        self,
        unit: str,
        model_a: str,
        model_b: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, str]:
        base_name = f"{self._sanitize_unit(unit)}_{model_a}_vs_{model_b}"
        out_prefix = Path(output_path) if output_path else (self._compare_viz_dir / base_name)

        artifacts = build_compare_guarded(
            df=self._get_comparison_dataframe(unit, model_a, model_b),
            dimensions=self._dimensions,
            out_prefix=str(out_prefix),
            model_A=model_a,
            model_B=model_b,
            tl_fracs=self._compare_tl_fracs,
            delta=self._compare_delta,
            interleave=self._phos_interleave,
            weights=self._phos_weights,
        )

        return {
            "summary": artifacts.get("summary", {}),
            "sweep": artifacts.get("sweep", {}),
            "diff_range": artifacts.get("diff_range"),
            "diff_image": f"{str(out_prefix)}_vpm_chosen_diff.png",
            "model_a_chosen": artifacts["sweep"].get(model_a, [{}])[-1].get("phos_path", ""),
            "model_b_chosen": artifacts["sweep"].get(model_b, [{}])[-1].get("phos_path", ""),
        }

    def generate_curriculum_visualization(self, output_path: Optional[str] = None) -> Dict[str, str]:
        all_metrics = []
        for unit, metrics in self._metrics_history.items():
            for metric in metrics:
                all_metrics.append({"unit": unit, **metric})
        if not all_metrics:
            return {}

        df = pd.DataFrame(all_metrics)
        ts_name = f"curriculum_progression_{int(time.time())}"
        out_prefix = Path(output_path) if output_path else (self._phos_viz_dir / ts_name)

        artifacts = build_vpm_phos_artifacts(
            df,
            model="curriculum",
            dimensions=self._dimensions,
            out_prefix=str(out_prefix),
            tl_frac=self._phos_tl_frac,
            interleave=self._phos_interleave,
            weights=self._phos_weights,
        )

        return {
            "curriculum_phos": artifacts["paths"]["phos"],
            "curriculum_raw": artifacts["paths"]["raw"],
            "metrics": json.dumps(artifacts["metrics"]),
        }

    def generate_raw_vpm_image(
        self,
        unit: str,
        step_idx: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> str:
        metrics_history = self._metrics_history.get(unit, [])
        if not metrics_history:
            return ""

        df = self._convert_to_dataframe(metrics_history)
        dims_to_use = [d for d in self._dimensions if d in df.columns]
        if not dims_to_use:
            df, dims_pref = self._with_model_prefixed_columns(df, "ssp", self._dimensions)
            dims_to_use = [d for d in dims_pref if d in df.columns]
            if not dims_to_use:
                self.logger.log("VPMVisualizationError", {
                    "event": "no_dims_found",
                    "unit": unit,
                    "df_cols": list(df.columns)
                })
                return ""

        df, dims_pref = self._with_model_prefixed_columns(df, "ssp", dims_to_use)
        vec = vpm_vector_from_df(
            df, model="ssp", dimensions=dims_to_use,
            interleave=self._raw_vpm_interleave, weights=self._raw_vpm_weights
        )

        self.logger.log("VPMDebug", {
            "unit": unit,
            "step_idx": step_idx,
            "vector_shape": getattr(vec, "shape", None),
            "vector_min": float(np.min(vec)) if len(vec) else 0.0,
            "vector_max": float(np.max(vec)) if len(vec) else 0.0,
        })

        vec = robust01(np.asarray(vec, np.float32))
        if not np.isfinite(vec).any() or float(vec.max() - vec.min()) < 1e-6:
            vec = np.full_like(vec, 0.5, dtype=np.float32)

        img, _ = to_square(vec)
        base_name = self._sanitize_unit(unit)
        out_path = Path(output_path) if output_path else (self._raw_viz_dir / f"{base_name}_raw_vpm_{int(time.time())}.png")
        save_img(img, str(out_path), title=f"SSP VPM (Raw) - {unit}")
        return str(out_path)

    def generate_phos_image(
        self,
        unit: str,
        step_idx: Optional[int] = None,
        output_path: Optional[str] = None,
        tl_frac: Optional[float] = None,
    ) -> str:
        metrics_history = self._metrics_history.get(unit, [])
        if not metrics_history:
            return ""

        df = self._convert_to_dataframe(metrics_history)
        dims_to_use = [d for d in self._dimensions if d in df.columns]
        if not dims_to_use:
            self.logger.log("VPMVisualizationError", {
                "event": "no_dims_found_phos",
                "unit": unit, "df_cols": list(df.columns)
            })
            return ""

        df, _ = self._with_model_prefixed_columns(df, "ssp", dims_to_use)
        vec = vpm_vector_from_df(
            df, model="ssp", dimensions=dims_to_use,
            interleave=self._phos_interleave, weights=self._phos_weights
        )

        self.logger.log("VPMDebug", {
            "unit": unit,
            "step_idx": step_idx,
            "vector_shape": getattr(vec, "shape", None),
            "vector_min": float(np.min(vec)) if len(vec) else 0.0,
            "vector_max": float(np.max(vec)) if len(vec) else 0.0,
        })

        vec = robust01(np.asarray(vec, np.float32))
        if not np.isfinite(vec).any() or float(vec.max() - vec.min()) < 1e-6:
            vec = np.full_like(vec, 0.5, dtype=np.float32)

        tl = float(tl_frac if tl_frac is not None else self._phos_tl_frac)
        img = phos_sort_pack(vec, tl_frac=tl)

        base_name = self._sanitize_unit(unit)
        out_path = Path(output_path) if output_path else (self._phos_viz_dir / f"{base_name}_phos_vpm_{int(time.time())}.png")
        save_img(img, str(out_path), title=f"SSP VPM (PHOS) - {unit}")
        return str(out_path)

    # ------------------------- internals ---------------------------------

    def _save_episode_data(self, unit: str, metrics_history: List[Dict]) -> None:
        """Save episode metrics under episode_data/."""
        try:
            episode = self._episode_traces.get(unit)
            if not episode:
                return
            data = episode.to_dict()
            data["unit"] = unit
            data["metrics_history"] = metrics_history

            out = self._episode_data_dir / f"{self._sanitize_unit(unit)}.json"
            out.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            self.logger.log("VPMVisualizationError", {"event": "episode_data_save_failed", "unit": unit, "error": str(e)})

    def _convert_to_dataframe(self, metrics_history: List[Dict]) -> pd.DataFrame:
        """Convert metrics history into wide DF (node_id x dims)."""
        try:
            rows: List[Dict[str, Any]] = []
            for rec in metrics_history:
                step_idx = int(rec.get("step_idx", 0))
                for k, v in rec.items():
                    if k == "step_idx":
                        continue
                    dim = _KEY_MAP.get(k, k)
                    try:
                        val = float(v)
                    except Exception:
                        continue
                    rows.append({"node_id": step_idx, "dimension": dim, "ssp": val})

            if not rows:
                return pd.DataFrame({"node_id": [], **{d: [] for d in self._dimensions}})

            df = pd.DataFrame(rows)
            df = df.groupby(["node_id", "dimension"], as_index=False, sort=True).agg({"ssp": "last"})
            wide = (
                df.pivot_table(index="node_id", columns="dimension", values="ssp", aggfunc="last", fill_value=0.0)
                .sort_index()
                .reset_index()
            )

            for dim in self._dimensions:
                if dim not in wide.columns:
                    wide[dim] = 0.0

            return wide
        except Exception as e:
            self.logger.log("VPMVisualizationError", {"event": "dataframe_conversion_failed", "error": str(e)})
            n = len(metrics_history)
            return pd.DataFrame({"node_id": list(range(n)), **{d: [0.0] * n for d in self._dimensions}})

    def _with_model_prefixed_columns(self, df: pd.DataFrame, model: str, dims: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Ensure df has columns like 'ssp.dim' for each dim in `dims`."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df, [f"{model}.{d}" for d in dims]

        df2 = df.copy()
        prefixed = []
        for d in dims:
            col_pref = f"{model}.{d}"
            if col_pref not in df2.columns and d in df2.columns:
                df2[col_pref] = df2[d]
            prefixed.append(col_pref)
        return df2, prefixed

    # ------------------------- rendering & snapshots ----------------------

    def _to_uint8_img(self, vec: np.ndarray, *, grid_side: int = 32, fixed_scale: bool = True) -> np.ndarray:
        v = np.nan_to_num(vec.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        v = np.clip(v, 0.0, 1.0)

        if not fixed_scale:
            vmin, vmax = float(v.min()), float(v.max())
            if vmax > vmin:
                v = (v - vmin) / (vmax - vmin)
            else:
                v[:] = 0.5

        n = v.shape[0]
        side = int(grid_side)
        if side * side < n:
            side = int(np.ceil(np.sqrt(n)))
        pad = side * side - n
        if pad > 0:
            v = np.pad(v, (0, pad), constant_values=0.0)
        return (v.reshape(side, side) * 255.0).astype(np.uint8)

    def snapshot_progress(self, *, unit: str, dims: Dict[str, float], step_idx: int, tag: str = "") -> str:
        """Save a progress frame under progress/<unit>/ and track metrics."""
        # maintain feature order
        order = self._feature_order.setdefault(unit, [])
        for k in dims.keys():
            if k not in order:
                order.append(k)

        vec = np.array([float(dims.get(k, 0.0)) for k in order], dtype=np.float32)
        grid_side = int(self.cfg.get("grid_side", self.cfg.get("vpm_viz", {}).get("grid_side", 32)))
        fixed_scale = bool(self.cfg.get("fixed_scale", self.cfg.get("vpm_viz", {}).get("fixed_scale", True)))
        img = self._to_uint8_img(vec, grid_side=grid_side, fixed_scale=fixed_scale)

        unit_dir = self._progress_unit_dir(unit)
        fname = f"{self._sanitize_unit(unit)}_step{step_idx:04d}{'_' + tag if tag else ''}.png"
        out_path = unit_dir / fname

        plt.figure(figsize=(4, 4))
        plt.imshow(img, interpolation="nearest", cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.title(f"{unit} • {tag or 'frame'} • {step_idx}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        _logger.info(f"VPM snapshot saved unit={unit}, step_idx={step_idx}, tag={tag}, path={str(out_path)}")

        # track metrics once
        try:
            vpm_row = VPMRow(
                unit=unit,
                kind="text",
                timestamp=time.time(),
                step_idx=step_idx,
                dims={k: float(dims.get(k, 0.0)) for k in order},
                meta={"tag": tag},
            )
            self._track_metrics(unit, vpm_row, step_idx)
        except Exception as e:
            try:
                self.logger.log("VPMVisualizationError", {"event": "track_metrics_failed", "error": str(e)})
            except Exception:
                pass

        self._progress_frames.setdefault(unit, []).append(out_path)
        return str(out_path)

    def generate_filmstrip(self, unit: str, *, rows: int = None, cols: int = None, dpi: int = None) -> str:
        cfg = self.cfg.get("filmstrip", self.cfg.get("vpm_viz", {}).get("filmstrip", {}))
        rows = rows or int(cfg.get("rows", 2))
        cols = cols or int(cfg.get("cols", 10))
        dpi = dpi or int(cfg.get("dpi", 300))

        unit_dir = self._progress_unit_dir(unit)
        frames = sorted(unit_dir.glob(f"{self._sanitize_unit(unit)}_step*.png"))
        if not frames:
            raise FileNotFoundError(f"No frames found for unit {unit} in {unit_dir}")

        imgs = [Image.open(p).convert("L") for p in frames[: rows * cols]]
        w, h = imgs[0].size
        grid = Image.new("L", (cols * w, rows * h))
        for idx, im in enumerate(imgs):
            r, c = divmod(idx, cols)
            grid.paste(im, (c * w, r * h))

        film_path = unit_dir / f"{self._sanitize_unit(unit)}_filmstrip.png"
        grid.save(film_path, dpi=(dpi, dpi))
        try:
            self.logger.info("VPM filmstrip saved", extra={"unit": unit, "path": str(film_path), "frames": len(imgs)})
        except Exception:
            pass
        return str(film_path)

    # ------------------------- metrics history ----------------------------

    def _track_metrics(self, unit: str, vpm_row: VPMRow, step_idx: Optional[int]) -> None:
        if unit not in self._metrics_history:
            self._metrics_history[unit] = []
        record = {"step_idx": step_idx if step_idx is not None else len(self._metrics_history[unit]),
                  **{k: float(v) for k, v in vpm_row.dims.items()}}
        self._metrics_history[unit].append(record)

        max_history = int(self.cfg.get("vpm_viz", {}).get("max_metrics_history", 100))
        if len(self._metrics_history[unit]) > max_history:
            self._metrics_history[unit] = self._metrics_history[unit][-max_history:]

    # ------------------------- comparison helpers -------------------------

    def _get_comparison_dataframe(self, unit: str, model_a: str, model_b: str) -> pd.DataFrame:
        model_a_metrics = self._get_model_metrics(unit, model_a)
        model_b_metrics = self._get_model_metrics(unit, model_b)

        df_a = self._convert_to_dataframe(model_a_metrics)
        df_b = self._convert_to_dataframe(model_b_metrics)

        df = df_a.merge(df_b, on="node_id", suffixes=("_a", "_b"))
        for dim in self._dimensions:
            a_col, b_col = f"{dim}_a", f"{dim}_b"
            if a_col in df.columns and b_col in df.columns:
                df[f"{model_a}.{dim}"] = df[a_col]
                df[f"{model_b}.{dim}"] = df[b_col]
                df.drop([a_col, b_col], axis=1, inplace=True)
        return df

    def _get_model_metrics(self, unit: str, model: str) -> List[Dict]:
        if model == "current" and unit in self._metrics_history:
            return self._metrics_history[unit]
        # Placeholder: add retrieval for baselines if needed
        return []

    # ------------------------- episode conversion -------------------------

    def _episode_to_vpm_row(self, unit: str, episode: EpisodeTrace, step_idx: Optional[int] = None) -> VPMRow:
        return VPMRow(
            unit=unit,
            kind="text",
            timestamp=time.time(),
            step_idx=step_idx,
            dims=self.episode_to_dims(episode),
            meta={
                "episode_id": episode.episode_id,
                "verified": episode.verified,
                "question": episode.question,
                "predicted_answer": episode.predicted_answer,
                "solver_steps": episode.solver_steps,
                "evidence_count": len(episode.evidence_docs),
                "difficulty": episode.difficulty,
            },
        )

    def episode_to_dims(self, ep: EpisodeTrace) -> Dict[str, float]:
        dims = {
            "verifier_score": float(ep.reward or 0.0),
            "verified": 1.0 if ep.verified else 0.0,
            "difficulty": float(ep.difficulty or 0.0),
            "question_len": min(1.0, len((ep.question or "").split()) / 128.0),
            "answer_len": min(1.0, len((ep.predicted_answer or "").split()) / 128.0),
            "evidence_count": min(1.0, float(len(ep.evidence_docs or [])) / 8.0),
            "solver_steps": min(1.0, float(ep.solver_steps or 0) / 64.0),
        }
        m = getattr(ep, "meta", {}) or {}
        for k in ("score", "best_score", "improvement", "depth", "novelty",
                  "coverage", "correctness", "coherence", "citation_support", "entity_consistency"):
            if k in m:
                try:
                    v = float(m[k])
                    dims[k] = max(0.0, min(1.0, v))
                except Exception:
                    pass
        for d in self._dimensions:
            dims.setdefault(d, 0.0)
        return dims

    def __repr__(self):
        return (
            f"<VPMVisualizationService status={'initialized' if self._initialized else 'uninitialized'} "
            f"viz_dir={self._viz_dir} dims={len(self._dimensions)} episodes={len(self._episode_traces)}>"
        )
