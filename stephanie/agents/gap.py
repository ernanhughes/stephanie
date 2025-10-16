# stephanie/agents/gap.py
# -----------------------------------------------------------------------------
# GAP Agent: HRM vs Tiny end-to-end comparison with PHOS + Frontier + Δ-map
# -----------------------------------------------------------------------------
# Review markers:
#   [S1] Imports & constants
#   [S2] Run manifest helpers (start_run)
#   [S3] GapAgent class: __init__
#   [S4] GapAgent.run(): setup + sample harvesting
#   [S5] GapAgent.run(): scoring loop
#   [S6] GapAgent.run(): timelines → GIFs, Frontier, Δ-map, intensity report
#   [S7] GapAgent.run(): PHOS guarded compare (single-vector VPMs)
#   [S8] GapAgent.run(): persist manifest and return
#   [S9] Internal helpers (flatten, dedupe, harvesting)
# -----------------------------------------------------------------------------

from __future__ import annotations

# [S1] Imports & constants -----------------------------------------------------
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import asyncio
import hashlib
import json
import logging
import os
import time

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.scoring.training.preference_pair_builder import PreferencePairBuilder
from stephanie.services.workers.metrics_worker import MetricsWorkerInline
from stephanie.services.workers.vpm_worker import VPMWorkerInline
from stephanie.utils.json_sanitize import dumps_safe

_logger = logging.getLogger(__name__)

# Candidate suffixes to “discover” metric columns in loose payloads
CAND_SUFFIXES = ["", ".score", ".aggregate", ".raw", ".value"]

# Force a non-interactive backend if needed (safe in servers/notebooks)
if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")


# [S2] Run manifest helpers (start_run) ---------------------------------------
@dataclass
class RunManifest:
    """Minimal, reproducible run identity + metadata saved to disk."""
    run_id: str
    dataset: str
    models: dict
    preproc_version: str = "v1"
    created_at: float = field(default_factory=lambda: time.time())


def start_run(run_id, dataset: str, models: dict, base_root: str = "data/gap_runs") -> RunManifest:
    """
    Create a reproducible run folder structure and a manifest.json.

    Layout:
      gap_runs/<run_id>/
        raw/ aligned/ visuals/ metrics/ reports/
        manifest.json
    """
    base = os.path.join(base_root, run_id)
    for sub in ("raw", "aligned", "visuals", "metrics", "reports"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    m = RunManifest(run_id=run_id, dataset=dataset, models=models)
    with open(os.path.join(base, "manifest.json"), "w", encoding="utf-8") as f:
        f.write(dumps_safe(asdict(m), indent=2))
    return m

# [S3.a] IO helpers for timelines/matrices -----------------------------------
def _load_triples_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # tolerate a bad line but keep going
                continue
    return rows


# [S4.a] helpers --------------------------------------------------------------
def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _update_manifest(*, base_root: str, run_id: str, patch: dict) -> None:
    mpath = os.path.join(base_root, run_id, "manifest.json")
    try:
        cur = _read_json(mpath)
    except Exception:
        cur = {}
    # shallow merge (dicts only)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(cur.get(k), dict):
            cur[k].update(v)
        else:
            cur[k] = v
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(dumps_safe(cur, indent=2))

def _safe_vec(metrics: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    """
    Extract (names, values) from a metrics payload.
    We prefer a flat 'vector' dict if present; else try ('columns','values').
    """
    vec = metrics.get("vector")
    if isinstance(vec, dict) and vec:
        names = list(vec.keys())
        vals = [float(vec[k]) for k in names]
        return names, vals

    cols, vals = metrics.get("columns"), metrics.get("values")
    if isinstance(cols, list) and isinstance(vals, list) and len(cols) == len(vals):
        try:
            return [str(c) for c in cols], [float(v) for v in vals]
        except Exception:
            pass

    return [], []  # empty fallback

# [S5.a] small helpers --------------------------------------------------------
def _try_load_df(raw_dir: str) -> pd.DataFrame | None:
    """
    Try to load rows_for_df from raw/. If not found, return None instead of raising.
    """
    pq = os.path.join(raw_dir, "rows_for_df.parquet")
    csv = os.path.join(raw_dir, "rows_for_df.csv")
    try:
        if os.path.exists(pq):
            return pd.read_parquet(pq)
        if os.path.exists(csv):
            return pd.read_csv(csv)
    except Exception:
        # fall through to None on any read error
        pass
    return None


def _pick_metric_column(df: pd.DataFrame, base: str) -> str | None:
    if base in df.columns:
        return base
    for suf in CAND_SUFFIXES:
        cand = f"{base}{suf}"
        if cand in df.columns:
            return cand
    pref = f"{base}."
    for c in df.columns:
        if isinstance(c, str) and c.startswith(pref):
            return c
    return None

def _project_dimensions(df_in: pd.DataFrame, dims: list[str], logger) -> pd.DataFrame:
    out = {"node_id": df_in["node_id"].values}
    missing = {"hrm": [], "tiny": []}
    for d in dims:
        h_col = _pick_metric_column(df_in, f"hrm.{d}")
        t_col = _pick_metric_column(df_in, f"tiny.{d}")

        if h_col is None:
            missing["hrm"].append(d)
            out[f"hrm.{d}"] = 0.0
        else:
            out[f"hrm.{d}"] = pd.to_numeric(df_in[h_col], errors="coerce").fillna(0.0).astype(float)

        if t_col is None:
            missing["tiny"].append(d)
            out[f"tiny.{d}"] = 0.0
        else:
            out[f"tiny.{d}"] = pd.to_numeric(df_in[t_col], errors="coerce").fillna(0.0).astype(float)

    logger.log("PHOSColumnDiscovery", {
        "rows": int(df_in.shape[0]),
        "dims": dims,
        "present_hrm": [c for c in df_in.columns if isinstance(c, str) and c.startswith("hrm.")],
        "present_tiny": [c for c in df_in.columns if isinstance(c, str) and c.startswith("tiny.")],
        "missing_hrm_dims": missing["hrm"],
        "missing_tiny_dims": missing["tiny"],
    })
    return pd.DataFrame(out)


def _write_matrix_and_names(base_root: str, run_id: str, tag: str, names: List[str], mat: np.ndarray) -> Dict[str, Any]:
    """
    Persist matrix + metric names to aligned/ and return a payload for manifest.
    """
    aligned_dir = os.path.join(base_root, run_id, "aligned")
    os.makedirs(aligned_dir, exist_ok=True)
    mpath = os.path.join(aligned_dir, f"{tag}_matrix.npy")
    npath = os.path.join(aligned_dir, f"{tag}_metric_names.json")
    np.save(mpath, mat.astype(np.float32))
    with open(npath, "w", encoding="utf-8") as f:
        f.write(dumps_safe(names, indent=2))
    return {"matrix": mpath, "names": npath, "shape": list(map(int, mat.shape))}


def _write_timeline_gif(base_root: str, run_id: str, tag: str, out_path_from_worker: str | None) -> str:
    """
    The worker already renders a GIF and returns its path. We copy it under visuals/.
    If the worker path is already there, just report it.
    """
    visuals_dir = os.path.join(base_root, run_id, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)
    dst = os.path.join(visuals_dir, f"{tag}_timeline.gif")
    try:
        if out_path_from_worker and os.path.exists(out_path_from_worker):
            import shutil
            shutil.copyfile(out_path_from_worker, dst)
            return dst
    except Exception:
        pass
    # fallback (worker path unknown): just record intended destination
    return dst


# [S6.a] Helpers: pick columns, monotone calibration, routing -----------------
def _find_first(names, candidates):
    for c in candidates:
        if c in names: return c
    return None

def _col_for_dim(names: list[str], model: str, dim: str) -> int | None:
    """
    Find a column index for {model}.{dim} score-like series.
    Preference: .score -> .aggregate -> exact -> any suffix match.
    """
    priorities = [
        f"{model}.{dim}.score",
        f"{model}.{dim}.aggregate",
        f"{model}.{dim}",
    ]
    choice = _find_first(names, priorities)
    if choice is None:
        # loose search fallback
        for i, n in enumerate(names):
            if n.startswith(f"{model}.{dim}") and n.endswith((".score", ".aggregate", ".value", ".raw")):
                return i
        return None
    return names.index(choice)

def _col_for_diag(names: list[str], model: str, key_parts: tuple[str, ...]) -> int | None:
    """
    Find diagnostic column like 'tiny.reasoning.attr.ood_hat' or 'tiny.uncertainty'.
    Matches if all key_parts appear in the column name (in order).
    """
    def has_parts(s: str) -> bool:
        pos = 0
        for kp in key_parts:
            j = s.find(kp, pos)
            if j < 0: return False
            pos = j + len(kp)
        return True
    for i, n in enumerate(names):
        if n.startswith(f"{model}.") and has_parts(n):
            return i
    return None

def _monotone_pl_calibration(x: np.ndarray, y: np.ndarray, *, n_knots: int = 21) -> dict:
    """
    Fit a simple monotone piecewise-linear map Tiny->HRM using quantile knots.
    - x: Tiny scores in [0,1]
    - y: HRM scores in [0,1]
    Returns {"x_knots": [...], "y_knots": [...]} where y is non-decreasing in x.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        return {"x_knots": [0.0, 1.0], "y_knots": [0.0, 1.0]}
    # define quantile grid on x; map each bin center to mean y in that bin
    qs = np.linspace(0, 1, n_knots)
    x_knots = np.quantile(x, qs).astype(float)
    # avoid duplicates (flat distributions)
    x_knots[0], x_knots[-1] = 0.0, 1.0
    # compute y at each knot via local averaging
    y_knots = []
    for q in qs:
        # window around q
        lo = np.quantile(x, max(0.0, q - 0.025))
        hi = np.quantile(x, min(1.0, q + 0.025))
        mask = (x >= lo) & (x <= hi)
        if mask.sum() < 8:  # fallback to nearest neighbors
            idx = np.argsort(np.abs(x - np.quantile(x, q)))[:16]
            y_knots.append(float(np.mean(y[idx])))
        else:
            y_knots.append(float(np.mean(y[mask])))
    # enforce monotonicity (isotonic projection)
    yk = np.array(y_knots, dtype=np.float64)
    for i in range(1, len(yk)):
        if yk[i] < yk[i-1]:
            yk[i] = yk[i-1]
    # clamp to [0,1]
    yk = np.clip(yk, 0.0, 1.0)
    return {"x_knots": x_knots.tolist(), "y_knots": yk.tolist()}

def _apply_monotone_pl(x: np.ndarray, calib: dict) -> np.ndarray:
    if not calib or "x_knots" not in calib or "y_knots" not in calib:
        return x
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    xk = np.asarray(calib["x_knots"], dtype=np.float64)
    yk = np.asarray(calib["y_knots"], dtype=np.float64)
    # linear interp with clipping
    return np.interp(xx, xk, yk).astype(np.float64)

def _safe01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    return np.clip(a, 0.0, 1.0)

def _mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0: return 0.0
    return float(np.mean(np.abs(a - b)))

def _write_progress(base_root: str, run_id: str, *, phase: str, done: int, total: int, extra: dict | None = None):
    progress_dir = os.path.join(base_root, run_id, "metrics")
    os.makedirs(progress_dir, exist_ok=True)
    payload = {
        "phase": phase,
        "done": int(done),
        "total": int(total),
        "percent": float(0 if total == 0 else 100.0 * done / total),
        "ts": time.time(),
    }
    if extra:
        payload.update(extra)
    with open(os.path.join(progress_dir, "progress.json"), "w", encoding="utf-8") as f:
        f.write(dumps_safe(payload, indent=2))


# [S3] GapAgent class: __init__ -----------------------------------------------
class GapAgent(BaseAgent):
    """
    Compare HRM vs Tiny via:
      - Timelines (GIFs)
      - Frontier map (aligned difference field)
      - Inter-model Δ-map (HRM − Tiny) with metadata
      - PHOS-guarded VPMs (single-vector packed images)
      - Intensity report (top rows/cols, overlap, Δ-mass)

    Outputs are written under:
      - data/vpm/…  (legacy PHOS paths)
      - gap_runs/<run_id>/…  (stable GAP paths)
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        # Core config
        self.dimensions = list(
            cfg.get(
                "dimensions",
                ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"],
            )
        )
        self.hrm_scorers = list(cfg.get("hrm_scorers", ["hrm"]))
        self.tiny_scorers = list(cfg.get("tiny_scorers", ["tiny"]))

        # Output roots
        self.out_dir = Path(cfg.get("out_dir", "data/vpm"))              # legacy/PHOS
        self.base_dir = Path(cfg.get("gap_base_dir", "data/gap_runs"))        # GAP root

        # Misc config
        self.interleave = bool(cfg.get("interleave", False))
        self.progress_log_every = int(cfg.get("progress_log_every", 25))
        self.progress_manifest_every = int(cfg.get("progress_manifest_every", 25))
        self.progress_enable_tqdm = bool(cfg.get("progress_enable_tqdm", True))
        self.limit = int(cfg.get("limit", 100))  # optional limit on samples


    # [S4] GapAgent.run(): setup + sample harvesting --------------------------
    async def run(self, context: dict) -> dict:
        """
        Main entrypoint. Produces artifacts and returns paths + metrics
        in the agent output key.
        """
        eval_stats: Dict[str, Any] = {}

        # 4.1 Tools
        scoring_service = self.container.get("scoring")
        zm = self.container.get("zeromodel")
        hrm_worker = MetricsWorkerInline(scoring_service, self.hrm_scorers, self.dimensions)
        tiny_worker = MetricsWorkerInline(scoring_service, self.tiny_scorers, self.dimensions)
        vpm_worker = VPMWorkerInline(zm, self.logger)

        # 4.2 Reproducible run identity
        dataset_name = context.get("dataset", "unknown")
        models = {"hrm": self.hrm_scorers[0], "tiny": self.tiny_scorers[0]}
        pipeline_run_id = context.get("pipeline_run_id")
        if not pipeline_run_id:
            raise ValueError("GapAgent requires context['pipeline_run_id']")

        manifest = start_run(run_id=pipeline_run_id, dataset=dataset_name, models=models, base_root=str(self.base_dir))
        # after manifest = start_run(...)
        base = os.path.join(str(self.base_dir), manifest.run_id)
        manifest_dict = asdict(manifest)
        manifest_dict["dimensions"] = self.dimensions
        manifest_dict["paths"] = {
            "root": base,
            "raw": os.path.join(base, "raw"),
            "aligned": os.path.join(base, "aligned"),
            "visuals": os.path.join(base, "visuals"),
            "metrics": os.path.join(base, "metrics"),
            "reports": os.path.join(base, "reports"),
        }

        with open(os.path.join(base, "manifest.json"), "w", encoding="utf-8") as f:
            f.write(dumps_safe(manifest_dict, indent=2))

        # 4.3 Open timelines
        hrm_run_id = f"{pipeline_run_id}_hrm"
        tiny_run_id = f"{pipeline_run_id}_tiny"
        zm.timeline_open(run_id=hrm_run_id)
        zm.timeline_open(run_id=tiny_run_id)

        # 4.4 Collect + dedupe samples
        pair_builder = PreferencePairBuilder(self.memory, self.logger)
        triples_by_dim: Dict[str, List[Tuple[str, str, float]]] = {}
        total_raw = 0

        for dimension in self.dimensions:
            pairs_by_dim = pair_builder.get_training_pairs_by_dimension(dimension=dimension, limit=100)
            samples_full = pairs_by_dim.get(dimension, [])
            if not samples_full:
                self.logger.log("NoSamplesFound", {"dimension": dimension})
                triples_by_dim[dimension] = []
                continue
            triples = _flatten_samples_for_eval(samples_full)
            triples_by_dim[dimension] = triples
            total_raw += len(triples)

        deduped = _dedupe_triples_by_dimension(
            triples_by_dim,
            policy=self.cfg.get("dedupe_policy", "first_wins"),
            per_dim_cap=self.cfg.get("per_dim_cap")  # e.g., 400
        )

        # [S2.b] Materialize the triples index --------------------------------------
        policy_used = self.cfg.get("dedupe_policy", "first_wins")
        triples_info = _write_triples_index(
            base_root=str(self.base_dir),
            run_id=pipeline_run_id,
            deduped=deduped,
            policy=policy_used,
            write_parquet=True,
        )

        # Update manifest with inputs + counts
        _update_manifest(
            base_root=str(self.base_dir),
            run_id=pipeline_run_id,
            patch={
                "inputs": {
                    "triples_index": triples_info["jsonl_path"],
                    "triples_index_parquet": triples_info.get("parquet_path"),
                    "triples_head": triples_info["head_path"],
                    "triples_rows": int(triples_info["rows"]),
                },
                "stats": {
                    "samples_total_unique": int(triples_info["rows"]),
                    "per_dim_counts": {k: int(v) for k, v in triples_info["per_dim_counts"].items()},
                },
            },
        )

        # Optional: log a friendly breadcrumb
        self.logger.log("GapTriplesIndex", {
            "run_id": pipeline_run_id,
            "path": triples_info["jsonl_path"],
            "rows": int(triples_info["rows"]),
            "per_dim_counts": {k: int(v) for k, v in triples_info["per_dim_counts"].items()},
        })


        total_triples = sum(len(v) for v in deduped.values())
        if total_triples == 0:
            self.logger.log("PhosHRMNoSamples", {"dimensions": self.dimensions})
            context[self.output_key] = {"msg": "no samples"}
            return context

        self.logger.log("PhosHRMDedupeSummary", {
            "run_id": pipeline_run_id,
            "total_raw": total_raw,
            "total_unique": total_triples,
            "per_dim_counts": {d: len(deduped[d]) for d in self.dimensions}
        })

        # [S5] GapAgent.run(): scoring loop -----------------------------------
        processed = 0
        rows_for_df: List[Dict[str, float]] = []   # per-turn, per-dimension table for PHOS
        raw_dir = os.path.join(str(self.base_dir), pipeline_run_id, "raw")
        os.makedirs(raw_dir, exist_ok=True)


        # [S3.b] Timelines + matrices from triples -----------------------------------
        triples_path = os.path.join(str(self.base_dir), pipeline_run_id, "raw", "triples.jsonl")
        triples = _load_triples_jsonl(triples_path)
        T = len(triples)
        if T == 0:
            self.logger.log("GapStep3NoTriples", {"run_id": pipeline_run_id, "path": triples_path})
            context[self.output_key] = {"msg": "no triples for step 3"}
            return context

        # open fresh timelines (distinct from HRM/Tiny scoring session IDs if needed)
        hrm_run_id = f"{pipeline_run_id}_hrm"
        tiny_run_id = f"{pipeline_run_id}_tiny"
        zm.timeline_open(run_id=hrm_run_id)
        zm.timeline_open(run_id=tiny_run_id)

        # We'll accumulate rows → matrices
        hrm_names: List[str] = []
        tiny_names: List[str] = []
        hrm_rows: List[List[float]] = []
        tiny_rows: List[List[float]] = []

        _write_progress(str(self.base_dir), pipeline_run_id, phase="step3_scoring", done=0, total=T)

        with tqdm(total=T, desc="[GAP] step 3: timelines & matrices", unit="turn") as pbar:
            for i, row in enumerate(triples):
                goal_text = row.get("goal_text", "")
                output_text = row.get("output_text", "")
                node_id = row.get("node_id") or f"{pipeline_run_id}|{i:06d}"

                scorable = Scorable(output_text, ScorableType.CONVERSATION_TURN)

                # score with HRM and Tiny
                hrm_metrics = await hrm_worker.score(scorable, goal_text, hrm_run_id)
                tiny_metrics = await tiny_worker.score(scorable, goal_text, tiny_run_id)
                # harvest per-dimension hrm./tiny. columns for PHOS (robust to suffixes)
                row_for_df = {"node_id": node_id}
                _harvest_model_dim_scores(row_for_df, hrm_metrics, model_name="hrm", dims=self.dimensions)
                _harvest_model_dim_scores(row_for_df, tiny_metrics, model_name="tiny", dims=self.dimensions)
                rows_for_df.append(row_for_df)
                # ----------------------------------------------


                # append to timelines for GIFs
                await vpm_worker.append(hrm_run_id, node_id, hrm_metrics)
                await vpm_worker.append(tiny_run_id, node_id, tiny_metrics)

                # extract vectors → matrices
                h_names, h_vals = _safe_vec(hrm_metrics)
                t_names, t_vals = _safe_vec(tiny_metrics)

                if i == 0:
                    hrm_names = h_names[:]  # lock order on first example
                    tiny_names = t_names[:]
                else:
                    # defensive: if names differ in later rows, align by first-row names
                    # (missing keys become 0.0, extras are ignored)
                    if h_names != hrm_names:
                        name2val = {n: v for n, v in zip(h_names, h_vals)}
                        h_vals = [float(name2val.get(n, 0.0)) for n in hrm_names]
                    if t_names != tiny_names:
                        name2val = {n: v for n, v in zip(t_names, t_vals)}
                        t_vals = [float(name2val.get(n, 0.0)) for n in tiny_names]

                hrm_rows.append([float(v) for v in h_vals])
                tiny_rows.append([float(v) for v in t_vals])

                processed += 1
                if self.progress_enable_tqdm:
                    pbar.update(1)
                    # keep the bar visually fresh even in async contexts
                    if ((processed % 5) == 0) or (processed == T):
                        pbar.refresh()

                # fire logs *early* and regularly
                if (processed == 1) or (processed % self.progress_log_every == 0) or (processed == T):
                    self.logger.log("GapStep3Progress", {
                        "run_id": pipeline_run_id,
                        "done": processed,
                        "total": T,
                        "hrm_cols": len(hrm_names),
                        "tiny_cols": len(tiny_names),
                        "percent": round(100.0 * processed / max(1, T), 2),
                    })
                # write the progress heartbeat for UIs to poll
                if (processed == 1) or (processed % self.progress_manifest_every == 0) or (processed == T):
                    _write_progress(
                        base_root=str(self.base_dir),
                        run_id=pipeline_run_id,
                        phase="step3_scoring",
                        done=processed,
                        total=T,
                        extra={"hrm_cols": len(hrm_names), "tiny_cols": len(tiny_names)}
                    )
                await asyncio.sleep(0)  # cooperative

        _write_progress(str(self.base_dir), pipeline_run_id, phase="step3_scoring_done", done=T, total=T)

        raw_dir = os.path.join(str(self.base_dir), pipeline_run_id, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        df_rows = pd.DataFrame(rows_for_df)
        # prefer parquet; fallback to csv
        parquet_path = os.path.join(raw_dir, "rows_for_df.parquet")
        csv_path     = os.path.join(raw_dir, "rows_for_df.csv")
        try:
            df_rows.to_parquet(parquet_path, index=False)
        except Exception:
            df_rows.to_csv(csv_path, index=False)


        # finalize GIFs via worker (returns paths)
        hrm_gif_path = await vpm_worker.finalize(hrm_run_id, f"vpm_phos_run_{hrm_run_id}.gif")
        tiny_gif_path = await vpm_worker.finalize(tiny_run_id, f"vpm_phos_run_{tiny_run_id}.gif")

        # persist matrices + names
        hrm_mat = np.array(hrm_rows, dtype=np.float32) if hrm_rows else np.zeros((0,0), np.float32)
        tiny_mat = np.array(tiny_rows, dtype=np.float32) if tiny_rows else np.zeros((0,0), np.float32)
        hrm_pack = _write_matrix_and_names(str(self.base_dir), pipeline_run_id, "hrm", hrm_names, hrm_mat)
        tiny_pack = _write_matrix_and_names(str(self.base_dir), pipeline_run_id, "tiny", tiny_names, tiny_mat)

        # copy GIFs to visuals/
        hrm_gif = _write_timeline_gif(str(self.base_dir), pipeline_run_id, "hrm", hrm_gif_path.get("output_path") if isinstance(hrm_gif_path, dict) else hrm_gif_path)
        tiny_gif = _write_timeline_gif(str(self.base_dir), pipeline_run_id, "tiny", tiny_gif_path.get("output_path") if isinstance(tiny_gif_path, dict) else tiny_gif_path)

        # --- persist rows_for_df so PHOS can read it later ---
        try:
            if rows_for_df:
                df_rows = pd.DataFrame(rows_for_df)
                df_rows.to_parquet(os.path.join(raw_dir, "rows_for_df.parquet"), index=False)
                df_rows.to_csv(os.path.join(raw_dir, "rows_for_df.csv"), index=False)
                _update_manifest(
                    base_root=str(self.base_dir),
                    run_id=pipeline_run_id,
                    patch={"inputs": {"rows_for_df_parquet": os.path.join(raw_dir, "rows_for_df.parquet"),
                                    "rows_for_df_csv": os.path.join(raw_dir, "rows_for_df.csv")}}
                )
                self.logger.log("GapRowsForDFWritten", {
                    "run_id": pipeline_run_id,
                    "rows": len(df_rows),
                    "path_parquet": os.path.join(raw_dir, "rows_for_df.parquet"),
                    "path_csv": os.path.join(raw_dir, "rows_for_df.csv"),
                })
            else:
                self.logger.log("GapRowsForDFEmpty", {"run_id": pipeline_run_id})
        except Exception as e:
            self.logger.log("GapRowsForDFWriteError", {"run_id": pipeline_run_id, "error": str(e)})
        # ----------------------------------------------------


        # manifest patch
        _update_manifest(
            base_root=str(self.base_dir),
            run_id=pipeline_run_id,
            patch={
                "timelines": {
                    "hrm_gif": hrm_gif,
                    "tiny_gif": tiny_gif,
                },
                "aligned": {
                    "hrm_matrix": hrm_pack["matrix"],
                    "tiny_matrix": tiny_pack["matrix"],
                    "hrm_metric_names": hrm_pack["names"],
                    "tiny_metric_names": tiny_pack["names"],
                    "hrm_shape": hrm_pack["shape"],
                    "tiny_shape": tiny_pack["shape"],
                },
            },
        )

        # also return in eval_stats for convenience
        eval_stats.update({
            "timelines": {"hrm_gif": hrm_gif, "tiny_gif": tiny_gif},
            "aligned": {
                "hrm": {"matrix": hrm_pack["matrix"], "names": hrm_pack["names"], "shape": hrm_pack["shape"]},
                "tiny": {"matrix": tiny_pack["matrix"], "names": tiny_pack["names"], "shape": tiny_pack["shape"]},
            }
        })

        # [S4.b] Frontier & Δ-map from aligned artifacts ------------------------------
        base = str(self.base_dir)
        aligned_dir = os.path.join(base, pipeline_run_id, "aligned")
        visuals_dir = os.path.join(base, pipeline_run_id, "visuals")
        metrics_dir = os.path.join(base, pipeline_run_id, "metrics")
        os.makedirs(visuals_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        # Load aligned matrices and names
        hrm_mat = np.load(os.path.join(aligned_dir, "hrm_matrix.npy"))
        tiny_mat = np.load(os.path.join(aligned_dir, "tiny_matrix.npy"))
        hrm_names = _read_json(os.path.join(aligned_dir, "hrm_metric_names.json"))
        tiny_names = _read_json(os.path.join(aligned_dir, "tiny_metric_names.json"))

        # 4.1 FRONTIER (aligned difference field)
        # Uses your ZeroModel renderer; produces a PNG + summary meta.
        frontier_out_dir = visuals_dir  # keep in visuals
        frontier_meta = zm.render_frontier_map(
            hrm_mat, tiny_mat,
            out_dir=str(frontier_out_dir),
            pos_label="HRM",
            neg_label="Tiny",
            k_latent=20  # or your default
        )
        # Expect frontier_meta to include file path(s) and summary stats.

        # 4.2 Inter-model Δ-meta (numbers you cite in the blog)
        delta_out_dir = metrics_dir
        delta_meta = zm.render_intermodel_delta(
            hrm_mat, tiny_mat,
            names_A=hrm_names,
            names_B=tiny_names,
            output_dir=str(delta_out_dir),
            pos_label="HRM",
            neg_label="Tiny",
        )

        # Optional: quick |Δ| heat for eyeballing (grayscale)
        try:
            # align by names before subtracting
            A2, B2, shared_names, diag = _align_mats_by_names(hrm_mat, hrm_names, tiny_mat, tiny_names, mode="intersection")

            # write a small debug artifact so you can see what didn’t match
            with open(os.path.join(metrics_dir, "name_alignment_debug.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "shared_count": len(shared_names),
                    "missing_in_tiny": diag.get("missing_in_B", []),
                    "missing_in_hrm": diag.get("missing_in_A", []),
                    "mode": diag.get("mode", "intersection"),
                }, f, indent=2)

            if A2.shape[1] == 0:
                # no overlap — avoid crash and leave a breadcrumb
                self.logger.log("DeltaHeatNoOverlap", {
                    "run_id": pipeline_run_id,
                    "hrm_cols": len(hrm_names),
                    "tiny_cols": len(tiny_names),
                })
            else:
                Dabs = np.abs(A2 - B2)
                plt.figure(figsize=(8, 6))
                plt.imshow(Dabs, cmap="gray", aspect="auto")
                plt.title("|HRM − Tiny| (aligned-by-names)")
                plt.axis("off")
                abs_path = os.path.join(visuals_dir, "delta_heat.png")
                plt.savefig(abs_path, dpi=160, bbox_inches="tight")
                plt.close()
                delta_meta["delta_abs_heat"] = abs_path
        except Exception as e:
            self.logger.log("DeltaHeatError", {"err": str(e)})

        # Persist + expose in manifest/eval_stats
        delta_path = os.path.join(metrics_dir, "intermodel_delta.json")
        with open(delta_path, "w", encoding="utf-8") as f:
            f.write(dumps_safe(delta_meta, indent=2))

        # Collect key fields you’ll cite in the blog
        delta_mass = float(delta_meta.get("delta_mass", np.nan))
        overlap    = float(delta_meta.get("overlap", np.nan))
        top_cols   = delta_meta.get("top_cols", [])
        top_rows   = delta_meta.get("top_rows", [])
        frontier_png = frontier_meta.get("frontier_png") or frontier_meta.get("image_path")  # depending on your impl

        # Update manifest and eval_stats
        _update_manifest(
            base_root=base,
            run_id=pipeline_run_id,
            patch={
                "frontier": {
                    "image": frontier_png,
                    **{k: v for k, v in frontier_meta.items() if k != "image"}  # keep extra fields
                },
                "intermodel_delta": {
                    "json": delta_path,
                    "delta_mass": delta_mass,
                    "overlap": overlap,
                    "top_cols": top_cols,
                    "top_rows": top_rows,
                }
            },
        )

        eval_stats.update({
            "frontier": frontier_meta,
            "intermodel_delta": {
                "path": delta_path,
                "delta_mass": delta_mass,
                "overlap": overlap,
                "top_cols": top_cols,
                "top_rows": top_rows,
            }
        })

        # [S5.b] PHOS-guarded VPMs + Intensity Report --------------------------------
        base = str(self.base_dir)
        raw_dir     = os.path.join(base, pipeline_run_id, "raw")
        aligned_dir = os.path.join(base, pipeline_run_id, "aligned")
        visuals_dir = os.path.join(base, pipeline_run_id, "visuals")
        metrics_dir = os.path.join(base, pipeline_run_id, "metrics")
        os.makedirs(visuals_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        # 5.1 Load the per-turn rows DataFrame and project to canonical hrm./tiny. cols
        try:
            df_raw = _try_load_df(raw_dir)
        except Exception as e:
            self.logger.log("PHOSRowsMissing", {"error": str(e)})
            df_raw = None

        if df_raw is not None:
            keep = ["node_id"] + [c for c in df_raw.columns
                                if isinstance(c, str) and (c.startswith("hrm.") or c.startswith("tiny."))]
            df_raw = df_raw[keep]
            df_proj = _project_dimensions(df_raw, self.dimensions, self.logger)

            # 5.2 Build PHOS-guarded artifacts using your zeromodel helper
            from stephanie.zeromodel.vpm_phos import build_hrm_vs_tiny_guarded
            vpm_prefix = os.path.join(visuals_dir, "vpm")  # prefix for per-model sweep outputs
            phos_res = build_hrm_vs_tiny_guarded(
                df_proj,
                dimensions=self.dimensions,
                out_prefix=vpm_prefix,
                tl_fracs=(0.25, 0.16, 0.36, 0.09),
                delta=0.02,
                interleave=bool(self.interleave),
                weights=None,
            )

            # Standardize “chosen” copies to stable names in visuals/
            # (build_hrm_vs_tiny_guarded already writes *_chosen.png; we also expose them plainly)
            import shutil
            hrm_chosen = phos_res.get("hrm_chosen", {})
            tiny_chosen = phos_res.get("tiny_chosen", {})
            if hrm_chosen.get("phos_path"):
                shutil.copyfile(hrm_chosen["phos_path"], os.path.join(visuals_dir, "hrm_vpm_phos.png"))
            if tiny_chosen.get("phos_path"):
                shutil.copyfile(tiny_chosen["phos_path"], os.path.join(visuals_dir, "tiny_vpm_phos.png"))
            # Raw versions (best effort)
            if hrm_chosen.get("raw_path"):
                shutil.copyfile(hrm_chosen["raw_path"], os.path.join(visuals_dir, "hrm_vpm_raw.png"))
            if tiny_chosen.get("raw_path"):
                shutil.copyfile(tiny_chosen["raw_path"], os.path.join(visuals_dir, "tiny_vpm_raw.png"))

            # Persist sweep summaries to metrics/
            with open(os.path.join(metrics_dir, "hrm_vpm_guard_metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"model": "hrm", **{"chosen": hrm_chosen}, **{"sweep": phos_res.get("sweep", {}).get("hrm", [])}}, f, indent=2)
            with open(os.path.join(metrics_dir, "tiny_vpm_guard_metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"model": "tiny", **{"chosen": tiny_chosen}, **{"sweep": phos_res.get("sweep", {}).get("tiny", [])}}, f, indent=2)
            with open(os.path.join(metrics_dir, "guard_compare.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "delta": 0.02,
                    "hrm_chosen": hrm_chosen,
                    "tiny_chosen": tiny_chosen
                }, f, indent=2)

            # Expose result in eval_stats
            eval_stats["phos_guarded"] = {
                "hrm": {k: v for k, v in hrm_chosen.items() if k in ("tl_frac", "raw_path", "phos_path", "raw_conc", "phos_conc", "improved")},
                "tiny": {k: v for k, v in tiny_chosen.items() if k in ("tl_frac", "raw_path", "phos_path", "raw_conc", "phos_conc", "improved")},
            }
        else:
            self.logger.log("PHOSRowsMissing", {"run_id": pipeline_run_id, "raw_dir": raw_dir})

        # 5.3 Intensity Report (Top rows/cols, overlap, Δ-mass) — from aligned matrices
        hrm_mat = np.load(os.path.join(aligned_dir, "hrm_matrix.npy"))
        tiny_mat = np.load(os.path.join(aligned_dir, "tiny_matrix.npy"))
        hrm_names = _read_json(os.path.join(aligned_dir, "hrm_metric_names.json"))
        tiny_names = _read_json(os.path.join(aligned_dir, "tiny_metric_names.json"))

        intensity = zm.build_intensity_report(
            hrm_matrix=hrm_mat,
            tiny_matrix=tiny_mat,
            hrm_metric_names=hrm_names,
            tiny_metric_names=tiny_names,
            out_dir=metrics_dir,        # write straight under metrics/
            top_k=25,
        )
        # intensity returns { path: <json>, ... }
        eval_stats["intensity_report"] = {"path": intensity.get("path")}

        _update_manifest(
            base_root=str(self.base_dir),
            run_id=pipeline_run_id,
            patch={
                "vpm": {
                    "hrm_phos_png": os.path.join(visuals_dir, "hrm_vpm_phos.png"),
                    "tiny_phos_png": os.path.join(visuals_dir, "tiny_vpm_phos.png"),
                    "hrm_raw_png": os.path.join(visuals_dir, "hrm_vpm_raw.png"),
                    "tiny_raw_png": os.path.join(visuals_dir, "tiny_vpm_raw.png"),
                },
                "intensity_report": {
                    "path": os.path.join(metrics_dir, "intensity_report.json")
                }
            }
        )

        # [S6.b] Calibration + Routing This is something that happened Simulation ------------------------------------
        aligned_dir = os.path.join(str(self.base_dir), pipeline_run_id, "aligned")
        metrics_dir = os.path.join(str(self.base_dir), pipeline_run_id, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        # Load aligned matrices + names
        hrm_mat = np.load(os.path.join(aligned_dir, "hrm_matrix.npy"))
        tiny_mat = np.load(os.path.join(aligned_dir, "tiny_matrix.npy"))
        hrm_names = _read_json(os.path.join(aligned_dir, "hrm_metric_names.json"))
        tiny_names = _read_json(os.path.join(aligned_dir, "tiny_metric_names.json"))

        dims = self.dimensions  # e.g., ["reasoning","knowledge","clarity","faithfulness","coverage"]

        # 6.1 Extract per-dimension aggregate scores (HRM reference vs Tiny to calibrate)
        calib_params = {}
        dim_stats = []

        for dim in dims:
            i_hrm = _col_for_dim(hrm_names, "hrm", dim)
            i_tny = _col_for_dim(tiny_names, "tiny", dim)
            if i_hrm is None or i_tny is None:
                # record missing and continue
                dim_stats.append({
                    "dimension": dim, "status": "missing",
                    "hrm_col": i_hrm, "tiny_col": i_tny
                })
                continue

            hrm_scores = _safe01(hrm_mat[:, i_hrm])
            tiny_scores = _safe01(tiny_mat[:, i_tny])

            # Fit monotone piecewise-linear calibration Tiny->HRM
            calib = _monotone_pl_calibration(tiny_scores, hrm_scores, n_knots=21)
            calib_params[dim] = calib

            # Evaluate pre/post calibration MAE to HRM
            mae_pre = _mae(tiny_scores, hrm_scores)
            tiny_cal = _apply_monotone_pl(tiny_scores, calib)
            mae_post = _mae(tiny_cal, hrm_scores)

            dim_stats.append({
                "dimension": dim,
                "status": "ok",
                "mae_pre": round(mae_pre, 6),
                "mae_post": round(mae_post, 6),
                "improved": bool(mae_post < mae_pre - 1e-6),
                "hrm_col": int(i_hrm),
                "tiny_col": int(i_tny),
            })

        # Persist calibration params
        with open(os.path.join(metrics_dir, "calibration_params.json"), "w", encoding="utf-8") as f:
            json.dump({"per_dimension": calib_params, "stats": dim_stats}, f, indent=2)

        # 6.2 Build routing mask from Tiny diagnostics (OOD/uncertainty)
        # Try to find diag signals on the Tiny side; fall back to zeros if missing.
        def _maybe_col(name_parts, default=None):
            idx = _col_for_diag(tiny_names, "tiny", name_parts)
            return (tiny_mat[:, idx] if idx is not None else default)

        tiny_unc = _maybe_col(("uncertainty",), default=np.zeros(tiny_mat.shape[0]))
        tiny_ood = _maybe_col(("ood_hat",), default=np.zeros(tiny_mat.shape[0]))
        # Optional extras:
        tiny_cons = _maybe_col(("consistency", "hat"), default=np.zeros(tiny_mat.shape[0]))

        tiny_unc = _safe01(tiny_unc)
        tiny_ood = _safe01(tiny_ood)
        tiny_cons = _safe01(tiny_cons)

        # Default thresholds; make them configurable via cfg if you want
        thr_unc = float(self.cfg.get("route_threshold_uncertainty", 0.6))
        thr_ood = float(self.cfg.get("route_threshold_ood", 0.7))

        use_hrm_mask = (tiny_unc > thr_unc) | (tiny_ood > thr_ood)

        # 6.3 Simulate routed final scores and compute summary
        usage_rate = float(np.mean(use_hrm_mask))
        routed_stats = {"usage_rate": round(usage_rate, 6), "thresholds": {"uncertainty": thr_unc, "ood": thr_ood}}

        per_dim_results = []
        for dim in dims:
            # Only compute if calibration done
            if dim not in calib_params:
                per_dim_results.append({"dimension": dim, "status": "skipped"})
                continue

            i_hrm = _col_for_dim(hrm_names, "hrm", dim)
            i_tny = _col_for_dim(tiny_names, "tiny", dim)
            hrm_scores = _safe01(hrm_mat[:, i_hrm])
            tiny_scores = _safe01(tiny_mat[:, i_tny])
            tiny_cal   = _apply_monotone_pl(tiny_scores, calib_params[dim])

            final = np.where(use_hrm_mask, hrm_scores, tiny_cal)

            mae_vs_hrm = _mae(final, hrm_scores)
            mae_tiny   = _mae(tiny_scores, hrm_scores)
            mae_cal    = _mae(tiny_cal, hrm_scores)

            per_dim_results.append({
                "dimension": dim,
                "status": "ok",
                "mae_vs_hrm_routed": round(mae_vs_hrm, 6),
                "mae_vs_hrm_tiny": round(mae_tiny, 6),
                "mae_vs_hrm_calibrated_tiny": round(mae_cal, 6),
            })

        # 6.4 Persist routing results
        with open(os.path.join(metrics_dir, "routing_detail.json"), "w", encoding="utf-8") as f:
            json.dump({"per_dimension": per_dim_results}, f, indent=2)

        with open(os.path.join(metrics_dir, "routing_summary.json"), "w", encoding="utf-8") as f:
            # Global headline: mean across dimensions of routed MAE vs HRM
            maes = [r.get("mae_vs_hrm_routed") for r in per_dim_results if r.get("status") == "ok"]
            avg_mae = float(np.mean(maes)) if maes else 0.0
            json.dump({
                "usage_rate": routed_stats["usage_rate"],
                "avg_mae_vs_hrm": round(avg_mae, 6),
                "thresholds": routed_stats["thresholds"],
            }, f, indent=2)

        # 6.5 Surface in eval_stats (for the manifest + upstream)
        eval_stats["calibration"] = {
            "params_path": os.path.join(metrics_dir, "calibration_params.json"),
            "routing_summary_path": os.path.join(metrics_dir, "routing_summary.json"),
            "routing_detail_path": os.path.join(metrics_dir, "routing_detail.json"),
        }

        context[self.output_key] = eval_stats
        return context


# ---- Free functions (shared helpers) ----------------------------------------
def _flatten_samples_for_eval(samples: List[dict]) -> List[Tuple[str, str, float]]:
    """
    Normalize various sample schemas into (goal_text, output_text, target_value).
    Supports:
      - {"title"/"goal_text", "output", "score"}
      - {"title"/"goal_text", "output_a"/"output_b", "value_a"/"value_b"}
      - {"goal_text", "scorable_text", "target_score" or "score"}
    """
    triples = []
    for s in samples:
        title = (s.get("goal_text") or s.get("title") or "").strip()

        # singleton
        if "output" in s and ("score" in s or "target_score" in s):
            out = (s.get("output") or "").strip()
            val = s.get("target_score", s.get("score"))
            if title and out and val is not None:
                triples.append((title, out, float(val)))
            continue

        # pairwise
        if all(k in s for k in ("output_a", "output_b", "value_a", "value_b")):
            a_out, b_out = (s.get("output_a") or "").strip(), (s.get("output_b") or "").strip()
            a_val, b_val = s.get("value_a", None), s.get("value_b", None)
            if title and a_out and a_val is not None:
                triples.append((title, a_out, float(a_val)))
            if title and b_out and b_val is not None:
                triples.append((title, b_out, float(b_val)))
            continue

        # explicit HRM/MRQ form
        if ("goal_text" in s and "scorable_text" in s and ("target_score" in s or "score" in s)):
            out = (s.get("scorable_text") or "").strip()
            val = s.get("target_score", s.get("score"))
            if title and out and val is not None:
                triples.append((title, out, float(val)))
    return triples

# ---- add near other helpers (top of file is fine) ---------------------------
def _align_mats_by_names(
    A: np.ndarray, names_A: list[str],
    B: np.ndarray, names_B: list[str],
    *, mode: str = "intersection"  # or "union"
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """
    Align two (T x K) matrices by column names.

    Returns:
      A2, B2: aligned matrices with same (T x K_shared) shape
      shared: ordered list of shared (or union) names used
      info:   diagnostics (missing_A, missing_B)
    """
    idxA = {n: i for i, n in enumerate(names_A)}
    idxB = {n: i for i, n in enumerate(names_B)}

    if mode == "intersection":
        shared = [n for n in names_A if n in idxB]
        A2 = A[:, [idxA[n] for n in shared]] if shared else np.zeros((A.shape[0], 0), A.dtype)
        B2 = B[:, [idxB[n] for n in shared]] if shared else np.zeros((B.shape[0], 0), B.dtype)
        info = {
            "missing_in_B": [n for n in names_A if n not in idxB],
            "missing_in_A": [n for n in names_B if n not in idxA],
            "mode": mode,
        }
        return A2, B2, shared, info

    # union with zero-fill
    union = []
    seen = set()
    for n in names_A + names_B:
        if n not in seen:
            seen.add(n); union.append(n)

    KA = A.shape[1]; KB = B.shape[1]
    A2 = np.zeros((A.shape[0], len(union)), dtype=A.dtype)
    B2 = np.zeros((B.shape[0], len(union)), dtype=B.dtype)
    for j, n in enumerate(union):
        if n in idxA: A2[:, j] = A[:, idxA[n]]
        if n in idxB: B2[:, j] = B[:, idxB[n]]
    info = {"mode": mode, "union_size": len(union)}
    return A2, B2, union, info


def _fill_from_columns(row: Dict[str, float], cols: List[str], vals: List[float], model_name: str, dims: List[str]) -> None:
    """
    Extract per-dimension score for keys like:
      - '{model}.{dim}'
      - '{model}.{dim}.score'
      - '{model}.{dim}.aggregate}'
    Prefers a strict match in that order; falls back gracefully.
    """
    name2val = {str(c): float(v) for c, v in zip(cols, vals)}
    for dim in dims:
        preferred = [
            f"{model_name}.{dim}",
            f"{model_name}.{dim}.score",
            f"{model_name}.{dim}.aggregate",
        ]
        val = None
        for key in preferred:
            if key in name2val:
                val = name2val[key]
                break
        # loose fallback
        if val is None:
            for k in name2val.keys():
                if k.endswith(f".{dim}") or f".{dim}." in k:
                    val = name2val[k]
                    break
        if val is not None:
            row[f"{model_name}.{dim}"] = float(val)


def _harvest_model_dim_scores(row: Dict[str, float], metrics: Dict[str, Any], *, model_name: str, dims: List[str]) -> None:
    """
    Populate `row` with best-effort per-dimension scores under keys like
    '{model}.{dimension}' from a metrics payload that can be either:
      - ScoreBundle-like: {'bundle': <ScoreBundle>}  (uses flatten())
      - Flat: {'columns': [...], 'values': [...]}    (aligns by name)
    """
    bundle = metrics.get("bundle")
    if bundle is not None and hasattr(bundle, "flatten"):
        cols, vals = bundle.flatten()
        _fill_from_columns(row, cols, vals, model_name, dims)
        return

    cols = metrics.get("columns")
    vals = metrics.get("values")
    if isinstance(cols, list) and isinstance(vals, (list, tuple)) and len(cols) == len(vals):
        _fill_from_columns(row, cols, vals, model_name, dims)
        return
    return


def _fingerprint(goal_text: str, output_text: str) -> str:
    h = hashlib.sha1()
    h.update((goal_text.strip() + "\n␟\n" + output_text.strip()).encode("utf-8"))
    return h.hexdigest()


def _dedupe_triples_by_dimension(
    triples_by_dim: Dict[str, List[Tuple[str, str, float]]],
    policy: str = "first_wins",            # or "round_robin"
    per_dim_cap: int | None = None
) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Ensure no (goal, output) pair appears in more than one dimension.

    policy:
      - "first_wins": keep the first dimension (by iteration order) that sees a sample
      - "round_robin": build a global pool and assign unique items evenly across dims
    """
    dims = list(triples_by_dim.keys())

    if policy == "first_wins":
        seen: set[str] = set()
        deduped: Dict[str, List[Tuple[str, str, float]]] = {d: [] for d in dims}
        for d in dims:
            for (g, o, v) in triples_by_dim[d]:
                key = _fingerprint(g, o)
                if key in seen:
                    continue
                deduped[d].append((g, o, v))
                seen.add(key)
            if per_dim_cap is not None and len(deduped[d]) > per_dim_cap:
                deduped[d] = deduped[d][:per_dim_cap]
        return deduped

    elif policy == "round_robin":
        pool: Dict[str, Tuple[str, str, float]] = {}
        for d in dims:
            for (g, o, v) in triples_by_dim[d]:
                key = _fingerprint(g, o)
                if key not in pool:
                    pool[key] = (g, o, v)

        keys = list(pool.keys())
        deduped = {d: [] for d in dims}
        i = 0
        for k in keys:
            d = dims[i % len(dims)]
            if per_dim_cap is None or len(deduped[d]) < per_dim_cap:
                deduped[d].append(pool[k])
                i += 1
        return deduped

    else:
        raise ValueError(f"Unknown policy: {policy}")

# [S2.a] Utilities for manifest updates and IO -------------------------------
def _update_manifest(base_root: str, run_id: str, patch: dict) -> None:
    """Read/patch/write manifest.json safely."""
    mpath = os.path.join(base_root, run_id, "manifest.json")
    try:
        with open(mpath, "r", encoding="utf-8") as f:
            m = json.load(f)
    except Exception:
        m = {}
    # deep-merge (shallow is fine for our keys)
    def _merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _merge(dst[k], v)
            else:
                dst[k] = v
    _merge(m, patch)
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(dumps_safe(m, indent=2))


def _write_triples_index(
    base_root: str,
    run_id: str,
    deduped: Dict[str, List[Tuple[str, str, float]]],
    *,
    policy: str,
    write_parquet: bool = True,
    head_n: int = 200,
) -> Dict[str, Any]:
    """
    Materialize the deduped triples to raw/triples.jsonl (and optional parquet).
    Returns paths + basic counts for manifest updates.
    """
    base = os.path.join(base_root, run_id)
    raw_dir = os.path.join(base, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    jsonl_path = os.path.join(raw_dir, "triples.jsonl")
    head_path  = os.path.join(raw_dir, "triples.head.json")
    parquet_path = os.path.join(raw_dir, "triples.parquet")

    rows = []
    total = 0
    per_dim_counts = {}

    for dim, triples in deduped.items():
        per_dim_counts[dim] = len(triples)
        for idx, (goal_text, output_text, target_val) in enumerate(triples):
            node_id = f"{run_id}|{dim}|{idx:06d}"
            rows.append({
                "node_id": node_id,
                "dimension": dim,
                "goal_text": goal_text,
                "output_text": output_text,
                "target": float(target_val) if target_val is not None else None,
                "turn_idx": idx,
                "fingerprint": _fingerprint(goal_text, output_text),
                "policy": policy,
            })
        total += len(triples)

    # Write JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(dumps_safe(r))
            f.write("\n")

    # Quick head sample for inspection
    with open(head_path, "w", encoding="utf-8") as f:
        f.write(dumps_safe(rows[:head_n], indent=2))

    # Optional Parquet for fast downstream loading
    if write_parquet:
        try:
            pd.DataFrame(rows).to_parquet(parquet_path, index=False)
        except Exception:
            parquet_path = None

    return {
        "jsonl_path": jsonl_path,
        "parquet_path": parquet_path,
        "head_path": head_path,
        "rows": total,
        "per_dim_counts": per_dim_counts,
    }
