# stephanie/agents/phoshrm.py  (patched)
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import json
import hashlib
from typing import DefaultDict
from collections import defaultdict

import matplotlib

from stephanie.utils.json_sanitize import dumps_safe
if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.scoring.training.preference_pair_builder import PreferencePairBuilder
from stephanie.utils.id_utils import make_numeric_id
import asyncio
from stephanie.services.workers.vpm_worker import VPMWorkerInline
from stephanie.services.workers.metrics_worker import MetricsWorkerInline
from stephanie.zeromodel.vpm_phos import build_hrm_vs_tiny_guarded

import logging
_logger = logging.getLogger(__name__)

CAND_SUFFIXES = ["", ".score", ".aggregate", ".raw", ".value"]

class PhoshrmAgent(BaseAgent):
    """
    Compare HRM vs Tiny via PHOS VPMs over ~N chat responses.
    Produces:
      - hrm_vpm_raw.png / tiny_vpm_raw.png
      - hrm_vpm_phos.png / tiny_vpm_phos.png
      - vpm_phos_diff.png
      - manifest.json (metrics + paths)
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.dimensions = list(
            cfg.get(
                "dimensions",
                ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"],
            )
        )
        self.hrm_scorers = list(cfg.get("hrm_scorers", ["hrm"]))
        self.tiny_scorers = list(cfg.get("tiny_scorers", ["tiny"]))
        self.out_dir = Path(cfg.get("out_dir", "data/vpm"))
        self.interleave = bool(cfg.get("interleave", False))
        self.progress_log_every = int(cfg.get("progress_log_every", 25))  # NEW

    async def run(self, context: dict) -> dict:
        eval_stats: Dict[str, Any] = {}

        # 1) Prepare tools
        scoring_service = self.container.get("scoring")
        zm = self.container.get("zeromodel")
        hrm_worker = MetricsWorkerInline(scoring_service, self.hrm_scorers, self.dimensions)
        tiny_worker = MetricsWorkerInline(scoring_service, self.tiny_scorers, self.dimensions)
        vpm_worker = VPMWorkerInline(zm, self.logger)

        pipeline_run_id = context.get("pipeline_run_id") or "phoshrm"
        hrm_run_id = f"{pipeline_run_id}_hrm"
        tiny_run_id = f"{pipeline_run_id}_tiny"
        zm.timeline_open(run_id=hrm_run_id)
        zm.timeline_open(run_id=tiny_run_id)

        # For PHOS guarded compare: accumulate rows per sample
        rows_for_df: List[Dict[str, float]] = []  # <-- NEW

        # 2) Gather all samples per dimension, then dedupe globally
        pair_builder = PreferencePairBuilder(self.memory, self.logger)
        triples_by_dim: Dict[str, List[Tuple[str, str, float]]] = {}
        total_raw = 0

        for dimension in self.dimensions:
            pairs_by_dim = pair_builder.get_training_pairs_by_dimension(dimension=dimension)
            samples_full = pairs_by_dim.get(dimension, [])
            if not samples_full:
                self.logger.log("NoSamplesFound", {"dimension": dimension})
                triples_by_dim[dimension] = []
                continue

            triples = _flatten_samples_for_eval(samples_full)
            triples_by_dim[dimension] = triples
            total_raw += len(triples)

        # 🔒 dedupe across dimensions (choose policy and optional caps)
        deduped = _dedupe_triples_by_dimension(
            triples_by_dim,
            policy=self.cfg.get("dedupe_policy", "first_wins"),
            per_dim_cap=self.cfg.get("per_dim_cap")  # e.g., 400
        )
        # Update totals post-dedupe and proceed
        total_triples = sum(len(v) for v in deduped.values())
        if total_triples == 0:
            self.logger.log("PhosHRMNoSamples", {"dimensions": self.dimensions})
            context[self.output_key] = {"msg": "no samples"}
            return context

        self.logger.log("PhosHRMDedupeSummary", {
            "total_raw": total_raw,
            "total_unique": total_triples,
            "per_dim_counts": {d: len(deduped[d]) for d in self.dimensions}
        })

        # 3) Main scoring loop now uses deduped per-dimension lists
        processed = 0
        rows_for_df = []
        with tqdm(total=total_triples, desc="[PhosHRM] overall", unit="sample") as pbar_all:
            for d in self.dimensions:
                triples = deduped.get(d, [])
                if not triples:
                    tqdm.write(f"[PhosHRM] ⚠️ No (unique) samples for dimension: {d}")
                    continue

                with tqdm(total=len(triples), desc=f"[PhosHRM] {d}", unit="node", leave=False) as pbar_dim:
                    for idx, (goal_text, output_text, target_val) in enumerate(triples):
                        node_id = make_numeric_id(pipeline_run_id, d, idx)
                        scorable = Scorable(output_text, ScorableType.CONVERSATION_TURN)
                        hrm_metrics = await hrm_worker.score(scorable, goal_text, hrm_run_id)

                        tiny_metrics = await tiny_worker.score(scorable, goal_text, tiny_run_id)

                        row = {"node_id": node_id}
                        row.update({k: float(v) for k, v in hrm_metrics.get("vector", {}).items()})
                        row.update({k: float(v) for k, v in tiny_metrics.get("vector", {}).items()})
                        rows_for_df.append(row)

                        await vpm_worker.append(hrm_run_id, node_id, hrm_metrics)
                        await vpm_worker.append(tiny_run_id, node_id, tiny_metrics)

                        processed += 1
                        pbar_dim.update(1)
                        pbar_all.update(1)
                        if (processed % self.progress_log_every) == 0 or processed == total_triples:
                            self.logger.log("PhosHRMProgress", {
                                "run_id": pipeline_run_id,
                                "dimension": d,
                                "processed": processed,
                                "total": total_triples,
                                "percent": round(100.0 * processed / total_triples, 2),
                            })
                        await asyncio.sleep(0)
        # 4) Finalize timelines (unchanged)
        hrm_gif = f"vpm_phos_run_{hrm_run_id}.gif"
        tiny_gif = f"vpm_phos_run_{tiny_run_id}.gif"
        hrm_final = await vpm_worker.finalize(hrm_run_id, hrm_gif)
        tiny_final = await vpm_worker.finalize(tiny_run_id, tiny_gif)

        # hrm_final and tiny_final came from vpm_worker.finalize(...)
        # in PhoshrmAgent.run after hrm_final / tiny_final are ready
        hrm_mat = np.asarray(hrm_final["matrix"])
        tiny_mat = np.asarray(tiny_final["matrix"])
        hrm_names = hrm_final.get("metric_names", [])
        tiny_names = tiny_final.get("metric_names", [])

        frontier_dir = Path(self.out_dir) / "frontier" / f"run_{pipeline_run_id}"

        frontier_meta = zm.render_frontier_map(
            hrm_mat, tiny_mat,
            out_dir=str(frontier_dir),
            pos_label="HRM",
            neg_label="Tiny",
            k_latent=20
        )

        eval_stats["frontier"] = frontier_meta


        delta_meta = zm.render_intermodel_delta(
            hrm_mat, tiny_mat,
            names_A=hrm_names,
            names_B=tiny_names,
            output_dir=str(Path(self.out_dir, "intermodel_delta", f"run_{pipeline_run_id}")),
            pos_label="HRM",
            neg_label="Tiny",
        )

        # Include in manifest
        eval_stats["intermodel_delta"] = delta_meta

        # Build per-model intensity JSON + cross-model comparison
        intensity = zm.build_intensity_report(
            hrm_matrix=hrm_final["matrix"],
            tiny_matrix=tiny_final["matrix"],
            hrm_metric_names=hrm_final.get("metric_names", []),
            tiny_metric_names=tiny_final.get("metric_names", []),
            out_dir=str(Path(self.out_dir) / f"phos_reports/run_{pipeline_run_id}"),
            top_k=20,
        )
        eval_stats["intensity_report"] = {"path": intensity["path"]}

        tqdm.write(f"[PhosHRM] ✅ Timelines closed → {hrm_gif} / {tiny_gif}")


        def _pick_metric_column(df: pd.DataFrame, base: str) -> str | None:
            # Try exact base first, then candidate suffixes, then a loose fallback
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

        # 5) Build DataFrame and run PHOS guarded comparison
        if rows_for_df:
            df_raw = pd.DataFrame(rows_for_df)
            # Keep only model columns + node_id (defensive)
            keep = ["node_id"] + [c for c in df_raw.columns
                                if isinstance(c, str) and (c.startswith("hrm.") or c.startswith("tiny."))]
            df_raw = df_raw[keep]

            # 🔧 Project to canonical columns hrm.{dim} / tiny.{dim}
            df_proj = _project_dimensions(df_raw, self.dimensions, self.logger)

            out_prefix = str(Path(self.out_dir) / f"phos_guard/run_{pipeline_run_id}")
            Path(self.out_dir, "phos_guard").mkdir(parents=True, exist_ok=True)

            phos_res = build_hrm_vs_tiny_guarded(
                df_proj,                                # ← use projected DF
                dimensions=self.dimensions,
                out_prefix=out_prefix,
                tl_fracs=(0.25, 0.16, 0.36, 0.09),
                delta=0.02,
                interleave=bool(self.interleave),
                weights=None,
            )
        else:
            phos_res = {"status": "no_rows_for_df"}

        # 6) Return enriched context
        eval_stats.update({
            "hrm": {"gif": hrm_final.get("output_path"), "shape": hrm_final.get("shape")},
            "tiny": {"gif": tiny_final.get("output_path"), "shape": tiny_final.get("shape")},
            "phos_guarded": phos_res,
        })

        # Persist manifest
        manifest_dir = Path(self.out_dir) / "phoshrm_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"phoshrm_manifest_{pipeline_run_id}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(dumps_safe(eval_stats, indent=2))

        _logger.info(f"[PhosHRM] Manifest saved → {manifest_path}")

        context[self.output_key] = eval_stats
        return context


    def _analyze_vpms(self, vpms: Dict[str, Any]):
        """Run contrastive analysis on actual matrices returned by timeline_finalize."""
        good = vpms.get("good", {})
        mixed = vpms.get("medium", {})
        bad = vpms.get("opposite", {})

        # Safely extract matrices only if the entry is a dict
        good_mat = good.get("matrix") if isinstance(good, dict) else None
        mixed_mat = mixed.get("matrix") if isinstance(mixed, dict) else None
        bad_mat = bad.get("matrix") if isinstance(bad, dict) else None

        results = {}
        output_dir = Path(f"data/vpms/phos_diffs/run_{self.run_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        if good_mat is not None and bad_mat is not None:
            meta = self.zm.generate_epistemic_field_phos_ordered(
                pos_matrices=[good_mat],
                neg_matrices=[bad_mat],
                output_dir=str(output_dir / "good_vs_bad"),
                aggregate=True,
                metric_names=self.metric_names,
            )
            results["good_vs_bad"] = meta
            _logger.debug(
                "ΔMass=%s, Overlap=%s",
                meta["delta_mass"],
                meta["overlap_score"],
            )
            # 🧠 Analyze the differential field
            diff_matrix = meta.get("diff_matrix")
            metric_names = meta.get("metric_names_reordered", [])
            ranked_metrics = self.zm.analyze_differential_field(
                diff_matrix, metric_names, output_dir / "good_vs_bad"
            )
            self.logger.log("metrics", ranked_metrics)

        if good_mat is not None and mixed_mat is not None:
            meta = self.generate_epistemic_field_phos_ordered(
                pos_matrices=[good_mat], 
                neg_matrices=[mixed_mat],
                output_dir=str(output_dir / "good_vs_mixed"),
            )
            results["good_vs_mixed"] = meta
            _logger.info(
                f"ΔMass={meta['delta_mass']:.4f}, Overlap={meta['overlap_score']:.4f}"
            )

        return results


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

def _fill_from_columns(row: Dict[str, float], cols: List[str], vals: List[float], model_name: str, dims: List[str]) -> None:
    """
    Extract per-dimension score for keys like:
      - '{model}.{dim}'
      - '{model}.{dim}.score'
      - '{model}.{dim}.aggregate'
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
        # If still missing, try a loose search (any column that endswith dim or contains f".{dim}.")
        if val is None:
            for k in name2val.keys():
                if k.endswith(f".{dim}") or f".{dim}." in k:
                    val = name2val[k]
                    break
        if val is not None:
            row[f"{model_name}.{dim}"] = float(val)

# -------------------------
# Helpers for PHOS compare
# -------------------------
def _harvest_model_dim_scores(row: Dict[str, float], metrics: Dict[str, Any], *, model_name: str, dims: List[str]) -> None:
    """
    Populate `row` with best-effort per-dimension scores under keys like
    '{model}.{dimension}' from a metrics payload that can be either:
      - ScoreBundle-like: {'bundle': <ScoreBundle>}  (uses flatten())
      - Flat: {'columns': [...], 'values': [...]}    (aligns by name)
    It tolerates variants like '{model}.{dim}', '{model}.{dim}.score'.
    """
    # Case A: ScoreBundle present and has flatten()
    bundle = metrics.get("bundle")
    if bundle is not None and hasattr(bundle, "flatten"):
        cols, vals = bundle.flatten()  # expected to return (List[str], List[float])
        _fill_from_columns(row, cols, vals, model_name, dims)
        return

    # Case B: columns/values dict (as in your example)
    cols = metrics.get("columns")
    vals = metrics.get("values")
    if isinstance(cols, list) and isinstance(vals, (list, tuple)) and len(cols) == len(vals):
        _fill_from_columns(row, cols, vals, model_name, dims)
        return

    # Fallback: nothing we can do
    return


def _fingerprint(goal_text: str, output_text: str) -> str:
    h = hashlib.sha1()
    h.update((goal_text.strip() + "\n␟\n" + output_text.strip()).encode("utf-8"))
    return h.hexdigest()

def _dedupe_triples_by_dimension(
    triples_by_dim: Dict[str, List[Tuple[str, str, float]]],
    policy: str = "first_wins",            # or "round_robin"
    per_dim_cap: int | None = None         # optional cap per dimension
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
        # 1) global unique pool preserving first occurrence value
        pool: Dict[str, Tuple[str, str, float]] = {}
        for d in dims:
            for (g, o, v) in triples_by_dim[d]:
                key = _fingerprint(g, o)
                if key not in pool:
                    pool[key] = (g, o, v)

        # 2) assign evenly across dimensions
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
