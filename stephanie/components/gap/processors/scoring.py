from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from stephanie.components.gap.services.scm_service import SCM_FEATURE_KEYS

import numpy as np
import pandas as pd
from tqdm import tqdm

from stephanie.components.gap.io.manifest import GapRunManifest
from stephanie.components.gap.models import GapConfig, TripleSample
from stephanie.components.gap.services.scm_service import SCMService
from stephanie.components.gap.shared_scm import (
    SCM_COLUMNS,
    scm_from_vector,
    scm_row,
)
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.services.workers.metrics_worker import MetricsWorkerInline
from stephanie.services.workers.vpm_worker import VPMWorkerInline
from stephanie.utils.progress_mixin import ProgressMixin

logger = logging.getLogger(__name__)


class ScoringProcessor(ProgressMixin):
    """Minimal, readable scoring + alignment pipeline with SCM injection."""

    def __init__(self, config: GapConfig, container, logger):
        self.config = config
        self.container = container
        self.logger = logger
        self._init_progress(container, logger)  # <-- ProgressService hookup

    # ---------- public API --------------------------------------------------
    async def prepare_samples(
        self, dimensions: List[str], memory
    ) -> Dict[str, List[TripleSample]]:
        from stephanie.scoring.training.preference_pair_builder import (
            PreferencePairBuilder,
        )

        pb = PreferencePairBuilder(memory, self.logger)
        by_dim: Dict[str, List[TripleSample]] = {}
        for d in dimensions:
            pairs = pb.get_training_pairs_by_dimension(
                dimension=d, text_max=2000
            )
            samples = pairs.get(d, [])
            by_dim[d] = self._flatten_samples(samples, d)
        return self._dedupe(by_dim)

    async def execute_scoring(
        self,
        triples_data: Dict[str, List[TripleSample]],
        run_id: str,
        manifest: GapRunManifest,
    ) -> Dict[str, Any]:
        scoring_service = self.container.get("scoring")
        zm = self.container.get("zeromodel")

        # start stage on the manifest
        total_rows = sum(len(v) for v in triples_data.values())

        # 1) Start stage (let stage_start set started_at itself)
        manifest.stage_start(
            "scoring",
            total=total_rows,
            run_id=run_id,  # fine as metadata
            status="running",  # optional, stage_start already sets "running"
        )

        # helper to tick progress into the manifest every so often
        storage = self.container.get("gap_storage")

        hrm_worker = MetricsWorkerInline(
            scoring_service, self.config.hrm_scorers, self.config.dimensions
        )
        tiny_worker = MetricsWorkerInline(
            scoring_service, self.config.tiny_scorers, self.config.dimensions
        )
        result = await self._score_all_triples(
            triples_data, hrm_worker, tiny_worker, run_id
        )

        # 3) End stage (let stage_end set finished_at itself)
        manifest.stage_end(
            "scoring",
            status="ok",
            triples_count=result.get("triples_count"),
            columns=result.get("hrm_names"),
            hrm_label=result.get("hrm_label"),
            tiny_label=result.get("tiny_label"),
            artifacts={
                "hrm_gif": result.get("hrm_gif"),
                "tiny_gif": result.get("tiny_gif"),
                "rows_for_df": result.get("rows_for_df_path"),
                "hrm_scorers": self.config.hrm_scorers,
                "tiny_scorers": self.config.tiny_scorers,
            },
        )

        # Persist the final scoring stage into the canonical manifest file
        try:
            storage.patch_manifest(
                run_id, {"stages": {"scoring": manifest.get_stage("scoring")}}
            )
        except Exception:
            pass

        # Optional: keep your quick snapshot (debug aid)
        storage.save_json(
            run_id, "metrics", "scoring_stage.json", manifest.to_dict()
        )

        return result


    async def _score_model_pass(
        self,
        *,
        model_label: str,                 # "hrm" or "tiny"
        worker: MetricsWorkerInline,
        triples: List[TripleSample],
        timeline_id: str,
        task_name: str,
    ) -> Dict[str, Any]:
        """Score ALL samples with ONE model to avoid VRAM thrash."""
        zm = self.container.get("zeromodel")
        vpm_worker = VPMWorkerInline(zm, self.logger)
        scm_service: SCMService = self.container.get("scm_service")

        zm.timeline_open(run_id=timeline_id)

        names: List[str] = []
        rows: List[List[float]] = []
        scm_rows: List[List[float]] = []
        rows_for_df: List[Dict[str, float]] = []
        keep_mask: List[bool] = []
        kept_indices: List[int] = []

        log_every = max(1, self.config.progress_log_every)

        with tqdm(total=len(triples), desc=task_name, unit="turn") as pbar:
            for i, triple in enumerate(triples):
                scorable = Scorable(triple.output_text, ScorableType.CONVERSATION_TURN)

                metrics = await worker.score(scorable, triple.goal_text, timeline_id)

                # 1) Pull raw vector from the scorer output and normalize key space
                vec_raw = self._to_vector(metrics)
                vec_canon = self._promote_scm_keys(vec_raw)

                # 2) Resolve the alias used by the scorer (e.g., "hf_TinyLama")
                model_alias = _first_or_str(metrics.get("model_alias", model_label))

                # 3) Ask SCM service to build the enriched vector bundle
                scm_bundle = scm_service.build_vector(
                    model_alias=model_alias,
                    attrs=vec_canon,
                    dimensions=self.config.dimensions,  
                )
                # scm_bundle: {"vector": {...}, "columns": [...], "values": [...]}
                scm_vec = scm_bundle["vector"]

                # 4) Merge original metrics with SCM vector bundle for the timeline payload
                merged = self._merge_for_timeline(metrics, scm_bundle)

                # 5) Append to the timeline video
                vpm_worker.append(timeline_id, triple.node_id, merged)

                # 6) Build the aligned row across all keys observed so far
                vec = merged["vector"]
                if not names:
                    names = list(vec.keys())
                rows.append(self._align_row(vec, names))

                # 7) Build SCM row in fixed order (SCM_FEATURE_KEYS)
                #    (Previously you passed a bundle to scm_row, which is why you got zeros.)
                scm_rows.append([float(scm_vec.get(k, 0.0)) for k in SCM_FEATURE_KEYS])

                # 8) PHOS row (model side): copy the per-dimension scalar scores from scm_vec
                r = {"node_id": triple.node_id}
                for d in self.config.dimensions:
                    r[f"{model_alias}.{d}"] = float(scm_vec.get(f"scm.{d}.score01", 0.0))
                rows_for_df.append(r)

                keep_mask.append(True)
                kept_indices.append(i)

                if ((i + 1) % log_every) == 0 or (i + 1) == len(triples):
                    self.logger.log(
                        "ScoringProgress",
                        {"model": model_label, "processed": i + 1, "total": len(triples)},
                    )
                pbar.update(1)
                await asyncio.sleep(0)

        # finalize GIF under standard visuals dir
        out_dir = str(self.config.base_dir / timeline_id.split("_")[0] / "visuals")
        hrm_or_tiny_gif = await vpm_worker.finalize(
            timeline_id, f"{out_dir}/vpm_phos_run_{timeline_id}.gif"
        )

        return {
            "names": names,
            "rows": np.asarray(rows, np.float32),
            "scm_rows": np.asarray(scm_rows, np.float32),
            "rows_for_df": rows_for_df,
            "keep_mask": np.array(keep_mask, dtype=bool),
            "kept_indices": kept_indices,
            "gif": hrm_or_tiny_gif,
        }

    # ---------- core pipeline ----------------------------------------------
    async def _score_all_triples(
        self,
        triples_data: Dict[str, List[TripleSample]],
        hrm_worker,
        tiny_worker,
        run_id: str,
    ) -> Dict[str, Any]:
        # 0) flatten in a stable order used by both passes
        all_triples: List[TripleSample] = [
            t for ts in triples_data.values() for t in ts
        ]
        T = len(all_triples)

        overall = f"scoring:{run_id}"
        self.pstart(
            overall,
            total=T * 2,
            meta={
                "dims": len(triples_data),
                "passes": 2,
                "console_echo": False,
            },
        )

        # --- PASS 1: HRM only
        hrm_tl = f"{run_id}_hrm"
        hrm_res = await self._score_model_pass(
            model_label="hrm",
            worker=hrm_worker,
            triples=all_triples,
            timeline_id=hrm_tl,
            task_name=f"{overall}:hrm",
        )

        # Unload HuggingFace models (free VRAM) before loading Tiny
        # This is the critical change that prevents repeated OOM.
        self._unload_hf_scorers()

        # --- PASS 2: Tiny only
        tiny_tl = f"{run_id}_tiny"
        tiny_res = await self._score_model_pass(
            model_label="tiny",
            worker=tiny_worker,
            triples=all_triples,
            timeline_id=tiny_tl,
            task_name=f"{overall}:tiny",
        )

        # Update structured progress
        self.ptick(overall, done=T * 2, total=T * 2)

        # --- Align common subset of rows (both passes must have kept them)
        hrm_keep = hrm_res["keep_mask"]
        tny_keep = tiny_res["keep_mask"]
        if hrm_keep.shape != tny_keep.shape:
            raise RuntimeError("Internal: mask shapes differ between models.")
        both_keep_mask = hrm_keep & tny_keep
        kept_idx = np.nonzero(both_keep_mask)[0].tolist()

        # Build index maps from original -> compacted row within each pass
        hrm_idx_map = {
            orig: j for j, orig in enumerate(hrm_res["kept_indices"])
        }
        tny_idx_map = {
            orig: j for j, orig in enumerate(tiny_res["kept_indices"])
        }

        def _select_rows(
            pass_rows: np.ndarray, idx_map: Dict[int, int], orig_idx: List[int]
        ) -> np.ndarray:
            if pass_rows.size == 0 or not orig_idx:
                return np.zeros((0, 0), dtype=np.float32)
            sel = [idx_map[i] for i in orig_idx if i in idx_map]
            return (
                pass_rows[sel, :]
                if sel
                else np.zeros((0, pass_rows.shape[1]), dtype=pass_rows.dtype)
            )

        # matrices
        hrm_names = hrm_res["names"]
        tiny_names = tiny_res["names"]
        hrm_matrix_raw = _select_rows(hrm_res["rows"], hrm_idx_map, kept_idx)
        tiny_matrix_raw = _select_rows(tiny_res["rows"], tny_idx_map, kept_idx)

        # preferred columns (aggregate + dims) → shared order
        pref_hrm = self._preferred_indices(
            hrm_names, "hrm", self.config.dimensions
        )
        pref_tny = self._preferred_indices(
            tiny_names, "tiny", self.config.dimensions
        )
        canonical_order = ["aggregate"] + list(self.config.dimensions)
        shared = [
            k for k in canonical_order if k in pref_hrm and k in pref_tny
        ]
        if not shared:
            storage = self.container.get("gap_storage")
            storage.save_json(
                run_id,
                "metrics",
                "name_alignment_debug.json",
                {
                    "hrm_names": hrm_names,
                    "tiny_names": tiny_names,
                    "preferred_hrm": pref_hrm,
                    "preferred_tiny": pref_tny,
                    "dims": self.config.dimensions,
                },
            )
            raise RuntimeError("No shared preferred metrics across models.")
        h_cols = [pref_hrm[k] for k in shared]
        t_cols = [pref_tny[k] for k in shared]
        hrm_matrix = hrm_matrix_raw[:, h_cols]
        tiny_matrix = tiny_matrix_raw[:, t_cols]

        # SCM matrices (canonical order)
        hrm_scm_sel = _select_rows(hrm_res["scm_rows"], hrm_idx_map, kept_idx)
        tiny_scm_sel = _select_rows(
            tiny_res["scm_rows"], tny_idx_map, kept_idx
        )

        # PHOS rows merged
        def _select_rows_for_df(
            rows_for_df: List[Dict[str, float]],
            idx_map: Dict[int, int],
            orig_idx: List[int],
        ) -> List[Dict[str, float]]:
            out = []
            for i in orig_idx:
                j = idx_map.get(i)
                if j is not None:
                    out.append(rows_for_df[j])
            return out

        hrm_df_rows = _select_rows_for_df(
            hrm_res["rows_for_df"], hrm_idx_map, kept_idx
        )
        tiny_df_rows = _select_rows_for_df(
            tiny_res["rows_for_df"], tny_idx_map, kept_idx
        )
        merged_rows: List[Dict[str, float]] = []
        for rh, rt in zip(hrm_df_rows, tiny_df_rows):
            row = {"node_id": rh.get("node_id") or rt.get("node_id")}
            row.update({k: v for k, v in rh.items() if k != "node_id"})
            row.update({k: v for k, v in rt.items() if k != "node_id"})
            merged_rows.append(row)
        df_rows = pd.DataFrame(merged_rows)

        # GIFs (already saved by each pass)
        hrm_gif = hrm_res["gif"]
        tiny_gif = tiny_res["gif"]

        # Persist artifacts (same locations & tags as before)
        storage = self.container.get("gap_storage")
        storage.save_matrix(hrm_matrix, shared, run_id, tag="hrm")
        storage.save_matrix(tiny_matrix, shared, run_id, tag="tiny")
        storage.save_matrix(
            hrm_scm_sel.astype(np.float32), SCM_COLUMNS, run_id, tag="hrm_scm"
        )
        storage.save_matrix(
            tiny_scm_sel.astype(np.float32),
            SCM_COLUMNS,
            run_id,
            tag="tiny_scm",
        )

        raw_paths = storage.save_rows_df(df_rows, run_id, name="rows_for_df")

        # row-level provenance for the *kept* rows
        provenance = []
        for row_idx, orig_i in enumerate(kept_idx):
            t = all_triples[orig_i]
            provenance.append(
                {
                    "row_index": row_idx,
                    "orig_index": orig_i,
                    "node_id": t.node_id,
                    "dimension": t.dimension,
                    "goal_text": t.goal_text,
                    "output_text": t.output_text,
                }
            )
        storage.save_json(run_id, "raw", "row_provenance.json", provenance)

        # Labels
        scoring_service = self.container.get("scoring")

        def _disp(n):
            s = scoring_service._scorers.get(n)
            return (
                s.get_display_name()
                if s and hasattr(s, "get_display_name")
                else n
            )

        hrm_label = ", ".join(_disp(n) for n in self.config.hrm_scorers)
        tiny_label = ", ".join(_disp(n) for n in self.config.tiny_scorers)

        self.pdone(overall, extra={"rows": len(kept_idx)})
        return {
            "hrm_vectors": hrm_matrix,
            "tiny_vectors": tiny_matrix,
            "hrm_names": shared,
            "hrm_label": hrm_label,
            "tiny_names": shared,
            "tiny_label": tiny_label,
            "hrm_scm_matrix": hrm_scm_sel.astype(np.float32),
            "tiny_scm_matrix": tiny_scm_sel.astype(np.float32),
            "scm_names": SCM_COLUMNS,
            "hrm_gif": hrm_gif,
            "tiny_gif": tiny_gif,
            "triples_count": len(kept_idx),
            "rows_for_df_path": str(
                storage.base_dir / run_id / "raw" / "rows_for_df.parquet"
            ),
            **raw_paths,
        }

    # ---------- helpers -----------------------------------------------------
    def _flatten_samples(
        self, samples: List[Dict[str, Any]], dim: str
    ) -> List[TripleSample]:
        triples: List[TripleSample] = []
        for i, s in enumerate(samples):
            title = (s.get("goal_text") or s.get("title") or "").strip()
            # singleton
            if "output" in s and ("score" in s or "target_score" in s):
                out = (s.get("output") or "").strip()
                val = s.get("target_score", s.get("score"))
                if title and out and val is not None:
                    triples.append(
                        TripleSample(
                            node_id=f"{dim}|{i:06d}",
                            dimension=dim,
                            goal_text=title,
                            output_text=out,
                            target_value=float(val),
                            fingerprint=self._fingerprint(title, out),
                        )
                    )
                continue
            # pairwise
            if all(
                k in s for k in ("output_a", "output_b", "value_a", "value_b")
            ):
                for suf in ("a", "b"):
                    out = (s.get(f"output_{suf}") or "").strip()
                    val = s.get(f"value_{suf}")
                    if title and out and val is not None:
                        triples.append(
                            TripleSample(
                                node_id=f"{dim}|{i:06d}_{suf}",
                                dimension=dim,
                                goal_text=title,
                                output_text=out,
                                target_value=float(val),
                                fingerprint=self._fingerprint(title, out),
                            )
                        )
        return triples

    def _promote_scm_keys(self, vec: Dict[str, float]) -> Dict[str, float]:
        """
        Return a copy of vec where any nested '... .scm.X' is also available as
        a canonical top-level 'scm.X'. Never overwrites existing canonical keys.
        """
        out = dict(vec)
        for k, v in vec.items():
            kl = k.lower()
            if ".scm." in kl:
                # take everything after the first ".scm."
                _, after = kl.split(".scm.", 1)
                canon = "scm." + after
                if canon not in out:
                    out[canon] = float(v)
        return out

    def _dedupe(
        self, by_dim: Dict[str, List[TripleSample]]
    ) -> Dict[str, List[TripleSample]]:
        policy = self.config.dedupe_policy
        cap = self.config.per_dim_cap
        if policy == "first_wins":
            seen = set()
            out = {d: [] for d in by_dim}
            for d, ts in by_dim.items():
                for t in ts:
                    if t.fingerprint in seen:
                        continue
                    out[d].append(t)
                    seen.add(t.fingerprint)
                if cap and len(out[d]) > cap:
                    out[d] = out[d][:cap]
            return out
        elif policy == "round_robin":
            pool: Dict[str, TripleSample] = {}
            for ts in by_dim.values():
                for t in ts:
                    pool.setdefault(t.fingerprint, t)
            dims = list(by_dim.keys())
            out = {d: [] for d in dims}
            i = 0
            for t in pool.values():
                d = dims[i % len(dims)]
                if cap is None or len(out[d]) < cap:
                    out[d].append(t)
                    i += 1
            return out
        else:
            raise ValueError(f"Unknown dedupe policy: {policy}")

    def _fingerprint(self, g: str, o: str) -> str:
        return hashlib.sha1(
            (g.strip() + "\n␟\n" + o.strip()).encode("utf-8")
        ).hexdigest()

    def _to_vector(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        vec = metrics.get("vector")
        if isinstance(vec, dict) and vec:
            return {str(k): float(v) for k, v in vec.items()}
        cols = metrics.get("columns")
        vals = metrics.get("values")
        if (
            isinstance(cols, list)
            and isinstance(vals, list)
            and len(cols) == len(vals)
        ):
            return {str(c): float(v) for c, v in zip(cols, vals)}
        return {}

    def _align_row(
        self, vec: Dict[str, float], names: List[str]
    ) -> List[float]:
        return [float(vec.get(n, 0.0)) for n in names]

    def _merge_for_timeline(self, metrics: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge 'metrics' (may contain vector and/or columns+values) with 'extra', which can be
        a flat dict or an SCM-style {vector, columns, values}. Extras win on collisions.
        Drops non-finite values (NaN/±inf) and keeps a stable column order.
        """
        def _num(x):
            try:
                fx = float(x)
                if fx == fx and fx not in (float("inf"), float("-inf")):
                    return fx
            except Exception:
                pass
            return None  # drop NaN/inf/non-numeric

        # 1) Seed from metrics.vector
        base: Dict[str, float] = {}
        if isinstance(metrics.get("vector"), dict):
            for k, v in metrics["vector"].items():
                nv = _num(v)
                if nv is not None:
                    base[str(k)] = nv

        # 2) Merge metrics.columns/values (remember original order)
        metric_cols = metrics.get("columns") if isinstance(metrics.get("columns"), list) else []
        metric_vals = metrics.get("values") if isinstance(metrics.get("values"), list) else []
        if metric_cols and metric_vals and len(metric_cols) == len(metric_vals):
            for c, v in zip(metric_cols, metric_vals):
                nv = _num(v)
                if nv is not None:
                    base[str(c)] = nv

        # 3) Normalize 'extra' into vector + (optional) ordering
        extra_vec: Dict[str, float] = {}
        extra_cols: List[str] = []

        if isinstance(extra, dict) and ("vector" in extra or "columns" in extra or "values" in extra):
            if isinstance(extra.get("vector"), dict):
                for k, v in extra["vector"].items():
                    nv = _num(v)
                    if nv is not None:
                        extra_vec[str(k)] = nv
            ecols = extra.get("columns") if isinstance(extra.get("columns"), list) else []
            evals = extra.get("values") if isinstance(extra.get("values"), list) else []
            if ecols and evals and len(ecols) == len(evals):
                for c, v in zip(ecols, evals):
                    nv = _num(v)
                    if nv is not None:
                        extra_vec[str(c)] = nv
                extra_cols = [str(c) for c in ecols]
        elif isinstance(extra, dict):
            for k, v in extra.items():
                nv = _num(v)
                if nv is not None:
                    extra_vec[str(k)] = nv

        # 4) Overlay extras (extras WIN on collisions)
        base.update(extra_vec)

        # 5) Build final column order safely
        #    a) keep metrics.columns order, but only those that survived into base
        ordered = [c for c in (metric_cols or []) if c in base]

        #    b) then append NEW keys from extra, preferring extra_cols order if provided
        seen = set(ordered)
        if extra_cols:
            for c in extra_cols:
                if c in base and c not in seen:
                    ordered.append(c); seen.add(c)
        else:
            for c in extra_vec.keys():
                if c in base and c not in seen:
                    ordered.append(c); seen.add(c)

        #    c) finally append any remaining base keys not yet included
        for k in base.keys():
            if k not in seen:
                ordered.append(k)

        # 6) Values aligned to columns (use .get as a guard, though all should exist)
        final_cols = ordered
        final_vals = [base.get(c, 0.0) for c in final_cols]

        return {"columns": final_cols, "values": final_vals, "vector": base}

    def _preferred_indices(
        self, names: List[str], model: str, dims: List[str]
    ) -> Dict[str, int]:
        name_idx = {n: i for i, n in enumerate(names)}

        def exact(k: str):
            return name_idx.get(k)

        def seek(*patterns: str):
            for i, n in enumerate(names):
                s = n.lower()
                if all(p in s for p in patterns):
                    return i
            return None

        out = {}

        # Aggregate: prefer canonical SCM, then model-prefixed.
        agg = (
            exact("scm.aggregate01")
            or exact(f"{model}.aggregate01")
            or exact(f"{model}.aggregate")
            or seek(model + ".", ".aggregate01")
            or seek(model + ".", ".aggregate")
        )
        if agg is not None:
            out["aggregate"] = agg

        for d in dims:
            idx = (
                exact(f"scm.{d}.score01")  # gold path
                or exact(f"{model}.{d}.score01")
                or seek(model + ".", f".{d}", ".score01")
                or exact(f"{model}.{d}.score")
                or seek(model + ".", f".{d}", ".score")
            )
            if idx is not None:
                out[d] = idx
        return out

    # --- VRAM / RAM cleanup helpers -----------------------------------------
    def _unload_hf_scorers(self):
        """
        Best-effort unload of HuggingFace-based scorers to release VRAM/RAM.
        Works if scorers expose .unload() or .close(); otherwise drop heavy attrs.
        """
        scoring = self.container.get("scoring")
        for name, scorer in list(getattr(scoring, "_scorers", {}).items()):
            try:
                # Preferred explicit API if your scorer implements it
                if hasattr(scorer, "unload") and callable(scorer.unload):
                    scorer.unload()
                elif hasattr(scorer, "close") and callable(scorer.close):
                    scorer.close()

                # Defensive: drop common heavy attrs
                for attr in (
                    "model",
                    "tokenizer",
                    "pipeline",
                    "hf_model",
                    "hf_tokenizer",
                ):
                    if hasattr(scorer, attr):
                        try:
                            setattr(scorer, attr, None)
                        except Exception:
                            pass
            except Exception:
                pass

        # Finally, ask CUDA/GC to give memory back
        self._free_accelerator_memory()

    def _free_accelerator_memory(self):
        try:
            import torch, gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def _detect_model_prefix(vec: dict, fallback: str) -> str:
    """
    Find the actual root prefix used in the metric keys.
    If fallback ('hrm'/'tiny') doesn't exist, pick a close match like 'hf_hrm'.
    """
    roots = {k.split(".", 1)[0] for k in vec.keys() if "." in k}
    if fallback in roots:
        return fallback
    # Heuristic: pick a root that contains the fallback token
    for r in sorted(roots, key=len, reverse=True):
        if fallback in r:
            return r
    # Last resort: just return a deterministic root
    return sorted(roots)[0] if roots else fallback


def extract_raw_dim_and_scm(
    vec: dict, model_label: str, dims: list[str]
) -> dict[str, float]:
    """
    For each dim in dims:
      - include any raw score fields:
          {prefix}.{dim}.score01 / score100 / score / aggregate01 / aggregate
      - include *all* SCM fields that live under:
          {prefix}.{dim}.attr.scm.*
      Values are taken verbatim (float-cast only). No normalization.
    Returns a flat dict of key->float.
    """
    out: dict[str, float] = {}
    prefix = _detect_model_prefix(
        vec, model_label
    )  # e.g. 'hf_hrm' or 'hf_mistral'

    def _take(k: str):
        try:
            out[k] = float(vec[k])
        except Exception:
            pass

    for dim in dims:
        # 1) grab raw score-ish fields for this dim (whatever exists)
        candidates = [
            f"{prefix}.{dim}.score01",
            f"{prefix}.{dim}.score100",
            f"{prefix}.{dim}.score",
            f"{prefix}.{dim}.aggregate01",
            f"{prefix}.{dim}.aggregate",
        ]
        for k in candidates:
            if k in vec:
                _take(k)

        # 2) grab *all* SCM fields under this dim
        scm_root = f"{prefix}.{dim}.attr.scm."
        for k, v in vec.items():
            if k.startswith(scm_root):
                # Re-key to a stable, neutral form: {prefix}.{dim}.scm.<tail>
                tail = k[len(scm_root) :]
                try:
                    out[f"{prefix}.{dim}.scm.{tail}"] = float(v)
                except Exception:
                    pass

    return out

def _first_or_str(x):
    # metrics["model_alias"] can be ["hf_TinyLama"] or "hf_TinyLama"
    if isinstance(x, (list, tuple)) and x:
        return str(x[0])
    return str(x)

