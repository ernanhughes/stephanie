from __future__ import annotations
import asyncio, hashlib, logging
from typing import Any, Dict, List, Tuple, Callable, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd

from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.services.workers.metrics_worker import MetricsWorkerInline
from stephanie.services.workers.vpm_worker import VPMWorkerInline
from stephanie.components.gap.models import GapConfig, TripleSample 
from stephanie.components.gap.io.manifest import GapRunManifest
from stephanie.components.gap.shared_scm import scm_from_vector, scm_row, SCM_COLUMNS
from stephanie.utils.progress_mixin import ProgressMixin
import time

logger = logging.getLogger(__name__)

class ScoringProcessor(ProgressMixin):
    """Minimal, readable scoring + alignment pipeline with SCM injection."""

    def __init__(self, config: GapConfig, container, logger):
        self.config = config
        self.container = container
        self.logger = logger
        self._init_progress(container, logger)  # <-- ProgressService hookup

    # ---------- public API --------------------------------------------------
    async def prepare_samples(self, dimensions: List[str], memory) -> Dict[str, List[TripleSample]]:
        from stephanie.scoring.training.preference_pair_builder import PreferencePairBuilder
        pb = PreferencePairBuilder(memory, self.logger)
        by_dim: Dict[str, List[TripleSample]] = {}
        for d in dimensions:
            pairs = pb.get_training_pairs_by_dimension(dimension=d)
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
        manifest.stage_start(
            "scoring",
            total=total_rows,
            run_id=run_id,
            status="running",
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        hrm_worker = MetricsWorkerInline(scoring_service, self.config.hrm_scorers, self.config.dimensions)
        tiny_worker = MetricsWorkerInline(scoring_service, self.config.tiny_scorers, self.config.dimensions)
        vpm_worker = VPMWorkerInline(zm, self.logger)

        result = await self._score_all_triples(triples_data, hrm_worker, tiny_worker, vpm_worker, run_id)

        manifest.stage_end(
                "scoring",
                status="ok",
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                triples_count=result.get("triples_count"),
                columns=result.get("hrm_names"),
                hrm_label=result.get("hrm_label"),
                tiny_label=result.get("tiny_label"),
                artifacts={
                    "hrm_gif": result.get("hrm_gif"),
                    "tiny_gif": result.get("tiny_gif"),
                    "rows_for_df": result.get("rows_for_df_path"),
                },
            )

        # optional: also persist a flat copy for quick inspection
        storage = self.container.get("gap_storage")
        storage.save_json(run_id, "metrics", "scoring_stage.json", manifest.to_dict())

        return result

    # ---------- core pipeline ----------------------------------------------
    async def _score_model_pass(
        self,
        model_label: str,                         # "hrm" or "tiny"
        worker,                                   # MetricsWorkerInline bound to that model
        triples: List[TripleSample],
        timeline_id: str,
            *,
        parent_task: str,              # overall task name (e.g., f"scoring:{run_id}")
        base_done: int,                # offset into the overall progress (0 for HRM, T for Tiny)
        grand_total: int,              # overall total = T*2
    ) -> Dict[str, Any]:
        """
        Score ALL samples for ONE model in a single pass.

        WHY THIS EXISTS:
        - Keeps ONE HF model loaded on the GPU for the whole pass → avoids VRAM thrash.
        - Much more stable on 12 GB GPUs (no interleaved memory spikes).
        - Better kernel reuse (attention impl stays hot).
        """
        sub_task = f"{parent_task}:{model_label}"
        self.pstart(sub_task, total=len(triples), meta={"model": model_label, "timeline": timeline_id})

        zm = self.container.get("zeromodel")
        vpm_worker = VPMWorkerInline(zm, self.logger)

        zm.timeline_open(run_id=timeline_id)

        names: List[str] = []
        rows: List[List[float]] = []
        scm_rows: List[List[float]] = []

        keep_mask: List[bool] = []         # one bool per original triple index
        kept_indices: List[int] = []       # original indices we kept (for merges)
        rows_for_df: List[Dict[str, float]] = []   # PHOS-friendly rows (model side)

        log_every = max(1, self.config.progress_log_every)

        for i, triple in enumerate(triples):
            scorable = Scorable(triple.output_text, ScorableType.CONVERSATION_TURN)

            # SHARED GATE: identical length guard for both models so masks intersect cleanly.
            if len(triple.goal_text) > 4000 or len(scorable.text) > 4000:
                keep_mask.append(False)
                continue

            # Score this model
            metrics = await worker.score(scorable, triple.goal_text, timeline_id)

            # Extract vectors and SCM
            vec_raw = self._to_vector(metrics)
            scm = scm_from_vector(vec_raw, model_prefix=model_label)    # e.g. "scm.reasoning.score01"
            merged = self._merge_for_timeline(metrics, scm)

            # Timeline (for GIFs)
            vpm_worker.append(timeline_id, triple.node_id, merged)

            # Lock first-row column order for this model, align subsequent rows
            vec = merged["vector"]
            if not names:
                names = list(vec.keys())
            rows.append(self._align_row(vec, names))
            scm_rows.append(scm_row(scm))

            # PHOS row (only this model’s side for now; we’ll merge later)
            r = {"node_id": triple.node_id}
            for d in self.config.dimensions:
                r[f"{model_label}.{d}"] = float(scm.get(f"scm.{d}.score01", 0.0))
            rows_for_df.append(r)

            keep_mask.append(True)
            kept_indices.append(i)

            # structured progress
            if ((i + 1) % log_every) == 0 or (i + 1) == len(triples):
                self.logger.log("ScoringProgress", {
                    "model": model_label, "processed": i + 1, "total": len(triples)
                })
            await asyncio.sleep(0)

        gif_path = await vpm_worker.finalize(timeline_id, f"vpm_phos_run_{timeline_id}.gif")

        return {
            "names": names,
            "rows": np.asarray(rows, np.float32),
            "scm_rows": np.asarray(scm_rows, np.float32),
            "keep_mask": np.array(keep_mask, dtype=bool),
            "kept_indices": kept_indices,           # original indices retained (in order)
            "gif": gif_path,
            "rows_for_df": rows_for_df,             # one entry per kept index (same order as kept_indices)
        }

    async def _score_all_triples(
        self,
        triples_data: Dict[str, List[TripleSample]],
        hrm_worker, tiny_worker, vpm_worker, run_id: str,
    ) -> Dict[str, Any]:

        # 0) materialize flattened list (stable order for both passes)
        all_triples: List[TripleSample] = [t for ts in triples_data.values() for t in ts]
        T = len(all_triples)

        task = f"scoring:{run_id}"
        self.pstart(task, total=T * 2, meta={
            "dims": len(triples_data),
            "passes": 2,
            "console_echo": False
        })


        # 1) PER-MODEL PASSES (keeps one model resident on GPU at a time)
        hrm_tl  = f"{run_id}_hrm"
        tiny_tl = f"{run_id}_tiny"

        # HRM pass updates overall [0 .. T]
        hrm_res = await self._score_model_pass(
            "hrm", hrm_worker, all_triples, hrm_tl,
            parent_task=task, base_done=0, grand_total=T * 2
        )

        # Tiny pass updates overall [T .. 2T]
        tiny_res = await self._score_model_pass(
            "tiny", tiny_worker, all_triples, tiny_tl,
            parent_task=task, base_done=T, grand_total=T * 2
        )

        hrm_names = hrm_res["names"]
        tiny_names = tiny_res["names"]

        # 2) INTERSECT MASKS to ensure both models scored the same rows
        #    We intersect by ORIGINAL INDEX; we then select corresponding rows from each pass.
        hrm_keep = hrm_res["keep_mask"]
        tny_keep = tiny_res["keep_mask"]
        if hrm_keep.shape != tny_keep.shape:
            raise RuntimeError("Internal: mask shapes differ between models.")

        both_keep_mask = hrm_keep & tny_keep
        kept_idx = np.nonzero(both_keep_mask)[0].tolist()     # original indices we keep

        # Build compaction index maps: original index -> compacted row index within each pass
        # (helper since each pass only kept some rows)
        hrm_idx_map = {orig: j for j, orig in enumerate(hrm_res["kept_indices"])}
        tny_idx_map = {orig: j for j, orig in enumerate(tiny_res["kept_indices"])}

        # 3) SLICE MATRICES to the common subset
        def _select_rows(pass_rows: np.ndarray, idx_map: Dict[int,int], orig_idx: List[int]) -> np.ndarray:
            sel = [idx_map[i] for i in orig_idx if i in idx_map]
            return pass_rows[sel, :] if len(sel) else np.zeros((0, pass_rows.shape[1]), dtype=pass_rows.dtype)

        hrm_matrix_raw  = _select_rows(hrm_res["rows"],  hrm_idx_map, kept_idx)
        tiny_matrix_raw = _select_rows(tiny_res["rows"], tny_idx_map, kept_idx)

        # 4) Resolve preferred columns ONCE, align columns
        pref_hrm = self._preferred_indices(hrm_names,  "hrm",  self.config.dimensions)
        pref_tny = self._preferred_indices(tiny_names, "tiny", self.config.dimensions)

        canonical_order = ["aggregate"] + list(self.config.dimensions)
        shared = [k for k in canonical_order if k in pref_hrm and k in pref_tny]
        if not shared:
            storage = self.container.get("gap_storage")
            storage.save_json(run_id, "metrics", "name_alignment_debug.json", {
                "hrm_names": hrm_names, "tiny_names": tiny_names,
                "preferred_hrm": pref_hrm, "preferred_tiny": pref_tny,
                "dims": self.config.dimensions,
            })
            raise RuntimeError("No shared preferred metrics across models.")

        h_cols = [pref_hrm[k] for k in shared]
        t_cols = [pref_tny[k] for k in shared]
        hrm_matrix  = hrm_matrix_raw[:, h_cols]
        tiny_matrix = tiny_matrix_raw[:, t_cols]

        # 5) SCM matrices (already in canonical SCM_COLUMNS order)
        #    Select the same rows by original index for both passes
        from stephanie.components.gap.shared_scm import SCM_COLUMNS
        hrm_scm_sel  = _select_rows(hrm_res["scm_rows"],  hrm_idx_map, kept_idx)
        tiny_scm_sel = _select_rows(tiny_res["scm_rows"], tny_idx_map, kept_idx)

        # 6) Build PHOS rows by merging per-model dicts on the common subset
        #    Note: rows_for_df in each pass is only for kept rows in that pass; reindex via kept indices.
        def _select_rows_for_df(rows_for_df: List[Dict[str,float]], idx_map: Dict[int,int], orig_idx: List[int]) -> List[Dict[str,float]]:
            sel = []
            for i in orig_idx:
                j = idx_map.get(i, None)
                if j is not None:
                    sel.append(rows_for_df[j])
            return sel

        hrm_df_rows  = _select_rows_for_df(hrm_res["rows_for_df"],  hrm_idx_map, kept_idx)
        tiny_df_rows = _select_rows_for_df(tiny_res["rows_for_df"], tny_idx_map, kept_idx)

        merged_rows = []
        for rh, rt in zip(hrm_df_rows, tiny_df_rows):
            # both contain node_id; keep one
            row = {"node_id": rh.get("node_id") or rt.get("node_id")}
            row.update({k:v for k,v in rh.items() if k != "node_id"})
            row.update({k:v for k,v in rt.items() if k != "node_id"})
            merged_rows.append(row)

        df_rows = pd.DataFrame(merged_rows)

        # 7) Timelines → GIF paths
        hrm_gif = hrm_res["gif"]
        tiny_gif = tiny_res["gif"]

        # 8) Persist artifacts
        storage = self.container.get("gap_storage")
        storage.save_matrix(hrm_matrix, shared, run_id, tag="hrm")
        storage.save_matrix(tiny_matrix, shared, run_id, tag="tiny")
        storage.save_matrix(hrm_scm_sel.astype(np.float32),  SCM_COLUMNS, run_id, tag="hrm_scm")
        storage.save_matrix(tiny_scm_sel.astype(np.float32), SCM_COLUMNS, run_id, tag="tiny_scm")

        raw_paths = storage.save_rows_df(df_rows, run_id, name="rows_for_df")

        # Row-level provenance: filter by kept indices for a 1:1 mapping
        provenance = []
        for row_idx, orig_i in enumerate(kept_idx):
            t = all_triples[orig_i]
            provenance.append({
                "row_index": row_idx,
                "orig_index": orig_i,
                "node_id": t.node_id,
                "dimension": t.dimension,
                "goal_text": t.goal_text,
                "output_text": t.output_text,
            })
        storage.save_json(run_id, "raw", "row_provenance.json", provenance)

        # Labels (human display)
        scoring_service = self.container.get("scoring")
        def _disp(n):
            s = scoring_service._scorers.get(n)
            return s.get_display_name() if s and hasattr(s, "get_display_name") else n
        hrm_label  = ", ".join(_disp(n) for n in self.config.hrm_scorers)
        tiny_label = ", ".join(_disp(n) for n in self.config.tiny_scorers)

        self.pdone(task, extra={"rows": len(kept_idx)})

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
            "rows_for_df_path": str(storage.base_dir / run_id / "raw" / "rows_for_df.parquet"),
            **raw_paths,
        }
    # ---------- helpers -----------------------------------------------------
    def _flatten_samples(self, samples: List[Dict[str, Any]], dim: str) -> List[TripleSample]:
        triples: List[TripleSample] = []
        for i, s in enumerate(samples):
            title = (s.get("goal_text") or s.get("title") or "").strip()
            # singleton
            if "output" in s and ("score" in s or "target_score" in s):
                out = (s.get("output") or "").strip()
                val = s.get("target_score", s.get("score"))
                if title and out and val is not None:
                    triples.append(TripleSample(
                        node_id=f"{dim}|{i:06d}", dimension=dim,
                        goal_text=title, output_text=out,
                        target_value=float(val),
                        fingerprint=self._fingerprint(title, out),
                    ))
                continue
            # pairwise
            if all(k in s for k in ("output_a","output_b","value_a","value_b")):
                for suf in ("a","b"):
                    out = (s.get(f"output_{suf}") or "").strip()
                    val = s.get(f"value_{suf}")
                    if title and out and val is not None:
                        triples.append(TripleSample(
                            node_id=f"{dim}|{i:06d}_{suf}", dimension=dim,
                            goal_text=title, output_text=out,
                            target_value=float(val),
                            fingerprint=self._fingerprint(title, out),
                        ))
        return triples

    def _dedupe(self, by_dim: Dict[str, List[TripleSample]]) -> Dict[str, List[TripleSample]]:
        policy = self.config.dedupe_policy
        cap = self.config.per_dim_cap
        if policy == "first_wins":
            seen = set(); out = {d: [] for d in by_dim}
            for d, ts in by_dim.items():
                for t in ts:
                    if t.fingerprint in seen: continue
                    out[d].append(t); seen.add(t.fingerprint)
                if cap and len(out[d]) > cap: out[d] = out[d][:cap]
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
                    out[d].append(t); i += 1
            return out
        else:
            raise ValueError(f"Unknown dedupe policy: {policy}")

    def _fingerprint(self, g: str, o: str) -> str:
        return hashlib.sha1((g.strip()+"\n␟\n"+o.strip()).encode("utf-8")).hexdigest()

    def _to_vector(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        vec = metrics.get("vector")
        if isinstance(vec, dict) and vec:
            return {str(k): float(v) for k, v in vec.items()}
        cols = metrics.get("columns"); vals = metrics.get("values")
        if isinstance(cols, list) and isinstance(vals, list) and len(cols)==len(vals):
            return {str(c): float(v) for c, v in zip(cols, vals)}
        return {}

    def _align_row(self, vec: Dict[str, float], names: List[str]) -> List[float]:
        return [float(vec.get(n, 0.0)) for n in names]

    def _merge_for_timeline(self, metrics: Dict[str, Any], extra: Dict[str, float]) -> Dict[str, Any]:
        """
        Build a flat mapping from the incoming metrics (vector or columns/values),
        merge in `extra`, and return BOTH representations so downstream callers
        always find 'columns'/'values'.
        """
        base: Dict[str, float] = {}

        # 1) Start from vector if present
        vec = metrics.get("vector")
        if isinstance(vec, dict) and vec:
            base.update({str(k): float(v) for k, v in vec.items()})

        # 2) Merge any columns/values if present
        cols = metrics.get("columns")
        vals = metrics.get("values")
        if isinstance(cols, list) and isinstance(vals, list) and len(cols) == len(vals):
            base.update({str(c): float(v) for c, v in zip(cols, vals)})

        # 3) Overlay extras (SCM, etc.)
        for k, v in extra.items():
            base[str(k)] = float(v)

        out_cols = list(base.keys())
        out_vals = [base[c] for c in out_cols]
        return {"columns": out_cols, "values": out_vals, "vector": base}

    def _preferred_indices(self, names: List[str], model: str, dims: List[str]) -> Dict[str, int]:
        name_idx = {n: i for i, n in enumerate(names)}

        def exact(k: str): return name_idx.get(k)

        def seek(*patterns: str):
            for i, n in enumerate(names):
                s = n.lower()
                if all(p in s for p in patterns):
                    return i
            return None

        out = {}

        # aggregate: prefer model-prefixed, else SCM
        agg = (exact(f"{model}.aggregate01")
            or exact(f"{model}.aggregate")
            or seek(model+".", ".aggregate01")
            or seek(model+".", ".aggregate")
            or exact("scm.aggregate01"))                       # <<< NEW
        if agg is not None:
            out["aggregate"] = agg

        for d in dims:
            idx = (exact(f"{model}.{d}.score01")
                or seek(model+".", f".{d}", ".score01")
                or exact(f"{model}.{d}.score")            # legacy 0–1
                or seek(model+".", f".{d}", ".score")
                or exact(f"scm.{d}.score01"))             # <<< NEW (gold path)
            if idx is not None:
                out[d] = idx
        return out
