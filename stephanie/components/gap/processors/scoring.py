from __future__ import annotations
import asyncio, hashlib, logging
from typing import Any, Dict, List, Tuple, Callable, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd

from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.services.workers.metrics_worker import MetricsWorkerInline
from stephanie.services.workers.vpm_worker import VPMWorkerInline
from stephanie.components.gap.models import GapConfig, TripleSample, GapRunManifest
from stephanie.components.gap.shared_scm import scm_from_vector, scm_row, SCM_COLUMNS

logger = logging.getLogger(__name__)

class ScoringProcessor:
    """Minimal, readable scoring + alignment pipeline with SCM injection."""

    def __init__(self, config: GapConfig, container, logger):
        self.config = config
        self.container = container
        self.logger = logger

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
        *,
        progress_cb: Optional[Callable[[int,int,Optional[Dict[str,Any]]], None]] = None,
    ) -> Dict[str, Any]:
        scoring_service = self.container.get("scoring")
        zm = self.container.get("zeromodel")

        hrm_worker = MetricsWorkerInline(scoring_service, self.config.hrm_scorers, self.config.dimensions)
        tiny_worker = MetricsWorkerInline(scoring_service, self.config.tiny_scorers, self.config.dimensions)
        vpm_worker = VPMWorkerInline(zm, self.logger, progress_cb=progress_cb)

        return await self._score_all_triples(triples_data, hrm_worker, tiny_worker, vpm_worker, run_id, progress_cb=progress_cb)

    # ---------- core pipeline ----------------------------------------------
    async def _score_all_triples(
        self,
        triples_data: Dict[str, List[TripleSample]],
        hrm_worker, tiny_worker, vpm_worker, run_id: str,
        *, progress_cb=None
    ) -> Dict[str, Any]:

        # 0) materialize list
        all_triples: List[TripleSample] = [t for ts in triples_data.values() for t in ts]
        T = len(all_triples)

        # 1) timelines
        zm = self.container.get("zeromodel")
        hrm_tl = f"{run_id}_hrm"
        tiny_tl = f"{run_id}_tiny"
        zm.timeline_open(run_id=hrm_tl)
        zm.timeline_open(run_id=tiny_tl)

        # 2) accumulators
        hrm_names: List[str] = []
        tiny_names: List[str] = []
        hrm_rows: List[List[float]] = []
        tiny_rows: List[List[float]] = []

        hrm_scm_rows: List[List[float]] = []
        tiny_scm_rows: List[List[float]] = []

        rows_for_df: List[Dict[str, float]] = []  # PHOS-friendly per turn

        log_every = max(1, self.config.progress_log_every)

        with tqdm(total=T, desc="[GAP] Scoring", unit="turn") as pbar:
            for i, triple in enumerate(all_triples):
                scorable = Scorable(triple.output_text, ScorableType.CONVERSATION_TURN)

                # score both models
                hrm_metrics = await hrm_worker.score(scorable, triple.goal_text, hrm_tl)
                tiny_metrics = await tiny_worker.score(scorable, triple.goal_text, tiny_tl)

                # extract flat vectors
                h_vec = self._to_vector(hrm_metrics)
                t_vec = self._to_vector(tiny_metrics)

                # build SCM dicts (same source of truth as vectors we save)
                h_scm = scm_from_vector(h_vec, model_prefix="hrm")
                t_scm = scm_from_vector(t_vec, model_prefix="tiny")

                # append SCM-merged payloads to timelines (IMPORTANT)
                hrm_for_tl = self._merge_for_timeline(hrm_metrics, h_scm)
                tiny_for_tl = self._merge_for_timeline(tiny_metrics, t_scm)
                vpm_worker.append(hrm_tl, triple.node_id, hrm_for_tl)
                vpm_worker.append(tiny_tl, triple.node_id, tiny_for_tl)

                # lock first-row names; align subsequent rows by first-row order
                if i == 0:
                    hrm_names = list(h_vec.keys())
                    tiny_names = list(t_vec.keys())

                hrm_rows.append(self._align_row(h_vec, hrm_names))
                tiny_rows.append(self._align_row(t_vec, tiny_names))
                hrm_scm_rows.append(scm_row(h_scm))
                tiny_scm_rows.append(scm_row(t_scm))

                # PHOS rows (per-dim stable)
                phos_row = {"node_id": triple.node_id}
                for d in self.config.dimensions:
                    phos_row[f"hrm.{d}"]  = float(h_scm.get(f"scm.{d}.score01", 0.0))
                    phos_row[f"tiny.{d}"] = float(t_scm.get(f"scm.{d}.score01", 0.0))
                rows_for_df.append(phos_row)

                # progress
                if ((i+1) % log_every) == 0 or (i+1) == T:
                    self.logger.log("ScoringProgress", {"processed": i+1, "total": T})
                    if progress_cb:
                        try: progress_cb(i+1, T, None)
                        except Exception: pass
                pbar.update(1)
                await asyncio.sleep(0)

        # 3) finalize timelines (GIFs)
        hrm_gif = await vpm_worker.finalize(hrm_tl, f"vpm_phos_run_{hrm_tl}.gif")
        tiny_gif = await vpm_worker.finalize(tiny_tl, f"vpm_phos_run_{tiny_tl}.gif")

        # 4) convert to matrices
        hrm_matrix_raw = np.asarray(hrm_rows, dtype=np.float32)
        tiny_matrix_raw = np.asarray(tiny_rows, dtype=np.float32)

        # 5) pick preferred indices and align (aggregate + per-dim)
        pref_hrm = self._preferred_indices(hrm_names, "hrm", self.config.dimensions)
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
        hrm_matrix = hrm_matrix_raw[:, h_cols]
        tiny_matrix = tiny_matrix_raw[:, t_cols]

        # 6) persist artifacts
        storage = self.container.get("gap_storage")
        storage.save_matrix(hrm_matrix, shared, run_id, tag="hrm")
        storage.save_matrix(tiny_matrix, shared, run_id, tag="tiny")
        storage.save_matrix(np.asarray(hrm_scm_rows, np.float32), SCM_COLUMNS, run_id, tag="hrm_scm")
        storage.save_matrix(np.asarray(tiny_scm_rows, np.float32), SCM_COLUMNS, run_id, tag="tiny_scm")

        # rows_for_df for PHOS
        df_rows = pd.DataFrame(rows_for_df)
        raw_paths = storage.save_rows_df(df_rows, run_id, name="rows_for_df")

        if progress_cb:
            try: progress_cb(T, T, {"done": True})
            except Exception: pass

        return {
            "hrm_vectors": hrm_matrix,
            "tiny_vectors": tiny_matrix,
            "hrm_names": shared,
            "tiny_names": shared,
            "hrm_scm_matrix": np.asarray(hrm_scm_rows, np.float32),
            "tiny_scm_matrix": np.asarray(tiny_scm_rows, np.float32),
            "scm_names": SCM_COLUMNS,
            "hrm_gif": hrm_gif,
            "tiny_gif": tiny_gif,
            "triples_count": T,
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

        def exact(key: str) -> Optional[int]:
            return name_idx.get(key)

        def loose(parts: List[str]) -> Optional[int]:
            for i, n in enumerate(names):
                s = n.lower()
                if all(p in s for p in parts): return i
            return None

        out: Dict[str, int] = {}
        # aggregate
        idx = exact(f"{model}.aggregate") or loose([model+".", "aggregate"])
        if idx is not None: out["aggregate"] = idx
        # per-dim
        for d in dims:
            cands = [f"{model}.{d}.score", f"{model}.{d}.aggregate", f"{model}.{d}"]
            idx = None
            for c in cands:
                if c in name_idx: idx = name_idx[c]; break
            if idx is None:
                idx = loose([model+".", f".{d}", ".score"]) or loose([model+".", f".{d}", ".aggregate"])
            if idx is not None: out[d] = idx
        return out
