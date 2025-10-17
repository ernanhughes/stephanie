# stephanie/components/gap/processors/scoring.py
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Tuple, Callable, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd

from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.services.workers.metrics_worker import MetricsWorkerInline
from stephanie.services.workers.vpm_worker import VPMWorkerInline
from stephanie.components.gap.models import GapConfig, TripleSample, GapRunManifest

logger = logging.getLogger(__name__)

class ScoringProcessor:
    """Handles sample preparation, scoring, and timeline generation."""
    
    def __init__(self, config: GapConfig, container, logger):
        self.config = config
        self.container = container
        self.logger = logger
    
    async def prepare_samples(self, dimensions: List[str], memory) -> Dict[str, List[TripleSample]]:
        """Collect and deduplicate training samples across dimensions."""
        from stephanie.scoring.training.preference_pair_builder import PreferencePairBuilder
        
        pair_builder = PreferencePairBuilder(memory, self.logger)
        triples_by_dim: Dict[str, List[TripleSample]] = {}
        
        for dimension in dimensions:
            pairs = pair_builder.get_training_pairs_by_dimension(dimension=dimension)
            samples = pairs.get(dimension, [])
            triples = self._flatten_samples(samples, dimension)
            triples_by_dim[dimension] = triples
        
        return self._deduplicate_samples(triples_by_dim)
    
    async def execute_scoring(
        self,
        triples_data: Dict[str, List[TripleSample]],
        run_id: str,
        manifest: GapRunManifest,
        *,
        progress_cb: Optional[Callable[[int, int, Optional[Dict[str, Any]]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute model scoring and generate timelines."""
        scoring_service = self.container.get("scoring")
        zm = self.container.get("zeromodel")
        
        # Initialize workers
        hrm_worker = MetricsWorkerInline(scoring_service, self.config.hrm_scorers, self.config.dimensions)
        tiny_worker = MetricsWorkerInline(scoring_service, self.config.tiny_scorers, self.config.dimensions)
        vpm_worker = VPMWorkerInline(zm, self.logger)
        
        # Execute scoring pipeline
        results = await self._score_all_triples(
            triples_data, hrm_worker, tiny_worker, vpm_worker, run_id, progress_cb=progress_cb
        )
        
        return results
    
    def _flatten_samples(self, samples: List[Dict[str, Any]], dimension: str) -> List[TripleSample]:
        """Convert raw samples to TripleSample objects."""
        triples: List[TripleSample] = []
        
        for i, sample in enumerate(samples):
            goal_text = (sample.get("goal_text") or sample.get("title") or "").strip()
            
            # singleton
            if "output" in sample and ("score" in sample or "target_score" in sample):
                output_text = (sample.get("output") or "").strip()
                value = sample.get("target_score", sample.get("score"))
                if goal_text and output_text and value is not None:
                    fingerprint = self._compute_fingerprint(goal_text, output_text)
                    triples.append(TripleSample(
                        node_id=f"{dimension}|{i:06d}",
                        dimension=dimension,
                        goal_text=goal_text,
                        output_text=output_text,
                        target_value=float(value),
                        fingerprint=fingerprint,
                    ))
                continue

            # pairwise
            if all(k in sample for k in ("output_a", "output_b", "value_a", "value_b")):
                for suffix in ("a", "b"):
                    output_text = (sample.get(f"output_{suffix}") or "").strip()
                    value = sample.get(f"value_{suffix}")
                    if goal_text and output_text and value is not None:
                        fingerprint = self._compute_fingerprint(goal_text, output_text)
                        triples.append(TripleSample(
                            node_id=f"{dimension}|{i:06d}_{suffix}",
                            dimension=dimension,
                            goal_text=goal_text,
                            output_text=output_text,
                            target_value=float(value),
                            fingerprint=fingerprint,
                        ))
                continue
        
        return triples
    
    def _deduplicate_samples(self, triples_by_dim: Dict[str, List[TripleSample]]) -> Dict[str, List[TripleSample]]:
        """Deduplicate samples across dimensions using configured policy."""
        if self.config.dedupe_policy == "first_wins":
            return self._deduplicate_first_wins(triples_by_dim)
        elif self.config.dedupe_policy == "round_robin":
            return self._deduplicate_round_robin(triples_by_dim)
        raise ValueError(f"Unknown deduplication policy: {self.config.dedupe_policy}")
    
    def _deduplicate_first_wins(self, triples_by_dim: Dict[str, List[TripleSample]]) -> Dict[str, List[TripleSample]]:
        seen = set()
        deduped = {dim: [] for dim in triples_by_dim.keys()}
        for dim, triples in triples_by_dim.items():
            for t in triples:
                if t.fingerprint in seen:
                    continue
                deduped[dim].append(t)
                seen.add(t.fingerprint)
            if self.config.per_dim_cap and len(deduped[dim]) > self.config.per_dim_cap:
                deduped[dim] = deduped[dim][: self.config.per_dim_cap]
        return deduped
    
    def _deduplicate_round_robin(self, triples_by_dim: Dict[str, List[TripleSample]]) -> Dict[str, List[TripleSample]]:
        unique: Dict[str, TripleSample] = {}
        for dim, triples in triples_by_dim.items():
            for t in triples:
                unique.setdefault(t.fingerprint, t)
        dims = list(triples_by_dim.keys())
        deduped = {d: [] for d in dims}
        i = 0
        for t in unique.values():
            d = dims[i % len(dims)]
            if self.config.per_dim_cap is None or len(deduped[d]) < self.config.per_dim_cap:
                deduped[d].append(t)
                i += 1
        return deduped
    
    def _compute_fingerprint(self, goal_text: str, output_text: str) -> str:
        content = (goal_text.strip() + "\n␟\n" + output_text.strip()).encode("utf-8")
        return hashlib.sha1(content).hexdigest()
    

    def _canon_name(self, name: str) -> str:
        """
        Canonicalize a metric name so HRM and Tiny variants can align.
        - Lowercase
        - Strip leading model prefixes: hrm., tiny.
        - Strip trailing suffixes: .score, .aggregate, .value, .raw
        - Collapse duplicate dots
        """
        s = str(name or "").strip().lower()
        for pref in ("hrm.", "tiny."):
            if s.startswith(pref):
                s = s[len(pref):]
        # collapse accidental duplicate dots
        while ".." in s:
            s = s.replace("..", ".")
        for suf in (".score", ".aggregate", ".value", ".raw"):
            if s.endswith(suf):
                s = s[: -len(suf)]
        return s

    async def _score_all_triples(
        self,
        triples_data: Dict[str, List[TripleSample]],
        hrm_worker,
        tiny_worker,
        vpm_worker,
        run_id: str,
        *,
        progress_cb: Optional[Callable[[int, int, Optional[Dict[str, Any]]], None]] = None,
    ) -> Dict[str, Any]:
        """Score all triples with both models and generate timelines."""
        # Combine and count
        all_triples: List[TripleSample] = []
        for dim_triples in triples_data.values():
            all_triples.extend(dim_triples)
        total = len(all_triples)

        # Initialize timelines
        zm = self.container.get("zeromodel")
        hrm_timeline_id = f"{run_id}_hrm"
        tiny_timeline_id = f"{run_id}_tiny"
        zm.timeline_open(run_id=hrm_timeline_id)
        zm.timeline_open(run_id=tiny_timeline_id)
        
        # Containers
        hrm_vectors: List[List[float]] = []
        tiny_vectors: List[List[float]] = []
        hrm_names: List[str] = []
        tiny_names: List[str] = []
        rows_for_df: List[Dict[str, Any]] = []

        log_every = max(1, self.config.progress_log_every)

        with tqdm(total=total, desc="[GAP] Scoring triples", unit="turn") as pbar:
            for i, triple in enumerate(all_triples):
                scorable = Scorable(triple.output_text, ScorableType.CONVERSATION_TURN)
                
                # Score with both models
                hrm_metrics = await hrm_worker.score(scorable, triple.goal_text, hrm_timeline_id)
                tiny_metrics = await tiny_worker.score(scorable, triple.goal_text, tiny_timeline_id)
                
                # Append to timelines
                await vpm_worker.append(hrm_timeline_id, triple.node_id, hrm_metrics)
                await vpm_worker.append(tiny_timeline_id, triple.node_id, tiny_metrics)
                
                # Extract vectors
                h_vec = self._extract_metrics_vector(hrm_metrics)
                t_vec = self._extract_metrics_vector(tiny_metrics)
                
                # Lock names on first example; align subsequent rows
                if i == 0:
                    hrm_names = list(h_vec.keys())
                    tiny_names = list(t_vec.keys())
                hrm_vectors.append(self._align_vector(h_vec, hrm_names))
                tiny_vectors.append(self._align_vector(t_vec, tiny_names))

                # Also build a row for PHOS later (we’ll align columns post-loop)
                base_row = {"node_id": triple.node_id}
                base_row.update({f"hrm.{k}": float(v) for k, v in h_vec.items()})
                base_row.update({f"tiny.{k}": float(v) for k, v in t_vec.items()})
                rows_for_df.append(base_row)
                
                # Progress hooks
                if ((i + 1) % log_every) == 0 or (i + 1) == total:
                    self.logger.log("ScoringProgress", {"processed": i + 1, "total": total})
                    if progress_cb:
                        try:
                            progress_cb(i + 1, total, None)
                        except Exception:
                            pass

                pbar.update(1)
                await asyncio.sleep(0)  # cooperative yield

        # Finalize timelines
        hrm_gif = await vpm_worker.finalize(hrm_timeline_id, f"vpm_phos_run_{hrm_timeline_id}.gif")
        tiny_gif = await vpm_worker.finalize(tiny_timeline_id, f"vpm_phos_run_{tiny_timeline_id}.gif")
        
        # Convert to matrices
        hrm_matrix_raw = np.array(hrm_vectors, dtype=np.float32)
        tiny_matrix_raw = np.array(tiny_vectors, dtype=np.float32)

        # Align by shared names (preserve HRM order)
        def _select_preferred_metric_indices(names: List[str], model: str, dims: List[str]) -> Dict[str, int]:
            """
            Return a map canonical_name -> column_index for:
              - 'aggregate'
              - each dim in dims (final score), canonicalized to the dim name
            Preference is strict exacts first, then loose patterns.
            """
            name_idx = {n: i for i, n in enumerate(names)}
            can_map: Dict[str, int] = {}

            def _find_exact(key: str):
                return name_idx.get(key)

            def _find_loose(patterns: List[str]):
                for i, n in enumerate(names):
                    s = n.lower()
                    if all(p in s for p in patterns):
                        return i
                return None

            # 1) aggregate
            agg_exact = _find_exact(f"{model}.aggregate")
            if agg_exact is None:
                agg_exact = _find_loose([model+".", "aggregate"])
            if agg_exact is not None:
                can_map["aggregate"] = agg_exact

            # 2) per-dimension final scores
            for d in dims:
                # strict preference order
                candidates = [
                    f"{model}.{d}.score",
                    f"{model}.{d}.aggregate",
                    f"{model}.{d}",
                ]
                idx = None
                for c in candidates:
                    if c in name_idx:
                        idx = name_idx[c]; break
                if idx is None:
                    # loose patterns as fallback (e.g. hrm.reasoning.score.v2)
                    idx = _find_loose([model+".", f".{d}", ".score"])
                    if idx is None:
                        idx = _find_loose([model+".", f".{d}", ".aggregate"])
                if idx is not None:
                    # canonical key is just the dimension name
                    can_map[d] = idx

            return can_map

        # Build preferred maps
        preferred_hrm = _select_preferred_metric_indices(hrm_names, "hrm", self.config.dimensions)
        preferred_tny = _select_preferred_metric_indices(tiny_names, "tiny", self.config.dimensions)

        # Shared canonical keys, preserving HRM order (aggregate first if present)
        canonical_order = ["aggregate"] + list(self.config.dimensions)
        shared_canon = [c for c in canonical_order if c in preferred_hrm and c in preferred_tny]

        if not shared_canon:
            # Dump names to help debugging on real payloads
            try:
                storage = self.container.get("gap_storage")
                raw_dir = storage.base_dir / run_id / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                import json
                with open(raw_dir / "debug_metric_names.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "hrm_names": hrm_names,
                        "tiny_names": tiny_names,
                        "preferred_hrm": preferred_hrm,
                        "preferred_tiny": preferred_tny,
                        "dims": self.config.dimensions,
                    }, f, indent=2)
            except Exception:
                pass
            raise RuntimeError(
                "No shared preferred metrics (aggregate + per-dimension .score). "
                "See raw/debug_metric_names.json for details."
            )

        # Slice matrices to shared_canon
        hrm_cols = [preferred_hrm[c] for c in shared_canon]
        tiny_cols = [preferred_tny[c] for c in shared_canon]
        hrm_matrix = hrm_matrix_raw[:, hrm_cols]
        tiny_matrix = tiny_matrix_raw[:, tiny_cols]

        # Final shared names are canonical (aggregate + dims)
        shared_names = shared_canon

        # Persist aligned matrices + names via storage
        storage = self.container.get("gap_storage")
        storage.save_matrix(hrm_matrix, shared_names, run_id, tag="hrm")
        storage.save_matrix(tiny_matrix, shared_names, run_id, tag="tiny")

        # Build rows_for_df as a DataFrame with canonical columns
        # Normalize: create hrm.<suffix> / tiny.<suffix> pairs from shared_names
        rows_for_df = []
        for row_i, triple in enumerate(all_triples):
            row = {"node_id": triple.node_id}
            for j, suffix in enumerate(shared_names):
                # suffix is either 'aggregate' or a dim like 'reasoning'
                if suffix == "aggregate":
                    row["hrm.aggregate"]  = float(hrm_matrix[row_i, j])
                    row["tiny.aggregate"] = float(tiny_matrix[row_i, j])
                else:
                    row[f"hrm.{suffix}.score"]  = float(hrm_matrix[row_i, j])
                    row[f"tiny.{suffix}.score"] = float(tiny_matrix[row_i, j])
            rows_for_df.append(row)

        df_rows = pd.DataFrame(rows_for_df)
        raw_paths = storage.save_rows_df(df_rows, run_id, name="rows_for_df")

        # Final progress tick
        if progress_cb:
            try:
                progress_cb(total, total, {"done": True})
            except Exception:
                pass

        # return aligned content (not raw)
        return {
            "hrm_vectors": hrm_matrix,
            "tiny_vectors": tiny_matrix,
            "hrm_names": shared_names,
            "tiny_names": shared_names,
            "hrm_gif": hrm_gif,
            "tiny_gif": tiny_gif,
            "triples_count": total,
            "rows_for_df_path": str(storage.base_dir / run_id / "raw" / "rows_for_df.parquet"),
            **raw_paths,
        }
    
    def _extract_metrics_vector(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract metrics vector from metrics payload."""
        vec = metrics.get("vector")
        if isinstance(vec, dict) and vec:
            return {k: float(v) for k, v in vec.items()}
        
        cols = metrics.get("columns")
        vals = metrics.get("values")
        if isinstance(cols, list) and isinstance(vals, list) and len(cols) == len(vals):
            return {str(c): float(v) for c, v in zip(cols, vals)}
        
        return {}
    
    def _align_vector(self, vector: Dict[str, float], target_names: List[str]) -> List[float]:
        """Align vector to target names, filling missing values with 0.0."""
        return [float(vector.get(name, 0.0)) for name in target_names]
