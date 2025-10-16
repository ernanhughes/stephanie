# stephanie/components/gap/processors/scoring.py
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.services.workers.metrics_worker import MetricsWorkerInline
from stephanie.services.workers.vpm_worker import VPMWorkerInline
from stephanie.components.gap.models import GapConfig, TripleSample, GapRunManifest

logger = logging.getLogger(__name__)


class ScoringProcessor:
    """Handles sample preparation, scoring, alignment, and timeline generation."""

    def __init__(self, config: GapConfig, container, logger):
        self.config = config
        self.container = container
        self.logger = logger

    # ---------------------------
    # Public API
    # ---------------------------
    async def prepare_samples(self, dimensions: List[str], memory) -> Dict[str, List[TripleSample]]:
        """Collect and deduplicate training samples across dimensions."""
        from stephanie.scoring.training.preference_pair_builder import PreferencePairBuilder

        pair_builder = PreferencePairBuilder(memory, self.logger)
        triples_by_dim: Dict[str, List[TripleSample]] = {}

        for dimension in dimensions:
            pairs = pair_builder.get_training_pairs_by_dimension(dimension=dimension, limit=None)
            samples = pairs.get(dimension, [])
            triples = self._flatten_samples(samples, dimension)
            triples_by_dim[dimension] = triples

        return self._deduplicate_samples(triples_by_dim)

    async def execute_scoring(
        self,
        triples_data: Dict[str, List[TripleSample]],
        run_id: str,
        manifest: GapRunManifest | None = None,
    ) -> Dict[str, Any]:
        """
        Execute model scoring, produce aligned HRM/Tiny matrices, rows_for_df, and timelines.
        Returns file paths only (no heavy arrays), suitable for downstream processors.
        """
        scoring_service = self.container.get("scoring")
        zm = self.container.get("zeromodel")

        hrm_worker = MetricsWorkerInline(scoring_service, self.config.hrm_scorers, self.config.dimensions)
        tiny_worker = MetricsWorkerInline(scoring_service, self.config.tiny_scorers, self.config.dimensions)
        vpm_worker = VPMWorkerInline(zm, self.logger)

        return await self._score_all_triples(triples_data, hrm_worker, tiny_worker, vpm_worker, run_id)

    # ---------------------------
    # Internals: sample handling
    # ---------------------------
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
                    triples.append(
                        TripleSample(
                            node_id=f"{dimension}|{i:06d}",
                            dimension=dimension,
                            goal_text=goal_text,
                            output_text=output_text,
                            target_value=float(value),
                            fingerprint=self._compute_fingerprint(goal_text, output_text),
                        )
                    )
                continue

            # pairwise
            if all(k in sample for k in ("output_a", "output_b", "value_a", "value_b")):
                for suf in ("a", "b"):
                    output_text = (sample.get(f"output_{suf}") or "").strip()
                    value = sample.get(f"value_{suf}")
                    if goal_text and output_text and value is not None:
                        triples.append(
                            TripleSample(
                                node_id=f"{dimension}|{i:06d}_{suf}",
                                dimension=dimension,
                                goal_text=goal_text,
                                output_text=output_text,
                                target_value=float(value),
                                fingerprint=self._compute_fingerprint(goal_text, output_text),
                            )
                        )
                continue

            # explicit MRQ/HRM form
            if ("goal_text" in sample and "scorable_text" in sample and ("target_score" in sample or "score" in sample)):
                out = (sample.get("scorable_text") or "").strip()
                val = sample.get("target_score", sample.get("score"))
                if goal_text and out and val is not None:
                    triples.append(
                        TripleSample(
                            node_id=f"{dimension}|{i:06d}",
                            dimension=dimension,
                            goal_text=goal_text,
                            output_text=out,
                            target_value=float(val),
                            fingerprint=self._compute_fingerprint(goal_text, out),
                        )
                    )

        return triples

    def _deduplicate_samples(self, triples_by_dim: Dict[str, List[TripleSample]]) -> Dict[str, List[TripleSample]]:
        """Deduplicate samples across dimensions using configured policy."""
        policy = self.config.dedupe_policy
        if policy == "first_wins":
            return self._deduplicate_first_wins(triples_by_dim)
        if policy == "round_robin":
            return self._deduplicate_round_robin(triples_by_dim)
        raise ValueError(f"Unknown deduplication policy: {policy}")

    def _deduplicate_first_wins(self, triples_by_dim: Dict[str, List[TripleSample]]) -> Dict[str, List[TripleSample]]:
        seen: set[str] = set()
        deduped = {d: [] for d in triples_by_dim.keys()}
        for d, triples in triples_by_dim.items():
            for t in triples:
                if t.fingerprint in seen:
                    continue
                deduped[d].append(t)
                seen.add(t.fingerprint)
            if self.config.per_dim_cap and len(deduped[d]) > self.config.per_dim_cap:
                deduped[d] = deduped[d][: self.config.per_dim_cap]
        return deduped

    def _deduplicate_round_robin(self, triples_by_dim: Dict[str, List[TripleSample]]) -> Dict[str, List[TripleSample]]:
        pool: Dict[str, TripleSample] = {}
        for d, triples in triples_by_dim.items():
            for t in triples:
                if t.fingerprint not in pool:
                    pool[t.fingerprint] = t
        dims = list(triples_by_dim.keys())
        out = {d: [] for d in dims}
        i = 0
        for fp, t in pool.items():
            target = dims[i % len(dims)]
            if self.config.per_dim_cap is None or len(out[target]) < self.config.per_dim_cap:
                out[target].append(t)
                i += 1
        return out

    def _compute_fingerprint(self, goal_text: str, output_text: str) -> str:
        payload = (goal_text.strip() + "\n␟\n" + output_text.strip()).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    # ---------------------------
    # Internals: scoring/alignment
    # ---------------------------
    async def _score_all_triples(
        self,
        triples_data: Dict[str, List[TripleSample]],
        hrm_worker,
        tiny_worker,
        vpm_worker,
        run_id: str,
    ) -> Dict[str, Any]:
        """Score triples with both models, align on a shared schema, persist artifacts."""
        # Flatten across dims
        all_triples: List[TripleSample] = []
        for lst in triples_data.values():
            all_triples.extend(lst)

        # Open timelines
        zm = self.container.get("zeromodel")
        hrm_tid = f"{run_id}_hrm"
        tiny_tid = f"{run_id}_tiny"
        zm.timeline_open(run_id=hrm_tid)
        zm.timeline_open(run_id=tiny_tid)

        # Buffers
        hrm_names_0: List[str] = []
        tiny_names_0: List[str] = []
        hrm_rows: List[List[float]] = []
        tiny_rows: List[List[float]] = []

        # Progress
        total = len(all_triples)
        with tqdm(total=total, desc="[GAP] Scoring triples", unit="turn") as pbar:
            for i, triple in enumerate(all_triples):
                scorable = Scorable(triple.output_text, ScorableType.CONVERSATION_TURN)

                # Score both
                hrm_metrics = await hrm_worker.score(scorable, triple.goal_text, hrm_tid)
                tiny_metrics = await tiny_worker.score(scorable, triple.goal_text, tiny_tid)

                # Append to timelines
                await vpm_worker.append(hrm_tid, triple.node_id, hrm_metrics)
                await vpm_worker.append(tiny_tid, triple.node_id, tiny_metrics)

                # Extract vectors (flat dict name->value)
                h_vec = self._extract_metrics_vector(hrm_metrics)
                t_vec = self._extract_metrics_vector(tiny_metrics)

                if i == 0:
                    hrm_names_0 = list(h_vec.keys())
                    tiny_names_0 = list(t_vec.keys())

                # Align per first-row schema for each model (fill missing with 0)
                hrm_rows.append(self._align_vector(h_vec, hrm_names_0))
                tiny_rows.append(self._align_vector(t_vec, tiny_names_0))

                # periodic log
                if ((i + 1) % self.config.progress_log_every) == 0 or (i + 1) == total:
                    self.logger.log("ScoringProgress", {"run_id": run_id, "processed": i + 1, "total": total})
                pbar.update(1)
                await asyncio.sleep(0)

        # Finalize timelines (worker may return str OR dict)
        hrm_gif_obj = await vpm_worker.finalize(hrm_tid, f"vpm_phos_run_{hrm_tid}.gif")
        tiny_gif_obj = await vpm_worker.finalize(tiny_tid, f"vpm_phos_run_{tiny_tid}.gif")
        hrm_gif = hrm_gif_obj.get("output_path") if isinstance(hrm_gif_obj, dict) else hrm_gif_obj
        tiny_gif = tiny_gif_obj.get("output_path") if isinstance(tiny_gif_obj, dict) else tiny_gif_obj

        # Convert to arrays
        hrm_matrix_raw = np.array(hrm_rows, dtype=np.float32) if hrm_rows else np.zeros((0, 0), np.float32)
        tiny_matrix_raw = np.array(tiny_rows, dtype=np.float32) if tiny_rows else np.zeros((0, 0), np.float32)

        # ---- Alignment to a shared schema (CRITICAL) -------------------------
        # Normalize names to suffixes (strip leading "hrm."/"tiny." if present).
        def _suffix(names: List[str]) -> List[str]:
            out = []
            for n in names:
                if n.startswith("hrm."):
                    out.append(n.split(".", 1)[1])
                elif n.startswith("tiny."):
                    out.append(n.split(".", 1)[1])
                else:
                    out.append(n)
            return out

        hrm_suffix = _suffix(hrm_names_0)
        tiny_suffix = _suffix(tiny_names_0)

        # intersection preserves HRM order (deterministic)
        shared_suffix = [n for n in hrm_suffix if n in set(tiny_suffix)]
        if not shared_suffix:
            raise RuntimeError("No shared metric suffixes between HRM and Tiny. Cannot align for delta/frontier.")

        hrm_idx = [hrm_suffix.index(n) for n in shared_suffix]
        tiny_idx = [tiny_suffix.index(n) for n in shared_suffix]

        hrm_matrix = hrm_matrix_raw[:, hrm_idx]
        tiny_matrix = tiny_matrix_raw[:, tiny_idx]

        # Persist matrices + shared names via storage
        storage = self.container.get("storage")
        hrm_pack = storage.save_matrix(hrm_matrix, shared_suffix, run_id, tag="hrm")
        tiny_pack = storage.save_matrix(tiny_matrix, shared_suffix, run_id, tag="tiny")

        # Build rows_for_df for PHOS: node_id + hrm.{suffix} / tiny.{suffix}
        rows_for_df: List[Dict[str, Any]] = []
        for r, triple in enumerate(all_triples):
            row = {"node_id": triple.node_id}
            for c, name in enumerate(shared_suffix):
                row[f"hrm.{name}"] = float(hrm_matrix[r, c])
                row[f"tiny.{name}"] = float(tiny_matrix[r, c])
            rows_for_df.append(row)

        raw_dir = storage.base_dir / run_id / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        rows_path_pq = raw_dir / "rows_for_df.parquet"
        rows_path_csv = raw_dir / "rows_for_df.csv"
        pd.DataFrame(rows_for_df).to_parquet(rows_path_pq, index=False)
        pd.DataFrame(rows_for_df).to_csv(rows_path_csv, index=False)

        # Return file paths for downstream processors
        return {
            "hrm_matrix_path": hrm_pack.get("matrix_path") or hrm_pack.get("matrix"),
            "tiny_matrix_path": tiny_pack.get("matrix_path") or tiny_pack.get("matrix"),
            "metric_names_path": hrm_pack.get("names_path") or hrm_pack.get("names"),
            "hrm_timeline_gif": hrm_gif,
            "tiny_timeline_gif": tiny_gif,
            "rows_for_df_path": str(rows_path_pq),
            "triples_count": len(all_triples),
        }

    # ---------------------------
    # Metric extraction helpers
    # ---------------------------
    def _extract_metrics_vector(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract metrics vector from payload; supports {'vector':{...}} or columns/values."""
        vec = metrics.get("vector")
        if isinstance(vec, dict) and vec:
            return {str(k): float(v) for k, v in vec.items()}

        cols, vals = metrics.get("columns"), metrics.get("values")
        if isinstance(cols, list) and isinstance(vals, list) and len(cols) == len(vals):
            try:
                return {str(c): float(v) for c, v in zip(cols, vals)}
            except Exception:
                return {}
        return {}

    def _align_vector(self, vector: Dict[str, float], target_names: List[str]) -> List[float]:
        """Align a dict vector to a fixed name order, filling missing with 0.0."""
        return [float(vector.get(n, 0.0)) for n in target_names]
