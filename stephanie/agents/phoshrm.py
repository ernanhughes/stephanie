# stephanie/agents/phoshrm.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")

from tqdm import tqdm  # <-- NEW

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.scoring.training.preference_pair_builder import PreferencePairBuilder
from stephanie.utils.id_utils import make_numeric_id
import asyncio
from stephanie.services.workers.vpm_worker import VPMWorkerInline
from stephanie.services.workers.metrics_worker import MetricsWorkerInline

import logging
_logger = logging.getLogger(__name__)


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
        self.hrm_scorers = list(cfg.get("hrm_scorers", ["hrm", "sicql", "mrq"]))
        self.tiny_scorers = list(cfg.get("tiny_scorers", ["tiny"]))
        self.out_dir = Path(cfg.get("out_dir", "data/vpm"))
        self.interleave = bool(cfg.get("interleave", False))
        self.progress_log_every = int(cfg.get("progress_log_every", 25))  # NEW

    async def run(self, context: dict) -> dict:
        eval_stats: Dict[str, Any] = {}

        # 1) Prepare tools
        scoring_service = self.container.get("scoring")  # ScoringService
        zm = self.container.get("zeromodel")
        hrm_worker = MetricsWorkerInline(scoring_service, self.hrm_scorers, self.dimensions)
        tiny_worker = MetricsWorkerInline(scoring_service, self.tiny_scorers, self.dimensions)
        vpm_worker = VPMWorkerInline(zm, self.logger)

        pipeline_run_id = context.get("pipeline_run_id") or "phoshrm"
        hrm_run_id = f"{pipeline_run_id}_hrm"
        tiny_run_id = f"{pipeline_run_id}_tiny"
        zm.timeline_open(run_id=hrm_run_id)
        zm.timeline_open(run_id=tiny_run_id)

        # 2) Gather all samples so we know totals (for tqdm bars)
        pair_builder = PreferencePairBuilder(self.memory, self.logger)
        triples_by_dim: Dict[str, List[Tuple[str, str, float]]] = {}
        total_triples = 0

        dimension = self.dimensions[0]
        pairs_by_dim = pair_builder.get_training_pairs_by_dimension(dimension=dimension)
        triples = _flatten_samples_for_eval(pairs_by_dim.get(dimension, []))
        triples_by_dim[dimension] = triples
        total_triples += len(triples)

        # Guard (nothing to do)
        if total_triples == 0:
            self.logger.log("PhosHRMNoSamples", {"dimensions": self.dimensions})
            context[self.output_key] = {"msg": "no samples"}
            return context

        # 3) Main scoring loop with tqdm (overall + per-dimension)
        processed = 0
        with tqdm(total=total_triples, desc="[PhosHRM] overall", unit="sample") as pbar_all:
            for d in self.dimensions:
                triples = triples_by_dim.get(d, [])
                if not triples:
                    tqdm.write(f"[PhosHRM] ⚠️ No samples for dimension: {d}")
                    continue

                with tqdm(total=len(triples), desc=f"[PhosHRM] {d}", unit="node", leave=False) as pbar_dim:
                    for idx, (goal_text, output_text, target_val) in enumerate(triples):
                        node_id = make_numeric_id(pipeline_run_id, d, idx)
                        scorable = Scorable(output_text, ScorableType.CONVERSATION_TURN)

                        # score (HRM & Tiny)
                        hrm_metrics = await hrm_worker.score(scorable, goal_text, hrm_run_id)
                        tiny_metrics = await tiny_worker.score(scorable, goal_text, tiny_run_id)

                        # append to VPM timelines
                        await vpm_worker.append(hrm_run_id, node_id, hrm_metrics)
                        await vpm_worker.append(tiny_run_id, node_id, tiny_metrics)

                        # progress
                        processed += 1
                        pbar_dim.update(1)
                        pbar_all.update(1)

                        if (processed % self.progress_log_every) == 0 or processed == total_triples:
                            # structured progress for logs/dashboards
                            self.logger.log("PhosHRMProgress", {
                                "run_id": pipeline_run_id,
                                "dimension": d,
                                "processed": processed,
                                "total": total_triples,
                                "percent": round(100.0 * processed / total_triples, 2),
                            })

                        # give the event loop a breath (keeps UI responsive)
                        # and avoids starving any in-process workers
                        await asyncio.sleep(0)

        # 4) Finalize timelines (renders GIFs)
        hrm_gif = f"vpm_phos_run_{hrm_run_id}.gif"
        tiny_gif = f"vpm_phos_run_{tiny_run_id}.gif"
        await vpm_worker.finalize(hrm_run_id, hrm_gif)
        await vpm_worker.finalize(tiny_run_id, tiny_gif)
        tqdm.write(f"[PhosHRM] ✅ Timelines closed → {hrm_gif} / {tiny_gif}")

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
            meta = self.zm.generate_epistemic_field(
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
            meta = self.zm.generate_epistemic_field(
                pos_matrices=[good_mat],
                neg_matrices=[mixed_mat],
                output_dir=str(output_dir / "good_vs_mixed"),
                aggregate=True,
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
