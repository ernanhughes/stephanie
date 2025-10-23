# stephanie/agents/phos.py
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import text
from tqdm import tqdm

from stephanie.agents.agentic_tree_search import \
    SolutionNode  # assuming you have this
from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import PIPELINE_RUN_ID
from stephanie.scoring.scorable import Scorable, ScorableFactory, ScorableType
from stephanie.services.scoring_service import ScoringService
from stephanie.services.workers.metrics_worker import MetricsWorkerInline
from stephanie.services.workers.vpm_worker import VPMWorkerInline
from stephanie.services.zeromodel_service import ZeroModelService

_logger = logging.getLogger(__name__)


class PhosAgent(BaseAgent):
    """
    PhōsAgent — Illuminates semantic structure by generating contrastive VPMs.
    Inline version (no NATS/inproc bus).
    """

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.run_id = context.get(PIPELINE_RUN_ID)
        goal_text = context["goal"]["goal_text"]
        self._goal_text = goal_text
        self.ref_goal_id = context["goal"].get("id", 0)

        # --- inline workers ---
        scoring_cfg = self.cfg.get("metrics", {})
        scorers = scoring_cfg.get("scorers", ["sicql", "mrq", "ebt"])
        dims = scoring_cfg.get("dimensions", ["alignment", "clarity", "implementability", "novelty", "relevance"])
        self.scorer: ScoringService = self.container.get("scoring")
        self.zm: ZeroModelService = self.container.get("zeromodel")
        metrics_worker = MetricsWorkerInline(self.scorer, scorers, dims)
        self.metric_names = []
        vpm_worker = VPMWorkerInline(self.zm, self.logger)

        # --- load bands ---
        good_rows = await self._gather_runs(self.ref_goal_id)
        good = [self._make_scorable(r["response"], 1.0, "good") for r in good_rows]
        medium_rows = self.memory.embedding.search_scorables_in_similarity_band(goal_text, ScorableType.RESPONSE, 0.15, 0.80, 300)
        medium = [self._make_scorable(r["text"], r["similarity"], "medium") for r in medium_rows]
        opposite_rows = self.memory.embedding.search_unrelated_scorables(goal_text, ScorableType.RESPONSE, top_k=300)
        opposite = [self._make_scorable(r["text"], r["score"], "opposite") for r in opposite_rows]

        datasets = {"good": good, "medium": medium, "opposite": opposite}

        # --- generate timelines inline ---
        results = {}
        for label, scorables in datasets.items():
            if not scorables:
                _logger.warning(f"[PhosAgent] ⚠️ Empty band: {label}")
                continue

            out_path = f"vpm_phos_run{self.run_id}_{label}.gif"
            self.zm.timeline_open(run_id=self.run_id)
            with tqdm(total=len(scorables), desc=f"[PhosAgent] {label} band") as pbar:
                for idx, scorable in enumerate(scorables):
                    node_id = self._make_numeric_id(self.run_id, label, idx)
                    metrics = await metrics_worker.score(scorable, goal_text, self.run_id)
                    self.metric_names = metrics["columns"]
                    await vpm_worker.append(self.run_id, node_id, metrics)
                    pbar.update(1)
                    await asyncio.sleep(0)  # yield loop

            final = await vpm_worker.finalize(self.run_id, out_path)
            results[label] = final
            _logger.info(f"[PhosAgent] ✅ Timeline closed for {label} → {out_path}")

        # --- differential analysis ---
        diffs = self._analyze_vpms(results)
        context["phos_outputs"] = diffs
        return context


    async def _gather_runs(self, goal_id: int) -> List[Dict]:
        """Pull ATS runs matching or near the goal."""
        q = text(f"""
            SELECT id, response_text, *
            FROM prompts
            WHERE goal_id = {goal_id}
            GROUP BY pipeline_run_id, id
            ORDER BY id DESC
            LIMIT 300;
        """)

        # Use the sessionmaker to create a new session
        with self.memory.session() as session:
            rows = session.execute(q).fetchall()

        return [
            {"response": self._strip_think_blocks(r.response_text)}
            for r in rows
        ]

    async def _render_vpms(
        self, datasets: Dict[str, List[Dict]]
    ) -> Dict[str, np.ndarray]:
        vpms = {}
        for label, data in datasets.items():
            metrics = []
            score = {}
            for i, row in enumerate(data):  # sample limit
                score = self.scorer.score(row["goal"], row["response"])

                metrics.append([i] + list(score.values()))
            matrix = np.array(metrics)
            self.zm.render_timeline_from_matrix(
                matrix=matrix,
                out_path=f"vpm_phos_{label}.gif",
                metric_names=["id"] + list(score.keys()),
            )
            vpms[label] = matrix
        return vpms

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

    def _strip_think_blocks(self, text: str) -> str:
        if not text:
            return text
        """Remove <think>...</think> sections from text."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _make_scorable(
        self, text: str, similarity: float, label: str
    ) -> Scorable:
        """
        Create a Scorable with similarity stored in its meta data.
        """
        sc = ScorableFactory.from_text(
            text=text,
            target_type=ScorableType.RESPONSE,
            meta={"similarity": similarity, "source_label": label},
        )
        return sc

    # --------------------------------------------------------------------------- #
    # 🧠 PLAN PROCESSOR — uses same logic as AgenticTreeSearch._process_plan
    # --------------------------------------------------------------------------- #
    async def _process_plan_for_phos(
        self,
        index: int,
        scorable: Scorable,
        label: str,
        context: Dict[str, Any],
        parent_node: Optional[SolutionNode] = None,
        node_type: str = "draft",
    ) -> SolutionNode:
        """
        Execute a text (scorable) as a plan and emit results to ZeroModel.
        Used to simulate the AgenticTreeSearch behavior for contrastive sets.
        """
        # 1. Compute the evaluation (metric + vector)

        plan = scorable.text
        ts = time.time()

        # 2. Create a minimal SolutionNode-like dict for emission
        node_id = self._make_numeric_id(self.run_id, label, index)
        _logger.debug(f"Processing plan {node_id} ({label}): {plan[:60]}...")
        node = {
            "id": node_id,
            "plan": plan,
            "summary": plan[:75],
            "output": plan,
            "timestamp": ts,
            "type": node_type,
            "label": label,
            "run_id": self.run_id,
        }

        # 4. Emit directly to ZeroModel timeline (if in-process mode)
        try:
            await self._timeline_sink("node", node)
        except Exception as e:
            _logger.warning("[PhosAgent] Emit failed (%s): %s", label, e)

        return node

    async def _simulate_band_processing(
        self,
        datasets: Dict[str, List[Dict[str, Any]]],
        context: Dict[str, Any], 
    ) -> Dict[str, str]:
        """
        Simulate AgenticTreeSearch execution for each contrastive dataset
        and produce three ZeroModel timelines (good / medium / opposite).
        """
        results = {}
        for label, scorables in datasets.items():
            run_id = self.run_id
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_run = str(self.run_id)[:8]
            out_path = f"vpm_phos_{timestamp}_run{safe_run}_{label}.gif"

            # ✅ Guard against empty or invalid bands early
            if not scorables:
                _logger.warning(
                    "[PhosAgent] ⚠️ No scorables found for %s, skipping band.", label
                )
                continue

            total = len(scorables)

            if total == 0:
                _logger.warning(
                    "[PhosAgent] ⚠️ No scorables found for %s", label
                )
                continue

            # 1️⃣ Open a fresh timeline stream
            try:
                self.zm.timeline_open(run_id=run_id)
                _logger.info(
                    f"Timeline opened for {label} ({total} scorables)"
                )
            except Exception as e:
                _logger.warning("[PhosAgent] Timeline open failed (%s): %s", label, e)

            # 2️⃣ Process each scorable with visible progress bar
            with tqdm(
                total=total, desc=f"[PhosAgent] {label} band", unit="node"
            ) as pbar:
                for idx, scorable in enumerate(scorables):
                    await self._process_plan_for_phos(
                        index=idx,
                        scorable=scorable,
                        label=label,
                        context=context,
                    )

                    pbar.update(1)

                    # Structured progress event for dashboards / metrics
                    if idx % 10 == 0 or idx == total - 1:
                        progress_ratio = (idx + 1) / total
                        _logger.info(
                            json.dumps(
                                {
                                    "event": "PhosProgress",
                                    "label": label,
                                    "run_id": self.run_id,
                                    "completed": idx + 1,
                                    "total": total,
                                    "percent": round(progress_ratio * 100, 1),
                                }
                            )
                        )
                    await asyncio.sleep(
                        0.1
                    )  # small delay for event 
                    await self.memory.bus.flush()
                    await asyncio.sleep(2.0)  # give worker loop time to process
                    await self._wait_for_timeline_ready(run_id, expected=len(scorables), timeout=20.0)


            # 3️⃣ Close the stream and render to GIF
            try:
                await self._wait_for_timeline_fill(run_id, min_rows=5)
                # 🧩 Wait for bus to deliver all metric rows
                await self.memory.bus.flush()
                await self._wait_for_timeline_ready(run_id, expected=len(scorables), timeout=15.0)

                # 🧠 Finalize timeline only after rows arrive
                final_res = await self.zm.timeline_finalize(run_id=run_id, out_path=out_path)

                # ✅ Only store valid results that actually contain a matrix
                if (
                    final_res
                    and isinstance(final_res, dict)
                    and final_res.get("matrix") is not None
                ): 
                    results[label] = final_res
                    self.metric_names = final_res.get("metric_names", [])
                    _logger.info(
                        "[PhosAgent] ✅ Timeline closed for %s → %s", label, out_path
                    )
                else:
                    _logger.warning(
                        "[PhosAgent] ⚠️ No valid matrix for %s, skipping analysis.", label
                    )
            except Exception as e:
                _logger.warning(
                    "[PhosAgent] Timeline close failed (%s): %s", label, e
                )

        return results

    async def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        """
        Emit event to ZeroModel + EmitBroadcaster (if configured).
        """
        run_id = payload.get("run_id")
        # 1. Send to ZeroModel timeline directly
        try:
            if event == "node":
                node = payload.get("node", payload)
                node_id = node.get("id")
                key = (self.run_id, node_id)
                if key in self._published_nodes:
                    return  # already published, skip to avoid loop
                self._published_nodes.add(key)

                vector = payload.get("vector", {})
                metrics_columns = list(vector.keys())
                metrics_values = list(vector.values())
                if len(metrics_columns):
                    self.zm. OK (
                        run_id=run_id,
                        metrics_columns=metrics_columns,
                        metrics_values=metrics_values,
                    )
            elif event == "report":
                await self.zm.timeline_finalize(run_id)
        except Exception as e:
            self.logger.warning(
                f"[PhosAgent] ZeroModel timeline emit failed: {e}"
            )

        # 2. Notify EmitBroadcaster
        if self.emit_cb:
            try:
                await self.emit_cb(event, payload)
            except Exception as e:
                self.logger.debug(f"[PhosAgent] EmitBroadcaster failed: {e}")

    async def _timeline_sink(
        self, event: str, payload: Dict[str, Any]
    ) -> None:
        """Centralized event sink for ZeroModel + metrics bus."""
        if not self.run_id:
            return

        try:
            if event == "node":
                node = payload.get("node", payload)
                # emit into ZeroModel’s in-process timeline if available
                try:
                    await self._emit("node", node)
                except Exception as e:
                    self.logger.warning(
                        f"[PhosAgent] ZeroModel emit failed: {e}"
                    )

                # publish metrics job for async worker
                _logger.debug(
                    f"[PhosAgent] -> 'arena.metrics.request' job for node={node}"
                )

                await self.memory.bus.publish(
                    subject="arena.metrics.request",
                    payload={
                        "run_id": self.run_id,
                        "node_id": node.get("id"),
                        "parent_id": node.get("parent_id"),
                        "action_type": node.get("type", "draft"),
                        "goal_text": self._goal_text,
                        "prompt_text": node.get("plan"),
                        "best_metric": node.get("metric"),
                        "bug": bool(node.get("bug", False)),
                        "ts_enqueued": time.time(),
                    },
                )

            elif event == "report":
                await self.memory.bus.publish(
                    subject="arena.ats.report",
                    payload={"run_id": self.run_id},
                )

        except Exception as e:
            _logger.warning("[PhosAgent] Timeline sink error: %s", e)

    def _make_numeric_id(self, run_id: Any, label: str, index: int) -> int:
        """
        Build a sortable integer ID:
            [run_code][label_code][index:04d]
        Ensures ascending order by run → label → index.
        """
        # 1️⃣ derive a 4-digit run code from the run_id (UUID-safe)
        if isinstance(run_id, str):
            run_hash = abs(hash(run_id)) % 10_000
        else:
            run_hash = int(run_id) % 10_000

        # 2️⃣ map label to a stable small integer
        label_code = {"good": 1, "medium": 2, "opposite": 3}.get(label, 9)

        # 3️⃣ combine into a single sortable integer
        numeric_id = int(f"{run_hash:04d}{label_code}{index:04d}")
        return numeric_id

    def _emit_to_logger(self, event: str, payload: Dict[str, Any]) -> None:
        _logger.debug(f"Phos::{event} --> {payload}")

    async def _wait_for_bus_drain(self, timeout: float = 10.0):
        """
        Wait for the bus to finish processing all messages for the current run.
        Returns early if the bus drains successfully.
        """
        try:
            _logger.info(f"[PhosAgent] Waiting for bus to drain (run_id={self.run_id})...")
            ok = await self.memory.bus.flush(timeout=timeout)
            if ok:
                _logger.info("[PhosAgent] ✅ Bus flush complete.")
            else:
                _logger.warning("[PhosAgent] ⚠️ Bus flush incomplete (timeout or unsupported).")
        except Exception as e:
            _logger.warning(f"[PhosAgent] Bus flush failed: {e}")


    async def _wait_for_timeline_fill(self, run_id: str, min_rows: int = 5, timeout: float = 10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            sess = self.zm._sessions.get(run_id)
            if sess and len(sess.rows) >= min_rows:
                return True
            await asyncio.sleep(0.2)
        _logger.warning(f"[PhosAgent] Timed out waiting for timeline fill ({min_rows} rows).")
        return False


    async def _wait_for_timeline_ready(self, run_id: str, expected: int, timeout: float = 10.0):
        """
        Wait until the ZeroModel timeline for `run_id` has at least `expected` rows
        or until timeout. Returns True if successful.
        """
        start = time.time()
        while time.time() - start < timeout:
            sess = self.zm._sessions.get(run_id)
            count = len(sess.rows) if sess else 0
            if count >= expected:
                _logger.info(f"[PhosAgent] ✅ Timeline filled ({count}/{expected}) for run {run_id}.")
                return True
            await asyncio.sleep(0.25)
        _logger.warning(f"[PhosAgent] ⚠️ Timed out waiting for timeline fill ({count}/{expected})")
        return False

    async def _wait_for_workers_ready(self, timeout: float = 5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            ok1 = getattr(self.metrics_worker, "_running", False)
            ok2 = getattr(self.vpm_worker, "_running", False)
            if ok1 and ok2:
                _logger.info("[PhosAgent] 🟢 Workers ready.")
                return True
            await asyncio.sleep(0.2)
        _logger.warning("[PhosAgent] ⚠️ Workers not ready after timeout.")
        return False
