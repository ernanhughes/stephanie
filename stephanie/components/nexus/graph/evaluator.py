from __future__ import annotations

import asyncio
import statistics as stats
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from stephanie.components.vericot.vericot_verifier import \
    VericotVerifier  # your final version
from stephanie.core.manifest import ManifestManager
from stephanie.memory.memcube_store import MemCubeStore
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_processor import ScorableProcessor
from stephanie.services.epistemic_guard_service import EpistemicGuardService
from stephanie.services.scoring_service import ScoringService


@dataclass
class EvalWeights:
    solve: float = 0.5
    faithful: float = 0.2
    uncert_gain: float = 0.2
    support_eff: float = 0.1


class GraphAugmentEvaluator:
    """
    Evaluate Δ(d | G, H) = J(G ∪ {d}, H) - J(G, H)
    Assumes your pipeline has a 'solve(H)' that uses the current graph/index.
    """

    def __init__(
        self, container, memory, logger, weights: EvalWeights = EvalWeights()
    ):
        self.c = container
        self.m = memory
        self.log = logger
        self.w = weights

        self.sp: ScorableProcessor = ScorableProcessor(
            container, memory, logger
        )
        self.eg: EpistemicGuardService = self.c.get("ep_guard")
        self.vericot: VericotVerifier = self.c.get(
            "vericot"
        )  # register if not yet
        self.scoring: ScoringService = self.c.get("scoring")
        self.memcubes: MemCubeStore = self.m.memcubes
        self.manifest = ManifestManager(base_root="data/graph_aug_runs")

        # you likely have a Nexus adapter; keep it abstract here:
        self.nexus = self.c.get("nexus_graph")

    # ---- public -------------------------------------------------------------
    async def evaluate_candidate(
        self,
        run_id: str,
        hypothesis: Dict[str, Any],
        queries: List[Dict[str, Any]],
        candidate_doc: Dict[str, Any],  # raw doc dict or Scorable-like
        *,
        accept_threshold: float = 0.0,
        n_bootstrap: int = 3,
    ) -> Dict[str, Any]:
        """
        Returns a dict with Δ and component deltas; also persists a MemCube "graph_gain".
        """
        self.manifest.start_run(
            run_id=run_id, dataset="adhoc", models={"solve": "default"}
        )
        try:
            # 0) Baseline score J0
            J0, parts0 = await self._score_graph(hypothesis, queries)

            # 1) Apply candidate (add node + edges)
            node_id = await self._add_candidate_to_graph(candidate_doc)
            try:
                # 2) Score with candidate: small bootstrap for judge variance
                Js, Ps = [], []
                for _ in range(max(1, n_bootstrap)):
                    J1, parts1 = await self._score_graph(hypothesis, queries)
                    Js.append(J1)
                    Ps.append(parts1)
                J1 = float(stats.mean(Js))
                parts1 = self._mean_parts(Ps)

                dJ = J1 - J0
                accept = dJ > accept_threshold

                result = {
                    "run_id": run_id,
                    "hypothesis": hypothesis.get("id")
                    or hypothesis.get("text")[:80],
                    "candidate_id": node_id,
                    "delta": dJ,
                    "baseline": {"J": J0, **parts0},
                    "with_candidate": {"J": J1, **parts1},
                    "accept": accept,
                }

                # 3) Persist to MemCube under dimension "graph_gain"
                if self.memcubes:
                    payload = {
                        "hypothesis_id": hypothesis.get("id"),
                        "candidate_node": node_id,
                        "delta": dJ,
                        "baseline": parts0,
                        "with_candidate": parts1,
                        "weights": self.w.__dict__,
                    }
                    self.memcubes.store_calibration(
                        kind="graph_gain",
                        payload=payload,
                        source="augment",
                        version="v1",
                        sensitivity="internal",
                    )

                # Manifest drop
                self.manifest.finish_run(run_id, result=result)
                return result
            finally:
                # optionally roll back the candidate to keep G clean
                await self._remove_candidate_from_graph(node_id)
        except Exception as e:
            self.log.error(f"[GraphAugment] failed: {e}")
            self.manifest.finish_run(run_id, result={"error": str(e)})
            raise

    # ---- internals ----------------------------------------------------------
    async def _score_graph(
        self, H: Dict[str, Any], queries: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Runs your solve on queries tied to H and aggregates parts:
          - solve_score: chat judge avg
          - faithful: (1 - contradictions) + EG faithfulness
          - uncert_gain: -EBT_energy + SICQL_advantage
          - support_eff: -unsat_core_size
        """
        solve_scores, faithfuls, energies, advantages, core_sizes = (
            [],
            [],
            [],
            [],
            [],
        )

        for q in queries:
            # Your existing "Solve" path should use current graph/index internally.
            # If you don’t have a single entry point, call your agents here.
            out = await self.c.get("solve_runner").solve(H, q)

            # Chat judge (0..100)
            solve_scores.append(float(out["judge_score"]))

            # EpistemicGuard metrics
            eg = out.get("eg_metrics", {}) or {}
            faithfuls.append(float(eg.get("faithfulness", 0.0)))

            # EBT/SICQL
            energies.append(float(out.get("ebt_energy", 0.0)))
            advantages.append(float(out.get("sicql_advantage", 0.0)))

            # VERICOT: minimal premises / contradictions
            vc = out.get("vericot", {}) or {}
            core_sizes.append(float(vc.get("unsat_core_size", 0.0)))

            await asyncio.sleep(0)

        # aggregate parts
        solve_score = float(stats.mean(solve_scores)) if solve_scores else 0.0
        faithful = float(stats.mean(faithfuls)) if faithfuls else 0.0
        uncert_gain = (0.0 - float(stats.mean(energies))) + float(
            stats.mean(advantages or [0.0])
        )
        support_eff = 0.0 - float(stats.mean(core_sizes or [0.0]))

        J = (
            self.w.solve * solve_score
            + self.w.faithful * faithful
            + self.w.uncert_gain * uncert_gain
            + self.w.support_eff * support_eff
        )
        parts = {
            "solve_score": solve_score,
            "faithful": faithful,
            "uncert_gain": uncert_gain,
            "support_eff": support_eff,
        }
        return float(J), parts

    async def _add_candidate_to_graph(self, doc: Dict[str, Any]) -> str:
        """
        Adds a doc node via ScorableProcessor → creates edges (entities/similarities/citations).
        Returns node_id.
        """
        sc = Scorable.from_dict(doc) if not isinstance(doc, Scorable) else doc
        # hydrate domains/entities/embeddings (fast)
        await self.sp.process(sc, context={"pipeline_run_id": "graph-augment"})
        # ask Nexus to insert node + heuristic edges
        node_id = await self.nexus.add_document_node(sc)
        await self.nexus.link_new_node(node_id)  # e.g., entity/cosine edges
        return node_id

    async def _remove_candidate_from_graph(self, node_id: str) -> None:
        try:
            await self.nexus.remove_node(node_id)
        except Exception:
            pass

    @staticmethod
    def _mean_parts(list_of_parts: List[Dict[str, float]]) -> Dict[str, float]:
        keys = list(list_of_parts[0].keys()) if list_of_parts else []
        return {
            k: float(stats.mean([p[k] for p in list_of_parts])) for k in keys
        }
