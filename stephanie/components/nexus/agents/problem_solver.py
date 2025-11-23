from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.vericot.vericot_verifier import VericotVerifier
from stephanie.core.manifest import ManifestManager
from stephanie.memory.memcube_store import MemCubeStore
from stephanie.scoring.metrics.scorable_processor import ScorableProcessor
from stephanie.services.epistemic_guard_service import EpistemicGuardService
from stephanie.services.scoring_service import ScoringService


@dataclass
class Goal:
    id: Optional[str]
    text: str
    queries: Optional[List[str]] = None
    domain: Optional[str] = None

@dataclass
class SolverWeights:
    solve: float = 0.5
    faithful: float = 0.25
    uncert: float = 0.15   # -EBT energy + SICQL advantage
    support_eff: float = 0.10  # -VERICOT unsat core size

class ProblemSolver(BaseAgent):
    """
    Goal-conditioned solve over a Nexus graph of Scorables.
    Stages: focus_subgraph → evidence_selection → reason → verify → score.
    """

    def __init__(self, container, memory, logger, *, weights: SolverWeights = SolverWeights()):
        super().__init__(container=container, memory=memory, logger=logger)
        self.w = weights

        # Required services (use container so you can swap implementations)
        self.sp: ScorableProcessor = container.get("scorable_processor")
        self.eg: EpistemicGuardService = container.get("ep_guard")
        self.vericot: VericotVerifier = container.get("vericot")
        self.scoring: ScoringService = container.get("scoring")
        self.memcubes: MemCubeStore = memory.memcubes

        # Graph + LLM adapters you provide (see adapter section)
        self.nexus = container.get("nexus_graph")
        self.llm = container.get("llm")  # any callable: (prompt:str) -> str

        self.manifest = ManifestManager(base_root="data/nexus_goal_runs")

        # Embedding interface (already in your memory)
        self.embed = memory.embedding

    # ------------------ public API ------------------

    async def solve(self, goal: Goal, *, run_id: str, k: int = 12, hops: int = 2) -> Dict[str, Any]:
        """
        End-to-end goal solve. Returns dict with answer, evidence, metrics, artifacts.
        """
        self.manifest.start_run(run_id=run_id, dataset="adhoc", models={"llm":"default"})
        t0 = time.perf_counter()

        try:
            # 1) Focus subgraph (goal-conditioned)
            seeds = await self._seed_nodes(goal, top_k=k)
            subgraph = await self._expand(seeds, hops=hops)

            # 2) Evidence selection
            evid = await self._select_evidence(goal, subgraph, target_k=min(16, len(subgraph)))

            # 3) Compose context + Reason
            answer_text, gen_meta = await self._reason(goal, evid)

            # 4) Verify & (optionally) repair
            eg_metrics, vc_metrics, artifacts = await self._verify(goal, answer_text, evid)

            # 5) Score (judge, energy, advantage, support)
            metrics = await self._score(goal, answer_text, evid, eg_metrics, vc_metrics)

            result = {
                "run_id": run_id,
                "goal": {"id": goal.id, "text": goal.text},
                "answer": answer_text,
                "evidence": [{"id": e["id"], "title": e.get("title"), "score": e.get("score")} for e in evid],
                "metrics": metrics,
                "artifacts": artifacts,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

            # Persist a MemCube “solve” record (optional but useful)
            if self.memcubes:
                payload = {
                    "goal_id": goal.id,
                    "metrics": metrics,
                    "evidence_ids": [e["id"] for e in evid],
                }
                self.memcubes.store_calibration(kind="nexus_solve", payload=payload, source="solve", sensitivity="internal")

            self.manifest.finish_run(run_id, result)
            return result

        except Exception as e:
            self.logger.error(f"[NexusGoalSolver] failed: {e}")
            self.manifest.finish_run(run_id, {"error": str(e)})
            raise

    # ------------------ internals ------------------

    async def _seed_nodes(self, goal: Goal, top_k: int) -> List[Dict[str, Any]]:
        """Pick seed nodes by embedding similarity & (optional) domain match."""
        g_emb = self.embed.get_or_create(goal.text)
        # Adapter: nexus.search_by_embedding returns list of {"id","score",...}
        candidates = await self.nexus.search_by_embedding(g_emb, top_k=top_k * 3)

        # Optional domain filter
        if goal.domain:
            candidates = [c for c in candidates if goal.domain in (c.get("domains") or [])]

        # Return top_k after lightweight rerank with entity/keyword overlap
        seeds = []
        for c in candidates:
            sc = await self.nexus.get_scorable(c["id"])
            over = self._token_overlap(goal.text, sc.text or "")
            seeds.append({"id": c["id"], "score": 0.8 * float(c["score"]) + 0.2 * over})
        seeds.sort(key=lambda x: x["score"], reverse=True)
        return seeds[:top_k]

    async def _expand(self, seeds: List[Dict[str, Any]], *, hops: int) -> List[Dict[str, Any]]:
        """k-hop expansion with edge weights; dedupe; return node dicts with a working score."""
        frontier = [s["id"] for s in seeds]
        seen = set(frontier)
        scored: Dict[str, float] = {s["id"]: float(s["score"]) for s in seeds}

        for h in range(hops):
            nbrs = await self.nexus.neighbors(frontier)
            next_frontier = []
            for src, outs in nbrs.items():
                for dst, w in outs:
                    if dst in seen:
                        continue
                    # attenuate by hop distance & edge weight
                    gain = (scored.get(src, 0.0)) * float(w) * (0.6 ** (h + 1))
                    if gain <= 0:
                        continue
                    scored[dst] = max(scored.get(dst, 0.0), gain)
                    seen.add(dst)
                    next_frontier.append(dst)
            frontier = next_frontier
            if not frontier:
                break

        # materialize nodes
        nodes = []
        for nid, s in scored.items():
            sc = await self.nexus.get_scorable(nid)
            nodes.append({"id": nid, "title": (sc.meta or {}).get("title"), "text": sc.text, "score": s})
        nodes.sort(key=lambda x: x["score"], reverse=True)
        return nodes

    async def _select_evidence(self, goal: Goal, nodes: List[Dict[str, Any]], *, target_k: int) -> List[Dict[str, Any]]:
        """Diversity-aware selection (MMR/facility-location surrogate)."""
        if not nodes:
            return []
        g_emb = self.embed.get_or_create(goal.text)

        picked: List[int] = []
        embeds = [self.embed.get_or_create(n["text"] or "") for n in nodes]

        def sim(a, b):
            # cosine for unit-normalized vectors; replace with your fast IP
            import numpy as np
            a = np.asarray(a); b = np.asarray(b)
            num = float((a * b).sum()); den = (float((a*a).sum())**0.5) * (float((b*b).sum())**0.5) + 1e-9
            return num / den

        lamb = 0.7  # relevance vs diversity
        scores = [sim(g_emb, e) for e in embeds]

        while len(picked) < min(target_k, len(nodes)):
            best_i, best_val = -1, -1e9
            for i in range(len(nodes)):
                if i in picked: 
                    continue
                div_pen = 0.0
                for j in picked:
                    div_pen = max(div_pen, sim(embeds[i], embeds[j]))
                val = lamb * scores[i] - (1 - lamb) * div_pen
                if val > best_val:
                    best_val, best_i = val, i
            picked.append(best_i)

        return [nodes[i] for i in picked]

    async def _reason(self, goal: Goal, evid: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """Compose prompt and call your LLM. Swap with your existing Reasoner if you have one."""
        ctx = "\n\n".join([f"[{i+1}] {e.get('title') or e['id']}\n{(e['text'] or '')[:1200]}" for i, e in enumerate(evid)])
        prompt = (
            "You are a precise, grounded assistant.\n"
            f"Goal: {goal.text}\n\n"
            "Use only the evidence below to answer. Cite with [#] where # is the evidence index.\n\n"
            f"EVIDENCE:\n{ctx}\n\n"
            "Answer:"
        )
        ans = self.llm(prompt)  # container-provided
        return str(ans or "").strip(), {"prompt_len": len(prompt), "evidence_count": len(evid)}

    async def _verify(self, goal: Goal, answer: str, evid: List[Dict[str, Any]]):
        """EpistemicGuard + VERICOT verification; return metrics + artifact paths."""
        try:
            eg_in = {
                "trace_id": f"goal-{hash(goal.text)%10_000_000}",
                "question": goal.text,
                "hypothesis": answer,
                "reference": "\n\n".join([e["text"] or "" for e in evid]),
            }
            gout = await self.eg.assess_dict(eg_in)  # add .assess_dict in your service or map to GuardInput
            eg_metrics = gout.metrics or {}
            artifacts = {
                "vpm": gout.vpm_path, "legend": gout.legend_path,
                "field": gout.field_path, "badge": gout.badge_path
            }
        except Exception:
            eg_metrics, artifacts = {}, {}

        try:
            vc = self.vericot.verify(answer, [e["text"] or "" for e in evid])
            vc_metrics = {
                "unsat_core_size": vc.unsat_core_size,
                "contradictions": int(bool(vc.contradiction)),
                "ungrounded": int(vc.status == "ungrounded"),
            }
        except Exception:
            vc_metrics = {}

        return eg_metrics, vc_metrics, artifacts

    async def _score(
        self,
        goal: Goal,
        answer: str,
        evid: List[Dict[str, Any]],
        eg: Dict[str, Any],
        vc: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Aggregate Solve (judge), Faithfulness, Uncertainty, Support-efficiency into a composite J.
        """
        # Judge (0..100) via your ScoringService judge model
        judge = self.scoring.judge_answer(goal=goal.text, answer=answer, context=evid)

        # Uncertainty: -EBT energy + SICQL advantage, if available
        ebt_energy = self.scoring.get_attr("ebt_energy") or 0.0
        sicql_adv = self.scoring.get_attr("sicql_advantage") or 0.0
        uncert = (0.0 - float(ebt_energy)) + float(sicql_adv)

        faithful = float(eg.get("faithfulness", 0.0))
        support_eff = 0.0 - float(vc.get("unsat_core_size", 0.0))

        J = (
            self.w.solve * float(judge)
            + self.w.faithful * faithful
            + self.w.uncert * uncert
            + self.w.support_eff * support_eff
        )
        return {
            "judge": float(judge),
            "faithful": faithful,
            "uncert": uncert,
            "support_eff": support_eff,
            "J": float(J),
        }

    # ------------------ tiny utils ------------------

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        A = set(a.lower().split()); B = set(b.lower().split())
        return len(A & B) / max(1, len(A))
