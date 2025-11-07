# stephanie/components/nexus/agents/nexus_improver.py
from __future__ import annotations
import math
import statistics as stats
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from stephanie.scoring.scorable import Scorable
from stephanie.services.scoring_service import ScoringService

from stephanie.components.nexus.blossom.runner import BlossomRunnerAgent

@dataclass
class NexusImproverConfig:
    k_candidates: int = 2
    sharpen_iters: int = 1
    novelty_tau: float = 0.12
    promote_margin: float = 0.02
    dims: List[str] = None
    scorers: List[str] = None
    scorer_weights: Dict[str, float] = None
    dimension_weights: Dict[str, float] = None
    persist_scores: bool = True
    use_llm_heuristic: bool = True

class NexusImproverAgent:
    """
    Cohort improver:
      - evaluates baseline scorables
      - blossoms each into k candidates (+optional sharpen)
      - evaluates candidates and does local promotion with margin
      - aggregates cohort-level improvement
      - emits pairwise training signals
    """
    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        self.cfg = NexusImproverConfig(
            k_candidates=int(cfg.get("k_candidates", 2)),
            sharpen_iters=int(cfg.get("sharpen_iters", 1)),
            novelty_tau=float(cfg.get("novelty_tau", 0.12)),
            promote_margin=float(cfg.get("promote_margin", 0.02)),
            dims=list(cfg.get("dims", ["alignment","faithfulness","coverage","clarity","coherence"])),
            scorers=list(cfg.get("scorers", ["sicql","mrq","hrm"])),
            scorer_weights=dict(cfg.get("scorer_weights", {})),
            dimension_weights=dict(cfg.get("dimension_weights", {})),
            persist_scores=bool(cfg.get("persist_scores", True)),
            use_llm_heuristic=bool(cfg.get("use_llm_heuristic", True)),
        )
        self.memory = memory
        self.container = container
        self.logger = logger
        self.scoring: ScoringService = container.get("scoring")
        self.blossom = BlossomRunnerAgent(cfg.get("blossom", {}), memory, container, logger)

    # --------------- public entry -----------------
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        goal_text = (context.get("goal") or {}).get("goal_text", "") or ""
        items: List[Scorable] = context.get("scorables", [])
        if not items:
            return {"status": "no_items"}

        # 1) Baseline evaluation
        base_evals: Dict[str, float] = {}
        base_vectors: Dict[str, Dict[str, float]] = {}
        for s in items:
            eval_res = self._eval_state(s, context)
            base_evals[str(s.id)] = eval_res["overall"]
            base_vectors[str(s.id)] = eval_res["dims"]

            if self.cfg.persist_scores:
                self._persist_dims(s, eval_res, context)

        baseline_summary = self._cohort_summary(list(base_evals.values()))

        # 2) Blossom + candidate evaluation
        decisions = []
        for s in items:
            children = await self._blossom_and_refine(s, context, self.cfg.k_candidates)
            # filter by novelty vs parent
            children = self._novel_children(s, children, tau=self.cfg.novelty_tau)

            cand_evals = []
            for c in children:
                res = self._eval_state(c, context)
                cand_evals.append((c, res))
                if self.cfg.persist_scores:
                    self._persist_dims(c, res, context)

            # local selection
            parent_overall = base_evals[str(s.id)]
            winner, win_eval = self._select_winner(s, parent_overall, cand_evals, margin=self.cfg.promote_margin)

            # record decision
            decisions.append({
                "parent_id": s.id, "parent_overall": parent_overall,
                "winner_id": getattr(winner, "id", None),
                "winner_overall": (win_eval or {}).get("overall", parent_overall),
                "lift": ((win_eval or {}).get("overall", parent_overall) - parent_overall)
            })

            # link & persist blossom graphically
            try:
                self.memory.blossoms.link_parent_children(parent_id=s.id, child_ids=[c.id for c,_ in cand_evals])
                if winner and (win_eval["overall"] - parent_overall) >= self.cfg.promote_margin:
                    self.memory.nexus.promote(parent_id=s.id, child_id=winner.id, reason="local_improvement")
            except Exception as e:
                self.logger and self.logger.log("NexusPromoteError", {"parent": s.id, "error": str(e)})

            # learning signals (pairwise)
            try:
                for c, res in cand_evals:
                    self._emit_pairwise_training(s, base_evals[str(s.id)], c, res["overall"], context)
            except Exception as e:
                self.logger and self.logger.log("TrainingEmitError", {"parent": s.id, "error": str(e)})

        # 3) Global verdict
        wins = sum(1 for d in decisions if d["lift"] > 0.0)
        win_rate = wins / max(1, len(decisions))
        lifts = [d["lift"] for d in decisions]
        cohort_lift = float(stats.mean(lifts)) if lifts else 0.0
        topk = sorted(base_evals.values(), reverse=True)[: max(1, math.ceil(0.1*len(base_evals)))]
        # recompute top-k after applying promotions (approx: use winner_overall if lift>0)
        promoted_overalls = [max(d["parent_overall"], d["winner_overall"]) for d in decisions]
        topk_new = sorted(promoted_overalls, reverse=True)[: max(1, math.ceil(0.1*len(promoted_overalls)))]
        topk_lift = (float(stats.mean(topk_new)) - float(stats.mean(topk))) if topk and topk_new else 0.0

        report = {
            "baseline": baseline_summary,
            "decision_count": len(decisions),
            "win_rate": float(win_rate),
            "mean_lift": float(cohort_lift),
            "topk_lift": float(topk_lift),
            "decisions": decisions,
            "blossom_success_rate": self._blossom_success_rate(decisions),
            "mean_blossom_diversity": float(stats.mean([
                d.get("diversity", 0.0) for d in decisions if "diversity" in d
            ])) if decisions else 0.0,
        }
        return report

    def _blossom_diversity(self, children: list["Scorable"]) -> float:
        if len(children) < 2: 
            return 0.0
        try:
            embs = [np.asarray(self.memory.embedding.get_or_create(c.text), dtype=float) for c in children]
            embs = [e / (np.linalg.norm(e) + 1e-9) for e in embs]
            sims = []
            for i in range(len(embs)):
                for j in range(i+1, len(embs)):
                    sims.append(float(np.dot(embs[i], embs[j])))
            # diversity = 1 - mean cosine similarity
            return float(max(0.0, 1.0 - (sum(sims)/max(1,len(sims)))))
        except Exception:
            return 0.0

    def _blossom_success_rate(self, decisions: list[dict]) -> float:
        ok = sum(1 for d in decisions if d.get("lift", 0.0) > 0.0)
        return float(ok / max(1, len(decisions)))

    # --------------- helpers -----------------
    def _eval_state(self, scorable: Scorable, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.scoring.evaluate_state(
            scorable=scorable,
            context=context,
            scorers=self.cfg.scorers,
            dimensions=self.cfg.dims,
            scorer_weights=self.cfg.scorer_weights,
            dimension_weights=self.cfg.dimension_weights,
            include_llm_heuristic=self.cfg.use_llm_heuristic,
            fuse_mode="weighted_mean",
            clamp_01=True,
        )

    async def _blossom_and_refine(self, parent: Scorable, context: Dict[str, Any], k: int) -> List[Scorable]:
        # Use your existing blossom runner; inject 1-step sharpening by config
        return await self.blossom.generate_candidates(parent, context, k=k, sharpen_iters=self.cfg.sharpen_iters)

    def _novel_children(self, parent: Scorable, children: List[Scorable], tau: float) -> List[Scorable]:
        try:
            pe = self.memory.embedding.get_or_create(parent.text)
            keep = []
            for c in children:
                ce = self.memory.embedding.get_or_create(c.text)
                # cosine distance ~ 1 - sim
                sim = float((pe @ ce) / ( (pe**2).sum()**0.5 * (ce**2).sum()**0.5 + 1e-9 ))
                if (1.0 - sim) >= tau:
                    keep.append(c)
            return keep or children[:1]  # never drop all
        except Exception:
            return children

    def _select_winner(self, parent: Scorable, parent_overall: float, cand_evals: List[Tuple[Scorable, Dict]], margin: float):
        if not cand_evals:
            return None, None
        best = max(cand_evals, key=lambda x: x[1]["overall"])
        if (best[1]["overall"] - parent_overall) >= margin:
            return best[0], best[1]
        return parent, {"overall": parent_overall, "dims": {}}

    def _persist_dims(self, scorable: Scorable, eval_res: Dict[str, Any], context: Dict[str, Any]):
        # Persist fused overall as HRM canonical, and dims as score rows
        try:
            self.scoring.save_hrm_score(
                scorable_id=scorable.id, scorable_type=scorable.target_type, value=eval_res["overall"],
                **{k: context.get(k) for k in ("goal_id","plan_trace_id","pipeline_run_id") if k in context}
            )
            for d, v in (eval_res.get("dims") or {}).items():
                self.scoring.save_score(
                    scorable_id=scorable.id, scorable_type=scorable.target_type,
                    score_type=d, score_value=float(v), source="state_evaluator",
                    **{k: context.get(k) for k in ("goal_id","plan_trace_id","pipeline_run_id") if k in context}
                )
        except Exception as e:
            self.logger and self.logger.log("PersistDimsError", {"id": scorable.id, "error": str(e)})

    def _emit_pairwise_training(self, parent: Scorable, parent_overall: float, child: Scorable, child_overall: float, context: Dict[str, Any]):
        try:
            prefer_child = child_overall > parent_overall
            pos, neg = (child, parent) if prefer_child else (parent, child)
            w = max(0.1, min(1.0, abs(child_overall - parent_overall) + 0.2))
            # pairwise
            self.memory.training_events.insert_pairwise(
                model_key="ranker.sicql.v1",
                dimension="alignment",
                query_text=(context.get("goal") or {}).get("goal_text",""),
                pos_text=pos.text, neg_text=neg.text,
                weight=w, trust=w*0.6,
                goal_id=context.get("goal",{}).get("id"),
                pipeline_run_id=context.get("pipeline_run_id"),
                agent_name="NexusImproverAgent",
                source="nexus_improver",
                meta={"delta_overall": child_overall - parent_overall}
            )
            # pointwise
            for s, v in ((child, child_overall), (parent, parent_overall)):
                self.memory.training_events.add_pointwise(
                    model_key="retriever.mrq.v1",
                    dimension="alignment",
                    query_text=(context.get("goal") or {}).get("goal_text",""),
                    cand_text=s.text, label=1 if v >= parent_overall else 0,
                    weight=max(0.1, v), trust=0.5,
                    goal_id=context.get("goal",{}).get("id"),
                    pipeline_run_id=context.get("pipeline_run_id"),
                    agent_name="NexusImproverAgent",
                    source="nexus_improver"
                )
        except Exception as e:
            self.logger and self.logger.log("TrainingEmitError", {"parent": parent.id, "child": getattr(child,'id',None), "error": str(e)})

    def _cohort_summary(self, vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0}
        vals = list(vals)
        vals_sorted = sorted(vals)
        p90 = vals_sorted[max(0, int(0.9 * (len(vals_sorted)-1)))]
        return {"mean": float(stats.mean(vals)), "median": float(stats.median(vals)), "p90": float(p90)}
