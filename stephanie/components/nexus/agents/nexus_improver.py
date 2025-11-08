# stephanie/components/nexus/agents/nexus_improver.py
from __future__ import annotations

import math
import logging
import statistics as stats
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from stephanie.scoring.scorable import Scorable
from stephanie.services.scoring_service import ScoringService
from stephanie.components.nexus.blossom.runner import BlossomRunnerAgent
from stephanie.agents.base_agent import BaseAgent

@dataclass
class NexusImproverConfig:
    k_candidates: int = 2
    sharpen_iters: int = 1
    novelty_tau: float = 0.12
    promote_margin: float = 0.02
    dims: List[str] = field(default_factory=lambda: ["alignment", "faithfulness", "coverage", "clarity", "coherence"])
    scorers: List[str] = field(default_factory=lambda: ["sicql", "mrq", "hrm"])
    scorer_weights: Dict[str, float] = field(default_factory=dict)
    dimension_weights: Dict[str, float] = field(default_factory=dict)
    use_llm_heuristic: bool = False
    use_vpm_phi: bool = False
    persist_scores: bool = False
    blossom_cfg: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "NexusImproverConfig":
        return NexusImproverConfig(
            k_candidates=int(cfg.get("k_candidates", 2)),
            sharpen_iters=int(cfg.get("sharpen_iters", 1)),
            novelty_tau=float(cfg.get("novelty_tau", 0.12)),
            promote_margin=float(cfg.get("promote_margin", 0.02)),
            dims=list(cfg.get("dims", ["alignment","faithfulness","coverage","clarity","coherence"])),
            scorers=list(cfg.get("scorers", ["sicql","mrq","hrm"])),
            scorer_weights=dict(cfg.get("scorer_weights", {})),
            dimension_weights=dict(cfg.get("dimension_weights", {})),
            use_llm_heuristic=bool(cfg.get("use_llm_heuristic", False)),
            use_vpm_phi=bool(cfg.get("use_vpm_phi", False)),
            persist_scores=bool(cfg.get("persist_scores", True)),
            blossom_cfg=dict(cfg.get("blossom", {})),
        )


class NexusImproverAgent(BaseAgent):
    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.cfg = NexusImproverConfig.from_cfg(cfg or {})
        self.scoring: ScoringService = container.get("scoring")
        self.blossom = BlossomRunnerAgent(self.cfg.blossom_cfg, memory, container, logger)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        goal_text = (context.get("goal") or {}).get("goal_text", "") or ""
        items: List[Scorable] = list(context.get("scorables", []) or [])
        if not items:
            context[self.output_key] = {"status": "no_items"}
            return context

        # collect per-episode training telemetry here
        context.setdefault("blossom_runs", [])

        # sync runner defaults (defensive)
        try:
            # how many winners to surface for each parent
            self.blossom.run_cfg.return_top_k = int(self.cfg.k_candidates)
            # sharpen only if configured
            self.blossom.run_cfg.sharpen_top_k = int(self.cfg.k_candidates if self.cfg.sharpen_iters > 0 else 0)
        except Exception:
            pass

        use_vpm_phi = bool(context.get("use_vpm_phi", self.cfg.use_vpm_phi))

        # ---------------- 1) Baseline eval ----------------
        base_evals: Dict[str, float] = {}
        base_vectors: Dict[str, Dict[str, float]] = {}

        for s in items:
            scorable = Scorable.from_dict(s)
            eval_res = self._eval_state(scorable, context, use_vpm_phi=use_vpm_phi)
            sid = str(getattr(scorable, "id", None))
            base_evals[sid] = float(eval_res["overall"])
            base_vectors[sid] = dict(eval_res["dims"])

            if self.cfg.persist_scores:
                self._persist_dims(scorable, eval_res, context)

        baseline_summary = self._cohort_summary(list(base_evals.values()))

        # ---------------- 2) Blossom + local select --------
        decisions: List[Dict[str, Any]] = []

        for s in items:
            scorable = Scorable.from_dict(s)
            children, blossom_meta = await self._blossom_and_refine(s, context, self.cfg.k_candidates)

            # diversity before/after novelty
            diversity_raw = self._blossom_diversity(children)
            filtered_children = self._novel_children(scorable, children, tau=self.cfg.novelty_tau)
            diversity_post = self._blossom_diversity(filtered_children)

            # evaluate filtered candidates
            cand_evals: List[Tuple[Scorable, Dict[str, Any]]] = []
            for c in filtered_children:
                res = self._eval_state(c, context, use_vpm_phi=use_vpm_phi)
                cand_evals.append((c, res))
                if self.cfg.persist_scores:
                    self._persist_dims(c, res, context)

            parent_overall = float(base_evals[str(getattr(scorable, "id", None))])
            winner, win_eval = self._select_winner(scorable, parent_overall, cand_evals, margin=self.cfg.promote_margin)
            winner_overall = float((win_eval or {}).get("overall", parent_overall))
            lift = float(winner_overall - parent_overall)

            # attach blossom summary to the decision
            decisions.append({
                "parent_id": getattr(scorable, "id", None),
                "parent_overall": parent_overall,
                "winner_id": getattr(winner, "id", None),
                "winner_overall": winner_overall,
                "lift": lift,
                "k_generated": len(children),
                "k_after_novelty": len(filtered_children),
                "diversity_raw": float(diversity_raw),
                "diversity_post": float(diversity_post),

                # ---- blossom linkage (compact per-decision) ----
                "blossom_episode_id": blossom_meta.get("episode_id"),
                "blossom_winner_leaf_ids": [w.get("leaf_id") for w in blossom_meta.get("winners", [])],
                "blossom_winner_rewards": [float(w.get("reward", 0.0)) for w in blossom_meta.get("winners", [])],
                "blossom_winner_paths": [w.get("path") for w in blossom_meta.get("winners", [])],
                "blossom_top_reward": float(blossom_meta.get("top_reward", 0.0)),
            })

            # persist the full meta for training/replay
            context["blossom_runs"].append(blossom_meta)

            # link & promote
            self._safe_link_and_promote(scorable, [getattr(c, "id", None) for c, _ in cand_evals], winner, lift)

            # training events
            self._safe_emit_training(parent=scorable, parent_overall=parent_overall, cand_evals=cand_evals, context=context)

        # ---------------- 3) Cohort verdict ----------------
        wins = sum(1 for d in decisions if d["lift"] > 0.0)
        win_rate = float(wins / max(1, len(decisions)))
        lifts = [float(d["lift"]) for d in decisions] or [0.0]
        cohort_lift = float(stats.mean(lifts))
        topk = sorted(base_evals.values(), reverse=True)[: max(1, math.ceil(0.1 * len(base_evals)))]
        promoted_overalls = [max(d["parent_overall"], d["winner_overall"]) for d in decisions]
        topk_new = sorted(promoted_overalls, reverse=True)[: max(1, math.ceil(0.1 * len(promoted_overalls)))]
        topk_lift = float((stats.mean(topk_new) - stats.mean(topk))) if topk and topk_new else 0.0

        report = {
            "status": "ok",
            "goal_preview": goal_text[:120],
            "decision_count": len(decisions),
            "baseline": baseline_summary,
            "win_rate": win_rate,
            "mean_lift": cohort_lift,
            "topk_lift": topk_lift,
            "mean_blossom_diversity_raw": float(stats.mean([d["diversity_raw"] for d in decisions])) if decisions else 0.0,
            "mean_blossom_diversity_post": float(stats.mean([d["diversity_post"] for d in decisions])) if decisions else 0.0,
            "blossom_success_rate": self._blossom_success_rate(decisions),
            "decisions": decisions,
        }
        context[self.output_key] = report
        return context

    # --------------------------- CORE HELPERS ---------------------------

    def _eval_state(self, scorable: Scorable, context: Dict[str, Any], *, use_vpm_phi: bool) -> Dict[str, Any]:
        """
        Preferred path uses ScoringService.evaluate_state (if available).
        Falls back to local fusion across configured scorers/dimensions.
        Returns:
          {
            "overall": float in [0,1],
            "dims": {dim: float in [0,1], ...},
            "by_scorer": {scorer: {"aggregate": float, "per_dimension": {...}}, ...},
            "components": {}
          }
        """
        svc = getattr(self, "scoring", None)
        if svc and hasattr(svc, "evaluate_state") and callable(svc.evaluate_state):
            try:
                return svc.evaluate_state(
                    scorable=scorable,
                    context=context,
                    scorers=self.cfg.scorers,
                    dimensions=self.cfg.dims,
                    scorer_weights=self.cfg.scorer_weights,
                    dimension_weights=self.cfg.dimension_weights,
                    include_llm_heuristic=self.cfg.use_llm_heuristic,
                    include_vpm_phi=use_vpm_phi,
                    fuse_mode="weighted_mean",
                    clamp_01=True,
                )
            except Exception as e:
                self._slog("EvaluateStateError", {"id": getattr(scorable, "id", None), "error": str(e)})

        # Fallback if the fused API isn't present or failed
        return self._local_fuse_state(scorable, context)

    def _local_fuse_state(self, scorable: Scorable, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Local fusion when ScoringService.evaluate_state is unavailable.
        - Calls score() per-scorer
        - Weighted mean per dimension across scorers
        - Weighted mean across dimensions -> overall
        Clamps to [0,1].
        """
        dims = list(self.cfg.dims) or ["alignment"]
        sw = {k: float(v) for k, v in (self.cfg.scorer_weights or {}).items()}
        dw = {d: float(v) for d, v in (self.cfg.dimension_weights or {}).items()}

        per_dim_vals: Dict[str, List[Tuple[float, float]]] = {d: [] for d in dims}
        by_scorer: Dict[str, Any] = {}

        for name in (self.cfg.scorers or []):
            try:
                bundle = self.scoring.score(
                    scorer_name=name,
                    scorable=scorable,
                    context=context,
                    dimensions=dims,
                )
                agg = float(bundle.aggregate())
                # If a dimension is missing from results, fall back to aggregate
                per = {
                    d: float(bundle.results[d].score) if d in bundle.results else agg
                    for d in dims
                }
                by_scorer[name] = {"aggregate": agg, "per_dimension": per}

                w = sw.get(name, 1.0)
                for d in dims:
                    per_dim_vals[d].append((per[d], w))
            except Exception as e:
                self._slog("LocalFuseScorerError", {"scorer": name, "error": str(e)})

        fused_dim: Dict[str, float] = {}
        for d in dims:
            if not per_dim_vals[d]:
                fused_dim[d] = 0.0
                continue
            num = sum(v * w for (v, w) in per_dim_vals[d])
            den = sum(w for (_, w) in per_dim_vals[d]) or 1.0
            fused_dim[d] = float(num / den) * float(dw.get(d, 1.0))

        dw_total = sum(abs(dw.get(d, 1.0)) for d in dims) or float(len(dims))
        overall = float(sum(fused_dim[d] for d in dims) / dw_total)

        # clamp to [0,1]
        overall = float(max(0.0, min(1.0, overall)))
        for d in fused_dim:
            fused_dim[d] = float(max(0.0, min(1.0, fused_dim[d])))

        return {"overall": overall, "dims": fused_dim, "by_scorer": by_scorer, "components": {}}

    def _slog(self, event: str, payload: Dict[str, Any]):
        """Safe structured log used across helpers."""
        try:
            if hasattr(self.logger, "log"):
                self.logger.log(event, payload)
            else:
                logging.getLogger(__name__).warning("%s: %s", event, payload)
        except Exception:
            pass

    async def _blossom_and_refine(self, parent: Scorable, context: Dict[str, Any], k: int) -> Tuple[List[Scorable], Dict[str, Any]]:
        """
        Invoke BlossomRunnerAgent.run() and convert winners into Scorables.
        Also return a rich blossom_meta block for training/replay.
        """
        # sync runner per-call
        try:
            self.blossom.run_cfg.return_top_k = int(k)
            self.blossom.run_cfg.sharpen_top_k = int(k if self.cfg.sharpen_iters > 0 else 0)
        except Exception:
            pass
        goal_text = parent.get("goal_ref", {}).get("text", "") or context.get("goal", {}).get("goal_text", "")
        goal = self.memory.goals.get_or_create({"goal_text": goal_text})
        runner_ctx: Dict[str, Any] = {
            "goal": goal.to_dict(),
            "seed": {
                "seed_type": parent.get("scorable_type", "scorable"),
                "seed_id": parent.get("scorable_id", None),
                "plan_text": parent.get("text", None),
            },
            "pipeline_run_id": context.get("pipeline_run_id"),
        }

        try:
            out_ctx = await self.blossom.run(runner_ctx)
        except Exception as e:
            self._slog("BlossomRunError", {"parent": getattr(parent, "id", None), "error": str(e)})
            return [], {}

        bres = (out_ctx or {}).get("blossom_result") or {}
        episode_id = bres.get("episode_id")
        winners = list(bres.get("winners") or [])
        training_batch = bres.get("training_batch")
        top_reward = float(winners[0]["reward"]) if winners else 0.0

        # resolve texts for each winner (for immediate scoring or audit)
        resolved_children: List[Scorable] = []
        resolved_winners: List[Dict[str, Any]] = []

        for i, w in enumerate(winners):
            text, resolved_node_id = self._winner_to_text(w)
            if not text:
                continue

            child = Scorable(
                id=f"blossom:{episode_id}:{resolved_node_id or i}",
                text=str(text),
                target_type=getattr(parent, "target_type", "document"),
            )
            resolved_children.append(child)

            # keep rich winner meta for learning
            resolved_w = {
                "leaf_id": w.get("leaf_id"),
                "leaf_db_id": resolved_node_id,
                "reward": float(w.get("reward", 0.0)),
                "path": w.get("path"),
                "sharpened_meta": (w.get("sharpened") or {}),
                "text_len": len(text or ""),
            }
            resolved_winners.append(resolved_w)

        blossom_meta = {
            "episode_id": episode_id,
            "parent_id": getattr(parent, "id", None),
            "parent_text_len": len(getattr(parent, "text", "") or ""),
            "goal_text": (context.get("goal") or {}).get("goal_text", ""),
            "k_requested": int(k),
            "k_resolved": len(resolved_children),
            "top_reward": top_reward,
            "winners": resolved_winners,
            "training_batch_present": training_batch is not None,
            # (Optional) small cfg snapshot for analysis
            "runner_cfg": {
                "return_top_k": getattr(self.blossom.cfg, "return_top_k", None),
                "sharpen_top_k": getattr(self.blossom.cfg, "sharpen_top_k", None),
            },
        }

        return resolved_children, blossom_meta

    def _winner_to_text(self, w: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
        """
        Best-effort recovery of winner text:
          1) prefer w["sharpened"]["sharpened"]
          2) else try BlossomStore.get_node(leaf_db_id)
          3) else try BlossomStore.get_node(int(leaf_id)) if numeric
        Returns (text, resolved_node_id).
        """
        # 1) prefer sharpened payload
        sh = w.get("sharpened") or {}
        if isinstance(sh, dict) and sh.get("sharpened"):
            return str(sh["sharpened"]), None

        # 2) look up DB node
        node_id = w.get("leaf_db_id")
        try_ids: List[int] = []
        if node_id is not None:
            try:
                try_ids.append(int(node_id))
            except Exception:
                pass
        # 3) fallback: if leaf_id is numeric, also try it as DB pk
        lid = w.get("leaf_id")
        if lid is not None:
            try:
                try_ids.append(int(lid))
            except Exception:
                pass

        for nid in try_ids:
            try:
                node = self.memory.blossoms.get_node(int(nid))
                if node:
                    text = node.sharpened_text or node.state_text
                    if text:
                        return str(text), int(nid)
            except Exception as e:
                self._slog("BlossomFetchNodeError", {"leaf_candidate_id": nid, "error": str(e)})

        return None, None

    def _novel_children(self, parent: Scorable, children: List[Scorable], tau: float) -> List[Scorable]:
        """
        Keep children whose cosine distance from parent >= tau.
        If everything is filtered, keep the single best child (preserve progress).
        """
        if not children:
            return []
        try:
            pe = np.asarray(self.memory.embedding.get_or_create(parent.text), dtype=float)
            pn = float(np.linalg.norm(pe)) + 1e-9
            keep: List[Scorable] = []
            sims: List[Tuple[float, Scorable]] = []
            for c in children:
                ce = np.asarray(self.memory.embedding.get_or_create(c.text), dtype=float)
                cn = float(np.linalg.norm(ce)) + 1e-9
                sim = float(np.dot(pe, ce) / (pn * cn))
                sims.append((sim, c))
                if (1.0 - sim) >= tau:
                    keep.append(c)
            if keep:
                return keep
            # else keep the least similar (most novel) one
            sims.sort(key=lambda t: t[0])  # ascending by sim
            return [sims[0][1]]
        except Exception:
            return children

    def _select_winner(
        self,
        parent: Scorable,
        parent_overall: float,
        cand_evals: List[Tuple[Scorable, Dict[str, Any]]],
        margin: float,
    ) -> Tuple[Optional[Scorable], Optional[Dict[str, Any]]]:
        if not cand_evals:
            return None, None
        best = max(cand_evals, key=lambda x: float(x[1].get("overall", 0.0)))
        if (float(best[1]["overall"]) - parent_overall) >= margin:
            return best[0], best[1]
        return parent, {"overall": parent_overall, "dims": {}}

    def _persist_dims(self, scorable: Scorable, eval_res: Dict[str, Any], context: Dict[str, Any]):
        """
        Persist fused overall as HRM canonical + per-dimension fused scores.
        """
        try:
            self.scoring.save_hrm_score(
                scorable_id=scorable.id,
                scorable_type=scorable.target_type,
                value=float(eval_res["overall"]),
                **{k: context.get(k) for k in ("goal_id", "plan_trace_id", "pipeline_run_id") if k in context}
            )
            for d, v in (eval_res.get("dims") or {}).items():
                self.scoring.save_score(
                    scorable_id=scorable.id,
                    scorable_type=scorable.target_type,
                    score_type=str(d),
                    score_value=float(v),
                    source="state_evaluator",
                    **{k: context.get(k) for k in ("goal_id", "plan_trace_id", "pipeline_run_id") if k in context}
                )
        except Exception as e:
            self._slog("PersistDimsError", {"id": getattr(scorable, "id", None), "error": str(e)})

    def _safe_link_and_promote(self, parent: Scorable, child_ids: List[str], winner: Optional[Scorable], lift: float):
        try:
            if hasattr(self.memory, "blossoms") and hasattr(self.memory.blossoms, "link_parent_children"):
                self.memory.blossoms.link_parent_children(parent_id=parent.id, child_ids=list(child_ids))
        except Exception as e:
            self._slog("BlossomLinkError", {"parent": parent.get("id"), "error": str(e)})

        try:
            if winner and lift >= self.cfg.promote_margin:
                if hasattr(self.memory, "nexus") and hasattr(self.memory.nexus, "promote"):
                    self.memory.nexus.promote(parent_id=parent.id, child_id=winner.id, reason="local_improvement")
        except Exception as e:
            self._slog("NexusPromoteError", {"parent": parent.id, "error": str(e)})

    def _safe_emit_training(
        self,
        *,
        parent: Scorable,
        parent_overall: float,
        cand_evals: List[Tuple[Scorable, Dict[str, Any]]],
        context: Dict[str, Any],
    ):
        """
        Emit pairwise and pointwise training events for downstream ranker/retriever.
        """
        if not hasattr(self.memory, "training_events"):
            return
        for c, res in cand_evals:
            try:
                child_overall = float(res.get("overall", 0.0))
                prefer_child = child_overall > parent_overall
                pos, neg = (c, parent) if prefer_child else (parent, c)
                w = max(0.1, min(1.0, abs(child_overall - parent_overall) + 0.2))
                # pairwise
                self.memory.training_events.insert_pairwise({
                    "model_key": "ranker.sicql.v1",
                    "dimension": "alignment",
                    "query_text": (context.get("goal") or {}).get("goal_text", ""),
                    "pos_text": pos.text,
                    "neg_text": neg.text,
                    "weight": w,
                    "trust": w * 0.6,
                    "goal_id": (context.get("goal") or {}).get("id"),
                    "pipeline_run_id": context.get("pipeline_run_id"),
                    "agent_name": "NexusImproverAgent",
                    "source": "nexus_improver",
                    "meta": {"delta_overall": child_overall - parent_overall},
                }, dedup=True)

                # pointwise
                for s, v in ((c, child_overall), (parent, parent_overall)):
                    self.memory.training_events.insert_pointwise({
                        "model_key": "retriever.mrq.v1",
                        "dimension": "alignment",
                        "query_text": (context.get("goal") or {}).get("goal_text", ""),
                        "cand_text": s.text,
                        "label": 1 if v >= parent_overall else 0,
                        "weight": max(0.1, float(v)),
                        "trust": 0.5,
                        "goal_id": (context.get("goal") or {}).get("id"),
                        "pipeline_run_id": context.get("pipeline_run_id"),
                        "agent_name": "NexusImproverAgent",
                        "source": "nexus_improver",
                    }, dedup=True)
            except Exception as e:
                self._slog("TrainingEmitError", {"parent": parent.id, "child": getattr(c, 'id', None), "error": str(e)})

    # --------------------------- METRICS / UTIL ---------------------------

    def _blossom_diversity(self, children: List[Scorable]) -> float:
        """
        Diversity = 1 - mean cosine similarity among children embeddings.
        """
        if len(children) < 2:
            return 0.0
        try:
            embs = [np.asarray(self.memory.embedding.get_or_create(c.text), dtype=float) for c in children]
            # normalize
            embs = [e / (np.linalg.norm(e) + 1e-9) for e in embs]
            sims = []
            for i in range(len(embs)):
                ei = embs[i]
                for j in range(i + 1, len(embs)):
                    sims.append(float(np.dot(ei, embs[j])))
            return float(max(0.0, 1.0 - (sum(sims) / max(1, len(sims)))))
        except Exception:
            return 0.0

    def _blossom_success_rate(self, decisions: List[Dict[str, Any]]) -> float:
        ok = sum(1 for d in decisions if float(d.get("lift", 0.0)) > 0.0)
        return float(ok / max(1, len(decisions)))

    def _cohort_summary(self, vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0}
        vals_sorted = sorted(vals)
        p90 = vals_sorted[max(0, int(0.9 * (len(vals_sorted) - 1)))]
        return {"mean": float(stats.mean(vals_sorted)), "median": float(stats.median(vals_sorted)), "p90": float(p90)}
