# stephanie/components/nexus/agents/nexus_improver.py
from __future__ import annotations

import json
import logging
import math
import statistics as stats
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.nexus.blossom.runner import BlossomRunnerAgent
from stephanie.services.scoring_service import ScoringService
from stephanie.scoring.scorable import Scorable
from stephanie.utils.progress_mixin import ProgressMixin


# --------------------------- CONFIG ---------------------------

@dataclass
class NexusImproverAsyncConfig:
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
    persist_scores: bool = True
    blossom_cfg: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> NexusImproverAsyncConfig:
        return NexusImproverAsyncConfig(
            k_candidates=int(cfg.get("k_candidates", 2)),
            sharpen_iters=int(cfg.get("sharpen_iters", 1)),
            novelty_tau=float(cfg.get("novelty_tau", 0.12)),
            promote_margin=float(cfg.get("promote_margin", 0.02)),
            dims=list(cfg.get("dims", ["alignment", "faithfulness", "coverage", "clarity", "coherence"])),
            scorers=list(cfg.get("scorers", ["sicql", "mrq", "hrm"])),
            scorer_weights=dict(cfg.get("scorer_weights", {})),
            dimension_weights=dict(cfg.get("dimension_weights", {})),
            use_llm_heuristic=bool(cfg.get("use_llm_heuristic", False)),
            use_vpm_phi=bool(cfg.get("use_vpm_phi", False)),
            persist_scores=bool(cfg.get("persist_scores", True)),
            blossom_cfg=dict(cfg.get("blossom", {})),
        )


# --------------------------- AGENT ---------------------------

class NexusImproverAsyncAgent(ProgressMixin, BaseAgent):
    """
    Improves a cohort of Scorables via Blossom (generate -> refine -> select),
    evaluates lift, persists training events, and reports progress in real time.

    Progress tasks:
      - overall:  "nexus_improver:{pipeline_run_id|epoch}"
      - per-ep:   "blossom_episode:parent={parent_id}"
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self._init_progress(container, logger)
        self.cfg = NexusImproverAsyncConfig.from_cfg(cfg or {})
        self.scoring: ScoringService = container.get("scoring")
        self.blossom = BlossomRunnerAgent(self.cfg.blossom_cfg, memory, container, logger)

    # --------------------------- PUBLIC ---------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        goal_text = (context.get("goal") or {}).get("goal_text", "") or ""
        items: List[Scorable] = list(context.get("scorables", []) or [])
        if not items:
            context[self.output_key] = {"status": "no_items"}
            return context

        # collect per-episode telemetry
        context.setdefault("blossom_runs", [])

        # sync runner defaults
        try:
            self.blossom.run_cfg.return_top_k = int(self.cfg.k_candidates)
            self.blossom.run_cfg.sharpen_top_k = int(self.cfg.k_candidates if self.cfg.sharpen_iters > 0 else 0)
        except Exception:
            pass

        use_vpm_phi = bool(context.get("use_vpm_phi", self.cfg.use_vpm_phi))

        # ----- PROGRESS: overall task -----
        task_run = self._task_run_name(context)
        self.pstart(task_run, total=len(items))

        # ---------------- 1) Baseline eval ----------------
        self.pstage(task_run, stage="baseline")

        base_evals: Dict[str, float] = {}
        base_vectors: Dict[str, Dict[str, float]] = {}

        for s in items:
            scorable = Scorable.from_dict(s)
            eval_res = self._eval_state(scorable, context, use_vpm_phi=use_vpm_phi)
            sid = str(getattr(scorable, "id", None) or f"tmp-{idx}")
            base_evals[sid] = float(eval_res["overall"])
            base_vectors[sid] = dict(eval_res["dims"])
            if self.cfg.persist_scores:
                self._persist_dims(scorable, eval_res, context)

        baseline_summary = self._cohort_summary(list(base_evals.values()))

        # ---------------- 2) Blossom + local select --------
        self.pstage(task_run, stage="blossom")

        decisions: List[Dict[str, Any]] = []
        total = len(items)

        for idx, s in enumerate(items, start=1):
            scorable = Scorable.from_dict(s)

            # ---- per-episode subtask ----
            ep_task = self._task_episode_name(getattr(scorable, "id", "unknown"))
            est_total = max(1, self.cfg.k_candidates * max(1, self.cfg.sharpen_iters))
            self.pstart(ep_task, total=est_total)

            offload = bool(self.cfg.blossom_cfg.get("offload_prompts", False))
            if offload:
                tickets = await self._offload_candidate_prompts(
                    scorable,
                    {**context, "progress_task": ep_task},
                    k=self.cfg.k_candidates,
                    model=self.cfg.blossom_cfg.get("model", "gpt-4o-mini"),
                    target_pool=self.cfg.blossom_cfg.get("target_pool", "provider"),
                )
                # record and move on; don't wait
                context.setdefault("pending_prompt_tickets", []).extend(tickets)
                # mark this episode as pending
                decisions.append({
                    "parent_id": getattr(scorable, "id", None),
                    "status": "pending_prompts",
                    "k_requested": self.cfg.k_candidates
                })
                self.pstep(task_run, n=1)
                self.pdone(ep_task)  # ← add this

                # skip sync blossom path
                continue

            children, blossom_meta = await self._blossom_and_refine(
                s,
                {**context, "progress_task": ep_task},
                self.cfg.k_candidates,
            )

            # best-effort progress reconciliation (if runner didn't emit live)
            self._reconcile_episode_progress_from_events(ep_task, blossom_meta, est_total)

            # end subtask
            self.pdone(ep_task)

            # diversity + novelty
            diversity_raw = self._blossom_diversity(children)
            filtered_children = self._novel_children(scorable, children, tau=self.cfg.novelty_tau)
            diversity_post = self._blossom_diversity(filtered_children)

            # evaluate filtered
            cand_evals: List[Tuple[Scorable, Dict[str, Any]]] = []
            for c in filtered_children:
                res = self._eval_state(c, context, use_vpm_phi=use_vpm_phi)
                cand_evals.append((c, res))
                if self.cfg.persist_scores:
                    self._persist_dims(c, res, context)

            parent_overall = float(base_evals[str(getattr(scorable, "id", None))])
            winner, win_eval = self._select_winner(
                scorable, parent_overall, cand_evals, margin=self.cfg.promote_margin
            )
            winner_overall = float((win_eval or {}).get("overall", parent_overall))
            lift = float(winner_overall - parent_overall)

            # blossom linkage summary for the decision
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
                "blossom_episode_id": blossom_meta.get("episode_id"),
                "blossom_winner_leaf_ids": [w.get("leaf_id") for w in blossom_meta.get("winners", [])],
                "blossom_winner_rewards": [float(w.get("reward", 0.0)) for w in blossom_meta.get("winners", [])],
                "blossom_winner_paths": [w.get("path") for w in blossom_meta.get("winners", [])],
                "blossom_top_reward": float(blossom_meta.get("top_reward", 0.0)),
            })

            # persist episode meta
            context["blossom_runs"].append(blossom_meta)

            # link & promote
            self._safe_link_and_promote(
                scorable, [c for c, _ in cand_evals], winner, lift
            )


            # training events
            self._safe_emit_training(
                parent=scorable,
                parent_overall=parent_overall,
                cand_evals=cand_evals,
                context=context,
            )

            # overall progress step
            self.pstep(task_run, n=1)

        # ---------------- 3) Cohort verdict ----------------
        self.pstage(task_run, stage="finalize")

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
        try:
            run_dir = self._run_dir(context)
            (run_dir / "nexus_improver_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            self._slog("WriteReportError", {"error": str(e)})

        self.pdone(task_run)
        context[self.output_key] = report

        return context


    # inside NexusImproverAsyncAgent
    async def _offload_candidate_prompts(
        self,
        parent: Scorable,
        context: Dict[str, Any],
        *,
        k: int,
        model: str,
        target_pool: str,
    ) -> List[Dict[str, str]]:
        """
        Publish K prompt jobs for this parent and return a list of tickets:
        [{job_id, return_topic, parent_id, k_index}, ...]
        """
        from stephanie.prompts.prompt_client import PromptClient

        # Build K prompts (you likely already have a prompt template in Blossom)
        prompts = []
        for i in range(k):
            prompts.append({
                "messages": [
                    {"role": "system", "content": "You are a careful refiner."},
                    {"role": "user", "content": f"Refine this answer for goal:\n{(context.get('goal') or {}).get('goal_text','')}\n\n---\n{parent.text}\n"}
                ],
                # you can pour per-prompt params here
            })

        client = PromptClient()
        tickets = await client.offload_many(
            scorable_id=str(parent.id),
            prompts=prompts,
            model=model,
            target_pool=target_pool,
            priority="high",
            group_key=f"nexus:{context.get('pipeline_run_id')}",
            meta={"purpose": "nexus_blossom_refine", "parent_id": str(parent.id)},
            response_format="text",
        )
        # normalize for context
        return [{"job_id": jid, "return_topic": rt, "parent_id": str(parent.id), "k_index": str(i)} for i,(jid,rt) in enumerate(tickets)]

    # --------------------------- CORE HELPERS ---------------------------

    def _eval_state(self, scorable: Scorable, context: Dict[str, Any], *, use_vpm_phi: bool) -> Dict[str, Any]:
        """
        Preferred path uses ScoringService.evaluate_state (if available).
        Falls back to local fusion across configured scorers/dimensions.
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

        # Fallback if fused API isn't present or failed
        return self._local_fuse_state(scorable, context)

    def _local_fuse_state(self, scorable: Scorable, context: Dict[str, Any]) -> Dict[str, Any]:
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
                per = {d: float(bundle.results[d].score) if d in bundle.results else agg for d in dims}
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

    async def _blossom_and_refine(
        self, parent: Scorable, context: Dict[str, Any], k: int
    ) -> Tuple[List[Scorable], Dict[str, Any]]:
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

        # progress + events
        ep_task = context.get("progress_task")
        run_dir = self._run_dir(context)
        events_dir = run_dir / "blossom_events"
        events_dir.mkdir(parents=True, exist_ok=True)
        ev_path = events_dir / f"{str(getattr(parent, 'id', 'unknown'))}.jsonl"

        runner_ctx: Dict[str, Any] = {
            "goal": goal.to_dict(),
            "seed": {
                "seed_type": parent.get("scorable_type", "scorable"),
                "seed_id": parent.get("scorable_id", None),
                "plan_text": parent.get("text", None),
            },
            "pipeline_run_id": context.get("pipeline_run_id"),
            "events_jsonl": str(ev_path),
            "progress_task": ep_task,
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

        # resolve texts for each winner
        # resolve texts for each winner
        resolved_children: List[Scorable] = []
        resolved_winners: List[Dict[str, Any]] = []

        for i, w in enumerate(winners):
            text, resolved_node_id = self._winner_to_text(w)
            if not text:
                continue

            node_id = resolved_node_id

            # If we don't have a DB node for this text, create one so we get a real integer id
            if node_id is None:
                try:
                    if hasattr(self.memory, "blossoms") and hasattr(self.memory.blossoms, "create_node"):
                        node = self.memory.blossoms.create_node(
                            blossom_id=int(episode_id) if str(episode_id).isdigit() else episode_id,
                            state_text=text,
                            # NB: If your schema expects "parent is a NODE id", you may need to resolve that separately.
                            # Using a scorable id here only if it's actually a node id in your system.
                            # parent_id=None  # safer default unless you truly have the node id
                        )
                        node_id = int(node.id)
                except Exception as e:
                    self._slog("BlossomCreateNodeError", {"episode_id": episode_id, "error": str(e)})

            # Build the child once, with a stable id (prefer the real node id)
            scorable_id = str(node_id) if isinstance(node_id, int) else f"blossom:{episode_id}:{resolved_node_id or i}"
            child = Scorable(
                id=scorable_id,
                text=str(text),
                target_type=getattr(parent, "target_type", "document"),
            )
            # Persist the true DB node id (if any) for later linking/promote
            try:
                setattr(child, "blossom_node_id", node_id if isinstance(node_id, int) else None)
            except Exception:
                pass

            resolved_children.append(child)

            resolved_winners.append({
                "leaf_id": w.get("leaf_id"),
                "leaf_db_id": resolved_node_id,
                "reward": float(w.get("reward", 0.0)),
                "path": w.get("path"),
                "sharpened_meta": (w.get("sharpened") or {}),
                "text_len": len(text or ""),
                "node_id": node_id if isinstance(node_id, int) else None,
            })

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
            "events_jsonl": (out_ctx or {}).get("events_jsonl") or str(ev_path),
            "runner_cfg": {
                "return_top_k": getattr(self.blossom.cfg, "return_top_k", None) if hasattr(self.blossom, "cfg") else None,
                "sharpen_top_k": getattr(self.blossom.cfg, "sharpen_top_k", None) if hasattr(self.blossom, "cfg") else None,
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
        sh = w.get("sharpened") or {}
        if isinstance(sh, dict) and sh.get("sharpened"):
            return str(sh["sharpened"]), None

        node_id = w.get("leaf_db_id")
        try_ids: List[int] = []
        if node_id is not None:
            try:
                try_ids.append(int(node_id))
            except Exception:
                pass

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
            sims.sort(key=lambda t: t[0])  # ascending by similarity
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

    def _safe_link_and_promote(self, parent: Scorable, candidates: List[Scorable], winner: Optional[Scorable], lift: float):
        def _to_int(x) -> Optional[int]:
            try:
                if isinstance(x, int):
                    return x
                if isinstance(x, str) and x.isdigit():
                    return int(x)
            except Exception:
                pass
            return None

        # link parent->children using only integer node ids
        try:
            int_child_ids: List[int] = []
            for c in candidates:
                nid = getattr(c, "blossom_node_id", None)
                if not isinstance(nid, int):
                    nid = _to_int(getattr(c, "id", None))
                if isinstance(nid, int):
                    int_child_ids.append(nid)

            pid = getattr(parent, "blossom_node_id", None)
            if not isinstance(pid, int):
                pid = _to_int(getattr(parent, "id", None))

            if int_child_ids and pid is not None and hasattr(self.memory, "blossoms") and hasattr(self.memory.blossoms, "link_parent_children"):
                self.memory.blossoms.link_parent_children(parent_id=pid, child_ids=int_child_ids)
        except Exception as e:
            self._slog("BlossomLinkError", {"parent": getattr(parent, "id", None), "error": str(e)})

        # promote only if winner maps to a real integer node id
        try:
            if winner and lift >= self.cfg.promote_margin and hasattr(self.memory, "nexus") and hasattr(self.memory.nexus, "promote"):
                wid = getattr(winner, "blossom_node_id", None)
                if not isinstance(wid, int):
                    wid = _to_int(getattr(winner, "id", None))
                pid = getattr(parent, "blossom_node_id", None)
                if not isinstance(pid, int):
                    pid = _to_int(getattr(parent, "id", None))
                if isinstance(wid, int) and isinstance(pid, int):
                    self.memory.nexus.promote(parent_id=pid, child_id=wid, reason="local_improvement")
        except Exception as e:
            self._slog("NexusPromoteError", {"parent": getattr(parent, "id", None), "error": str(e)})

    def _safe_emit_training(
        self,
        *,
        parent: Scorable,
        parent_overall: float,
        cand_evals: List[Tuple[Scorable, Dict[str, Any]]],
        context: Dict[str, Any],
    ):
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

                # pointwise (both)
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
        if len(children) < 2:
            return 0.0
        try:
            embs = [np.asarray(self.memory.embedding.get_or_create(c.text), dtype=float) for c in children]
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
        return {
            "mean": float(stats.mean(vals_sorted)),
            "median": float(stats.median(vals_sorted)),
            "p90": float(p90),
        }

    def _reconcile_episode_progress_from_events(self, ep_task: str, blossom_meta: Dict[str, Any], est_total: int) -> None:
        """
        If the runner didn't emit live progress, tail the JSONL and set a sane final count.
        """
        try:
            events_path = blossom_meta.get("events_jsonl")
            if not events_path:
                return
            p = Path(events_path)
            if not p.exists():
                return
            made = 0
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                        if j.get("kind") == "add_node":
                            made += 1
                    except Exception:
                        continue
            if made:
                self.ptick(ep_task, done=made, total=max(made, est_total))
        except Exception:
            # best-effort only
            pass

    def _task_run_name(self, context: Dict[str, Any]) -> str:
        rid = context.get("pipeline_run_id") or int(time.time())
        return f"nexus_improver:{rid}"

    def _task_episode_name(self, parent_id: Any) -> str:
        return f"blossom_episode:parent={parent_id}"

    def _run_dir(self, context: Dict[str, Any]) -> Path:
        """
        Resolve a run directory suitable for logs/artifacts.
        """
        rid = context.get("pipeline_run_id") or int(time.time())
        root = Path(self.cfg.blossom_cfg.get("run_root", "runs"))
        d = root / f"run-{rid}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _slog(self, event: str, payload: Dict[str, Any]):
        try:
            if hasattr(self.logger, "log"):
                self.logger.log(event, payload)
            else:
                logging.getLogger(__name__).warning("%s: %s", event, payload)
        except Exception:
            pass
