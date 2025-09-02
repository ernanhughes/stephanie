"""
MementoAgent: Case-Based Reasoning with A/B Guardrails and Non-Regression

Key features:
- Casebook scoping by (pipeline × agent × tag) with flexible retrieval modes
- Guaranteed reuse (champion-first + recent-success + diverse-novel within a strict budget)
- Always-retain provenance for each run (even with zero hypotheses)
- MARS-based quality metric and non-regression champion promotion
- Automatic A/B validation: run "baseline" (no casebook) vs "CBR" and trust CBR only if it helps
- Seed locking for fair A/B comparisons; optional training freeze for baseline
- Defensive fallbacks if CaseBookStore helper methods are not yet implemented

Dependencies assumed (provided elsewhere in your codebase):
- CaseBookStore helpers (preferred): ensure_casebook_scope, get_cases_for_goal_in_casebook,
  get_cases_for_goal_scoped, get_goal_state, upsert_goal_state, get_recent_cases, get_pool_for_goal
- If not present, safe fallbacks are used.

"""

from __future__ import annotations


import random
import traceback
from typing import Dict, List, Optional, Tuple

# Optional: only import numpy/torch if seed locking may use them
import numpy as np
import torch

from stephanie.agents.dspy.mcts_reasoning import MCTSReasoningAgent
from stephanie.data.score_corpus import ScoreCorpus
from stephanie.scoring.calculations.mars_calculator import MARSCalculator
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.scoring.scorer.scorable_ranker import ScorableRanker
from stephanie.constants import GOAL, GOAL_TEXT, PIPELINE_RUN_ID

# at top of memento.py
from contextlib import contextmanager

CTX_NS = "_MEMENTO"          # single namespace key in the context dict
VARIANTS = "variants"        # bucket for per-variant scratch


class MementoAgent(MCTSReasoningAgent):
    """
    Case-Based Reasoning LATS Agent with A/B non-regression guardrails.
    """

    # -------------------------
    # Construction / Config
    # -------------------------
    def __init__(self, cfg, memory, logger):
        """
        cfg fields (all optional; sensible defaults):
            include_mars: bool = True
            casebook_tag: str = "default"
            retrieval_mode: str = "fallback"    # "strict" | "fallback" | "union"
            reuse_budget: int = 16
            novelty_k: int = 6
            improve_eps: float = 1e-6
            exploration_eps: float = 0.10
            ab_validation:
                mode: "periodic"               # "off" | "periodic" | "always"
                period: 5
                control_tag: "control"
                delta_eps: 1e-6
                seed_lock: True
                freeze_training: True
            quality_weights:
                mars: 1.0
                hrm: 0.5
                reward: 2.0
                llm: 0.25
        """
        super().__init__(cfg, memory, logger)

        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        self.ranker = ScorableRanker(cfg, memory, logger)
        self.include_mars = cfg.get("include_mars", True)
        if self.include_mars:
            self.mars = MARSCalculator(cfg, memory, logger)

        # Behavior knobs
        self.casebook_tag = cfg.get("casebook_tag", "default")
        self.retrieval_mode = cfg.get(
            "retrieval_mode", "fallback"
        )  # "strict" | "fallback" | "union"
        self.reuse_budget = int(cfg.get("reuse_budget", 16))
        self.novelty_k = int(cfg.get("novelty_k", 6))
        self.improve_eps = float(cfg.get("improve_eps", 1e-6))
        self.exploration_eps = float(cfg.get("exploration_eps", 0.10))

        self.ab_cfg = cfg.get("ab_validation", {}) or {}
        self.ab_mode = self.ab_cfg.get(
            "mode", "periodic"
        )  # "off" | "periodic" | "always"
        self.ab_period = int(self.ab_cfg.get("period", 5))
        self.ab_control_tag = self.ab_cfg.get("control_tag", "control")
        self.ab_delta_eps = float(self.ab_cfg.get("delta_eps", 1e-6))
        self.ab_seed_lock = bool(self.ab_cfg.get("seed_lock", True))
        self.ab_freeze_training = bool(
            self.ab_cfg.get("freeze_training", True)
        )

        self.qw = cfg.get("quality_weights", {}) or {
            "mars": 1.0,
            "hrm": 0.5,
            "reward": 2.0,
            "llm": 0.25,
        }

        # Internal switches set per _run_single
        self._training_enabled = True
        self._casebook_tag_runtime = self.casebook_tag
        self.dimensions = cfg.get("dimensions", ["alignment"])

    # -------------------------
    # Public entry
    # -------------------------
    async def run(self, context: dict) -> dict:
        goal = context[GOAL]
        home_casebook_id = self._home_casebook(context)

        try:
            run_ix = self._bump_and_get_run_counter(home_casebook_id, goal["id"])
            do_ab = self._ab_should_run(run_ix)
            if do_ab and self.ab_seed_lock:
                self._set_random_seed(context.get("seed") or 42)

            if do_ab:
                # A) BASELINE
                base_res = await self._run_single_variant(
                    context, variant="baseline", use_casebook=False,
                    retain=False, train_enabled=not self.ab_freeze_training,
                    casebook_tag=self.ab_control_tag
                )

                # B) CBR
                cbr_res = await self._run_single_variant(
                    context, variant="cbr", use_casebook=True,
                    retain=True, train_enabled=True,
                    casebook_tag=self.casebook_tag
                )

                q_base = base_res["metrics"]["quality"]
                q_cbr  = cbr_res["metrics"]["quality"]
                improved = q_cbr > (q_base + self.ab_delta_eps)
                self._update_non_regression(home_casebook_id, goal["id"], cbr_res, improved)

                # Project the winner's namespaced output to the canonical output key
                winner_variant = "cbr" if improved else "baseline"
                winner_okey = self._variant_output_key(winner_variant)
                context[self.output_key] = context.get(winner_okey, [])

                # Report A/B
                self.logger.log(
                    "MementoABCompare",
                    {
                        "goal_id": goal["id"],
                        "q_base": q_base,
                        "q_cbr": q_cbr,
                        "improved": improved,
                    },
                )
                self.report(
                    {
                        "event": "ab_compare",
                        "goal_id": goal["id"],
                        "q_base": q_base,
                        "q_cbr": q_cbr,
                        "improved": improved,
                    }
                )
                return context

            # No A/B
            single = await self._run_single_variant(
                context, variant="cbr", use_casebook=True, retain=True,
                train_enabled=True, casebook_tag=self.casebook_tag
            )
            # Project CBR namespaced output to canonical
            context[self.output_key] = context.get(self._variant_output_key("cbr"), [])
            return context

        except Exception as e:
            self.logger.error(
                "CBRRunFailed",
                {"error": str(e), "trace": traceback.format_exc()},
            )
            self.report(
                {"event": "error", "step": "MementoAgent", "details": str(e)}
            )
            return context

    async def _run_single_variant(
        self,
        context: dict,
        *,
        variant: str,
        use_casebook: bool,
        retain: bool,
        train_enabled: bool,
        casebook_tag: str,
    ) -> dict:
        goal = context[GOAL]
        home_casebook_id = self._home_casebook(context)

        self._training_enabled = train_enabled
        self._casebook_tag_runtime = casebook_tag

        # --- Retrieve/build reuse candidates *into namespaced bucket* ---
        past_cases = []
        reuse_candidates: List[str] = []
        if use_casebook:
            past_cases = self._retrieve_cases(context)
            reuse_candidates = self._build_reuse_candidates(home_casebook_id, goal["id"], past_cases)

        # Store scratch under our namespace (never at top-level)
        vb = self._vb(context, variant)
        vb["cases_found"] = len(past_cases)
        vb["reuse_candidates"] = reuse_candidates

        # Temporarily expose reuse_candidates for parent components that expect it.
        # Temporarily write outputs to a private namespaced output key.
        with self._temp_ctx_key(context, "reuse_candidates", reuse_candidates), \
            self._variant_output(variant):

            # REUSE → RANK → MARS (your existing flow)
            lats_result = await super().run(context)
            best_hypotheses = lats_result.get("hypotheses", []) or []
            ranked, corpus, mars_results, recommendations, scores_payload = (
                self._rank_and_analyze(context, best_hypotheses)
            )

            # Retain & micro-learning
            retained_case_id = None
            if retain:
                retained_case_id = self._retain_case(context, ranked, mars_results, scores_payload)
            if self._training_enabled:
                try:
                    self._online_learn(context, ranked, mars_results)
                except Exception as e:
                    self.logger.log("CBRTrainSkip", {"reason": str(e)})

            # Quality
            quality = self._compute_quality(mars_results, scores_payload)
            metrics = {"quality": quality, "variant": variant}

            # Save results only inside our namespace
            vb["ranked"] = ranked
            vb["mars_results"] = mars_results
            vb["recommendations"] = recommendations
            vb["retained_case"] = retained_case_id
            vb["metrics"] = metrics

            # Also put the ranked list in our private output key so downstream
            # pipeline stages can read it *if* they look at self.output_key during the run.
            context[self._variant_output_key(variant)] = ranked

        return vb  # return the namespaced results for convenience


    # -------------------------
    # Single-variant executor
    # -------------------------
    async def _run_single(
        self,
        context: dict,
        *,
        variant: str,
        use_casebook: bool,
        retain: bool,
        train_enabled: bool,
        casebook_tag: str,
    ) -> dict:
        goal = context[GOAL]
        home_casebook_id = self._home_casebook(context)

        self._training_enabled = train_enabled
        self._casebook_tag_runtime = casebook_tag

        # === 1) RETRIEVE (scoped; champion-first) ===
        past_cases = []
        reuse_candidates: List[str] = []
        if use_casebook:
            past_cases = self._retrieve_cases(context)  # list[CaseORM]
            reuse_candidates = self._build_reuse_candidates(
                home_casebook_id, goal["id"], past_cases
            )
        else:
            reuse_candidates = []

        context["reuse_candidates"] = reuse_candidates
        self.logger.log(
            "CBRRetrieve",
            {
                "goal_id": goal["id"],
                "variant": variant,
                "cases_found": len(past_cases),
                "reuse_candidates": len(reuse_candidates),
            },
        )
        self.report(
            {
                "event": "retrieve",
                "step": "MementoAgent",
                "variant": variant,
                "details": f"Reuse candidates: {len(reuse_candidates)}",
            }
        )

        # === 2) REUSE via MCTSReasoningAgent ===
        lats_result = await super().run(context)
        best_hypotheses = lats_result.get("hypotheses", []) or []
        self.logger.log(
            "CBRReuse",
            {
                "goal_id": goal["id"],
                "variant": variant,
                "count": len(best_hypotheses),
            },
        )
        self.report(
            {
                "event": "reuse",
                "step": "MementoAgent",
                "variant": variant,
                "details": f"Produced {len(best_hypotheses)} hypotheses",
            }
        )

        # === 3) RANK + SCORE + MARS ===
        ranked, corpus, mars_results, recommendations, scores_payload = (
            self._rank_and_analyze(context, best_hypotheses)
        )
        self.logger.log(
            "CBRRank",
            {
                "goal_id": goal["id"],
                "variant": variant,
                "ranked_count": len(ranked),
            },
        )
        if self.mars:
            self.logger.log(
                "CBRRevise",
                {
                    "goal_id": goal["id"],
                    "variant": variant,
                    "mars_summary_size": len(mars_results),
                    "recommendations": recommendations,
                },
            )
            self.report(
                {
                    "event": "revise",
                    "step": "MementoAgent",
                    "variant": variant,
                    "details": f"MARS results: {len(mars_results)}",
                    "recommendations": recommendations,
                }
            )

        # Attach MARS agreement to ranked items (if available)
        if ranked and mars_results:
            for r in ranked:
                r_id = r.get("id")
                r["mars_confidence"] = (mars_results.get(r_id, {}) or {}).get(
                    "agreement_score"
                )

        # === 4) RETAIN (always for provenance; but skip in baseline) ===
        retained_case_id = None
        if retain:
            retained_case_id = self._retain_case(
                context, ranked, mars_results, scores_payload
            )

        # === 5) MICRO-LEARNING (only when enabled) ===
        if self._training_enabled:
            try:
                self._online_learn(context, ranked, mars_results)
            except Exception as e:
                # learning is best-effort; never break the run
                self.logger.log("CBRTrainSkip", {"reason": str(e)})

        # === 6) QUALITY METRIC (A/B compare & champion) ===
        quality = self._compute_quality(mars_results, scores_payload)
        metrics = {"quality": quality, "variant": variant}

        # === 7) Output wiring ===
        context[self.output_key] = ranked
        primary = ranked[0]["text"] if ranked else "[no hypotheses generated]"
        self.set_scorable_details(
            input_text=goal.get(GOAL_TEXT, ""),
            output_text=primary,
            description=f"CBR hypotheses for goal: {goal.get(GOAL_TEXT, '')}",
            meta={
                "all_hypotheses": [h["text"] for h in ranked],
                "count": len(ranked),
            },
        )

        self.report(
            {
                "event": "conclusion",
                "step": "MementoAgent",
                "variant": variant,
                "goal": goal.get("goal_text", ""),
                "summary": f"Ranked {len(ranked)}; retained {retained_case_id if retain else 'no-retain'}; quality={quality:.4f}",
                "recommendations": recommendations,
            }
        )

        return {
            "ranked": ranked,
            "mars_results": mars_results,
            "recommendations": recommendations,
            "retained_case": retained_case_id,
            "metrics": metrics,
        }

    # -------------------------
    # Retrieval & Reuse build
    # -------------------------
    def _retrieve_cases(self, context: dict):
        """Retrieve past cases according to retrieval_mode with scoped casebooks."""
        goal = context[GOAL]
        pipeline_run_id = context[PIPELINE_RUN_ID]
        agent = self.name

        scopes = [(pipeline_run_id, agent, self.casebook_tag)]
        if self.retrieval_mode in ("fallback", "union"):
            scopes += [
                (None, agent, self.casebook_tag),  # agent-global
                (pipeline_run_id, None, self.casebook_tag),  # pipeline-global
                (None, None, self.casebook_tag),  # system-global
            ]

        # Prefer helper: get_cases_for_goal_scoped; else fallback to get_cases_for_goal
        try:
            if self.retrieval_mode == "strict":
                cb = self._ensure_casebook_scope(
                    pipeline_run_id, agent, self.casebook_tag
                )
                return self.memory.casebooks.get_cases_for_goal_in_casebook(
                    cb.id, goal["id"]
                )
            elif self.retrieval_mode == "fallback":
                for sc in scopes:
                    cases = self.memory.casebooks.get_cases_for_goal_scoped(
                        goal["id"], [sc]
                    )
                    if cases:
                        return cases
                return []
            else:  # union
                return self.memory.casebooks.get_cases_for_goal_scoped(
                    goal["id"], scopes
                )
        except AttributeError:
            # Fallback: unscoped retrieval
            return self.memory.casebooks.get_cases_for_goal(goal["id"])

    def _build_reuse_candidates(
        self, casebook_id: int, goal_id: str, cases
    ) -> List[str]:
        """
        Build a bounded list of scorable IDs:
          champion-first → recent-success → diverse-novel, capped at reuse_budget.
        Fallbacks gracefully if helper methods are missing.
        """
        ids: List[str] = []

        # Champion-first
        try:
            state = self.memory.casebooks.get_goal_state(casebook_id, goal_id)
            if state and state.champion_case_id:
                champ_case = next(
                    (c for c in cases if c.id == state.champion_case_id), None
                )
                if champ_case:
                    ids.extend(self._top_scorables_from_case(champ_case))
        except AttributeError:
            pass  # state table not present yet

        # Recent-success (best-effort)
        try:
            recent = self.memory.casebooks.get_recent_cases(
                casebook_id,
                goal_id,
                limit=max(1, self.reuse_budget // 2),
                only_accepted=True,
            )
            for c in recent:
                ids.extend(self._top_scorables_from_case(c))
        except AttributeError:
            # fallback: just iterate given cases by recency order (assuming .created_at desc)
            for c in cases[: max(1, self.reuse_budget // 2)]:
                ids.extend(self._top_scorables_from_case(c))

        # Diverse-novel pool
        pool_ids = set(ids)
        novel_pool = []
        try:
            pool = self.memory.casebooks.get_pool_for_goal(
                casebook_id,
                goal_id,
                exclude_ids=[getattr(c, "id", None) for c in cases],
                limit=200,
            )
            novel_pool = pool or []
        except AttributeError:
            # fallback: remaining cases
            novel_pool = [c for c in cases if c.id not in pool_ids]

        # Simple novelty: sample a few different cases (replace with MMR/k-center if available)
        random.shuffle(novel_pool)
        for c in novel_pool[: self.novelty_k]:
            ids.extend(self._top_scorables_from_case(c))

        # Exploration: occasionally inject fresh (if provided by context)
        if random.random() < float(self.exploration_eps):
            extra = (
                (cases[self.novelty_k : self.novelty_k + 2])
                if len(cases) > self.novelty_k
                else []
            )
            for c in extra:
                ids.extend(self._top_scorables_from_case(c))

        # Dedup and cap
        seen = set()
        deduped = []
        for x in ids:
            if x and x not in seen:
                deduped.append(x)
                seen.add(x)
            if len(deduped) >= self.reuse_budget:
                break
        return deduped

    @staticmethod
    def _top_scorables_from_case(case) -> List[str]:
        """
        Extract a small list of scorable IDs from a CaseORM, preferring output hypotheses.
        This assumes CaseORM.scorables is available with fields (scorable_id, role).
        """
        out: List[str] = []
        try:
            outs = [
                cs
                for cs in case.scorables
                if (getattr(cs, "role", "") or "").lower() == "output"
            ]
            outs.sort(
                key=lambda cs: getattr(cs, "rank", 1_000_000)
            )  # prefer ranked first
            for cs in outs[:3]:
                sid = getattr(cs, "scorable_id", None)
                if sid:
                    out.append(sid)
        except Exception:
            pass
        return out

    # -------------------------
    # Rank + Analyze
    # -------------------------
    def _rank_and_analyze(
        self,
        context: dict,
        best_hypotheses: List[dict],
    ) -> Tuple[List[dict], ScoreCorpus, Dict, List[str], Dict]:
        """
        Returns:
            ranked: list[dict]
            corpus: ScoreCorpus
            mars_results: dict
            recommendations: list[str]
            scores_payload: dict (ID -> serialized bundle or per-model scores)
        """
        goal = context[GOAL]

        # Build query scorable from the goal text
        query_scorable = Scorable(
            id=goal["id"], text=goal["goal_text"], target_type=TargetType.GOAL
        )

        ranked: List[dict] = []
        bundles: Dict[str, object] = {}

        if best_hypotheses:
            # Convert hypotheses into Scorables
            scorables = [
                Scorable(
                    id=h.get("id"),
                    text=h.get("text", ""),
                    target_type=TargetType.HYPOTHESIS,
                )
                for h in best_hypotheses
            ]

            # Rank
            ranked_raw = (
                self.ranker.rank(
                    query=query_scorable, candidates=scorables, context=context
                )
                or []
            )
            # Normalize to dicts if we got ORM objects
            ranked = [self._normalize_rank_item(r) for r in ranked_raw]

            # Collect per-hypothesis score bundles for MARS
            for scorable in scorables:
                try:
                    _scores, bundle = self._score(
                        context=context, scorable=scorable
                    )
                    bundles[scorable.id] = bundle
                except Exception as e:
                    self.logger.log(
                        "CBRScoreSkip",
                        {"scorable_id": scorable.id, "reason": str(e)},
                    )

        corpus = ScoreCorpus(bundles=bundles)

        # MARS
        mars_results: Dict = {}
        recommendations: List[str] = []
        if self.mars:
            try:
                mars_results = (
                    self.mars.calculate(corpus, context=context) or {}
                )
                recommendations = (
                    self.mars.generate_recommendations(mars_results) or []
                )
            except Exception as e:
                self.logger.log("CBRMarSkip", {"reason": str(e)})

        # Serialize bundles → scores_payload
        scores_payload = {}
        for sid, bundle in bundles.items():
            if hasattr(bundle, "to_dict"):
                try:
                    scores_payload[sid] = bundle.to_dict()
                except Exception:
                    scores_payload[sid] = {}
            else:
                scores_payload[sid] = {}

        return ranked, corpus, mars_results, recommendations, scores_payload

    @staticmethod
    def _normalize_rank_item(item) -> dict:
        """
        Convert different ranker return types into a common dict shape.
        Expected keys: id, text, rank (1-based).
        """
        if isinstance(item, dict):
            r = dict(item)
            if "rank" not in r:
                r["rank"] = r.get("position", 0) or r.get("order", 0) or 0
            return r
        # Fallback for ORM-like objects
        out = {
            "id": getattr(item, "id", None)
            or getattr(item, "scorable_id", None),
            "text": getattr(item, "text", "") or getattr(item, "content", ""),
            "rank": getattr(item, "rank", 0),
        }
        return out

    # -------------------------
    # Retain + Champion
    # -------------------------
    def _retain_case(
        self,
        context: dict,
        ranked: List[dict],
        mars_results: Dict,
        scores_payload: Dict,
    ) -> Optional[int]:
        """
        Persist a CaseORM with scorables and rich metadata.
        Always called in the CBR variant; baseline variant sets retain=False.
        """
        goal = context[GOAL]
        home_casebook_id = self._home_casebook(context)

        # Per-scorable payload
        scorables_payload = []
        for idx, r in enumerate(ranked):
            scorables_payload.append(
                {
                    "id": r.get("id"),
                    "type": "hypothesis",
                    "role": "output",
                    "rank": (idx + 1),
                    "meta": {
                        "text": r.get("text", ""),
                        "mars_confidence": r.get("mars_confidence"),
                    },
                }
            )

        # Retain
        try:
            case = self.memory.casebooks.add_case(
                casebook_id=home_casebook_id,
                goal_id=goal["id"],
                goal_text=goal["goal_text"],
                agent_name=self.__class__.__name__,
                mars_summary=mars_results,
                scores=scores_payload,
                metadata={
                    "pipeline_run_id": context.get(PIPELINE_RUN_ID),
                    "casebook_tag": self._casebook_tag_runtime,
                    "hypothesis_count": len(ranked),
                },
                scorables=scorables_payload,
            )
            self.logger.log(
                "CBRRetain", {"case_id": case.id, "goal_id": goal["id"]}
            )
            self.report(
                {
                    "event": "retain",
                    "step": "MementoAgent",
                    "details": f"Retained case {case.id} with {len(ranked)} outputs",
                }
            )
            return case.id
        except Exception as e:
            self.logger.log("CBRRetainError", {"error": str(e)})
            return None

    def _update_non_regression(
        self, casebook_id: int, goal_id: str, run_result: dict, improved: bool
    ):
        """
        Promote the champion only if improved; no regression allowed.
        """
        if not improved:
            # optional: decay trust / adjust reuse budget dynamically (not required to block)
            return
        try:
            case_id = run_result.get("retained_case")
            quality = run_result.get("metrics", {}).get("quality", 0.0)
            if case_id:
                self.memory.casebooks.upsert_goal_state(
                    casebook_id, goal_id, case_id, quality
                )
                self.logger.log(
                    "CBRChampionUpdate",
                    {
                        "goal_id": goal_id,
                        "case_id": case_id,
                        "quality": quality,
                    },
                )
        except AttributeError:
            # state table not implemented yet; silently ignore
            pass

    # -------------------------
    # Quality metric
    # -------------------------
    def _compute_quality(
        self, mars_results: Dict, scores_payload: Dict
    ) -> float:
        """
        Single scalar to compare variants and enforce non-regression.
        Extend here to include HRM, LLM grades, or real task rewards.
        """
        mars_agree = 0.0
        if mars_results:
            vals = [
                float(v.get("agreement_score", 0.0))
                for v in mars_results.values()
            ]
            mars_agree = (sum(vals) / len(vals)) if vals else 0.0

        # Placeholders for future sources
        hrm_score = 0.0
        llm_grade = 0.0
        task_reward = 0.0

        # Plug actual values here if present in scores_payload...
        # e.g., hrm_score = np.mean([...])

        return (
            float(self.qw.get("mars", 1.0)) * mars_agree
            + float(self.qw.get("hrm", 0.5)) * hrm_score
            + float(self.qw.get("llm", 0.25)) * llm_grade
            + float(self.qw.get("reward", 2.0)) * task_reward
        )

    # -------------------------
    # Helpers: scoping, counters, seeds
    # -------------------------
    def _home_casebook(self, context: dict):
        """
        Ensure and cache the 'home' casebook for (pipeline_run_id, agent_name, tag).
        """
        pk = context[PIPELINE_RUN_ID]
        agent = self.name
        try:
            cb = self._ensure_casebook_scope(pk, agent, self.casebook_tag)
            return cb.id
        except AttributeError:
            # Fallback: require cfg.casebook_id or raise
            casebook_id = self.cfg.get("casebook_id")
            if not casebook_id:
                raise RuntimeError(
                    "No scoped helpers and no cfg.casebook_id provided for MementoAgent."
                )
            return int(casebook_id)

    def _ensure_casebook_scope(
        self,
        pipeline_run_id: Optional[str],
        agent_name: Optional[str],
        tag: str,
    ):
        """
        Wrapper to CaseBookStore.ensure_casebook_scope with a fallback to ensure_casebook.
        """
        try:
            return self.memory.casebooks.ensure_casebook_scope(
                pipeline_run_id, agent_name, tag
            )
        except AttributeError:
            # Fallback to a single named casebook
            name = f"cb:{agent_name or 'all'}:{pipeline_run_id or 'all'}:{tag}"
            return self.memory.casebooks.ensure_casebook(
                name, description="Scoped fallback"
            )

    def _bump_and_get_run_counter(self, casebook_id: int, goal_id: str) -> int:
        """
        Increment and return a per-(casebook,goal) counter used to schedule A/B runs.
        Preferred store: case_goal_state table; fallback to memory-local counter.
        """
        try:
            state = self.memory.casebooks.get_goal_state(casebook_id, goal_id)
            if state is None:
                # bootstrap
                self.memory.casebooks.upsert_goal_state(
                    casebook_id, goal_id, case_id=None, quality=0.0
                )  # may create row
                state = self.memory.casebooks.get_goal_state(
                    casebook_id, goal_id
                )
            # piggy-back a "run_ix" in the state object if supported
            ix = getattr(state, "run_ix", 0) + 1
            setattr(state, "run_ix", ix)
            self.memory.casebooks.session.commit()  # type: ignore[attr-defined]
            return ix
        except Exception:
            # In-memory fallback
            key = f"{casebook_id}:{goal_id}"
            if not hasattr(self, "_run_counters"):
                self._run_counters = {}
            self._run_counters[key] = self._run_counters.get(key, 0) + 1
            return self._run_counters[key]

    def _ab_should_run(self, run_index: int) -> bool:
        if self.ab_mode == "off":
            return False
        if self.ab_mode == "always":
            return True
        # periodic
        return run_index % max(1, self.ab_period) == 0

    @staticmethod
    def _set_random_seed(seed: int):
        random.seed(seed)
        if np is not None:
            try:
                np.random.seed(seed)
            except Exception:
                pass
        if torch is not None:
            try:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except Exception:
                pass

    # -------------------------
    # Learning hooks (best-effort)
    # -------------------------
    def _online_learn(
        self, context: dict, ranked: List[dict], mars_results: Dict
    ):
        """
        Emit pairwise & pointwise learning signals and run tiny online updates.
        Supports:
        - DB-backed training store  (memory.training_store)
        - In-memory training buffers (memory.training_buffers)
        - Optional TrainingController (memory.training_controller)
        """
        # Respect A/B baseline or explicit disable
        if not self._training_enabled:
            return
        if not ranked or len(ranked) < 2:
            return

        goal = context[GOAL]
        goal_id = goal.get("id", "")
        goal_text = goal.get("goal_text", "")
        pipeline_run_id = context.get(PIPELINE_RUN_ID)
        agent_name = self.name

        # Config defaults
        tr_cfg = self.cfg.get("training", {}) or {}
        model_key_ranker = tr_cfg.get("buffers", {}).get(
            "model_key_ranker", "ranker.sicql.v1"
        )
        model_key_retriever = tr_cfg.get("buffers", {}).get(
            "model_key_retriever", "retriever.mrq.v1"
        )
        dimension = tr_cfg.get("dimension", "alignment")

        # Helper: MARS agreement as weight in [0,1]
        def agree(hid: str | None) -> float:
            if not hid:
                return 0.0
            try:
                return float(
                    (mars_results.get(hid) or {}).get("agreement_score") or 0.0
                )
            except Exception:
                return 0.0

        winner = ranked[0]
        pos_text = winner.get("text", "") or ""
        pos_id = winner.get("id")
        if not pos_text:
            return

        pos_w = max(
            0.1, agree(pos_id)
        )  # keep a small floor so we always learn a tiny bit

        # Build pairs for validator/controller (top-1 vs others)
        pairs_for_validation = []

        # Collect negatives & emit signals
        neg_texts = []
        for r in ranked[1:]:
            neg_text = r.get("text", "") or ""
            if not neg_text:
                continue
            neg_texts.append(neg_text)

            neg_id = r.get("id")
            neg_w = agree(neg_id)
            # Use a conservative weight (avg of pos/neg agreement, floored)
            pair_w = max(0.1, 0.5 * (pos_w + neg_w))

            # For controller’s validator
            pairs_for_validation.append(
                {"text_a": pos_text, "text_b": neg_text, "weight": pair_w}
            )

            # --- (A) DB-backed store ---
            self.memory.training_store.add_pairwise(
                model_key=model_key_ranker,
                dimension=dimension,
                query_text=goal_text,
                pos_text=pos_text,
                neg_text=neg_text,
                weight=pair_w,
                trust=neg_w,  # stash extra signal if you like
                goal_id=goal_id,
                pipeline_key=pipeline_run_id,
                agent_name=agent_name,
                source="memento",
                meta={"run_id": context.get("pipeline_run_id")},
            )
            # positive
            self.memory.training_store.add_pointwise(
                model_key=model_key_retriever,
                dimension=dimension,
                query_text=goal_text,
                cand_text=pos_text,
                label=1,
                weight=pos_w,
                trust=pos_w,
                goal_id=goal_id,
                pipeline_key=pipeline_run_id,
                agent_name=agent_name,
                source="memento",
                meta={"run_id": context.get("pipeline_run_id")},
            )
            # negatives
            for nt in neg_texts:
                self.memory.training_store.add_pointwise(
                    model_key=model_key_retriever,
                    dimension=dimension,
                    query_text=goal_text,
                    cand_text=nt,
                    label=0,
                    weight=0.5,  # flat weight for negatives; adjust if you want neg_w
                    trust=0.0,
                    goal_id=goal_id,
                    pipeline_key=pipeline_run_id,
                    agent_name=agent_name,
                    source="memento",
                    meta={"run_id": context.get("pipeline_run_id")},
                )

    def _ns(self, ctx: dict) -> dict:
        """Ensure and return our namespaced bucket."""
        return ctx.setdefault(CTX_NS, {VARIANTS: {}})

    def _vb(self, ctx: dict, variant: str) -> dict:
        """Ensure and return a per-variant subbucket."""
        return self._ns(ctx)[VARIANTS].setdefault(variant, {})

    def _variant_output_key(self, variant: str) -> str:
        """A private output key so baseline & cbr never overwrite each other."""
        return f"{CTX_NS}.{variant}.output"

    @contextmanager
    def _temp_ctx_key(self, ctx: dict, key: str, value):
        """
        Temporarily set a top-level key (e.g., 'reuse_candidates') for components
        that expect it; restore previous value afterwards.
        """
        _sentinel = object()
        old = ctx.get(key, _sentinel)
        ctx[key] = value
        try:
            yield
        finally:
            if old is _sentinel:
                ctx.pop(key, None)
            else:
                ctx[key] = old

    @contextmanager
    def _variant_output(self, variant: str):
        """
        Temporarily point self.output_key at a private, namespaced key,
        so super().run writes into our namespace instead of the canonical output.
        """
        prev = self.output_key
        self.output_key = self._variant_output_key(variant)
        try:
            yield
        finally:
            self.output_key = prev
