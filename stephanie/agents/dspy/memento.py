# stephanie/agents/dspy/memento.py
"""
MementoAgent: Case-Based Reasoning with A/B Guardrails and Non-Regression

- Namespaced context writes only under CTX_NS ("_MEMENTO"); top-level remains read-only
  except for the final projection to self.output_key.
- Casebook scoping, champion-first reuse, recent-success, and diversity within budget.
- Always-retain provenance (even with zero hypotheses) in the CBR branch.
- MARS-based quality metric + non-regression champion promotion.
- Optional A/B: baseline (no casebook, optional training freeze) vs. CBR, with seed lock.
- Optional micro-learning: emits training signals to either DB-backed store or in-memory
  buffers; kicks TrainingController for validate+maybe-train.

Assumed helpers (with safe fallbacks if missing):
- CaseBookStore.ensure_casebook_scope, get_cases_for_goal_in_casebook, get_cases_for_goal_scoped,
  get_goal_state, upsert_goal_state, get_recent_cases, get_pool_for_goal
"""

from __future__ import annotations

import hashlib
import random
import traceback
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.dspy.mcts_reasoning import MCTSReasoningAgent
from stephanie.constants import GOAL, PIPELINE_RUN_ID
from stephanie.data.score_corpus import ScoreCorpus
from stephanie.scoring.calculations.mars_calculator import MARSCalculator
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.scoring.scorer.scorable_ranker import ScorableRanker

# Namespace keys (all internal writes go under this)
CTX_NS = "_MEMENTO"
VARIANTS = "variants"
# Optional deps are loaded lazily inside _set_random_seed
np = None  # type: ignore
torch = None  # type: ignore


class MementoAgent(MCTSReasoningAgent):
    """
    Case-Based Reasoning LATS Agent with A/B non-regression guardrails.
    All runtime scratch is kept under context["_MEMENTO"].
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Ranker + (optional) MARS
        self.ranker = ScorableRanker(cfg, memory, container, logger)
        self.include_mars = cfg.get("include_mars", True)
        self.mars = (
            MARSCalculator(cfg, memory, container, logger)
            if self.include_mars
            else None
        )

        # Behavior knobs
        self.casebook_tag = cfg.get("casebook_tag", "default")
        self.retrieval_mode = cfg.get(
            "retrieval_mode", "fallback"
        )  # "strict"|"fallback"|"union"
        self.reuse_budget = int(cfg.get("reuse_budget", 16))
        self.novelty_k = int(cfg.get("novelty_k", 6))
        self.improve_eps = float(cfg.get("improve_eps", 1e-6))
        self.exploration_eps = float(cfg.get("exploration_eps", 0.10))

        # A/B validation
        self.ab_cfg = cfg.get("ab_validation", {}) or {}
        self.ab_mode = self.ab_cfg.get(
            "mode", "periodic"
        )  # "off"|"periodic"|"always"
        self.ab_period = int(self.ab_cfg.get("period", 5))
        self.ab_control_tag = self.ab_cfg.get("control_tag", "control")
        self.ab_delta_eps = float(self.ab_cfg.get("delta_eps", 1e-6))
        self.ab_seed_lock = bool(self.ab_cfg.get("seed_lock", True))
        self.ab_freeze_training = bool(
            self.ab_cfg.get("freeze_training", True)
        )

        # Quality weights
        self.qw = cfg.get("quality_weights", {}) or {
            "mars": 1.0,
            "hrm": 0.5,
            "reward": 2.0,
            "llm": 0.25,
        }

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
            run_ix = self._bump_and_get_run_counter(
                home_casebook_id, goal["id"]
            )
            do_ab = self._ab_should_run(run_ix)
            if do_ab and self.ab_seed_lock:
                self._set_random_seed(context.get("seed") or 42)

            if do_ab:
                self._report_event(
                    "ab_start", {"goal_id": goal["id"], "run_ix": run_ix}
                )

                # A) BASELINE (no casebook)
                base_res = await self._run_single_variant(
                    context,
                    variant="baseline",
                    use_casebook=False,
                    retain=False,
                    train_enabled=not self.ab_freeze_training,
                    casebook_tag=self.ab_control_tag,
                )

                # B) CBR (with casebook)
                cbr_res = await self._run_single_variant(
                    context,
                    variant="cbr",
                    use_casebook=True,
                    retain=True,
                    train_enabled=True,
                    casebook_tag=self.casebook_tag,
                )

                q_base = float(base_res["metrics"]["quality"])
                q_cbr = float(cbr_res["metrics"]["quality"])
                improved = q_cbr > (q_base + self.ab_delta_eps)

                # Non-regression champion update
                self._update_non_regression(
                    home_casebook_id, goal["id"], cbr_res, improved
                )

                # Project winner to canonical output
                winner_variant = "cbr" if improved else "baseline"
                winner_okey = self._variant_output_key(winner_variant)
                context[self.output_key] = context.get(winner_okey, [])

                payload = {
                    "goal_id": goal["id"],
                    "run_ix": run_ix,
                    "q_base": q_base,
                    "q_cbr": q_cbr,
                    "improved": improved,
                    "winner": winner_variant,
                }
                self.logger.log("MementoABCompare", payload)
                self._report_event("ab_compare", payload)
                self._report_event(
                    "ab_end", {"goal_id": goal["id"], "winner": winner_variant}
                )
                return context

            # No A/B → run CBR once
            self._report_event("cbr_single_start", {"goal_id": goal["id"]})
            _ = await self._run_single_variant(
                context,
                variant="cbr",
                use_casebook=True,
                retain=True,
                train_enabled=True,
                casebook_tag=self.casebook_tag,
            )
            context[self.output_key] = context.get(
                self._variant_output_key("cbr"), []
            )
            self._report_event("cbr_single_end", {"goal_id": goal["id"]})
            return context

        except Exception as e:
            self.logger.error(
                "CBRRunFailed",
                {"error": str(e), "trace": traceback.format_exc()},
            )
            self._report_event(
                "error", {"step": "MementoAgent", "details": str(e)}
            )
            return context

    # -------------------------
    # Variant executor (namespaced)
    # -------------------------
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
        """
        Runs one variant (baseline/CBR). All writes go inside context["_MEMENTO"].
        """
        goal = context[GOAL]
        home_casebook_id = self._home_casebook(context)

        self._training_enabled = train_enabled
        self._casebook_tag_runtime = casebook_tag

        # Retrieve → build reuse candidates
        past_cases = []
        reuse_candidates: List[str] = []
        if use_casebook:
            past_cases = self._retrieve_cases(context)
            reuse_candidates = self._build_reuse_candidates(
                home_casebook_id, goal["id"], past_cases
            )

        # Namespaced scratch, never top-level
        vb = self._vb(context, variant)
        vb["cases_found"] = len(past_cases)
        vb["reuse_candidates"] = reuse_candidates

        self.logger.log(
            "CBRVariantStart",
            {
                "goal_id": goal["id"],
                "variant": variant,
                "cases_found": len(past_cases),
                "reuse_candidates": len(reuse_candidates),
            },
        )
        self._report_event(
            "retrieve",
            {
                "variant": variant,
                "details": f"Reuse candidates: {len(reuse_candidates)}",
            },
        )

        # Temporarily expose reuse_candidates + redirect output into variant-local key
        with (
            self._temp_ctx_key(context, "reuse_candidates", reuse_candidates),
            self._variant_output(variant),
        ):
            # REUSE (super.run) → RANK → MARS
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
            self._report_event(
                "reuse",
                {
                    "variant": variant,
                    "details": f"Produced {len(best_hypotheses)} hypotheses",
                },
            )

            # Ensure hypotheses carry IDs for scoring/retention/training
            best_hypotheses = self._ensure_ids(best_hypotheses)

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
                self._report_event(
                    "revise",
                    {
                        "variant": variant,
                        "details": f"MARS results: {len(mars_results)}",
                        "recommendations": recommendations,
                    },
                )

            # Attach MARS agreement to ranked for downstream consumers
            if ranked and mars_results:
                for r in ranked:
                    rid = r.get("id")
                    r["mars_confidence"] = (
                        mars_results.get(rid, {}) or {}
                    ).get("agreement_score")

            # Retain & micro-learning
            retained_case_id = None
            if retain:
                retained_case_id = self._retain_case(
                    context, ranked, mars_results, scores_payload
                )

            if self._training_enabled:
                try:
                    self._online_learn(context, ranked, mars_results)
                except Exception as e:
                    self.logger.log("CBRTrainSkip", {"reason": str(e)})

            # Quality scalar for A/B
            quality = self._compute_quality(mars_results, scores_payload)
            metrics = {"quality": float(quality), "variant": variant}

            # Persist results in namespace and expose variant-local output key
            vb["ranked"] = ranked
            vb["mars_results"] = mars_results
            vb["recommendations"] = recommendations
            vb["retained_case"] = retained_case_id
            vb["metrics"] = metrics
            context[self._variant_output_key(variant)] = ranked

            self._report_event(
                "conclusion",
                {
                    "variant": variant,
                    "goal": goal.get("goal_text", ""),
                    "summary": f"Ranked {len(ranked)}; retained {retained_case_id if retain else 'no-retain'}; quality={quality:.4f}",
                    "recommendations": recommendations,
                },
            )
            self.logger.log(
                "CBRVariantEnd",
                {
                    "goal_id": goal["id"],
                    "variant": variant,
                    "quality": quality,
                    "retained_case_id": retained_case_id,
                    "ranked_count": len(ranked),
                },
            )

        return vb

    # -------------------------
    # Retrieval & Reuse build
    # -------------------------
    def _retrieve_cases(self, context: dict):
        """Retrieve past cases per configured retrieval_mode with scoped casebooks."""
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
            self.logger.log(
                "CBRRetrieveFallback", {"reason": "scoped_methods_missing"}
            )
            return self.memory.casebooks.get_cases_for_goal(goal["id"])

    def _build_reuse_candidates(
        self, casebook_id: int, goal_id: str, cases
    ) -> List[str]:
        """
        Build a bounded list of scorable IDs:
        champion-first → recent-success → diverse-novel, capped at reuse_budget.
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
            novel_pool = [c for c in cases if c.id not in pool_ids]

        random.shuffle(novel_pool)
        for c in novel_pool[: self.novelty_k]:
            ids.extend(self._top_scorables_from_case(c))

        # Exploration injection
        if random.random() < float(self.exploration_eps):
            extra = (
                (cases[self.novelty_k : self.novelty_k + 2])
                if len(cases) > self.novelty_k
                else []
            )
            for c in extra:
                ids.extend(self._top_scorables_from_case(c))

        # Dedup + cap
        seen, deduped = set(), []
        for x in ids:
            if x and x not in seen:
                deduped.append(x)
                seen.add(x)
            if len(deduped) >= self.reuse_budget:
                break
        return deduped

    @staticmethod
    def _top_scorables_from_case(case) -> List[str]:
        """Prefer 'output' roles and lower rank numbers."""
        out: List[str] = []
        try:
            outs = [
                cs
                for cs in case.scorables
                if (getattr(cs, "role", "") or "").lower() == "output"
            ]
            outs.sort(key=lambda cs: getattr(cs, "rank", 1_000_000))
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
            scores_payload: dict (id -> serialized bundle)
        """
        goal = context[GOAL]
        query_scorable = Scorable(
            id=goal["id"],
            text=goal["goal_text"],
            target_type=ScorableType.GOAL,
        )

        ranked: List[dict] = []
        bundles: Dict[str, object] = {}

        if best_hypotheses:
            # Convert to Scorables and rank
            scorables = [
                Scorable(
                    id=h.get("id"),
                    text=h.get("text", ""),
                    target_type=ScorableType.HYPOTHESIS,
                )
                for h in best_hypotheses
            ]

            ranked_raw = (
                self.ranker.rank(
                    query=query_scorable, candidates=scorables, context=context
                )
                or []
            )
            ranked = [self._normalize_rank_item(r) for r in ranked_raw]

            # Score bundles for MARS
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

        # Serialize bundles
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
        """Normalize different ranker item shapes to {id, text, rank}."""
        if isinstance(item, dict):
            r = dict(item)
            if "rank" not in r:
                r["rank"] = r.get("position", 0) or r.get("order", 0) or 0
            return r
        return {
            "id": getattr(item, "id", None)
            or getattr(item, "scorable_id", None),
            "text": getattr(item, "text", "") or getattr(item, "content", ""),
            "rank": getattr(item, "rank", 0),
        }

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
        """Persist a CaseORM with scorables and metadata (CBR only)."""
        goal = context[GOAL]
        home_casebook_id = self._home_casebook(context)

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

        try:
            case = self.memory.casebooks.add_case(
                casebook_id=home_casebook_id,
                goal_id=goal["id"],
                agent_name=self.name,
                scorables=scorables_payload,
                meta={
                    "pipeline_run_id": context.get(PIPELINE_RUN_ID),
                    "casebook_tag": self._casebook_tag_runtime,
                    "hypothesis_count": len(ranked),
                    "mars_summary": mars_results,
                    "scores": scores_payload,
                },
            )
            self.logger.log(
                "CBRRetain", {"case_id": case.id, "goal_id": goal["id"]}
            )
            self._report_event(
                "retain",
                {
                    "details": f"Retained case {case.id} with {len(ranked)} outputs"
                },
            )
            return case.id
        except Exception as e:
            self.logger.log("CBRRetainError", {"error": str(e)})
            return None

    def _update_non_regression(
        self, casebook_id: int, goal_id: str, run_result: dict, improved: bool
    ):
        """Promote champion only if improved (no regression)."""
        if not improved:
            return
        try:
            case_id = run_result.get("retained_case")
            quality = float(run_result.get("metrics", {}).get("quality", 0.0))
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
            pass

    # -------------------------
    # Quality metric
    # -------------------------
    def _compute_quality(
        self, mars_results: Dict, scores_payload: Dict
    ) -> float:
        """Combine MARS agreement (and future HRM/LLM/reward) into one scalar."""
        mars_agree = 0.0
        if mars_results:
            vals = [
                float(v.get("agreement_score", 0.0))
                for v in mars_results.values()
            ]
            mars_agree = (sum(vals) / len(vals)) if vals else 0.0
        hrm_score = 0.0
        llm_grade = 0.0
        task_reward = 0.0
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
        """Ensure and return the casebook id for (pipeline_run_id, agent_name, tag)."""
        pk = context[PIPELINE_RUN_ID]
        agent = self.name
        try:
            cb = self._ensure_casebook_scope(pk, agent, self.casebook_tag)
            return cb.id
        except AttributeError:
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
        """Prefer CaseBookStore.ensure_casebook_scope; fallback to a named casebook."""
        try:
            return self.memory.casebooks.ensure_casebook_scope(
                pipeline_run_id, agent_name, tag
            )
        except AttributeError:
            name = f"cb:{agent_name or 'all'}:{pipeline_run_id or 'all'}:{tag}"
            return self.memory.casebooks.ensure_casebook(
                name,
                pipeline_run_id=pipeline_run_id,
                description="Scoped fallback",
                tags=[tag],
            )

    def _bump_and_get_run_counter(self, casebook_id: int, goal_id: str) -> int:
        """
        Increment and return a per-(casebook,goal) counter to schedule A/B runs.
        Tries to piggy-back on CaseGoalState.run_ix; falls back to in-memory counter.
        """
        try:
            state = self.memory.casebooks.get_goal_state(casebook_id, goal_id)
            if state is None:
                self.memory.casebooks.upsert_goal_state(
                    casebook_id, goal_id, case_id=None, quality=0.0
                )
                state = self.memory.casebooks.get_goal_state(
                    casebook_id, goal_id
                )
            return self.memory.casebooks.bump_run_ix(casebook_id, goal_id)

        except Exception:
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
        return run_index % max(1, self.ab_period) == 0

    @staticmethod
    def _set_random_seed(seed: int):
        """Lock RNGs if available; tolerate missing libs."""
        random.seed(seed)
        global np, torch
        if np is None:
            try:
                import numpy as _np  # type: ignore

                np = _np
            except Exception:
                np = None
        if np is not None:
            try:
                np.random.seed(seed)  # type: ignore
            except Exception:
                pass
        if torch is None:
            try:
                import torch as _torch  # type: ignore

                torch = _torch
            except Exception:
                torch = None
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
        Emit training signals + kick tiny online steps.
        Prefers DB-backed TrainingStore; falls back to in-memory TrainingBuffers.
        Also nudges TrainingController for validate+maybe-train.
        """
        if not self._training_enabled or not ranked or len(ranked) < 2:
            return

        goal = context[GOAL]
        goal_id = goal.get("id", "")
        goal_text = goal.get("goal_text", "")
        pipeline_run_id = context.get(PIPELINE_RUN_ID)
        agent_name = self.name

        tr_cfg = self.cfg.get("training", {}) or {}
        model_key_ranker = tr_cfg.get("buffers", {}).get(
            "model_key_ranker", "ranker.sicql.v1"
        )
        model_key_retriever = tr_cfg.get("buffers", {}).get(
            "model_key_retriever", "retriever.mrq.v1"
        )
        dimension = tr_cfg.get("dimension", "alignment")

        def agree(hid: Optional[str]) -> float:
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

        pos_w = max(0.1, agree(pos_id))  # keep a small floor

        pairs_for_validation = []
        neg_texts: List[str] = []

        # Pairwise per negative; collect negatives for pointwise later
        for r in ranked[1:]:
            neg_text = r.get("text", "") or ""
            if not neg_text:
                continue
            neg_texts.append(neg_text)

            neg_id = r.get("id")
            neg_w = agree(neg_id)
            pair_w = max(0.1, 0.5 * (pos_w + neg_w))  # conservative

            pairs_for_validation.append(
                {"text_a": pos_text, "text_b": neg_text, "weight": pair_w}
            )

            try:
                self.memory.training_events.insert_pairwise({
                    "model_key": model_key_ranker,
                    "dimension": dimension,
                    "query_text": goal_text,
                    "pos_text": pos_text,
                    "neg_text": neg_text,
                    "weight": pair_w,
                    "trust": neg_w,
                    "goal_id": goal_id,
                    "pipeline_run_id": pipeline_run_id,
                    "agent_name": agent_name,
                    "source": "memento",
                    "meta": {"run_id": pipeline_run_id},
                })
            except Exception as e:
                self.logger.log(
                    "TrainStoreAddPairwiseError", {"error": str(e)}
                )

            try:
                self.memory.training_events.insert_pointwise({
                    "model_key": model_key_retriever,
                    "dimension": dimension,
                    "query_text": goal_text,
                    "cand_text": pos_text,
                    "label": 1,
                    "weight": pos_w,
                    "trust": pos_w,
                    "goal_id": goal_id,
                    "pipeline_run_id": pipeline_run_id,
                    "agent_name": agent_name,
                    "source": "memento",
                    "meta": {"run_id": pipeline_run_id},
                })
                for nt in neg_texts:
                    self.memory.training_events.insert_pointwise({
                        "model_key": model_key_retriever,
                        "dimension": dimension,
                        "query_text": goal_text,
                        "cand_text": nt,
                        "label": 0,
                        "weight": 0.5,
                        "trust": 0.0,
                        "goal_id": goal_id,
                        "pipeline_run_id": pipeline_run_id,
                        "agent_name": agent_name,
                        "source": "memento",
                        "meta": {"run_id": pipeline_run_id},
                    })
            except Exception as e:
                self.logger.log(
                    "TrainStoreAddPointwiseError", {"error": str(e)}
                )

        # Kick controller (validate+maybe-train)
        try:
            self.container.get("training").maybe_train(
                goal=goal_id, dimension=dimension, pairs=pairs_for_validation
            )
        except Exception as e:
            self.logger.log("TrainingControllerCallFailed", {"error": str(e)})

        # Tiny online steps (best-effort)
        try:
            steps = int(
                self.cfg.get("trainer", {})
                .get("sicql", {})
                .get("online_steps", 50)
            )
            self.container.get("training").trainers["sicql"].train_step(
                max_steps=steps
            )
        except Exception:
            pass
        try:
            steps = int(
                self.cfg.get("trainer", {})
                .get("mrq", {})
                .get("online_steps", 50)
            )
            self.container.get("training").trainers["mrq"].train_step(
                max_steps=steps
            )
        except Exception:
            pass

    # -------------------------
    # Small utilities
    # -------------------------
    def _ensure_ids(self, hypos: List[dict]) -> List[dict]:
        """
        Ensure each hypothesis has an 'id'. If missing, derive a stable 16-char id
        from its text; this plays nicely with MARS, retention, and training.
        """
        out = []
        for h in hypos:
            if h.get("id"):
                out.append(h)
                continue
            text = (h.get("text") or "").strip()
            sid = (
                hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
                if text
                else hashlib.sha1(
                    str(random.random()).encode("utf-8")
                ).hexdigest()[:16]
            )
            nh = dict(h)
            nh["id"] = sid
            out.append(nh)
        return out

    def _ns(self, ctx: dict) -> dict:
        """Ensure and return our namespaced bucket."""
        return ctx.setdefault(CTX_NS, {VARIANTS: {}})

    def _vb(self, ctx: dict, variant: str) -> dict:
        """Ensure and return a per-variant subbucket."""
        return self._ns(ctx)[VARIANTS].setdefault(variant, {})

    def _variant_output_key(self, variant: str) -> str:
        """A private output key so baseline & CBR never overwrite each other."""
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
        Temporarily point self.output_key at a private, namespaced key so
        super().run writes into our namespace instead of the canonical output.
        """
        prev = self.output_key
        self.output_key = self._variant_output_key(variant)
        try:
            yield
        finally:
            self.output_key = prev

    # -------------------------
    # Reporting convenience
    # -------------------------
    def _report_event(self, event: str, payload: Dict[str, Any]):
        """Thin wrapper around self.report with error-guard."""
        try:
            self.report({"event": event, **payload})
        except Exception:
            # Reporting must never break the pipeline
            if self.logger:
                self.logger.log("MementoReportSkip", {"event": event})
