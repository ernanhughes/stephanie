<!-- Merged Python Code Files -->


## File: __init__.py

`python
``n

## File: ab_validator.py

`python
# stephanie/cbr/ab_validator.py
from typing import Callable, Tuple, Dict

class DefaultABValidator:
    def __init__(self, cfg, memory, logger, ns, assessor):
        self.cfg, self.memory, self.logger = cfg, memory, logger
        self.ns, self.assessor = ns, assessor
        self.delta_eps = float(cfg.get("ab_validation",{}).get("delta_eps", 1e-6))

    async def run_two(self, ctx, run_cbr: Callable[[], dict], run_baseline: Callable[[], dict]) -> Tuple[str, Dict]:
        base = await run_baseline()
        cbr  = await run_cbr()
        q_base = float(base["metrics"]["quality"])
        q_cbr  = float(cbr["metrics"]["quality"])
        improved = q_cbr > (q_base + self.delta_eps)
        winner = "cbr" if improved else "baseline"
        return winner, {"q_base": q_base, "q_cbr": q_cbr, "improved": improved, "winner": winner}
``n

## File: case_selector.py

`python
# stephanie/cbr/case_selector.py
import random
from typing import List

class DefaultCaseSelector:
    def __init__(self, cfg, memory, logger):
        self.cfg, self.memory, self.logger = cfg, memory, logger

    @staticmethod
    def _top_scorables_from_case(case, k=3) -> List[str]:
        out = []
        try:
            outs = [cs for cs in case.scorables if (getattr(cs, "role", "") or "").lower() == "output"]
            outs.sort(key=lambda cs: getattr(cs, "rank", 1_000_000))
            for cs in outs[:k]:
                sid = getattr(cs, "scorable_id", None)
                if sid: out.append(sid)
        except Exception:
            pass
        return out

    def build_reuse_candidates(self, casebook_id: int, goal_id: str, cases, budget: int, novelty_k: int, exploration_eps: float) -> List[str]:
        ids: List[str] = []

        # Champion-first
        try:
            state = self.memory.casebooks.get_goal_state(casebook_id, goal_id)
            if state and state.champion_case_id:
                champion = next((c for c in cases if c.id == state.champion_case_id), None)
                if champion:
                    ids.extend(self._top_scorables_from_case(champion))
        except AttributeError:
            pass

        # Recent-success
        try:
            recent = self.memory.casebooks.get_recent_cases(casebook_id, goal_id, limit=max(1, budget // 2), only_accepted=True)
            for c in recent: ids.extend(self._top_scorables_from_case(c))
        except AttributeError:
            for c in cases[: max(1, budget // 2)]: ids.extend(self._top_scorables_from_case(c))

        # Diverse-novel
        pool_ids = set(ids)
        novel_pool = []
        try:
            pool = self.memory.casebooks.get_pool_for_goal(casebook_id, goal_id, exclude_ids=[getattr(c, "id", None) for c in cases], limit=200)
            novel_pool = pool or []
        except AttributeError:
            novel_pool = [c for c in cases if c.id not in pool_ids]
        random.shuffle(novel_pool)
        for c in novel_pool[:novelty_k]:
            ids.extend(self._top_scorables_from_case(c))

        # Exploration
        if random.random() < float(exploration_eps):
            extra = (cases[novelty_k : novelty_k + 2]) if len(cases) > novelty_k else []
            for c in extra: ids.extend(self._top_scorables_from_case(c))

        # Dedup + cap
        seen, out = set(), []
        for x in ids:
            if x and x not in seen:
                out.append(x); seen.add(x)
            if len(out) >= budget: break
        return out
``n

## File: casebook_scope_manager.py

`python
# stephanie/cbr/casebook_scope_manager.py
from typing import Any, Dict, List, Optional
from stephanie.constants import GOAL, PIPELINE_RUN_ID

class DefaultCasebookScopeManager:
    def __init__(self, cfg, memory, logger):
        self.cfg, self.memory, self.logger = cfg, memory, logger
        self.tag = cfg.get("casebook_tag", "default")
        self.retrieval_mode = cfg.get("retrieval_mode", "fallback")

    def ensure_scope(self, pipeline_run_id: Optional[str], agent: Optional[str], tag: str):
        try:
            return self.memory.casebooks.ensure_casebook_scope(pipeline_run_id, agent, tag)
        except AttributeError:
            name = f"cb:{agent or 'all'}:{pipeline_run_id or 'all'}:{tag}"
            return self.memory.casebooks.ensure_casebook(name, description="Scoped fallback")

    def home_casebook_id(self, ctx: Dict, agent_name: str, tag: str) -> int:
        cb = self.ensure_scope(ctx[PIPELINE_RUN_ID], agent_name, tag)
        return cb.id

    def get_cases(self, ctx: Dict, retrieval_mode: str, tag: str) -> List[Any]:
        goal_id = ctx[GOAL]["id"]
        pipeline_run_id = ctx[PIPELINE_RUN_ID]
        agent = ctx.get("agent_name") or "UnknownAgent"

        scopes = [(pipeline_run_id, agent, tag)]
        if retrieval_mode in ("fallback", "union"):
            scopes += [(None, agent, tag), (pipeline_run_id, None, tag), (None, None, tag)]

        try:
            if retrieval_mode == "strict":
                cb = self.ensure_scope(pipeline_run_id, agent, tag)
                return self.memory.casebooks.get_cases_for_goal_in_casebook(cb.id, goal_id)
            elif retrieval_mode == "fallback":
                for sc in scopes:
                    cases = self.memory.casebooks.get_cases_for_goal_scoped(goal_id, [sc])
                    if cases: return cases
                return []
            else:
                return self.memory.casebooks.get_cases_for_goal_scoped(goal_id, scopes)
        except AttributeError:
            return self.memory.casebooks.get_cases_for_goal(goal_id)
``n

## File: champion_promoter.py

`python
# stephanie/cbr/champion_promoter.py
class DefaultChampionPromoter:
    def __init__(self, cfg, memory, logger):
        self.cfg, self.memory, self.logger = cfg, memory, logger

    def maybe_promote(self, casebook_id: int, goal_id: str, retained_case_id: int | None, quality: float) -> None:
        if not retained_case_id: return
        try:
            self.memory.casebooks.upsert_goal_state(casebook_id, goal_id, retained_case_id, float(quality))
        except AttributeError:
            pass
``n

## File: context_namespacer.py

`python
# stephanie/cbr/context_namespacer.py
from __future__ import annotations
from contextlib import contextmanager
from typing import Dict, Any

CTX_NS = "_MEMENTO"
VARIANTS = "variants"

class DefaultContextNamespacer:
    def ns(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return ctx.setdefault(CTX_NS, {VARIANTS: {}})

    def variant_bucket(self, ctx: Dict[str, Any], variant: str) -> Dict[str, Any]:
        return self.ns(ctx)[VARIANTS].setdefault(variant, {})

    def variant_output_key(self, variant: str) -> str:
        return f"{CTX_NS}.{variant}.output"

    @contextmanager
    def temp_key(self, ctx: Dict[str, Any], key: str, value):
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
``n

## File: goal_state_tracker.py

`python
# stephanie/cbr/goal_state_tracker.py
class DefaultGoalStateTracker:
    def __init__(self, cfg, memory, logger):
        self.cfg, self.memory, self.logger = cfg, memory, logger
        self._mem = {}

    def bump_run_ix(self, casebook_id: int, goal_id: str) -> int:
        try:
            state = self.memory.casebooks.get_goal_state(casebook_id, goal_id)
            if state is None:
                self.memory.casebooks.upsert_goal_state(casebook_id, goal_id, case_id=None, quality=0.0)
                state = self.memory.casebooks.get_goal_state(casebook_id, goal_id)
            ix = getattr(state, "run_ix", 0) + 1
            setattr(state, "run_ix", ix)
            self.memory.casebooks.session.commit()  # type: ignore
            return ix
        except Exception:
            key = f"{casebook_id}:{goal_id}"
            self._mem[key] = self._mem.get(key, 0) + 1
            return self._mem[key]

    def should_run_ab(self, run_ix: int, mode: str, period: int) -> bool:
        if mode == "off": return False
        if mode == "always": return True
        return run_ix % max(1, period) == 0
``n

## File: interfaces.py

`python
# stephanie/cbr/interfaces.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

Context = Dict[str, Any]
ScorableLike = Dict[str, Any]  # {id, text, ...}

class ContextNamespacer(ABC):
    @abstractmethod
    def ns(self, ctx: Context) -> Context: ...
    @abstractmethod
    def variant_bucket(self, ctx: Context, variant: str) -> Context: ...
    @abstractmethod
    def variant_output_key(self, variant: str) -> str: ...
    @abstractmethod
    def temp_key(self, ctx: Context, key: str, value):  # contextmanager
        ...

class CasebookScopeManager(ABC):
    @abstractmethod
    def home_casebook_id(self, ctx: Context, agent_name: str, tag: str) -> int: ...
    @abstractmethod
    def ensure_scope(self, pipeline_run_id: Optional[str], agent: Optional[str], tag: str): ...
    @abstractmethod
    def get_cases(self, ctx: Context, retrieval_mode: str, tag: str) -> List[Any]: ...

class CaseSelector(ABC):
    @abstractmethod
    def build_reuse_candidates(self, casebook_id: int, goal_id: str, cases: List[Any],
                               budget: int, novelty_k: int, exploration_eps: float) -> List[str]: ...

class RankAndAnalyze(ABC):
    @abstractmethod
    def run(self, ctx: Context, hypotheses: List[ScorableLike]) -> Tuple[List[ScorableLike], Dict, Dict, List[str], Dict]:
        """returns (ranked, corpus, mars_results, recommendations, scores_payload)"""

class RetentionPolicy(ABC):
    @abstractmethod
    def retain(self, ctx: Context, ranked: List[ScorableLike], mars: Dict, scores: Dict) -> Optional[int]: ...

class QualityAssessor(ABC):
    @abstractmethod
    def quality(self, mars_results: Dict, scores_payload: Dict) -> float: ...

class ChampionPromoter(ABC):
    @abstractmethod
    def maybe_promote(self, casebook_id: int, goal_id: str, retained_case_id: Optional[int], quality: float) -> None: ...

class GoalStateTracker(ABC):
    @abstractmethod
    def bump_run_ix(self, casebook_id: int, goal_id: str) -> int: ...
    @abstractmethod
    def should_run_ab(self, run_ix: int, mode: str, period: int) -> bool: ...

class ABValidator(ABC):
    @abstractmethod
    async def run_two(self, ctx: Context, run_cbr, run_baseline) -> Tuple[str, Dict]:
        """Executes (baseline, cbr) variants, compares quality, returns (winner_variant, comparison_report)."""

class MicroLearner(ABC):
    @abstractmethod
    def learn(self, ctx: Context, ranked: List[ScorableLike], mars: Dict) -> None: ...
``n

## File: micro_learner.py

`python
# stephanie/cbr/micro_learner.py
from typing import Dict, List

class DefaultMicroLearner:
    def __init__(self, cfg, memory, logger):
        self.cfg, self.memory, self.logger = cfg, memory, logger

    def learn(self, ctx: Dict, ranked: List[Dict], mars: Dict) -> None:
        if not ranked or len(ranked) < 2: return
        # (Use your DB-backed training_store + controller as you already implemented)
        # Emit pairwise/pointwise + call controller.maybe_train(...)
        try:
            # … paste your fixed _online_learn guts here …
            pass
        except Exception as e:
            if self.logger: self.logger.log("MicroLearnError", {"error": str(e)})
``n

## File: middleware.py

`python
# stephanie/cbr/middleware.py
from typing import Dict, Any, Callable, Awaitable, List

class CBRMiddleware:
    def __init__(self, cfg, memory, logger,
                 ns, scope_mgr, selector, ranker, retention, assessor,
                 promoter, tracker, ab_validator, micro_learner):
        self.cfg, self.memory, self.logger = cfg, memory, logger
        self.ns, self.scope_mgr, self.selector = ns, scope_mgr, selector
        self.ranker, self.retention, self.assessor = ranker, retention, assessor
        self.promoter, self.tracker, self.ab = promoter, tracker, ab_validator
        self.micro = micro_learner
        self.tag = cfg.get("casebook_tag", "default")
        self.retrieval_mode = cfg.get("retrieval_mode","fallback")
        self.reuse_budget = int(cfg.get("reuse_budget",16))
        self.novelty_k = int(cfg.get("novelty_k",6))
        self.exploration_eps = float(cfg.get("exploration_eps",0.1))
        self.ab_cfg = cfg.get("ab_validation", {}) or {}

    async def run(self, ctx: Dict[str, Any], base_agent_run: Callable[[Dict], Awaitable[Dict]], output_key: str) -> Dict[str, Any]:
        # minimal info for scope mgr
        ctx["agent_name"] = ctx.get("agent_name") or ctx.get("AGENT_NAME") or "UnknownAgent"

        # Decide A/B
        casebook_id = self.scope_mgr.home_casebook_id(ctx, ctx["agent_name"], self.tag)
        goal_id = ctx["GOAL"]["id"]
        run_ix = self.tracker.bump_run_ix(casebook_id, goal_id)
        do_ab = self.tracker.should_run_ab(run_ix, self.ab_cfg.get("mode","periodic"), int(self.ab_cfg.get("period",5)))

        async def _run_variant(variant: str, use_casebook: bool, retain: bool, train_enabled: bool):
            vb = self.ns.variant_bucket(ctx, variant)
            cases = self.scope_mgr.get_cases(ctx, self.retrieval_mode, self.tag) if use_casebook else []
            reuse_candidates = self.selector.build_reuse_candidates(casebook_id, goal_id, cases, self.reuse_budget, self.novelty_k, self.exploration_eps) if use_casebook else []

            # expose reuse_candidates only temporarily
            with self.ns.temp_key(ctx, "reuse_candidates", reuse_candidates), self._variant_output_redirect(ctx, variant, output_key):
                # 1) run base agent (will look at context["reuse_candidates"] if it wants)
                lats_result = await base_agent_run(ctx)
                hypos = lats_result.get("hypotheses", []) or []
                hypos = self._ensure_ids(hypos)

                # 2) rank + analyze
                ranked, corpus, mars, recs, scores = self.ranker.run(ctx, hypos)
                for r in ranked:
                    rid = r.get("id"); r["mars_confidence"] = (mars.get(rid, {}) or {}).get("agreement_score")

                # 3) retain & micro-learn
                retained_case_id = self.retention.retain(ctx, ranked, mars, scores) if retain else None
                if train_enabled: self.micro.learn(ctx, ranked, mars)

                # 4) quality
                q = self.assessor.quality(mars, scores)
                vb.update(dict(
                    reuse_candidates=reuse_candidates, ranked=ranked,
                    mars_results=mars, recommendations=recs, retained_case=retained_case_id,
                    metrics={"quality": float(q), "variant": variant}
                ))
                return vb

        if do_ab:
            # baseline vs cbr
            base_res = await _run_variant("baseline", use_casebook=False, retain=False, train_enabled=not bool(self.ab_cfg.get("freeze_training",True)))
            cbr_res  = await _run_variant("cbr",      use_casebook=True,  retain=True,  train_enabled=True)
            winner, cmp = await self.ab.run_two(ctx, lambda: self._noop(base_res), lambda: self._noop(base_res))  # NOTE: plug real comparison if you want different policy
            # prefer the actual comparison:
            q_base, q_cbr = base_res["metrics"]["quality"], cbr_res["metrics"]["quality"]
            improved = q_cbr > q_base + float(self.ab_cfg.get("delta_eps",1e-6))
            winner = "cbr" if improved else "baseline"
            # champion promote only if CBR improved
            if improved: self.promoter.maybe_promote(casebook_id, goal_id, cbr_res.get("retained_case"), float(q_cbr))
            ctx[output_key] = ctx.get(self.ns.variant_output_key(winner), [])
            return ctx

        # Single CBR
        cbr_res = await _run_variant("cbr", use_casebook=True, retain=True, train_enabled=True)
        ctx[output_key] = ctx.get(self.ns.variant_output_key("cbr"), [])
        return ctx

    # -- helpers
    async def _noop(self, x): return x

    def _ensure_ids(self, hypos: List[dict]) -> List[dict]:
        import hashlib, random
        out = []
        for h in hypos:
            if h.get("id"): out.append(h); continue
            text = (h.get("text") or "").strip()
            sid = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16] if text else hashlib.sha1(str(random.random()).encode("utf-8")).hexdigest()[:16]
            nh = dict(h); nh["id"] = sid; out.append(nh)
        return out

    from contextlib import contextmanager
    @contextmanager
    def _variant_output_redirect(self, ctx, variant: str, output_key: str):
        prev = output_key
        # we can’t mutate agent.output_key here, so we write to namespaced output, and the caller will read it
        try:
            yield
        finally:
            pass
``n

## File: modular_memento.py

`python
# stephanie/agents/cbr/modular_memento.py
from stephanie.cbr.context_namespacer import DefaultContextNamespacer
from stephanie.cbr.casebook_scope_manager import DefaultCasebookScopeManager
from stephanie.cbr.case_selector import DefaultCaseSelector
from stephanie.cbr.rank_and_analyze import DefaultRankAndAnalyze
from stephanie.cbr.retention_policy import DefaultRetentionPolicy
from stephanie.cbr.quality_assessor import DefaultQualityAssessor
from stephanie.cbr.champion_promoter import DefaultChampionPromoter
from stephanie.cbr.goal_state_tracker import DefaultGoalStateTracker
from stephanie.cbr.ab_validator import DefaultABValidator
from stephanie.cbr.micro_learner import DefaultMicroLearner
from stephanie.cbr.middleware import CBRMiddleware
from stephanie.agents.dspy.mcts_reasoning import MCTSReasoningAgent
from stephanie.scoring.scorer.scorable_ranker import ScorableRanker
from stephanie.scoring.calculations.mars_calculator import MARSCalculator

class ModularMementoAgent(MCTSReasoningAgent):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        ns = DefaultContextNamespacer()
        scope = DefaultCasebookScopeManager(cfg, memory, logger)
        selector = DefaultCaseSelector(cfg, memory, logger)
        ranker = DefaultRankAndAnalyze(cfg, memory, logger, ranker=ScorableRanker(cfg, memory, logger),
                                       mars=MARSCalculator(cfg, memory, logger) if cfg.get("include_mars", True) else None)
        retention = DefaultRetentionPolicy(cfg, memory, logger, casebook_scope_mgr=scope)
        assessor = DefaultQualityAssessor(cfg, memory, logger)
        promoter = DefaultChampionPromoter(cfg, memory, logger)
        tracker = DefaultGoalStateTracker(cfg, memory, logger)
        ab = DefaultABValidator(cfg, memory, logger, ns=ns, assessor=assessor)
        micro = DefaultMicroLearner(cfg, memory, logger)

        self._cbr = CBRMiddleware(cfg, memory, logger, ns, scope, selector, ranker, retention, assessor, promoter, tracker, ab, micro)

    async def run(self, context: dict) -> dict:
        # This delegates “CBR extras” to the middleware, using this agent’s base run as the core.
        context["agent_name"] = self.name
        async def base_run(ctx):  # what CBR wraps: your monolithic base behavior
            return await super().run(ctx)
        result_ctx = await self._cbr.run(context, base_run, self.output_key)
        return result_ctx
``n

## File: quality_assessor.py

`python
# stephanie/cbr/quality_assessor.py
from typing import Dict

class DefaultQualityAssessor:
    def __init__(self, cfg, memory, logger):
        self.cfg, self.memory, self.logger = cfg, memory, logger
        self.qw = cfg.get("quality_weights", {}) or {"mars":1.0,"hrm":0.5,"reward":2.0,"llm":0.25}

    def quality(self, mars_results: Dict, scores_payload: Dict) -> float:
        mars_agree = 0.0
        if mars_results:
            vals = [float(v.get("agreement_score", 0.0)) for v in mars_results.values()]
            mars_agree = (sum(vals)/len(vals)) if vals else 0.0
        hrm_score = 0.0; llm_grade = 0.0; task_reward = 0.0
        return (self.qw["mars"]*mars_agree + self.qw["hrm"]*hrm_score +
                self.qw["llm"]*llm_grade + self.qw["reward"]*task_reward)
``n

## File: rank_and_analyze.py

`python
# stephanie/cbr/rank_and_analyze.py
from typing import Dict, List, Tuple
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.data.score_corpus import ScoreCorpus
from stephanie.constants import GOAL

class DefaultRankAndAnalyze:
    def __init__(self, cfg, memory, logger, ranker, mars=None):
        self.cfg, self.memory, self.logger = cfg, memory, logger
        self.ranker, self.mars = ranker, mars

    def _normalize(self, item) -> dict:
        if isinstance(item, dict):
            r = dict(item)
            r.setdefault("rank", r.get("position") or r.get("order") or 0)
            return r
        return {
            "id": getattr(item, "id", None) or getattr(item, "scorable_id", None),
            "text": getattr(item, "text", "") or getattr(item, "content", ""),
            "rank": getattr(item, "rank", 0),
        }

    def run(self, ctx, hypotheses: List[dict]) -> Tuple[List[dict], Dict, Dict, List[str], Dict]:
        goal = ctx[GOAL]
        ranked, bundles = [], {}

        if hypotheses:
            query = Scorable(id=goal["id"], text=goal["goal_text"], target_type=TargetType.GOAL)
            scorables = [Scorable(id=h.get("id"), text=h.get("text",""), target_type=TargetType.HYPOTHESIS) for h in hypotheses]
            ranked_raw = self.ranker.rank(query=query, candidates=scorables, context=ctx) or []
            ranked = [self._normalize(r) for r in ranked_raw]
            for s in scorables:
                try:
                    _scores, bundle = self.ranker._score(context=ctx, scorable=s)  # or inject scorer if separate
                    bundles[s.id] = bundle
                except Exception:
                    pass

        corpus = ScoreCorpus(bundles=bundles)
        mars_results, recommendations = {}, []
        if self.mars:
            mars_results = self.mars.calculate(corpus, context=ctx) or {}
            recommendations = self.mars.generate_recommendations(mars_results) or []

        scores_payload = {}
        for sid, bundle in bundles.items():
            if hasattr(bundle, "to_dict"):
                try: scores_payload[sid] = bundle.to_dict()
                except Exception: scores_payload[sid] = {}
            else:
                scores_payload[sid] = {}

        return ranked, corpus, mars_results, recommendations, scores_payload
``n

## File: retention_policy.py

`python
# stephanie/cbr/retention_policy.py
from typing import Dict, List, Optional
from stephanie.constants import GOAL, PIPELINE_RUN_ID

class DefaultRetentionPolicy:
    def __init__(self, cfg, memory, logger, casebook_scope_mgr):
        self.cfg, self.memory, self.logger = cfg, memory, logger
        self.scope_mgr = casebook_scope_mgr
        self.tag = cfg.get("casebook_tag", "default")

    def retain(self, ctx: Dict, ranked: List[Dict], mars: Dict, scores: Dict) -> Optional[int]:
        goal = ctx[GOAL]
        casebook_id = self.scope_mgr.home_casebook_id(ctx, ctx.get("agent_name") or "UnknownAgent", self.tag)

        scorables_payload = []
        for idx, r in enumerate(ranked):
            scorables_payload.append({
                "id": r.get("id"),
                "type": "hypothesis",
                "role": "output",
                "rank": (idx + 1),
                "meta": {"text": r.get("text",""), "mars_confidence": r.get("mars_confidence")},
            })

        try:
            case = self.memory.casebooks.add_case(
                casebook_id=casebook_id,
                goal_id=goal["id"],
                goal_text=goal["goal_text"],
                agent_name=ctx.get("agent_name") or "UnknownAgent",
                mars_summary=mars,
                scores=scores,
                metadata={
                    "pipeline_run_id": ctx.get(PIPELINE_RUN_ID),
                    "casebook_tag": self.tag,
                    "hypothesis_count": len(ranked),
                },
                scorables=scorables_payload,
            )
            return case.id
        except Exception:
            return None
``n
