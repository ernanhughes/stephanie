# stephanie/cbr/middleware.py
from typing import Dict, Any, Callable, Awaitable, List
from stephanie.constants import GOAL

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

    async def run(self, context: Dict[str, Any], base_agent_run: Callable[[Dict], Awaitable[Dict]], output_key: str) -> Dict[str, Any]:
        # minimal info for scope mgr
        context["agent_name"] = context.get("agent_name") or context.get("AGENT_NAME") or "UnknownAgent"

        # Decide A/B
        casebook_id = self.scope_mgr.home_casebook_id(context, context["agent_name"], self.tag)
        goal_id = context[GOAL]["id"]
        run_ix = self.tracker.bump_run_ix(casebook_id, goal_id)
        do_ab = self.tracker.should_run_ab(run_ix, self.ab_cfg.get("mode","periodic"), int(self.ab_cfg.get("period",5)))

        async def _run_variant(variant: str, use_casebook: bool, retain: bool, train_enabled: bool):
            vb = self.ns.variant_bucket(context, variant)
            cases = self.scope_mgr.get_cases(context, self.retrieval_mode, self.tag) if use_casebook else []
            reuse_candidates = self.selector.build_reuse_candidates(casebook_id, goal_id, cases, self.reuse_budget, self.novelty_k, self.exploration_eps) if use_casebook else []

            # expose reuse_candidates only temporarily
            with self.ns.temp_key(context, "reuse_candidates", reuse_candidates), self._variant_output_redirect(context, variant, output_key):
                # 1) run base agent (will look at context["reuse_candidates"] if it wants)
                lats_result = await base_agent_run(context)
                hypos = lats_result.get("hypotheses", []) or []
                hypos = self._ensure_ids(hypos)

                # 2) rank + analyze
                ranked, corpus, mars, recs, scores = self.ranker.run(context, hypos)
                for r in ranked:
                    rid = r.get("id"); r["mars_confidence"] = (mars.get(rid, {}) or {}).get("agreement_score")

                # 3) retain & micro-learn
                retained_case_id = self.retention.retain(context, ranked, mars, scores) if retain else None
                if train_enabled: self.micro.learn(context, ranked, mars)

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
            winner, cmp = await self.ab.run_two(context, lambda: self._noop(base_res), lambda: self._noop(base_res))  # NOTE: plug real comparison if you want different policy
            # prefer the actual comparison:
            q_base, q_cbr = base_res["metrics"]["quality"], cbr_res["metrics"]["quality"]
            improved = q_cbr > q_base + float(self.ab_cfg.get("delta_eps",1e-6))
            winner = "cbr" if improved else "baseline"
            # champion promote only if CBR improved
            if improved: self.promoter.maybe_promote(casebook_id, goal_id, cbr_res.get("retained_case"), float(q_cbr))
            context[output_key] = context.get(self.ns.variant_output_key(winner), [])
            return context

        # Single CBR
        cbr_res = await _run_variant("cbr", use_casebook=True, retain=True, train_enabled=True)
        context[output_key] = context.get(self.ns.variant_output_key("cbr"), [])
        return context

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
