# add imports
from typing import Any, Awaitable, Callable, Dict, List, Optional

import numpy as np

from stephanie.cbr.adaptor import DefaultAdaptor
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable import ScorableType
from stephanie.services.reporting_service import ReportingService


class CBRMiddleware:
    def __init__(
        self,
        cfg,
        memory,
        container,
        logger,
        ns,
        scope_mgr,
        selector,
        ranker,
        retention,
        assessor,
        promoter,
        tracker,
        ab_validator,
        micro_learner,
        scoring_service=None,
        adaptor: Optional[DefaultAdaptor] = None,
        reporter: ReportingService | None = None,
    ):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self.ns = ns
        self.scope_mgr = scope_mgr
        self.selector = selector
        self.ranker = ranker
        self.retention = retention
        self.assessor = assessor
        self.promoter = promoter
        self.tracker = tracker
        self.ab = ab_validator
        self.micro = micro_learner

        self.scoring = scoring_service  # may be None if ranker exposes a score_one; see _score_scorable below
        self.adaptor = adaptor or DefaultAdaptor(
            cfg.get("adapt", {}), logger=self.logger
        )

        self.tag = cfg.get("casebook_tag", "default")
        self.retrieval_mode = cfg.get("retrieval_mode", "fallback")
        self.reuse_budget = int(cfg.get("reuse_budget", 16))
        self.novelty_k = int(cfg.get("novelty_k", 6))
        self.exploration_eps = float(cfg.get("exploration_eps", 0.1))
        self.ab_cfg = cfg.get("ab_validation", {}) or {}

        # Adapt config
        a = cfg.get("adapt", {}) or {}
        self.adapt_enabled = bool(a.get("enabled", True))
        self.adapt_top_k = int(a.get("top_k", 1))
        self.adapt_improv_eps = float(a.get("improv_eps", 0.01))
        self.adapt_dimensions = a.get(
            "dimensions"
        )  # None → let scorer defaults apply
        self.adapt_scorer_name = a.get(
            "scorer_name"
        )  # None → use ranker/scoring defaults

        # Reporting
        self.reporter = container.get("reporting")

    async def run(
        self,
        context: Dict[str, Any],
        base_agent_run: Callable[[Dict], Awaitable[Dict]],
        output_key: str,
    ) -> Dict[str, Any]:
        await self.reporter.emit(
            ctx=context,
            stage="start",
            note="CBR entry",
            tag=self.tag,
            retrieval_mode=self.retrieval_mode,
        )
        context_key_agent = "agent_name"
        context[context_key_agent] = (
            context.get(context_key_agent) or "UnknownAgent"
        )

        casebook_id = self.scope_mgr.home_casebook_id(
            context, context[context_key_agent], self.tag
        )
        goal_id = context["goal"]["id"]
        run_ix = self.tracker.bump_run_ix(casebook_id, goal_id)
        do_ab = self.tracker.should_run_ab(
            run_ix,
            self.ab_cfg.get("mode", "periodic"),
            int(self.ab_cfg.get("period", 5)),
        )

        async def _run_variant(
            variant: str, use_casebook: bool, retain: bool, train_enabled: bool
        ):
            stage_name = f"variant:{variant}"
            vb = self.ns.variant_bucket(context, variant)
            cases = (
                self.scope_mgr.get_cases(
                    context, self.retrieval_mode, self.tag
                )
                if use_casebook
                else []
            )
            reuse_candidates = (
                self.selector.build_reuse_candidates(
                    casebook_id,
                    goal_id,
                    cases,
                    self.reuse_budget,
                    self.novelty_k,
                    self.exploration_eps,
                )
                if use_casebook
                else []
            )

            with (
                self.ns.temp_key(
                    context, "reuse_candidates", reuse_candidates
                ),
                self._variant_output_redirect(context, variant, output_key),
            ):
                await self.reporter.emit(
                    ctx=context,
                    stage=stage_name,
                    status="running",
                    summary=f"Variant {variant} begin",
                    event="variant_begin",
                    use_casebook=use_casebook,
                )
                # 1) Base agent produces a fresh scorable (or few)
                result = await base_agent_run(context)
                produced = result.get(output_key, []) or []
                await self.reporter.emit(
                    ctx=context,
                    stage=stage_name,
                    event="agent_output",
                    produced=len(produced),
                    preview=_safe_texts(produced),
                )

                # Ensure IDs
                produced = self._ensure_ids(produced)
                await self.reporter.emit(
                    ctx=context,
                    stage="agent_output",
                    variant=variant,
                    produced=len(produced),
                    preview=_safe_texts(produced),
                )

                # 2) Deduplicate semantically
                seeds = _dedupe_semantic(produced, self.memory.embedding, thresh=0.92)


                # 3) Rank + analyze (fresh + retrieved-as-scorables — ranker.run should already merge if you pass both)
                #    If your ranker expects the consolidated list, combine explicitly:



                # Convert 'reuse_candidates' (cases) to scorables if needed:
                seeds += [self._case_to_scorable(c) for c in reuse_candidates]
                ranked, corpus, mars, recs, scores = self.ranker.run(
                    context, seeds
                )
                for r in ranked:
                    rid = r.get("id")
                    r["mars_confidence"] = (mars.get(rid, {}) or {}).get(
                        "agreement_score"
                    )

                await self.reporter.emit(
                    ctx=context,
                    stage=stage_name,
                    event="rank_done",
                    n_ranked=len(ranked),
                    top_ids=[r.get("id") for r in ranked[:3]],
                    mars_dims=len(mars or {}),
                )

                # 4) ADAPT/REVISE (NEW): take top-k, produce revised items, score them, include if improved
                if self.adapt_enabled and ranked:
                    revised = await self._adapt_top_k(
                        context, ranked, top_k=self.adapt_top_k
                    )
                    if revised:
                        # re-rank with revised included
                        combined = ranked + revised
                        ranked, corpus, mars, recs, scores = self.ranker.run(
                            context, combined
                        )
                        for r in ranked:
                            rid = r.get("id")
                            r["mars_confidence"] = (
                                mars.get(rid, {}) or {}
                            ).get("agreement_score")
                        await self.reporter.emit(
                            ctx=context,
                            stage="rerank_after_adapt",
                            variant=variant,
                            n_ranked=len(ranked),
                        )

                # 5) retain & micro-learn
                retained_case_id = (
                    self.retention.retain(context, ranked, mars, scores)
                    if retain
                    else None
                )
                await self.reporter.emit(
                    ctx=context,
                    stage=stage_name,
                    event="retain",
                    retained=bool(retained_case_id),
                    retained_case_id=retained_case_id,
                )
                if train_enabled:
                    self.micro.learn(context, ranked, mars)
                    await self.reporter.emit(
                        ctx=context, stage="micro_learn", variant=variant
                    )

                # 6) quality
                q = self.assessor.quality(mars, scores)
                await self.reporter.emit(
                    ctx=context,
                    stage=stage_name,
                    event="quality",
                    quality=float(q),
                )
                await self.reporter.emit(
                    ctx=context,
                    stage=stage_name,
                    status="done",
                    summary=f"quality={q:.3f}, retained={bool(retained_case_id)}",
                    finalize=True,
                )
                vb.update(
                    dict(
                        reuse_candidates=reuse_candidates,
                        ranked=ranked,
                        mars_results=mars,
                        recommendations=recs,
                        retained_case=retained_case_id,
                        metrics={"quality": float(q), "variant": variant},
                    )
                )
                return vb

        if do_ab:
            base_res = await _run_variant(
                "baseline",
                use_casebook=False,
                retain=False,
                train_enabled=not bool(
                    self.ab_cfg.get("freeze_training", True)
                ),
            )
            cbr_res = await _run_variant(
                "cbr", use_casebook=True, retain=True, train_enabled=True
            )
            winner, _cmp = await self.ab.run_two(
                context,
                lambda: self._noop(base_res),
                lambda: self._noop(base_res),
            )
            q_base, q_cbr = (
                base_res["metrics"]["quality"],
                cbr_res["metrics"]["quality"],
            )
            improved = q_cbr > q_base + float(
                self.ab_cfg.get("delta_eps", 1e-6)
            )
            winner = "cbr" if improved else "baseline"
            if improved:
                self.promoter.maybe_promote(
                    casebook_id,
                    goal_id,
                    cbr_res.get("retained_case"),
                    float(q_cbr),
                )
            context[output_key] = context.get(
                self.ns.variant_output_key(winner), []
            )
            await self.reporter.emit(
                ctx=context,
                stage="CBR",
                event="ab_decision",
                q_base=float(q_base),
                q_cbr=float(q_cbr),
                winner=winner,
                improved=bool(improved),
            )
            return context

        cbr_res = await _run_variant(
            "cbr", use_casebook=True, retain=True, train_enabled=True
        )
        context[output_key] = context.get(
            self.ns.variant_output_key("cbr"), []
        )
        await self.reporter.emit(
            ctx=context,
            stage="CBR",
            status="done",
            summary="CBR middleware finished",
            finalize=True,
        )

        return context

    async def _adapt_top_k(
        self, context: Dict[str, Any], ranked: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Take top-k ranked items; for each, adapt it using the most relevant retrieved text (if any)."""
        goal_text = (context.get("goal") or {}).get("goal_text", "") or ""
        out: List[Dict[str, Any]] = []

        # pick the text to adapt *from*. Heuristic: top non-produced item or the next best text in ranked
        # If your ranker tags provenance, you can pick a retrieved-vs-produced explicitly.
        k = min(max(1, top_k), len(ranked))
        for i in range(k):
            base = ranked[i]
            base_text = (base.get("text") or "").strip()
            if not base_text:
                continue

            # choose a "retrieved" donor (fallback to next-best if no explicit retrieved)
            donor = (
                ranked[i + 1]
                if (i + 1) < len(ranked)
                else (ranked[0] if len(ranked) > 1 else None)
            )
            donor_text = (donor.get("text") or "").strip() if donor else ""

            # LLM edit-critique
            rev = self.adaptor.revise(
                goal_text=goal_text,
                candidate_text=base_text,
                retrieved_text=donor_text,
            )
            revised_text = (rev.get("revised") or "").strip()
            rationale = rev.get("rationale")

            if not revised_text or revised_text == base_text:
                # skip if nothing changed (or keep with a tiny tag if you prefer)
                continue

            revised = {
                "id": None,  # will be filled by _ensure_ids
                "text": revised_text,
                "target_type": base.get("target_type")
                or ScorableType.HYPOTHESIS,
                "metadata": {
                    "provenance": {
                        "type": "revised",
                        "from": [
                            base.get("id"),
                            donor.get("id") if donor else None,
                        ],
                        "rationale": rationale,
                    }
                },
            }

            # score revised
            agg_revised = self._score_scorable(context, revised)
            agg_base = base.get("aggregate_score") or base.get("score") or 0.0
            # Only keep if improved by epsilon
            if agg_revised >= float(agg_base) + self.adapt_improv_eps:
                revised["aggregate_score"] = float(agg_revised)
                out.append(self._ensure_ids([revised])[0])

        self.logger and self.logger.log(
            "CBRAdaptSummary",
            {
                "requested": top_k,
                "produced": len(out),
                "eps": self.adapt_improv_eps,
            },
        )
        return out

    def _score_scorable(
        self, context: Dict[str, Any], scorable_dict: Dict[str, Any]
    ) -> float:
        """
        Score a single scorable dict using either an injected scoring service or the ranker if it exposes one.
        Returns a 0..1 aggregate score (align with your ScoringService aggregate convention).
        """
        try:
            text = scorable_dict.get("text") or ""
            scorable = Scorable(
                text=text,
                id=scorable_dict.get("id") or "",
                target_type=scorable_dict.get("target_type") or "custom",
            )
            scorer_name = self.adapt_scorer_name or "sicql"
            dims = self.adapt_dimensions  # None → scorer default
            if self.scoring:
                bundle = self.container.get("scoring").score(
                    scorer_name,
                    scorable=scorable,
                    context=context,
                    dimensions=dims,
                )
                return float(bundle.aggregate()) / 100.0
            # fallback: if ranker exposes a similar primitive
            score_one = getattr(self.ranker, "score_one", None)
            if callable(score_one):
                agg = score_one(context, scorable, scorer_name, dims)
                return float(agg)
        except Exception as e:
            self.logger and self.logger.log(
                "CBRScoreRevisedError", {"error": str(e)}
            )
        return 0.0

    def _case_to_scorable(self, case):
        """
        Normalize case (could be str ID, dict, or ORM) into a Scorable dict.
        """
        # Case is just an ID string
        if isinstance(case, str):
            return {
                "id": case,
                "text": "",  # text will be resolved later
                "target_type": "document",
                "metadata": {},
            }

        # Case is already a dict
        if isinstance(case, dict):
            return {
                "id": case.get("id"),
                "text": case.get("text", ""),
                "target_type": case.get("target_type", "document"),
                "metadata": case.get("metadata", {}),
            }

        # Case is an ORM or object
        return {
            "id": getattr(case, "id", None),
            "text": getattr(case, "text", ""),
            "target_type": getattr(case, "target_type", "document"),
            "metadata": {
                "rank": getattr(case, "rank", None),
                "role": getattr(case, "role", None),
            },
        }

    # -- helpers
    async def _noop(self, x):
        return x

    def _ensure_ids(self, scorables: List[dict]) -> List[dict]:
        import hashlib
        import random

        out = []
        for h in scorables:
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

    from contextlib import contextmanager

    @contextmanager
    def _variant_output_redirect(self, ctx, variant: str, output_key: str):
        prev = output_key
        # we can’t mutate agent.output_key here, so we write to namespaced output, and the caller will read it
        try:
            yield
        finally:
            pass


def _dedupe_semantic(scorables, embed, thresh=0.92):
    kept, vecs = [], []
    for s in scorables:
        txt = (s.get("text") or "").strip()
        v = np.asarray(embed.get_or_create(txt), dtype=float)
        if vecs and max(_cos(v, u) for u in vecs) >= thresh:
            continue
        vecs.append(v); kept.append(s)
    return kept

# helper for tiny previews
def _safe_texts(items, max_items=2, max_len=140):
    out = []
    for it in items[:max_items]:
        t = (it.get("text") or "") if isinstance(it, dict) else str(it)
        if t:
            t = t.replace("\n", " ")
            out.append(t[:max_len] + ("…" if len(t) > max_len else ""))
    return out



def _cos(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

