# stephanie/agents/learning/evidence.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import asyncio
import json
import time

class Evidence:
    def __init__(self, cfg, memory, container, logger):
        self.cfg, self.memory, self.container, self.logger = (
            cfg, memory, container, logger
        )
        self.casebook_tag = cfg.get("casebook_action", "blog")
        self._last: Dict[str, Any] = {}   # NEW: last snapshot for delta reporting

    def _emit(self, event: str, **fields):
        payload = {"event": event, **fields}
        """
        Fire-and-forget reporting event using container.get('reporting').emit(...)
        Safe to call from sync code (no await).
        """
        try:
            reporter = self.container.get("reporting")
            coro = reporter.emit(ctx={}, stage="learning",  **payload)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(coro)
            except RuntimeError:
                # if no running loop (rare), just ignore to keep non-blocking
                pass
        except Exception:
            # never fail persistence due to reporting
            pass


    # -------------- util --------------
    @staticmethod
    def _percent_change(start: float, end: float) -> float:
        try:
            if start is None or end is None or start == 0:
                return 0.0
            return (end - start) / abs(start) * 100.0
        except Exception:
            return 0.0

    # -------------- longitudinal (unchanged logic; added emit) --------------
    def collect_longitudinal(self) -> Dict[str, Any]:
        out = {
            "total_papers": 0,
            "verification_scores": [],
            "iteration_counts": [],
            "avg_verification_score": 0.0,
            "avg_iterations": 0.0,
            "score_improvement_pct": 0.0,
            "iteration_reduction_pct": 0.0,
            "strategy_versions": [],
            "strategy_evolution_rate": 0.0,
        }
        try:
            casebooks = self.memory.casebooks.get_casebooks_by_tag(self.casebook_tag) or []
            for cb in casebooks:
                cases = self.memory.casebooks.get_cases_for_casebook(cb.id) or []
                for case in cases:
                    for s in self.memory.casebooks.list_scorables(case.id) or []:
                        if s.role != "metrics":
                            continue
                        try:
                            payload = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                            final_scores = (payload or {}).get("final_scores") or {}
                            overall = final_scores.get("overall")
                            iters = payload.get("refinement_iterations")
                            if overall is not None:
                                out["verification_scores"].append(float(overall))
                            if iters is not None:
                                out["iteration_counts"].append(int(iters))
                        except Exception:
                            continue

            vs, it = out["verification_scores"], out["iteration_counts"]
            out["total_papers"] = len(vs)
            if vs:
                out["avg_verification_score"] = sum(vs) / len(vs)
            if it:
                out["avg_iterations"] = sum(it) / len(it)
            if len(vs) >= 2:
                out["score_improvement_pct"] = self._percent_change(vs[0], vs[-1])
            if len(it) >= 2:
                out["iteration_reduction_pct"] = -1.0 * self._percent_change(it[0], it[-1])

            # EMIT longitudinal snapshot + deltas (NEW)
            self._emit("evidence.longitudinal",
                       at=time.time(),
                       total_papers=out["total_papers"],
                       avg_score=out["avg_verification_score"],
                       avg_iters=out["avg_iterations"],
                       score_trend_pct=out["score_improvement_pct"],
                       iter_trend_pct=out["iteration_reduction_pct"],
                       delta=self._delta_section("longitudinal", out))
        except Exception as e:
            try:
                self.logger.log("LfL_Longitudinal_Failed", {"err": str(e)})
            except Exception:
                pass
        return out

    # -------------- helpers used below (NEW) --------------
    def _delta_section(self, key: str, current: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        """Return {field: (prev, curr)} for changed fields and store snapshot."""
        prev = self._last.get(key, {})
        changed = {}
        for k, v in current.items():
            if isinstance(v, (int, float, str)) and prev.get(k) != v:
                changed[k] = (prev.get(k), v)
        self._last[key] = {**prev, **{k: current[k] for k in current}}
        return changed

    # -------------- cross-episode (added AR/AKL/RN/TR + emit) --------------
    def _get_paper_performance(self, casebook) -> Dict[str, float]:
        scores, iters = [], []
        for case in (self.memory.casebooks.get_cases_for_casebook(casebook.id) or []):
            for s in self.memory.casebooks.list_scorables(case.id) or []:
                if s.role == "metrics":
                    try:
                        rec = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                        final = (rec or {}).get("final_scores") or {}
                        if "overall" in final: scores.append(float(final["overall"]))
                        itv = rec.get("refinement_iterations")
                        if itv is not None: iters.append(int(itv))
                    except Exception:
                        pass
        return {
            "mean_overall": (sum(scores)/len(scores)) if scores else 0.0,
            "mean_iters": (sum(iters)/len(iters)) if iters else 0.0,
            "n": len(scores),
        }

    def _extract_winner_origins(self, casebook) -> Dict[str, int]:
        tally = {}
        for case in (self.memory.casebooks.get_cases_for_casebook(casebook.id) or []):
            for s in self.memory.casebooks.list_scorables(case.id) or []:
                if s.role == "arena_winner":
                    origin = (s.meta or {}).get("origin")
                    if origin: tally[origin] = tally.get(origin, 0) + 1
                    break
        return tally

    def _calculate_knowledge_transfer(self) -> Dict[str, Any]:
        cbs = self.memory.casebooks.get_casebooks_by_tag(self.casebook_tag) or []
        if len(cbs) < 2:
            return {"rate": 0.0, "examples": []}
        transfer, examples = 0, []
        for i in range(1, len(cbs)):
            prev_cb, curr_cb = cbs[i-1], cbs[i]
            prev = self._extract_winner_origins(prev_cb)
            curr = self._extract_winner_origins(curr_cb)
            reused = [k for k in prev.keys() if curr.get(k, 0) > 0]
            if reused:
                transfer += 1
                perf = self._get_paper_performance(curr_cb)
                examples.append({
                    "from_paper": getattr(prev_cb, "name", str(prev_cb.id)),
                    "to_paper": getattr(curr_cb, "name", str(curr_cb.id)),
                    "patterns_used": [{"name": k, "description": f"Winner origin reused: {k}"} for k in reused],
                    "performance_impact": perf["mean_overall"],
                })
        return {"rate": transfer / max(1, len(cbs)-1), "examples": examples}

    def _calculate_domain_learning(self) -> Dict[str, Any]:
        cbs = self.memory.casebooks.get_casebooks_by_tag(self.casebook_tag) or []
        vals = []
        for cb in cbs:
            perf = self._get_paper_performance(cb)
            vals.append(perf["mean_overall"])
        return {"all_mean": (sum(vals)/len(vals)) if vals else 0.0, "samples": len(vals)}

    def _calculate_meta_patterns(self) -> Dict[str, Any]:
        cbs = self.memory.casebooks.get_casebooks_by_tag(self.casebook_tag) or []
        rounds = 0; sections = 0
        for cb in cbs:
            for case in self.memory.casebooks.get_cases_for_casebook(cb.id) or []:
                sections += 1
                for s in self.memory.casebooks.list_scorables(case.id) or []:
                    if s.role == "arena_round_metrics":
                        try:
                            j = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                            rounds += len((j or {}).get("rounds", []))
                        except Exception:
                            pass
        return {"avg_rounds": (rounds/sections) if sections else 0.0, "sections": sections}

    def _calculate_adaptation_rate(self) -> float:
        return 0.0

    def _collect_improve_attributions(self):
        total_sections, supported_sections = 0, 0
        lifts = []
        ablation_pairs = []  # (score_normal, score_ablated)
        casebooks = self.memory.casebooks.get_casebooks_by_tag(self.casebook_tag) or []
        for cb in casebooks:
            for case in (self.memory.casebooks.get_cases_for_casebook(cb.id) or []):
                total_sections += 1
                has_supported = False
                final_overall = None
                ablated = False
                for s in self.memory.casebooks.list_scorables(case.id) or []:
                    if s.role == "metrics":
                        try:
                            rec = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                            final_overall = ((rec or {}).get("final_scores") or {}).get("overall")
                            # accept either path: directly in metrics, or nested in final_scores (your patch)
                            k_lift = (rec or {}).get("knowledge_applied_lift", None)
                            if k_lift is None:
                                k_lift = ((rec or {}).get("final_scores") or {}).get("knowledge_applied_lift", 0.0)
                            if k_lift:
                                lifts.append(float(k_lift))
                        except Exception:
                            pass
                    elif s.role == "improve_attribution":
                        has_supported = True
                    elif s.role == "ablation":
                        ablated = True
                if has_supported:
                    supported_sections += 1
                # naive pairing: compare ablated case to any non-ablated peer in same casebook
                if ablated and final_overall is not None:
                    peer = None
                    for c2 in (self.memory.casebooks.get_cases_for_casebook(cb.id) or []):
                        if c2.id == case.id:
                            continue
                        found_ablate = False
                        peer_final = None
                        for s2 in (self.memory.casebooks.list_scorables(c2.id) or []):
                            if s2.role == "ablation":
                                found_ablate = True
                            elif s2.role == "metrics":
                                try:
                                    r2 = json.loads(s2.text) if isinstance(s2.text, str) else (s2.text or {})
                                    peer_final = ((r2 or {}).get("final_scores") or {}).get("overall")
                                except Exception:
                                    pass
                        if not found_ablate and peer_final is not None:
                            peer = peer_final
                            break
                    if peer is not None:
                        ablation_pairs.append((peer, final_overall))
        AR = supported_sections / max(1, total_sections)
        AKL = (sum(lifts) / len(lifts)) if lifts else 0.0
        RN = None
        if ablation_pairs:
            diffs = [a - b for (a, b) in ablation_pairs]
            RN = sum(diffs) / len(diffs)
        return {
            "attribution_rate": AR,
            "applied_knowledge_lift": AKL,
            "retrieval_ablation_delta": RN,
        }

    def _strict_transfer_rate(self):
        cbs = self.memory.casebooks.get_casebooks_by_tag(self.casebook_tag) or []
        if len(cbs) < 2:
            return 0.0
        reused = 0; denom = 0
        prev_citations = set()
        for i, cb in enumerate(cbs):
            cites_here = set()
            for case in (self.memory.casebooks.get_cases_for_casebook(cb.id) or []):
                for s in self.memory.casebooks.list_scorables(case.id) or []:
                    if s.role == "arena_citations":
                        try:
                            j = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                            for c in (j or {}).get("citations", []):
                                key = (c.get("support_origin"), c.get("support_variant"))
                                if key[0] or key[1]:
                                    cites_here.add(key)
                        except Exception:
                            pass
            if i > 0:
                denom += 1
                if prev_citations & cites_here:
                    reused += 1
            prev_citations = cites_here
        return reused / max(1, denom)

    def cross_episode(self) -> Dict[str, Any]:
        kt = self._calculate_knowledge_transfer()
        dom = self._calculate_domain_learning()
        meta = self._calculate_meta_patterns()
        adapt_rate = self._calculate_adaptation_rate()
        ak = self._collect_improve_attributions()
        strict_tr = self._strict_transfer_rate()
        out = {
            "knowledge_transfer_rate": kt["rate"],
            "knowledge_transfer_examples": kt["examples"][:3],
            "domain_learning_patterns": dom,
            "meta_pattern_recognition": meta,
            "strategy_adaptation_rate": adapt_rate,
            "cross_episode_evidence_strength": self._calculate_evidence_strength(kt, dom, meta, adapt_rate),
            "attribution_rate": ak["attribution_rate"],
            "applied_knowledge_lift": ak["applied_knowledge_lift"],
            "retrieval_ablation_delta": ak["retrieval_ablation_delta"],
            "strict_transfer_rate": strict_tr,
        }
        # EMIT cross-episode snapshot + deltas (NEW)
        self._emit("evidence.cross_episode",
                   at=time.time(),
                   knowledge_transfer_rate=out["knowledge_transfer_rate"],
                   attribution_rate=out["attribution_rate"],
                   applied_knowledge_lift=out["applied_knowledge_lift"],
                   retrieval_ablation_delta=out["retrieval_ablation_delta"],
                   strict_transfer_rate=out["strict_transfer_rate"],
                   evidence_strength=out["cross_episode_evidence_strength"],
                   delta=self._delta_section("cross_episode", out))
        # one-line headline (NEW)
        self._emit("evidence.summary",
                   at=time.time(),
                   msg=("AR={:.0%} | AKL={:+.3f} | RNΔ={}".format(
                        out["attribution_rate"],
                        out["applied_knowledge_lift"],
                        "n/a" if out["retrieval_ablation_delta"] is None else f"{out['retrieval_ablation_delta']:+.3f}"
                   )))
        return out

    def _calculate_evidence_strength(self, kt: Dict[str, Any], dom: Dict[str, Any], meta: Dict[str, Any], adapt_rate: float) -> float:
        return max(0.0, min(1.0, 0.35*kt.get("rate",0.0) + 0.25*dom.get("all_mean",0.0) + 0.20*(meta.get("avg_rounds",0.0)/5.0) + 0.20*adapt_rate))

    def report(self, longitudinal: Dict[str, Any], cross: Dict[str, Any]) -> str:
        if not longitudinal or longitudinal.get("total_papers", 0) < 3:
            return ""
        score_trend = longitudinal.get("score_improvement_pct", 0.0)
        iter_trend  = longitudinal.get("iteration_reduction_pct", 0.0)
        arrow_score = "↑" if score_trend > 0 else "↓"
        arrow_iter  = "↓" if iter_trend > 0 else "↑"

        lines = []
        lines.append("## Learning from Learning: Evidence Report")
        lines.append("")
        lines.append(f"- **Total papers processed**: {longitudinal.get('total_papers', 0)}")
        lines.append(f"- **Verification score trend**: {score_trend:.1f}% {arrow_score}")
        lines.append(f"- **Average iterations trend**: {iter_trend:.1f}% {arrow_iter}")
        lines.append(f"- **Knowledge transfer rate**: {cross.get('knowledge_transfer_rate', 0.0):.0%}")
        if cross.get("strategy_ab_validation"):
            ab = cross["strategy_ab_validation"]
            lines.append(f"- **A/B delta (B−A)**: {ab.get('delta_B_minus_A', 0.0):+.3f} (A n={ab.get('samples_A',0)}, B n={ab.get('samples_B',0)})")
        lines.append(f"- **Cross-episode evidence strength**: {cross.get('cross_episode_evidence_strength', 0.0):.0%}")
        lines.append("")
        vs = longitudinal.get("verification_scores", [])
        it = longitudinal.get("iteration_counts", [])
        if vs and it:
            lines.append("### Snapshot")
            lines.append(f"- First: score={vs[0]:.2f}, iterations={it[0]}")
            lines.append(f"- Latest: score={vs[-1]:.2f}, iterations={it[-1]}")
        if cross.get("knowledge_transfer_examples"):
            ex = cross["knowledge_transfer_examples"][0]
            lines.append("")
            lines.append("### Cross-Episode Example")
            lines.append(f"- *{ex['from_paper']}* → *{ex['to_paper']}* reused patterns:")
            for p in ex.get("patterns_used", [])[:3]:
                lines.append(f" • {p['name']} – {p['description']}")
            lines.append(f"- Mean overall after transfer: {ex.get('performance_impact', 0.0):.3f}")

        lines.append(f"- **Attribution rate**: {cross.get('attribution_rate', 0.0):.0%}")
        lines.append(f"- **Applied-knowledge lift**: {cross.get('applied_knowledge_lift', 0.0):+.3f}")
        if cross.get("retrieval_ablation_delta") is not None:
            lines.append(f"- **Retrieval necessity (ablation Δ)**: {cross['retrieval_ablation_delta']:+.3f}")
        lines.append(f"- **Strict transfer rate**: {cross.get('strict_transfer_rate', 0.0):.0%}")

        md = "\n".join(lines)

        # also emit the final markdown block once per call (NEW)
        self._emit("evidence.report_md", markdown=md, at=time.time())
        return md
