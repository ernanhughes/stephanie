# stephanie/agents/learning/evidence.py
from __future__ import annotations
from typing import Dict, Any
import json

class Evidence:
    def __init__(self, cfg, memory, container, logger):
        self.cfg, self.memory, self.container, self.logger = cfg, memory, container, logger
        self.casebook_tag = cfg.get("casebook_action","blog")

    @staticmethod
    def _percent_change(start: float, end: float) -> float:
        try:
            if start is None or end is None or start == 0:
                return 0.0
            return (end - start) / abs(start) * 100.0
        except Exception:
            return 0.0

    def collect_longitudinal(self) -> Dict[str, Any]:
        out = {
            "total_papers": 0,
            "verification_scores": [],
            "iteration_counts": [],
            "avg_verification_score": 0.0,
            "avg_iterations": 0.0,
            "score_improvement_pct": 0.0,
            "iteration_reduction_pct": 0.0,
            "strategy_versions": [],           # left for compatibility
            "strategy_evolution_rate": 0.0,    # left for compatibility
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
            if vs: out["avg_verification_score"] = sum(vs) / len(vs)
            if it: out["avg_iterations"] = sum(it) / len(it)
            if len(vs) >= 2: out["score_improvement_pct"] = self._percent_change(vs[0], vs[-1])
            if len(it) >= 2: out["iteration_reduction_pct"] = -1.0 * self._percent_change(it[0], it[-1])
        except Exception as e:
            try: self.logger.log("LfL_Longitudinal_Failed", {"err": str(e)})
            except Exception: pass
        return out

    # ---------- Cross-episode helpers ----------
    def _get_paper_performance(self, casebook) -> Dict[str, float]:
        scores, iters = [], []
        for case in self.memory.casebooks.get_cases_for_casebook(casebook.id) or []:
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
        for case in self.memory.casebooks.get_cases_for_casebook(casebook.id) or []:
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
        # If you want, you can surface StrategyManager stats here via container.
        return 0.0

    def _calculate_evidence_strength(self, kt: Dict[str, Any], dom: Dict[str, Any], meta: Dict[str, Any], adapt_rate: float) -> float:
        return max(0.0, min(1.0, 0.35*kt.get("rate",0.0) + 0.25*dom.get("all_mean",0.0) + 0.20*(meta.get("avg_rounds",0.0)/5.0) + 0.20*adapt_rate))

    def cross_episode(self) -> Dict[str, Any]:
        kt = self._calculate_knowledge_transfer()
        dom = self._calculate_domain_learning()
        meta = self._calculate_meta_patterns()
        adapt_rate = self._calculate_adaptation_rate()
        return {
            "knowledge_transfer_rate": kt["rate"],
            "knowledge_transfer_examples": kt["examples"][:3],
            "domain_learning_patterns": dom,
            "meta_pattern_recognition": meta,
            "strategy_adaptation_rate": adapt_rate,
            "cross_episode_evidence_strength": self._calculate_evidence_strength(kt, dom, meta, adapt_rate),
        }

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
        return "\n".join(lines)
