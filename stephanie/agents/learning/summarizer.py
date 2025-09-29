# stephanie/agents/learning/summarizer.py
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from stephanie.utils.json_sanitize import dumps_safe


class Summarizer:
    def __init__(self, cfg, memory, container, logger, strategy, scoring, prompt_loader, call_llm):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self.strategy = strategy
        self.scoring = scoring
        self.prompt_loader = prompt_loader
        self.call_llm = call_llm

    def baseline(self, paper: Dict[str, Any], section: Dict[str, Any],
                 critical_msgs: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        merged = {
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "focus_section": section.get("section_name"),
            "focus_text": section.get("section_text", "")[:5000],
            # prefer assistant_text but fall back to text
            "hints": "\n".join((m.get("assistant_text") or m.get("text") or "")
                               for m in (critical_msgs[:6] if critical_msgs else [])),
            **context,
        }
        prompt = self.prompt_loader.from_file("baseline_summary", self.cfg, merged)
        return self.call_llm(prompt, merged)

    def improve_once(self, paper: Dict[str, Any], section: Dict[str, Any],
                     current_summary: str, context: Dict[str, Any],
                     return_attribution: bool=False):
        metrics = self.scoring.score_summary(current_summary, paper, section, context)
        merged_context = {
            "title": paper.get("title", ""),
            "section_name": section.get("section_name"),
            "section_text": section.get("section_text", "")[:6000],
            "current_summary": current_summary,
            "skeptic_weight": self.strategy.state.skeptic_weight,
            "editor_weight": self.strategy.state.editor_weight,
            "risk_weight": self.strategy.state.risk_weight,
            "weaknesses": json.dumps(metrics.get("weaknesses", []), ensure_ascii=False),
            **context,
        }
        prompt = self.prompt_loader.from_file("improve_summary", self.cfg, merged_context)
        improved = self.call_llm(prompt, merged_context)   # ← pass merged, not context

        if not return_attribution:
            return improved

        # --- attribution (AR/AKL) ---
        claims = self._extract_claim_sentences(improved)

        # Normalize retrieval & arena pools to a common {text, origin, variant} schema
        rpool = (context.get("retrieval_items") or []) + (context.get("arena_initial_pool") or [])
        norm_sources = self._normalize_sources(rpool)

        th = float(self.cfg.get("applied_knowledge", {}).get("attr_sim_threshold", 0.75))
        matches = self._attribute_claims(claims, norm_sources, th)

        # store a compact scorable for this improve step (if a case is present)
        try:
            case_id = context.get("case_id")
            if case_id and matches:
                payload = {"claims": matches, "threshold": th, "timestamp": time.time()}
                self.memory.casebooks.add_scorable(
                    case_id=case_id, role="improve_attribution",
                    text=dumps_safe(payload),
                    pipeline_run_id=context.get("pipeline_run_id"),
                    meta={"iteration": context.get("iteration")}
                )
        except Exception:
            pass

        return {"text": improved, "attribution": matches}

    def _normalize_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map mixed corpus/arena items to {text, origin, variant}."""
        out: List[Dict[str, Any]] = []
        for s in sources or []:
            # corpus items often store text under "assistant_text"
            txt = (s.get("text") or s.get("assistant_text") or "").strip()
            if not txt:
                continue
            origin = s.get("origin") or (s.get("meta") or {}).get("source")
            out.append({
                "text": txt,
                "origin": origin,
                "variant": s.get("variant")
            })
        return out

    def _cos_sim(self, a, b) -> float:
        num = sum(x*y for x,y in zip(a,b))
        na = (sum(x*x for x in a))**0.5 or 1.0
        nb = (sum(x*x for x in b))**0.5 or 1.0
        return max(0.0, min(1.0, num/(na*nb)))

    def _attribute_claims(self, claims: List[str], sources: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        emb = getattr(self.memory, "embedding", None) if self.cfg.get("applied_knowledge",{}).get("use_embeddings", True) else None
        out = []

        if emb:
            # pre-embed sources (cap for speed)
            S = [{"meta": s, "v": emb.get_or_create(s["text"][:2000])} for s in (sources[:50] if sources else [])]
        else:
            S = sources[:50] if sources else []

        import re
        for c in claims:
            if not c:
                continue
            best, best_sim = None, 0.0
            if emb and S:
                cv = emb.get_or_create(c)
                for s in S:
                    sim = self._cos_sim(cv, s["v"])
                    if sim > best_sim:
                        best_sim, best = sim, s["meta"]
            else:
                # fallback: token overlap
                ctoks = set(re.findall(r"\b\w+\b", c.lower()))
                for s in S:
                    stoks = set(re.findall(r"\b\w+\b", s["text"].lower()))
                    denom = float(max(1, max(len(ctoks), len(stoks))))  # ← fix
                    sim = len(ctoks & stoks) / denom
                    if sim > best_sim:
                        best_sim, best = sim, s

            if best and best_sim >= threshold:
                out.append({
                    "claim": c,
                    "support": {
                        "text": (best.get("text") or "")[:220],
                        "origin": best.get("origin"),
                        "variant": best.get("variant")
                    },
                    "similarity": round(best_sim, 3)
                })
        return out

    def verify_and_improve(self, baseline: str, paper: Dict[str, Any], section: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        current = baseline
        iters: List[Dict[str, Any]] = []
        max_iter = int(self.cfg.get("max_iterations", 3))
        last_attr_supported = False
        first_score_with_knowledge = None
        metrics = {"overall": 0.0}  # default

        for i in range(1, max_iter + 1):
            metrics = self.scoring.score_summary(current, paper, section, context)
            # annotate whether this iteration's text was influenced by knowledge (from prior improvement step)
            it = {"iteration": i, "score": metrics["overall"], "metrics": metrics, "knowledge_applied": bool(last_attr_supported)}
            iters.append(it)

            if last_attr_supported and first_score_with_knowledge is None:
                first_score_with_knowledge = metrics["overall"]

            if metrics["overall"] >= self.strategy.state.verification_threshold:
                break

            # run one improvement with attribution
            improved = self.improve_once(paper, section, current, {**context, "iteration": i}, return_attribution=True)
            if isinstance(improved, dict):
                current = improved["text"]
                last_attr_supported = bool(improved.get("attribution"))
            else:
                current = improved
                last_attr_supported = False

        # rollups that Persistence can store
        k_applied_iters = sum(1 for it in iters if it.get("knowledge_applied"))
        k_applied_lift = (metrics["overall"] - first_score_with_knowledge) if first_score_with_knowledge is not None else 0.0

        self.strategy.evolve(iters, context)
        return {"summary": current, "metrics": {**metrics, "knowledge_applied_iters": k_applied_iters, "knowledge_applied_lift": k_applied_lift}, "iterations": iters}

    def _extract_claim_sentences(self, text: str) -> List[str]:
        import re
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', (text or '').strip()) if s.strip()]
        triggers = ("show", "demonstrate", "evidence", "result", "increase", "decrease", "improve", "we propose", "we find")
        return [s for s in sents if any(t in s.lower() for t in triggers)][: int(self.cfg.get("applied_knowledge",{}).get("max_claims", 8))]

    def _attribute_claims(self, claims: List[str], sources: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        emb = getattr(self.memory, "embedding", None) if self.cfg.get("applied_knowledge",{}).get("use_embeddings", True) else None
        out = []
        # pre-embed sources (cap for speed)
        S = [{"meta": s, "v": (emb.get_or_create((s.get("text") or "")[:2000]) if emb else None)} for s in (sources[:50] if sources else [])]
        for c in claims:
            if not c: 
                continue
            best, best_sim = None, 0.0
            if emb and S:
                cv = emb.get_or_create(c)
                for s in S:
                    sim = self._cos_sim(cv, s["v"])
                    if sim > best_sim:
                        best_sim, best = sim, s["meta"]
            else:
                # fallback: token overlap
                import re
                ctoks = set(re.findall(r"\b\w+\b", c.lower()))
                for s in S or sources:
                    stoks = set(re.findall(r"\b\w+\b", (s.get("text","").lower())))
                    sim = len(ctoks & stoks) / float(max(1, len(ctoks)|len(stoks)))
                    if sim > best_sim:
                        best_sim, best = sim, s
            if best and best_sim >= threshold:
                out.append({"claim": c, "support": {"text": best.get("text","")[:220], "origin": best.get("origin"), "variant": best.get("variant")}, "similarity": round(best_sim, 3)})
        return out
