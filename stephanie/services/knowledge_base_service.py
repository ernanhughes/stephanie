from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class KBStrategyUpdate:
    when: str
    reason: str
    delta: Dict[str, float]  # e.g., {"skeptic": +0.05, "editor": -0.05, "verification_threshold": +0.01}

@dataclass
class KBDomainStats:
    successes: int = 0
    failures: int = 0
    avg_iters: float = 0.0
    n: int = 0

    def update(self, ok: bool, iters: int):
        self.n += 1
        if ok: self.successes += 1
        else:  self.failures  += 1
        # incremental avg
        self.avg_iters += (iters - self.avg_iters) / max(1, self.n)

@dataclass
class KnowledgeBaseState:
    strategy_updates: List[KBStrategyUpdate] = field(default_factory=list)
    domain_stats: Dict[str, KBDomainStats]   = field(default_factory=lambda: defaultdict(KBDomainStats))
    failure_modes: Counter                    = field(default_factory=Counter)  # text label → count
    solution_templates: Dict[str, Dict[str, Any]] = field(default_factory=dict) # template_id → {domain, outline, cues}
    # light pattern cache
    ngram_success: Counter                    = field(default_factory=Counter)  # "ngram::domain" → count
    ngram_fail: Counter                       = field(default_factory=Counter)

class KnowledgeBaseService:
    """
    Cross-paper memory for 'learning from learning'.
    Persists compact stats + templates and surfaces them as context for new papers.
    """
    def __init__(self, cfg: Dict[str, Any], memory, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.logger = logger
        self.path = self.cfg.get("knowledge_base_path", "data/kbase/knowledge_base.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.state = self._load()

    # --- persistence ---
    def _load(self) -> KnowledgeBaseState:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                return self._from_json(raw)
        except Exception as e:
            self.logger and self.logger.log("KBLoadError", {"error": str(e)})
        return KnowledgeBaseState()

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._to_json(self.state), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger and self.logger.log("KBSaveError", {"error": str(e)})

    # --- public API ---
    def update_from_paper(self, *, domain: str, summary_text: str, metrics: Dict[str, Any], iterations: List[Dict[str, Any]], strategy_delta: Optional[Dict[str, float]] = None):
        """
        Capture lessons after Track C completes.
        """
        ok = bool(metrics.get("overall", 0.0) >= float(self.cfg.get("kb_success_threshold", 0.80)))
        iters = int(len(iterations or []))
        domain = (domain or "unknown").lower()

        # 1) domain stats
        self.state.domain_stats[domain].update(ok, iters)

        # 2) failure modes (if not ok, label a rough cause)
        if not ok:
            cause = self._diagnose_failure(metrics)
            self.state.failure_modes[cause] += 1

        # 3) n-gram patterns (very light)
        grams = self._ngrams(summary_text.lower(), n=3)
        for g in grams:
            key = f"{g}::{domain}"
            (self.state.ngram_success if ok else self.state.ngram_fail)[key] += 1

        # 4) learn solution template from strong wins
        if ok and metrics.get("knowledge_verification", 0.0) >= 0.85:
            tid = f"tmpl_{len(self.state.solution_templates)+1:04d}"
            self.state.solution_templates[tid] = {
                "domain": domain,
                "outline": self._extract_outline(summary_text),
                "cues": ["cite figures for numerics", "cover all key claims", "avoid speculative language"],
            }

        # 5) record strategy change (if any)
        if strategy_delta:
            self.state.strategy_updates.append(KBStrategyUpdate(
                when=datetime.now(timezone.utc).isoformat(),
                reason="post_track_c_improvement",
                delta=strategy_delta
            ))

        self.save()

    def context_for_paper(self, *, title: str, abstract: str, domain: str) -> Dict[str, Any]:
        """
        Provide hints for verification prompt + weight nudges.
        """
        domain = (domain or "unknown").lower()
        dstat = self.state.domain_stats.get(domain) or KBDomainStats()
        # choose up to 2 best templates in this domain
        templates = [
            v for v in self.state.solution_templates.values()
            if v.get("domain") == domain
        ][:2]

        # n-gram gate: if a historically-failing ngram appears, advise 'skeptic++'
        grams = set(self._ngrams((title + " " + abstract).lower(), n=3))
        risk_flag = any(self.state.ngram_fail.get(f"{g}::{domain}", 0) > self.state.ngram_success.get(f"{g}::{domain}", 0) for g in grams)

        hints = []
        if dstat.n >= 5 and dstat.avg_iters > 2.5:
            hints.append("Prefer stronger claim grounding; prior papers needed >2.5 iterations on average.")
        if risk_flag:
            hints.append("Historically tricky phrasing detected; increase skeptic emphasis and require figure/table citations for numeric claims.")

        return {
            "domain_stats": {"n": dstat.n, "successes": dstat.successes, "failures": dstat.failures, "avg_iters": round(dstat.avg_iters, 2)},
            "templates": templates,
            "hints": hints,
            "weight_nudges": {"skeptic": 0.03 if risk_flag else 0.0, "editor": 0.0, "risk": 0.02 if risk_flag else 0.0}
        }

    # --- helpers ---
    @staticmethod
    def _ngrams(text: str, n=3) -> List[str]:
        toks = re.findall(r"[a-z0-9]+", text.lower())
        return [" ".join(toks[i:i+n]) for i in range(0, max(0, len(toks)-n+1))]

    @staticmethod
    def _extract_outline(text: str) -> List[str]:
        # very small heuristic: split into 5-8 short bullets
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        pick = [s.strip() for s in sents if s.strip()][:8]
        return [re.sub(r"\s+", " ", s) for s in pick]

    @staticmethod
    def _diagnose_failure(m: Dict[str, Any]) -> str:
        if float(m.get("hallucination_rate", 0.0)) > 0.25:
            return "hallucination"
        if float(m.get("claim_coverage", 0.0)) < 0.6:
            return "low_coverage"
        if float(m.get("knowledge_verification", 0.0)) < 0.6:
            return "weak_evidence"
        return "other"

    # --- (de)serialization ---
    def _to_json(self, st: KnowledgeBaseState) -> Dict[str, Any]:
        return {
            "strategy_updates": [dict(when=x.when, reason=x.reason, delta=x.delta) for x in st.strategy_updates],
            "domain_stats": {k: dict(successes=v.successes, failures=v.failures, avg_iters=v.avg_iters, n=v.n) for k, v in st.domain_stats.items()},
            "failure_modes": dict(st.failure_modes),
            "solution_templates": st.solution_templates,
            "ngram_success": dict(st.ngram_success),
            "ngram_fail": dict(st.ngram_fail),
        }

    def _from_json(self, raw: Dict[str, Any]) -> KnowledgeBaseState:
        st = KnowledgeBaseState()
        for r in raw.get("strategy_updates", []):
            st.strategy_updates.append(KBStrategyUpdate(when=r["when"], reason=r["reason"], delta=r["delta"]))
        for k, v in (raw.get("domain_stats") or {}).items():
            ds = KBDomainStats()
            ds.successes = int(v.get("successes", 0)); ds.failures = int(v.get("failures", 0))
            ds.avg_iters = float(v.get("avg_iters", 0.0)); ds.n = int(v.get("n", 0))
            st.domain_stats[k] = ds
        st.failure_modes.update(raw.get("failure_modes", {}))
        st.solution_templates = raw.get("solution_templates", {})
        st.ngram_success.update(raw.get("ngram_success", {}))
        st.ngram_fail.update(raw.get("ngram_fail", {}))
        return st
