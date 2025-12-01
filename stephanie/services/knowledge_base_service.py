# stephanie/services/kbase_service.py
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from stephanie.services.service_protocol import Service


# --------------------------- Datamodel ---------------------------

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
        # incremental average
        self.avg_iters += (iters - self.avg_iters) / max(1, self.n)

@dataclass
class KnowledgeBaseState:
    strategy_updates: List[KBStrategyUpdate] = field(default_factory=list)
    domain_stats: Dict[str, KBDomainStats]   = field(default_factory=lambda: defaultdict(KBDomainStats))
    failure_modes: Counter                    = field(default_factory=Counter)  # text label → count
    solution_templates: Dict[str, Dict[str, Any]] = field(default_factory=dict) # template_id → {domain, outline, cues}
    ngram_success: Counter                    = field(default_factory=Counter)  # "ngram::domain" → count
    ngram_fail: Counter                       = field(default_factory=Counter)
    version: int                              = 1  # state schema version


# --------------------------- Service -----------------------------

class KnowledgeBaseService(Service):
    """
    Cross-paper memory for 'learning from learning'.
    Persists compact stats + templates and surfaces them as context for new papers.

    Public API (used by agents):
      - update_from_paper(domain, summary_text, metrics, iterations, strategy_delta)
      - context_for_paper(title, abstract, domain) -> {"hints","templates","weight_nudges",...}
    """

    def __init__(self, cfg: Dict[str, Any], memory, logger):
        self.cfg = cfg or {}
        self.memory = memory
        self.logger = logger

        kb_cfg = (self.cfg.get("knowledge_base") or {})
        # Back-compat with old flat key:
        self.path = kb_cfg.get("state_path") or self.cfg.get("knowledge_base_path") or "data/kbase/knowledge_base.json"
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        self.enabled: bool = bool(kb_cfg.get("enabled", True))
        self.success_threshold: float = float(kb_cfg.get("success_threshold", self.cfg.get("kb_success_threshold", 0.80)))
        self.max_templates_per_domain: int = int(kb_cfg.get("templates_per_domain", 12))

        # nudge heuristics (can be tuned via config)
        nudges = (kb_cfg.get("nudges") or {})
        self.halluc_thresh = float(nudges.get("hallucination_rate_threshold", 0.25))
        self.coverage_thresh = float(nudges.get("low_coverage_threshold", 0.70))
        self.kv_thresh = float(nudges.get("knowledge_verification_threshold", 0.60))
        self.skeptic_nudge = float(nudges.get("skeptic_nudge", 0.05))
        self.editor_nudge  = float(nudges.get("editor_nudge", 0.03))
        self.risk_nudge    = float(nudges.get("risk_nudge", 0.03))

        self._state: KnowledgeBaseState = KnowledgeBaseState()
        self._initialized = False
        self._lock = threading.RLock()

    # -------- Service protocol --------

    @property
    def name(self) -> str:
        return "kbase"

    def initialize(self, **kwargs) -> None:
        if self._initialized or not self.enabled:
            return
        try:
            with self._lock:
                self._state = self._load()
                self._initialized = True
            self.logger and self.logger.log("KBaseInitialized", {
                "path": self.path,
                "domains": len(self._state.domain_stats),
                "templates": len(self._state.solution_templates),
            })
        except Exception as e:
            self.logger and self.logger.log("KBaseInitError", {"error": str(e)})
            # Don't raise; kbase is optional

    def shutdown(self) -> None:
        try:
            with self._lock:
                if self._initialized:
                    self._save()
                self._initialized = False
            self.logger and self.logger.log("KBaseShutdown", {})
        except Exception as e:
            self.logger and self.logger.log("KBaseShutdownError", {"error": str(e)})

    def health_check(self) -> Dict[str, Any]:
        with self._lock:
            st = self._state
            return {
                "status": "healthy" if (self._initialized and self.enabled) else "uninitialized",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": self.path,
                "version": st.version,
                "domains": len(st.domain_stats),
                "templates": len(st.solution_templates),
                "signals": {
                    "strategy_updates": len(st.strategy_updates),
                    "failure_modes": sum(st.failure_modes.values()),
                    "ngram_success": sum(st.ngram_success.values()),
                    "ngram_fail": sum(st.ngram_fail.values()),
                },
            }

    # -------- Public API (unchanged) --------

    def update_from_paper(
        self,
        *,
        domain: str,
        summary_text: str,
        metrics: Dict[str, Any],
        iterations: List[Dict[str, Any]],
        strategy_delta: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Capture lessons after Track C completes.
        """
        if not self.enabled:
            return

        ok = bool(float(metrics.get("overall", 0.0)) >= self.success_threshold)
        iters = int(len(iterations or []))
        domain = (domain or "unknown").lower()

        with self._lock:
            # 1) domain stats
            self._state.domain_stats[domain].update(ok, iters)

            # 2) failure modes (if not ok, label a rough cause)
            if not ok:
                cause = self._diagnose_failure(metrics)
                self._state.failure_modes[cause] += 1

            # 3) n-gram patterns
            grams = self._ngrams(summary_text.lower(), n=3)
            bucket = self._state.ngram_success if ok else self._state.ngram_fail
            for g in grams:
                bucket[f"{g}::{domain}"] += 1

            # 4) learn solution template from strong wins
            if ok and float(metrics.get("knowledge_verification", 0.0)) >= 0.85:
                tid = f"tmpl_{len(self._state.solution_templates)+1:04d}"
                self._state.solution_templates[tid] = {
                    "domain": domain,
                    "outline": self._extract_outline(summary_text),
                    "cues": ["cite figures for numerics", "cover all key claims", "avoid speculative language"],
                    "ts": time.time(),
                    "quality": float(metrics.get("overall", 0.0)),
                }
                # cap per domain
                self._cap_templates_per_domain(domain)

            # 5) record strategy change (if any)
            if strategy_delta:
                self._state.strategy_updates.append(KBStrategyUpdate(
                    when=datetime.now(timezone.utc).isoformat(),
                    reason="post_track_c_improvement",
                    delta=dict(strategy_delta)
                ))

            self._save()  # small file; safe to save synchronously

    def context_for_paper(self, *, title: str, abstract: str, domain: str) -> Dict[str, Any]:
        """
        Returns hints/templates/weight_nudges computed from prior signals.
        """
        domain = (domain or "unknown").lower()
        with self._lock:
            dstat = self._state.domain_stats.get(domain) or KBDomainStats()

            # choose up to 2 best templates in this domain
            templates = [
                v for v in self._state.solution_templates.values()
                if v.get("domain") == domain
            ]
            templates = sorted(templates, key=lambda t: (-float(t.get("quality", 0.0)), -float(t.get("ts", 0.0))))[:2]

            # n-gram risk gate
            grams = set(self._ngrams((title + " " + abstract).lower(), n=3))
            risk_flag = any(self._state.ngram_fail.get(f"{g}::{domain}", 0) >
                            self._state.ngram_success.get(f"{g}::{domain}", 0) for g in grams)

            hints: List[str] = []
            nudges: Dict[str, float] = {}

            if dstat.n >= 5 and dstat.avg_iters > 2.5:
                hints.append("Prior papers in this domain needed >2.5 iterations—strengthen claim grounding and structure.")
                nudges["editor"] = nudges.get("editor", 0.0) + self.editor_nudge

            # global trend from failures
            if sum(self._state.failure_modes.values()) >= 5:
                fm = self._state.failure_modes.most_common(1)[0][0]
                if fm == "hallucination":
                    hints.append("Hallucination is a frequent failure—prefer conservative phrasing and explicit evidence.")
                    nudges["skeptic"] = nudges.get("skeptic", 0.0) + self.skeptic_nudge
                elif fm == "low_coverage":
                    hints.append("Coverage is a common weakness—ensure all key claims are explicitly addressed.")
                    nudges["editor"] = nudges.get("editor", 0.0) + self.editor_nudge

            if risk_flag:
                hints.append("Historically tricky phrasing detected—require figure/table citations for numeric claims.")
                nudges["risk"] = nudges.get("risk", 0.0) + self.risk_nudge
                nudges["skeptic"] = nudges.get("skeptic", 0.0) + self.skeptic_nudge

            # normalize nudges to [-0.0, +0.4] window the agent expects
            for k in list(nudges.keys()):
                nudges[k] = max(0.0, min(0.4, float(nudges[k])))

            return {
                "domain_stats": {
                    "n": dstat.n,
                    "successes": dstat.successes,
                    "failures": dstat.failures,
                    "avg_iters": round(dstat.avg_iters, 2),
                },
                "templates": [{"outline": t.get("outline", []), "cues": t.get("cues", [])} for t in templates],
                "hints": hints,
                "weight_nudges": nudges,
            }

    # -------- Optional helpers (nice to have) --------

    def get_stats(self) -> Dict[str, Any]:
        """Quick snapshot for debugging or a /metrics endpoint."""
        with self._lock:
            return self.health_check()

    def reset_domain(self, domain: str) -> None:
        """Erase domain-specific memory (debug tool)."""
        d = (domain or "unknown").lower()
        with self._lock:
            self._state.domain_stats.pop(d, None)
            # remove templates for the domain
            self._state.solution_templates = {
                k: v for k, v in self._state.solution_templates.items() if v.get("domain") != d
            }
            # scrub ngrams for the domain
            for key in list(self._state.ngram_success.keys()):
                if key.endswith(f"::{d}"):
                    del self._state.ngram_success[key]
            for key in list(self._state.ngram_fail.keys()):
                if key.endswith(f"::{d}"):
                    del self._state.ngram_fail[key]
            self._save()

    # ------------------------ Internals ----------------------------

    def _cap_templates_per_domain(self, domain: str) -> None:
        """Keep only the top-k templates per domain by (quality, recency)."""
        items = [(k, v) for k, v in self._state.solution_templates.items() if v.get("domain") == domain]
        if len(items) <= self.max_templates_per_domain:
            return
        items = sorted(items, key=lambda kv: (-float(kv[1].get("quality", 0.0)), -float(kv[1].get("ts", 0.0))))
        keep = set(k for k, _ in items[:self.max_templates_per_domain])
        self._state.solution_templates = {k: v for k, v in self._state.solution_templates.items() if (v.get("domain") != domain) or (k in keep)}

    @staticmethod
    def _ngrams(text: str, n=3) -> List[str]:
        toks = re.findall(r"[a-z0-9]+", (text or "").lower())
        return [" ".join(toks[i:i+n]) for i in range(0, max(0, len(toks)-n+1))]

    @staticmethod
    def _extract_outline(text: str) -> List[str]:
        # very small heuristic: split into <=8 short bullets
        sents = re.split(r"(?<=[.!?])\s+", (text or "").strip())
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

    # -------- Persistence --------

    def _save(self) -> None:
        data = self._to_json(self._state)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def _load(self) -> KnowledgeBaseState:
        if not os.path.exists(self.path):
            return KnowledgeBaseState()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return self._from_json(raw)
        except Exception as e:
            self.logger and self.logger.log("KBLoadError", {"error": str(e)})
            return KnowledgeBaseState()

    @staticmethod
    def _to_json(st: KnowledgeBaseState) -> Dict[str, Any]:
        return {
            "version": st.version,
            "strategy_updates": [asdict(x) for x in st.strategy_updates],
            "domain_stats": {k: asdict(v) for k, v in st.domain_stats.items()},
            "failure_modes": dict(st.failure_modes),
            "solution_templates": st.solution_templates,
            "ngram_success": dict(st.ngram_success),
            "ngram_fail": dict(st.ngram_fail),
        }

    @staticmethod
    def _from_json(raw: Dict[str, Any]) -> KnowledgeBaseState:
        st = KnowledgeBaseState()
        st.version = int(raw.get("version", 1))
        for r in raw.get("strategy_updates", []):
            st.strategy_updates.append(KBStrategyUpdate(when=r["when"], reason=r["reason"], delta=dict(r["delta"])))
        for k, v in (raw.get("domain_stats") or {}).items():
            ds = KBDomainStats(
                successes=int(v.get("successes", 0)),
                failures=int(v.get("failures", 0)),
                avg_iters=float(v.get("avg_iters", 0.0)),
                n=int(v.get("n", 0)),
            )
            st.domain_stats[k] = ds
        st.failure_modes.update(raw.get("failure_modes", {}))
        st.solution_templates = raw.get("solution_templates", {})
        st.ngram_success.update(raw.get("ngram_success", {}))
        st.ngram_fail.update(raw.get("ngram_fail", {}))
        return st
