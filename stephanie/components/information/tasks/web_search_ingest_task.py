from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

log = logging.getLogger(__name__)


# -------------------------
# Config
# -------------------------

@dataclass(frozen=True)
class GoalQualityProfile:
    # weights for quality dimensions (0..1, sum not required; store normalizes)
    weights: Dict[str, float]
    # selection budgets per result_type
    budgets: Dict[str, int]
    # global cap
    max_total: int = 12


DEFAULT_PROFILES: Dict[str, GoalQualityProfile] = {
    "research": GoalQualityProfile(
        weights={
            "relevance": 0.35,
            "authority": 0.30,
            "evidence": 0.20,
            "depth": 0.15,
        },
        budgets={
            "paper_pdf": 5,
            "paper_html": 2,
            "documentation": 2,
            "pdf": 1,
            "blog": 2,
            "wiki": 1,
            "code_repo": 1,
            "unknown": 1,
        },
        max_total=12,
    ),
    "blog_write": GoalQualityProfile(
        weights={
            "relevance": 0.35,
            "accessibility": 0.25,
            "evidence": 0.20,
            "authority": 0.20,
        },
        budgets={
            "paper_pdf": 4,
            "documentation": 3,
            "blog": 3,
            "wiki": 1,
            "paper_html": 2,
            "unknown": 1,
        },
        max_total=12,
    ),
    "reproduce_results": GoalQualityProfile(
        weights={
            "authority": 0.35,
            "evidence": 0.25,
            "relevance": 0.25,
            "depth": 0.15,
        },
        budgets={
            "paper_pdf": 6,
            "paper_html": 2,
            "documentation": 2,
            "code_repo": 2,
            "pdf": 1,
            "unknown": 1,
        },
        max_total=12,
    ),
}


# -------------------------
# Heuristics
# -------------------------

ARXIV_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(?:v\d+)?", re.I)

PUBLISHER_DOMAINS = {
    "acm.org",
    "dl.acm.org",
    "ieee.org",
    "ieeexplore.ieee.org",
    "springer.com",
    "link.springer.com",
    "nature.com",
    "sciencedirect.com",
    "jmlr.org",
    "openreview.net",
}

DOC_DOMAINS = {
    "scikit-learn.org",
    "pytorch.org",
    "tensorflow.org",
    "docs.python.org",
    "readthedocs.io",
}

BLOG_DOMAINS = {
    "medium.com",
    "substack.com",
    "towardsdatascience.com",
    "wordpress.com",
}

CODE_DOMAINS = {"github.com", "gitlab.com", "bitbucket.org"}

WIKI_DOMAINS = {"wikipedia.org"}


def _domain(url: str) -> str:
    try:
        d = urlparse(url).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""


def guess_result_type(url: str) -> str:
    d = _domain(url)
    u = (url or "").lower()

    if "arxiv.org/pdf/" in u or (d == "arxiv.org" and u.endswith(".pdf")):
        return "paper_pdf"
    if d == "arxiv.org" and "/abs/" in u:
        return "paper_html"
    if d in CODE_DOMAINS:
        return "code_repo"
    if any(d.endswith(x) for x in WIKI_DOMAINS):
        return "wiki"
    if d in DOC_DOMAINS or d.endswith("readthedocs.io"):
        return "documentation"
    if d in BLOG_DOMAINS:
        return "blog"
    if d in PUBLISHER_DOMAINS:
        return "paper_html"
    if u.endswith(".pdf"):
        return "pdf"
    return "unknown"


def lexical_relevance(query: str, title: str, snippet: str) -> float:
    # cheap baseline: token overlap
    q = (query or "").lower()
    t = ((title or "") + " " + (snippet or "")).lower()
    q_toks = {w for w in re.split(r"\W+", q) if len(w) >= 3}
    if not q_toks:
        return 0.0
    hits = sum(1 for w in q_toks if w in t)
    return float(min(1.0, hits / max(6, len(q_toks))))


def authority_score(url: str, result_type: str) -> float:
    d = _domain(url)
    base = {
        "paper_pdf": 0.85,
        "paper_html": 0.75,
        "documentation": 0.70,
        "pdf": 0.55,
        "blog": 0.40,
        "wiki": 0.45,
        "code_repo": 0.50,
        "unknown": 0.35,
    }.get(result_type, 0.35)

    # domain boosts
    if d == "arxiv.org":
        base += 0.10
    if d in PUBLISHER_DOMAINS:
        base += 0.10
    if d in DOC_DOMAINS or d.endswith("readthedocs.io"):
        base += 0.05

    return float(max(0.0, min(1.0, base)))


def evidence_score(url: str, result_type: str, snippet: str) -> float:
    base = {
        "paper_pdf": 0.80,
        "paper_html": 0.65,
        "documentation": 0.55,
        "pdf": 0.50,
        "wiki": 0.45,
        "blog": 0.35,
        "code_repo": 0.45,
        "unknown": 0.30,
    }.get(result_type, 0.30)

    # light boosts if snippet suggests citations/math
    s = (snippet or "").lower()
    if any(x in s for x in ["doi", "arxiv", "et al", "theorem", "proof", "equation"]):
        base += 0.08

    return float(max(0.0, min(1.0, base)))


def depth_score(result_type: str, snippet: str) -> float:
    base = {
        "paper_pdf": 0.80,
        "paper_html": 0.65,
        "documentation": 0.55,
        "pdf": 0.55,
        "code_repo": 0.45,
        "wiki": 0.35,
        "blog": 0.35,
        "unknown": 0.30,
    }.get(result_type, 0.30)
    s = (snippet or "").lower()
    if any(x in s for x in ["kernel", "svm", "proof", "derivation", "complexity", "experiment"]):
        base += 0.05
    return float(max(0.0, min(1.0, base)))


def accessibility_score(url: str, result_type: str) -> float:
    # very rough: arxiv + docs are usually accessible
    d = _domain(url)
    base = {
        "paper_pdf": 0.80,
        "documentation": 0.80,
        "wiki": 0.80,
        "blog": 0.75,
        "code_repo": 0.70,
        "paper_html": 0.55,  # often paywalled
        "pdf": 0.55,
        "unknown": 0.50,
    }.get(result_type, 0.50)

    if d == "arxiv.org":
        base += 0.10
    if d in PUBLISHER_DOMAINS:
        base -= 0.10

    return float(max(0.0, min(1.0, base)))


def verifiability_score(url: str, result_type: str) -> float:
    u = (url or "").lower()
    if u.startswith("https://"):
        base = 0.70
    elif u.startswith("http://"):
        base = 0.55
    else:
        base = 0.40

    if result_type in ("paper_pdf", "documentation", "wiki"):
        base += 0.10
    if u.endswith(".pdf"):
        base += 0.05

    return float(max(0.0, min(1.0, base)))


# -------------------------
# Task
# -------------------------

class WebSearchIngestTask:
    """
    Inputs:
      - results: list[dict] with keys: url, title, snippet, rank (optional), provider(optional)
      - goal_type: e.g. "research", "blog_write", "reproduce_results"
      - pipeline_run_id: int
      - run_dir: artifact output directory

    Side effects:
      - creates Source records (via SourceStore) for urls
      - inserts SourceCandidate rows
      - inserts SourceQuality rows (heuristic_v1)
      - writes sources_ranked.json + fetch_plan.json

    Dependencies:
      - source_store: must expose get_or_create_source_id(source_type, locator, **kwargs)
      - candidate_store: SourceCandidateStore
      - quality_store: SourceQualityStore
    """

    def __init__(
        self,
        *,
        source_store: Any,
        candidate_store: Any,
        quality_store: Any,
        profiles: Optional[Dict[str, GoalQualityProfile]] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self.source_store = source_store
        self.candidate_store = candidate_store
        self.quality_store = quality_store
        self.profiles = profiles or DEFAULT_PROFILES
        self.logger = logger

    def run(
        self,
        *,
        pipeline_run_id: int,
        goal_type: str,
        query_text: str,
        results: List[Dict[str, Any]],
        run_dir: str,
    ) -> Dict[str, Any]:
        profile = self.profiles.get(goal_type) or self.profiles["research"]

        os.makedirs(run_dir, exist_ok=True)

        ranked_rows: List[Dict[str, Any]] = []

        # 1) record candidates + compute quality
        for i, r in enumerate(results):
            url = (r.get("url") or r.get("link") or "").strip()
            if not url:
                continue

            title = (r.get("title") or "").strip()
            snippet = (r.get("snippet") or r.get("summary") or "").strip()
            rank = r.get("rank")
            if rank is None:
                rank = i

            provider = r.get("provider")

            rtype = r.get("result_type") or guess_result_type(url)

            # canonical source
            src_id = self.source_store.get_or_create_source_id(
                source_type="url",
                locator=url,
                canonical_locator=url,
                meta={
                    "domain": _domain(url),
                    "result_type_guess": rtype,
                    "provider": provider,
                },
            )

            # candidate row (search episode)
            cand_id = self.candidate_store.upsert_candidate(
                pipeline_run_id=pipeline_run_id,
                goal_type=goal_type,
                query_text=query_text,
                source_id=src_id,
                rank=int(rank) if isinstance(rank, int) else None,
                title=title,
                snippet=snippet,
                result_type=rtype,
                provider=provider,
                status="pending",
                meta={"raw": {k: v for k, v in r.items() if k not in ("html", "content")}},
            )

            # 2) heuristic quality dimensions
            dims = {
                "relevance": lexical_relevance(query_text, title, snippet),
                "authority": authority_score(url, rtype),
                "evidence": evidence_score(url, rtype, snippet),
                "depth": depth_score(rtype, snippet),
                "accessibility": accessibility_score(url, rtype),
                "verifiability": verifiability_score(url, rtype),
            }

            # store quality rows
            for dim, sc in dims.items():
                self.quality_store.upsert_quality(
                    source_id=src_id,
                    goal_type=goal_type,
                    dimension=dim,
                    score=float(sc),
                    judge_type="heuristic_v1",
                    judge_version="v1",
                    pipeline_run_id=pipeline_run_id,
                    rationale=None,
                )

            total, breakdown = self.quality_store.compute_total(
                source_id=src_id,
                goal_type=goal_type,
                weights=profile.weights,
                judge_type="heuristic_v1",
            )

            # update candidate cached total
            self.candidate_store.upsert_candidate(
                pipeline_run_id=pipeline_run_id,
                goal_type=goal_type,
                query_text=query_text,
                source_id=src_id,
                rank=int(rank) if isinstance(rank, int) else None,
                title=title,
                snippet=snippet,
                result_type=rtype,
                provider=provider,
                status="pending",
                quality_total=total,
            )

            ranked_rows.append(
                {
                    "candidate_id": cand_id,
                    "source_id": src_id,
                    "url": url,
                    "domain": _domain(url),
                    "rank": rank,
                    "title": title,
                    "snippet": snippet,
                    "result_type": rtype,
                    "quality_total": total,
                    "quality_breakdown": breakdown,
                }
            )

        # sort by quality desc, then provider rank asc
        ranked_rows.sort(key=lambda x: (-float(x.get("quality_total") or 0.0), int(x.get("rank") or 999999)))

        # 3) selection: bucketed budgets + diversity by domain
        fetch_plan = self._select_fetch_plan(ranked_rows, profile)

        # mark selected candidates
        selected_ids = {x["candidate_id"] for x in fetch_plan}
        for row in ranked_rows:
            if row["candidate_id"] in selected_ids:
                self._set_candidate_status(pipeline_run_id, goal_type, query_text, row["source_id"], "selected")
            else:
                self._set_candidate_status(pipeline_run_id, goal_type, query_text, row["source_id"], "pending")

        # 4) emit artifacts
        ranked_path = os.path.join(run_dir, "sources_ranked.json")
        plan_path = os.path.join(run_dir, "fetch_plan.json")

        self._dump(ranked_path, {
            "pipeline_run_id": pipeline_run_id,
            "goal_type": goal_type,
            "query_text": query_text,
            "count": len(ranked_rows),
            "results": ranked_rows,
        })

        self._dump(plan_path, {
            "pipeline_run_id": pipeline_run_id,
            "goal_type": goal_type,
            "query_text": query_text,
            "selected_count": len(fetch_plan),
            "selected": fetch_plan,
        })

        if self.logger:
            self.logger.info(f"[WebSearchIngestTask] ranked={len(ranked_rows)} selected={len(fetch_plan)} → {plan_path}")
        else:
            log.info("[WebSearchIngestTask] ranked=%d selected=%d → %s", len(ranked_rows), len(fetch_plan), plan_path)

        return {
            "sources_ranked_path": ranked_path,
            "fetch_plan_path": plan_path,
            "selected": fetch_plan,
            "ranked": ranked_rows,
        }

    # -------------------------
    # internals
    # -------------------------

    def _dump(self, path: str, obj: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def _set_candidate_status(self, pipeline_run_id: int, goal_type: str, query_text: str, source_id: int, status: str) -> None:
        # re-upsert with status change (safe)
        self.candidate_store.upsert_candidate(
            pipeline_run_id=pipeline_run_id,
            goal_type=goal_type,
            query_text=query_text,
            source_id=source_id,
            status=status,
        )

    def _select_fetch_plan(self, ranked_rows: List[Dict[str, Any]], profile: GoalQualityProfile) -> List[Dict[str, Any]]:
        budgets = dict(profile.budgets or {})
        max_total = int(profile.max_total)

        used_by_type: Dict[str, int] = {}
        used_domains: Dict[str, int] = {}

        out: List[Dict[str, Any]] = []

        def can_take(rt: str) -> bool:
            b = budgets.get(rt, 0)
            return used_by_type.get(rt, 0) < b

        for row in ranked_rows:
            if len(out) >= max_total:
                break

            rt = row.get("result_type") or "unknown"
            dom = row.get("domain") or ""

            # enforce per-type budgets, fall back to unknown budget if needed
            if rt not in budgets:
                rt_budget = budgets.get("unknown", 0)
                if used_by_type.get("unknown", 0) >= rt_budget:
                    continue
                rt = "unknown"  # treat as unknown bucket

            if not can_take(rt):
                continue

            # diversity: avoid too many from same domain
            if dom and used_domains.get(dom, 0) >= 2:
                continue

            # choose parser route
            parser = self._choose_parser(row["url"], rt)

            out.append({
                "candidate_id": row["candidate_id"],
                "source_id": row["source_id"],
                "url": row["url"],
                "result_type": rt,
                "quality_total": row.get("quality_total"),
                "parser": parser,
                "reason": f"top_{rt}_by_quality",
            })

            used_by_type[rt] = used_by_type.get(rt, 0) + 1
            if dom:
                used_domains[dom] = used_domains.get(dom, 0) + 1

        return out

    def _choose_parser(self, url: str, result_type: str) -> str:
        # route into your existing toolchain
        if result_type == "paper_pdf":
            return "docling_pdf"
        if result_type == "paper_html":
            return "publisher_html"
        if result_type == "documentation":
            return "html_docs"
        if result_type == "wiki":
            return "html_wiki"
        if result_type == "blog":
            return "html_blog"
        if result_type == "code_repo":
            return "repo"
        if result_type == "pdf":
            return "generic_pdf"
        return "html_generic"
