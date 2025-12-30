from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from stephanie.memory.source_store import SourceStore


@dataclass
class IngestedSource:
    source_id: int
    candidate_id: Optional[int]
    trust_score: Optional[float]
    quality_score: Optional[float]
    verification: Optional[str]


class SourceIngestTask:
    """
    Convert search hits into:
      - SourceORM (canonical)
      - SourceCandidateORM (per-run/per-query)
      - SourceQualityORM (goal-conditioned heuristic priors)

    Later you can add LLM/HRM-based quality; this is the “good enough v1”.
    """

    def __init__(self, *, source_store: SourceStore, logger=None) -> None:
        self.sources = source_store
        self.logger = logger

    def ingest_hits(
        self,
        *,
        pipeline_run_id: int,
        goal_type: str,
        query_text: str,
        hits: List[Dict[str, Any]],
    ) -> List[IngestedSource]:
        out: List[IngestedSource] = []

        for idx, h in enumerate(hits):
            url = (h.get("url") or "").strip()
            if not url:
                continue

            provider = (h.get("source") or h.get("provider") or "").strip() or None
            result_type = (h.get("result_type") or "").strip() or None
            title = (h.get("title") or "").strip() or None
            snippet = (h.get("summary") or h.get("snippet") or "").strip() or None

            # v1: infer source_type + priors from URL
            source_type, canonical_uri = self._classify_uri(url)
            trust, qual, verification, rationale = self._heuristic_quality(
                goal_type=goal_type,
                source_type=source_type,
                canonical_uri=canonical_uri or url,
                provider=provider,
                result_type=result_type,
            )

            meta = {
                "provider": provider,
                "result_type": result_type,
                "pid": h.get("pid"),
                "raw": {k: v for k, v in h.items() if k not in ("embedding",)},
            }

            source_id = self.sources.get_or_create_source(
                source_type=source_type,
                source_uri=url,
                canonical_uri=canonical_uri,
                title=title,
                snippet=snippet,
                verification=verification,
                trust_score=trust,
                quality_score=qual,
                meta=meta,
            )

            cand_id = self.sources.add_candidate(
                pipeline_run_id=pipeline_run_id,
                query_text=query_text,
                source_id=source_id,
                provider=provider,
                result_type=result_type,
                rank=int(h.get("rank", idx)) if h.get("rank", idx) is not None else None,
                score=float(h["score"]) if isinstance(h.get("score"), (int, float)) else None,
                meta={"rationale": rationale},
            )

            self.sources.upsert_quality(
                pipeline_run_id=pipeline_run_id,
                goal_type=goal_type,
                source_id=source_id,
                trust_score=trust,
                quality_score=qual,
                verification=verification,
                method="heuristic",
                rationale=rationale,
            )

            out.append(
                IngestedSource(
                    source_id=source_id,
                    candidate_id=cand_id,
                    trust_score=trust,
                    quality_score=qual,
                    verification=verification,
                )
            )

        return out

    # -------------------------
    # heuristics
    # -------------------------

    def _classify_uri(self, uri: str) -> Tuple[str, Optional[str]]:
        """
        Return (source_type, canonical_uri).
        """
        u = uri.strip()
        if u.startswith("file://"):
            return "file", u
        if u.startswith("http://") or u.startswith("https://"):
            # normalize arxiv pdf urls a bit
            try:
                p = urlparse(u)
                host = (p.netloc or "").lower()
                if "arxiv.org" in host:
                    # canonicalize to pdf link form if possible
                    # keep it simple: store as canonical_uri = given
                    return "url", u
                return "url", u
            except Exception:
                return "url", u
        if "postgres://" in u or "mysql://" in u:
            return "db", u
        if u.startswith("llm://") or u.startswith("ai://"):
            return "ai", u
        return "unknown", u

    def _heuristic_quality(
        self,
        *,
        goal_type: str,
        source_type: str,
        canonical_uri: str,
        provider: Optional[str],
        result_type: Optional[str],
    ) -> Tuple[float, float, str, str]:
        """
        Return (trust_score, quality_score, verification, rationale).
        Scores are 0..1.
        """
        uri = (canonical_uri or "").lower()
        provider_l = (provider or "").lower()
        rt = (result_type or "").lower()

        # Generated content starts lower by default
        if source_type in ("ai", "generated"):
            return 0.15, 0.20, "generated", "source_type=ai|generated"

        # Strong priors for arXiv / papers
        if "arxiv.org" in uri or rt in ("paper", "pdf") or provider_l == "arxiv":
            return 0.85, 0.80, "verified", "arxiv/paper prior"

        # Wikipedia (good for definitions, weaker for cutting-edge claims)
        if "wikipedia.org" in uri or provider_l == "wikipedia" or rt == "wiki":
            return 0.70, 0.60, "unverified", "wikipedia prior"

        # GitHub repos (depends)
        if "github.com" in uri or rt == "repo":
            return 0.65, 0.60, "unverified", "github prior"

        # Blogs / random web
        if rt in ("blog", "web", "post") or provider_l == "web":
            return 0.40, 0.40, "unverified", "web/blog prior"

        return 0.50, 0.50, "unknown", "default prior"
