# stephanie/components/information/quality.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from stephanie.components.information.data import InformationSource
from stephanie.utils.date_utils import iso_now

log = logging.getLogger(__name__)



@dataclass
class SourceQualityScores:
    relevance: float          # 0.0–1.0
    novelty: float            # 0.0–1.0
    authority: float          # 0.0–1.0
    blog_usable: bool
    drop_reason: Optional[str] = None


class InformationQualityPass:
    """
    Offline / post-processing pass that scores and filters the
    InformationSources attached to an information MemCube.

    It expects MemCubes produced by InformationProcessor, i.e.:

        memcube.extra_data["information_sources"] = [
            { "kind": ..., "id": ..., "title": ..., "text": ..., "meta": {...} },
            ...
        ]

    and writes quality metadata back into each source's meta:

        meta["relevance_score"]
        meta["novelty_score"]
        meta["authority_score"]
        meta["blog_usable"]
        meta["drop_reason"]

    At the MemCube level, it can also set:

        extra_data["info_blog_ready_level"]
        extra_data["info_quality_summary"]

    You can run this from a batch agent or as a follow-up step
    in the ingest pipeline.
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg 
        self.memory = memory
        self.container = container
        self.logger = logger
        self.memcube_store = memory.memcubes

        # thresholds (can be tuned via cfg)
        self.min_relevance = float(self.cfg.get("min_relevance", 0.4))
        self.min_authority = float(self.cfg.get("min_authority", 0.3))
        self.min_novelty   = float(self.cfg.get("min_novelty", 0.1))  # avoid perfect dupes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_for_memcube_id(self, memcube_id: str) -> None:
        """
        Fetch a MemCube by id, score its information_sources, and write
        the updated extra_data back to the store.
        """
        cube = self.memcube_store.get_by_id(memcube_id)
        if cube is None:
            log.info("MemCube id %s not found", memcube_id)
            return

        updated_extra = self._score_sources_for_cube(cube)
        cube.extra_data = updated_extra
        data = cube.to_dict(include_extra=True)
        data["extra_data"] = updated_extra

        self.memcube_store.upsert(data, merge_extra=False)


        log.info("updated memcube_id %s %s ", memcube_id, updated_extra.get("info_blog_ready_level")           )

    def run_for_many(self, memcube_ids: List[str]) -> None:
        """
        Convenience helper to batch over a list of MemCube ids.
        """
        for mid in memcube_ids:
            self.run_for_memcube_id(mid)

    # ------------------------------------------------------------------
    # Core scoring logic
    # ------------------------------------------------------------------

    def _score_sources_for_cube(self, cube) -> Dict[str, Any]:
        extra = dict(cube.extra_data or {})
        sources_data = extra.get("information_sources") or []
        if not sources_data:
            # nothing to do
            return extra

        # First entry is the primary source (see InformationProcessor)
        primary_dict = sources_data[0]
        primary = self._to_source(primary_dict)

        related_dicts = sources_data[1:]
        related_sources = [self._to_source(d) for d in related_dicts]

        # Compute quality per source
        scored_sources: List[Dict[str, Any]] = []
        usable_count = 0
        dropped_count = 0

        for idx, src in enumerate([primary] + related_sources):
            scores = self._compute_quality(primary, src)

            if idx == 0:
                # primary is always usable; enforce that
                scores.blog_usable = True
                scores.drop_reason = None

            src_dict = self._update_source_meta(src, scores)
            scored_sources.append(src_dict)

            if scores.blog_usable:
                usable_count += 1
            else:
                dropped_count += 1

        extra["information_sources"] = scored_sources

        # Derive page-level summary
        total = len(scored_sources)
        noise_ratio = dropped_count / total if total else 0.0

        extra["info_quality_summary"] = {
            "total_sources": total,
            "usable_sources": usable_count,
            "dropped_sources": dropped_count,
            "noise_ratio": noise_ratio,
            "last_scored_at": iso_now(),
        }

        # Simple blog readiness heuristic:
        # 0 = raw ingest only
        # 1 = quality scored & filtered
        # 2+ reserved for future refiner stages
        extra["info_blog_ready_level"] = max(
            1, int(extra.get("info_blog_ready_level", 0))
        )

        return extra

    def _to_source(self, d: Dict[str, Any]) -> InformationSource:
        return InformationSource(
            kind=d.get("kind", ""),
            id=d.get("id", ""),
            title=d.get("title", ""),
            text=d.get("text", ""),
            meta=d.get("meta", {}) or {},
        )

    def _update_source_meta(
        self,
        src: InformationSource,
        scores: SourceQualityScores,
    ) -> Dict[str, Any]:
        meta = dict(src.meta or {})
        meta["relevance_score"] = scores.relevance
        meta["novelty_score"] = scores.novelty
        meta["authority_score"] = scores.authority
        meta["blog_usable"] = scores.blog_usable
        if scores.drop_reason:
            meta["drop_reason"] = scores.drop_reason

        return {
            "kind": src.kind,
            "id": src.id,
            "title": src.title,
            "text": src.text,
            "meta": meta,
        }

    # ------------------------------------------------------------------
    # Heuristics (can later be backed by embeddings / MRQ / etc.)
    # ------------------------------------------------------------------

    def _compute_quality(
        self,
        primary: InformationSource,
        src: InformationSource,
    ) -> SourceQualityScores:
        # 1) relevance: shared tokens in titles (+ text hint)
        rel = self._title_relevance(primary.title, src.title)

        # 2) novelty: different enough from primary body
        nov = self._novelty(primary.text, src.text)

        # 3) authority: simple kind-based heuristic
        auth = self._authority(src)

        # 4) blog_usable decision
        blog_usable, drop_reason = self._decide_usable(rel, nov, auth, src)

        return SourceQualityScores(
            relevance=rel,
            novelty=nov,
            authority=auth,
            blog_usable=blog_usable,
            drop_reason=drop_reason,
        )

    def _normalize_tokens(self, s: str) -> List[str]:
        s = (s or "").lower()
        s = re.sub(r"[^a-z0-9\s]+", " ", s)
        tokens = [t for t in s.split() if len(t) > 2]
        return tokens

    def _title_relevance(self, primary_title: str, src_title: str) -> float:
        ptoks = set(self._normalize_tokens(primary_title))
        stoks = set(self._normalize_tokens(src_title))
        if not ptoks or not stoks:
            return 0.0
        inter = len(ptoks & stoks)
        union = len(ptoks | stoks)
        return inter / union if union else 0.0

    def _novelty(self, primary_text: str, src_text: str) -> float:
        """
        Very rough novelty heuristic: compare lengths & token overlap.
        In practice, you'll replace this with embedding-based distance.
        """
        ptoks = set(self._normalize_tokens(primary_text))
        stoks = set(self._normalize_tokens(src_text))

        if not ptoks or not stoks:
            return 0.5  # unknown → neutral

        inter = len(ptoks & stoks)
        union = len(ptoks | stoks)
        jacc = inter / union if union else 0.0

        # novelty is "1 - similarity", but we clamp to avoid extremes
        nov = 1.0 - jacc
        return max(0.0, min(1.0, nov))

    def _authority(self, src: InformationSource) -> float:
        kind = (src.kind or "").lower()
        url = (src.meta or {}).get("url", "") or src.id or ""

        # Very crude ordering; tune or externalize later.
        if kind == "arxiv":
            return 0.9
        if kind == "wiki":
            return 0.6
        if "arxiv.org" in url:
            return 0.8
        if "aclanthology.org" in url or "neurips.cc" in url:
            return 0.8
        if "github.com" in url:
            return 0.5
        if "reddit.com" in url:
            return 0.3
        return 0.4  # generic blog / web page

    def _decide_usable(
        self,
        relevance: float,
        novelty: float,
        authority: float,
        src: InformationSource,
    ) -> Tuple[bool, Optional[str]]:
        kind = (src.kind or "").lower()
        title = (src.title or "").lower()

        # Hard drops: obviously off-topic wiki pages etc.
        if kind == "wiki":
            # crude guard against generic self-help / misc pages
            bad_keywords = [
                "twelve-step program",
                "gan",
                "generative adversarial network",
                "psychology",
            ]
            if any(k in title for k in bad_keywords):
                return False, "off_topic_wiki"

        # If everything is low, drop.
        if relevance < self.min_relevance and authority < self.min_authority:
            return False, "low_relevance_and_authority"

        # If it's near-duplicate of primary (very low novelty), drop.
        if novelty < self.min_novelty:
            return False, "near_duplicate"

        return True, None
