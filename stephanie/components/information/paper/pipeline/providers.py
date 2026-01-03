# stephanie/components/information/paper/pipeline/providers.py
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List

from stephanie.components.information.data import (PaperReferenceRecord)
from stephanie.components.information.tasks.reference_graph_task import (
    ReferenceProvider, SimilarPaperProvider)
from stephanie.tools.huggingface_tool import recommend_similar_papers

log = logging.getLogger(__name__)


class LocalJsonReferenceProvider(ReferenceProvider):
    """
    Reference provider that reads the references your PaperImportTool has
    already extracted and saved as `papers_root/<key>/references.json`.

    This means:
      - PaperImportTool remains the *only* place that parses PDFs
      - The graph/task code just consumes the structured JSON
    """

    def __init__(self, papers_root: Path, max_refs: int = 256) -> None:
        self.papers_root = Path(papers_root)
        self.max_refs = max_refs

    def _references_path_for(self, arxiv_id: str) -> Path:
        """
        By convention we store under:
            papers_root/<key>/references.json

        For arxiv IDs, <key> is just the ID. For local PDFs, it's the stem.
        """
        return self.papers_root / arxiv_id / "references.json"

    def get_references_for_arxiv(self, arxiv_id: str) -> List[PaperReferenceRecord]:
        path = self._references_path_for(arxiv_id)

        if not path.exists():
            log.debug(
                "LocalJsonReferenceProvider: no references.json for %s at %s",
                arxiv_id,
                path,
            )
            return []

        try:
            raw_list = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(
                "LocalJsonReferenceProvider: failed to read %s for %s: %s",
                path,
                arxiv_id,
                e,
            )
            return []

        refs: List[PaperReferenceRecord] = []
        for idx, item in enumerate(raw_list[: self.max_refs]):
            try:
                refs.append(
                    PaperReferenceRecord(
                        arxiv_id=item.get("arxiv_id"),
                        doi=item.get("doi"),
                        title=item.get("title"),
                        year=item.get("year"),
                        url=item.get("url"),
                        raw_citation=item.get("raw_citation"),
                    )
                )
            except TypeError as exc:
                # In case the JSON has extra keys that don't map cleanly
                log.warning(
                    "LocalJsonReferenceProvider: bad ref #%d in %s (%s): %s",
                    idx,
                    arxiv_id,
                    path,
                    exc,
                )
                continue

        log.info(
            "LocalJsonReferenceProvider: loaded %d references for %s from %s",
            len(refs),
            arxiv_id,
            path,
        )
        return refs


class HFSimilarPaperProvider(SimilarPaperProvider):
    """
    Similar-paper provider using your HuggingFace Tool.

    It calls recommend_similar_papers(paper_url) and maps the results into
    ReferenceRecord objects.
    """

    def __init__(self, max_limit: int = 16) -> None:
        self.max_limit = max_limit

    def get_similar_for_arxiv(
        self, arxiv_id: str, limit: int = 10
    ) -> List[PaperReferenceRecord]:
        limit = min(limit, self.max_limit)
        url = f"https://arxiv.org/abs/{arxiv_id}"

        try:
            hits = recommend_similar_papers(paper_url=url)
        except Exception as e:
            log.warning("HF similar papers failed for %s: %s", arxiv_id, e)
            return []

        recs: List[PaperReferenceRecord] = []
        for h in hits[:limit]:
            h_url = h.get("url") or h.get("paper_url") or ""
            if not h_url:
                continue

            # Try to extract an arxiv-like id from the URL.
            # Example patterns:
            #   https://arxiv.org/pdf/2505.08827.pdf
            #   https://arxiv.org/pdf/2505.08827
            m = re.search(r"/(\d{4}\.\d{4,5})(?:\.pdf)?$", h_url)
            if not m:
                # fall back to title if it looks like an id
                title = h.get("title", "")
                m2 = re.search(r"(\d{4}\.\d{4,5})", title)
                if not m2:
                    continue
                pid = m2.group(1)
            else:
                pid = m.group(1)

            recs.append(
                PaperReferenceRecord(
                    arxiv_id=pid,
                    title=h.get("title"),
                    url=h_url,
                    source="hf_similar",
                    raw=h,
                )
            )

        return recs
