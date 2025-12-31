# stephanie/components/information/linkers/base.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from stephanie.components.information.data import PaperSection
from stephanie.components.information.graph.paper_graph_abi import GraphEdge


class BaseSectionLinker:
    name: str = "base"

    def link(
        self,
        *,
        root_arxiv_id: str,
        root_sections: Sequence[PaperSection],
        corpus_sections: Sequence[PaperSection],
        context: Dict[str, Any],
    ) -> List[GraphEdge]:
        raise NotImplementedError

_ARXIV_PREFIX_RE = re.compile(r"^(?P<pid>\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})$", re.IGNORECASE)

def _norm_pid(x: str | None) -> str:
    x = (x or "").strip()
    return x.split("v")[0]  # drop version if present

def _infer_pid_from_section_id(section_id: str) -> str | None:
    head = (section_id or "").split("::", 1)[0].strip()
    if not head:
        return None
    return head if _ARXIV_PREFIX_RE.match(head) else None



def section_pid(s) -> str:
    # prefer declared paper_arxiv_id, but repair if it’s clearly wrong
    declared = _norm_pid(getattr(s, "paper_arxiv_id", None))
    inferred = _infer_pid_from_section_id(getattr(s, "id", "") or "")
    if inferred and declared and inferred != declared:
        return inferred
    return declared or inferred or ""
