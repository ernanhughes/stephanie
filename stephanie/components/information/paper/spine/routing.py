from typing import Any, Dict


def should_continue_docling(
    *,
    page_index: int,
    doctag_count: int,
    max_pages: int | None,
    cfg: Dict[str, Any],
) -> bool:
    """
    Heuristic gating for Docling continuation.
    """
    if max_pages and page_index >= max_pages:
        return False

    min_tags = cfg.get("min_doctags_per_page", 1)
    return doctag_count >= min_tags
