from typing import Dict, List

from stephanie.components.information.data import PaperSection


def ensure_section_page_fields(sections: List[PaperSection]) -> None:
    """
    Ensure start_page/end_page fields exist.
    """
    for s in sections:
        if getattr(s, "start_page", None) is None:
            s.start_page = None
        if getattr(s, "end_page", None) is None:
            s.end_page = None


def needs_page_fallback(sections: List[PaperSection]) -> bool:
    """
    True if sections lack page ranges.
    """
    if not sections:
        return True
    return any(s.start_page is None or s.end_page is None for s in sections)


def make_page_sections(
    *,
    arxiv_id: str,
    paper_role: str,
    num_pages: int,
    page_text_by_page: Dict[int, str],
) -> List[PaperSection]:
    """
    Build one section per page fallback.
    """
    sections: List[PaperSection] = []
    for page in range(1, num_pages + 1):
        sections.append(
            PaperSection(
                paper_id=arxiv_id,
                role=paper_role,
                title=f"Page {page}",
                start_page=page,
                end_page=page,
                text=page_text_by_page.get(page, ""),
            )
        )
    return sections
