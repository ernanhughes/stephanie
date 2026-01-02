from typing import List

from stephanie.components.information.utils.spine_dump import SpineDumper
from stephanie.components.information.data import PaperSection, DocumentElement


def dump_spine(
    *,
    dumper: SpineDumper,
    arxiv_id: str,
    sections: List[PaperSection],
    elements: List[DocumentElement],
    spine,
    proc_results,
):
    """
    Dump spine artifacts.
    """
    return dumper.dump(
        arxiv_id=arxiv_id,
        sections=sections,
        elements=elements,
        spine=spine,
        proc_results=proc_results,
    )
