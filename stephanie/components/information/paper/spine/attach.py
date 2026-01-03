from typing import List

from stephanie.components.information.data import (
    DocumentElement,
    PaperSection,
    attach_elements_to_sections,
)


def build_spine(
    sections: List[PaperSection],
    elements: List[DocumentElement],
):
    """
    Attach elements to sections and return spine object.
    """
    return attach_elements_to_sections(sections, elements)
