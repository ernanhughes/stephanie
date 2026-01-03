from typing import Any, Dict, List

from stephanie.components.information.data import PaperSection, DocumentElement


def emit_processing_signals(
    *,
    context: Dict[str, Any],
    proc_results: List[Any],
    sections: List[PaperSection],
    elements: List[DocumentElement],
):
    """
    Emit lightweight signals for downstream reporting.
    """
    signals = context.setdefault("paper_processing_signals", {})
    signals["spine"] = {
        "num_sections": len(sections),
        "num_elements": len(elements),
        "processors": [r.name for r in proc_results if getattr(r, "ran", False)],
    }
