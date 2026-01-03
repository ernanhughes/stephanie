from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ProcessorResult:
    name: str
    ran: bool
    num_new: int = 0
    meta: Dict[str, Any] | None = None


class BaseSpineProcessor:
    """
    Base class for all spine processors.
    """

    name: str = "base"

    async def run(
        self,
        *,
        arxiv_id: str,
        pdf_path,
        elements: list,
        context: Dict[str, Any],
    ) -> tuple[list, ProcessorResult]:
        raise NotImplementedError
