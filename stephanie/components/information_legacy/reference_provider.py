from abc import ABC, abstractmethod
from typing import Any, List

from stephanie.components.information_legacy.data import ReferenceRecord


class ReferenceProvider(ABC):
    """Abstract interface for getting references for a given paper."""

    @abstractmethod
    def get_references_for_arxiv(self, arxiv_id: str) -> List[ReferenceRecord]:
        """
        Return a list of references for the given arxiv_id.
        The implementation can call the arxiv API, OpenAlex, or use your own extractor.
        """
        raise NotImplementedError

class ArxivReferenceProvider(ReferenceProvider):
    def __init__(self, api_client: Any) -> None:
        self.api_client = api_client  # your real arxiv client

    def get_references_for_arxiv(self, arxiv_id: str) -> List[ReferenceRecord]:
        # TODO: implement using your real arxiv/OpenAlex client
        # This is just a skeleton.
        #
        # raw_refs = self.api_client.get_references(arxiv_id)
        # return [
        #     ReferenceRecord(
        #         arxiv_id=ref.get("arxiv_id"),
        #         doi=ref.get("doi"),
        #         title=ref.get("title"),
        #         year=ref.get("year"),
        #         url=ref.get("pdf_url"),
        #         raw_citation=ref.get("raw"),
        #     )
        #     for ref in raw_refs
        # ]
        return []
