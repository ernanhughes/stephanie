# stephanie/protocols/idea_extractor_protocol.py
from typing import Protocol

from stephanie.models.belief_cartridge import BeliefCartridge


class LearnableIdeaExtractor(Protocol):
    def extract(self, paper_text: str, metadata: dict) -> BeliefCartridge:
        """Extracts a learnable idea from a block of knowledge (e.g. a research paper)."""
