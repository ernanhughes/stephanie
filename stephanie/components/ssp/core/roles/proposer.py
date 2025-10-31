# stephanie/components/ssp/core/roles/proposer.py
"""
Proposer Role Interface

The Proposer is responsible for generating questions from seed answers.
In the SSP methodology, the Proposer:
- Takes a ground truth answer (seed)
- Generates a question that can be answered by that seed
- May perform retrieval to gather evidence while crafting the question
- Returns both the question and any evidence gathered

A key aspect of the paper's implementation is that the Proposer should be
"searching" - it actively gathers evidence while crafting the question,
rather than just generating a question in isolation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from stephanie.components.ssp.core.protocols import EpisodeContext


class Proposer(ABC):
    """
    Abstract interface for the Proposer component in SSP.

    The Proposer generates questions from seed answers, potentially using
    retrieval to gather evidence while crafting the question.
    """

    @abstractmethod
    async def propose(
        self, seed_answer: str, context: Dict[str, Any] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a question from a seed answer.

        Args:
            seed_answer: Ground truth answer to build a question around
            context: Additional context for the proposal

        Returns:
            Tuple of (question, metadata) where metadata should include:
            - evidence_snippets: List of evidence snippets gathered during proposal
            - difficulty: Estimated question difficulty (0-1)
            - other relevant metrics
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return information about the Proposer's capabilities.

        Returns:
            Dictionary describing capabilities like:
            - supports_search_during_proposal: bool
            - max_evidence_snippets: int
            - etc.
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return the current configuration of the Proposer.

        Returns:
            Dictionary of configuration parameters
        """
        pass
