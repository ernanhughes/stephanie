# stephanie/components/ssp/core/roles/proposer.py
"""
Proposer Role Interface

The Proposer is responsible for generating questions from seed answers.
In the SSP methodology, the Proposer:
- Takes a ground truth answer (seed)
- Performs retrieval to gather evidence WHILE crafting the question
- Returns both the question and the evidence gathered
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from stephanie.components.ssp.core.protocols import EpisodeContext


class Proposer(ABC):
    """
    Abstract interface for the Proposer component in SSP.
    
    The Proposer generates questions from seed answers WITH EVIDENCE GATHERING.
    This is the "searching proposer" described in the paper.
    """
    
    @abstractmethod
    async def propose(
        self,
        seed_answer: str,
        context: Optional[EpisodeContext] = None
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Generate a question from a seed answer WITH EVIDENCE GATHERING.
        
        Args:
            seed_answer: Ground truth answer to build a question around
            context: Additional context for the proposal
            
        Returns:
            Tuple of (question, evidence_snippets, metadata) where:
            - evidence_snippets: Evidence gathered DURING question creation
            - metadata should include difficulty, etc.
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return information about the Proposer's capabilities.
        
        Returns:
            Dictionary describing capabilities like:
            - supports_search_during_proposal: bool (should be True)
            - max_evidence_snippets: int
        """
        pass