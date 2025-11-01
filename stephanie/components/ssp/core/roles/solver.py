# stephanie/components/ssp/core/roles/solver.py
"""
Solver Role Interface

The Solver is responsible for answering questions using deep search.
In the SSP methodology, the Solver has TWO MODES:
1. Verification mode: answers using ONLY the proposer's evidence (NO SEARCH)
2. Deep search mode: performs full search to find the best answer
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from stephanie.components.ssp.core.protocols import EpisodeContext
from stephanie.components.ssp.core.protocols import VerificationResult


class Solver(ABC):
    """
    Abstract interface for the Solver component in SSP.
    
    The Solver answers questions using various search strategies.
    It operates in two modes:
    1. Verification mode: answers using ONLY the proposer's evidence (no search)
    2. Deep search mode: performs full search to find the best answer
    """
    
    @abstractmethod
    async def solve(
        self,
        question: str,
        seed_answer: str,
        context: Optional[EpisodeContext] = None,
        use_search: bool = True,
        evidence_snippets: Optional[List[str]] = None
    ) -> Tuple[str, List[str], int, Dict[str, Any]]:
        """
        Solve a question using the appropriate search strategy.
        
        Args:
            question: Question to answer
            seed_answer: Ground truth answer (for search guidance)
            context: Additional context for solving
            use_search: Whether to perform search (False for verification mode)
            evidence_snippets: Optional evidence to use (for verification mode)
            
        Returns:
            Tuple of (predicted_answer, evidence_used, steps_taken, metadata)
        """
        pass
    
    @abstractmethod
    async def verify_answer(
        self,
        question: str,
        seed_answer: str,
        evidence_snippets: List[str]
    ) -> VerificationResult:
        """
        Verify an answer using ONLY the provided evidence (no search).
        
        This implements the RAG-gated verification step from the paper.
        
        Args:
            question: Question to answer
            seed_answer: Ground truth answer to verify against
            evidence_snippets: Evidence gathered by the Proposer
            
        Returns:
            VerificationResult object with verification outcome
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return information about the Solver's capabilities.
        
        Returns:
            Dictionary describing capabilities like:
            - supports_verification_mode: bool
            - max_search_depth: int
        """
        pass