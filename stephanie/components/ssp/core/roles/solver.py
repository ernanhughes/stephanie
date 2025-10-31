# stephanie/components/ssp/core/roles/solver.py
"""
Solver Role Interface

The Solver is responsible for answering questions using deep search.
In the SSP methodology, the Solver:
- Takes a question and attempts to answer it
- Uses retrieval and tree search (like ATS) to find evidence
- May have two modes: 
  1. Verification mode (no search, using only proposer's evidence)
  2. Deep search mode (full search for final answer)
- Returns the predicted answer and evidence used
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from stephanie.components.ssp.core.protocols import EpisodeContext


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
        context: Dict[str, Any],
        use_search: bool = True,
        evidence_snippets: Optional[List[str]] = None
    ) -> Tuple[str, List[str], int]:
        """
        Solve a question using the appropriate search strategy.
        
        Args:
            question: Question to answer
            seed_answer: Ground truth answer (for search guidance)
            context: Additional context for solving
            use_search: Whether to perform search (False for verification mode)
            evidence_snippets: Optional evidence to use (for verification mode)
            
        Returns:
            Tuple of (predicted_answer, evidence_used, steps_taken)
        """
        pass
    
    @abstractmethod
    async def verify_answer(
        self,
        question: str,
        seed_answer: str,
        evidence_snippets: List[str]
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Verify an answer using ONLY the provided evidence (no search).
        
        This implements the RAG-gated verification step from the paper.
        
        Args:
            question: Question to answer
            seed_answer: Ground truth answer to verify against
            evidence_snippets: Evidence gathered by the Proposer
            
        Returns:
            Tuple of (is_correct, score, metadata)
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
            - etc.
        """
        pass