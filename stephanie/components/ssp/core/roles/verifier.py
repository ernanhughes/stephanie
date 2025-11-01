# stephanie/components/ssp/core/roles/verifier.py
"""
Verifier Role Interface

The Verifier is responsible for filtering questions before they
proceed to deep search. In the SSP methodology, the Verifier:
- Applies rule-based filters to questions
- Runs the RAG-gated verification (using Solver in no-search mode)
- Determines if a question should be accepted or rejected
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from stephanie.components.ssp.core.protocols import EpisodeContext


class Verifier(ABC):
    """
    Abstract interface for the Verifier component in SSP.
    
    The Verifier filters questions before they proceed to deep search.
    It implements the critical RAG-gated verification step from the paper.
    """
    
    @abstractmethod
    def apply_filters(
        self,
        question: str,
        evidence_snippets: List[str],
        seed_answer: str
    ) -> Tuple[bool, List[str]]:
        """
        Apply rule-based filters to a question.
        
        Args:
            question: Question to filter
            evidence_snippets: Evidence gathered by the Proposer
            seed_answer: Ground truth answer
            
        Returns:
            Tuple of (is_valid, list_of_failed_rules)
        """
        pass
    
    @abstractmethod
    async def verify(
        self,
        question: str,
        seed_answer: str,
        evidence_snippets: List[str],
        context: Optional[EpisodeContext] = None
    ) -> VerificationResult:
        """
        Verify a question meets all criteria for deep search.
        
        This implements the paper's RAG-gated verification:
        1. Apply rule-based filters
        2. Run verification using Solver with ONLY proposer's evidence
        3. Determine if question should be accepted
        
        Args:
            question: Question to verify
            seed_answer: Ground truth answer
            evidence_snippets: Evidence gathered by the Proposer
            context: Additional context for verification
            
        Returns:
            VerificationResult object with verification outcome
        """
        pass
    
    @abstractmethod
    def get_filter_rules(self) -> List[Dict[str, Any]]:
        """
        Return the current filter rules configuration.
        
        Returns:
            List of filter rule definitions
        """
        pass
    
    @abstractmethod
    def get_verification_threshold(self) -> float:
        """
        Return the current verification score threshold.
        
        Returns:
            Threshold value (0-1)
        """
        pass