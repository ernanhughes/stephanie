"""
RAG-Gated Verifier Implementation

This implementation performs the critical RAG-gated verification step
from the SSP paper:
1. Applies rule-based filters to the question
2. Runs verification using the solver in NO-SEARCH mode with proposer's evidence
3. Determines if the question should be accepted for deep search
"""

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple

from stephanie.components.ssp.core.roles import Verifier
from stephanie.components.ssp.core.protocols import EpisodeContext, VerificationResult
from stephanie.components.ssp.utils.filters import (
    check_question_length,
    check_answer_leakage,
    check_evidence_usage,
    check_tool_usage,
    check_format
)


class RAGVerifier(Verifier):
    """
    Verifier implementation that performs RAG-gated verification.
    
    This implements the paper's verification process:
    1. Apply rule-based filters
    2. Run verification using Solver with ONLY proposer's evidence
    3. Determine if question should be accepted
    """
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        memory: Any,
        container: Any,
        logger: Any = None,
        solver: Any = None
    ):
        """
        Initialize the RAGVerifier.
        
        Args:
            cfg: Configuration dictionary
            memory: Memory tool
            container: Dependency container
            logger: Logger instance
            solver: Optional pre-configured Solver instance
        """
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self.solver = solver or container.get("solver")
        
        if not self.solver:
            raise ValueError("Solver is required for verification")
        
        # Configuration parameters
        self.min_question_len = cfg.get("verify", {}).get("min_question_len", 20)
        self.forbid_answer_leak = cfg.get("verify", {}).get("forbid_answer_leak", True)
        self.min_evidence_count = cfg.get("verify", {}).get("min_evidence_count", 1)
        self.pass_threshold = cfg.get("verify", {}).get("pass_threshold", 0.75)
    
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
        failed_rules = []
        
        # Check question length
        if not check_question_length(question, self.min_question_len):
            failed_rules.append(f"question_too_short (< {self.min_question_len} chars)")
        
        # Check for answer leakage
        if self.forbid_answer_leak and check_answer_leakage(question, seed_answer):
            failed_rules.append("answer_leakage_detected")
        
        # Check evidence usage
        if not check_evidence_usage(evidence_snippets, self.min_evidence_count):
            failed_rules.append(f"insufficient_evidence (< {self.min_evidence_count})")
        
        # Check tool usage (evidence must come from search)
        if not check_tool_usage(evidence_snippets):
            failed_rules.append("no_tool_usage_detected")
        
        # Check format
        if not check_format(question):
            failed_rules.append("invalid_format")
        
        return len(failed_rules) == 0, failed_rules
    
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
        # 1. Apply rule-based filters
        filters_valid, failed_rules = self.apply_filters(
            question, 
            evidence_snippets,
            seed_answer
        )
        
        if not filters_valid:
            return VerificationResult(
                is_valid=False,
                score=0.0,
                reason=f"Failed rule-based filters: {', '.join(failed_rules)}",
                filter_results={rule.split('(')[0]: False for rule in failed_rules},
                verification_details={"failed_rules": failed_rules}
            )
        
        # 2. Run RAG-gated verification (solver with no search)
        verification_result = await self.solver.verify_answer(
            question,
            seed_answer,
            evidence_snippets
        )
        
        # 3. Determine if question should be accepted
        is_valid = verification_result.is_valid
        
        return VerificationResult(
            is_valid=is_valid,
            score=verification_result.score,
            reason=verification_result.reason,
            filter_results={rule.split('(')[0]: True for rule in failed_rules} if filters_valid else 
                         {rule.split('(')[0]: False for rule in failed_rules},
            verification_details=verification_result.verification_details
        )
    
    def get_filter_rules(self) -> List[Dict[str, Any]]:
        """Return the current filter rules configuration."""
        return [
            {
                "name": "question_length",
                "description": f"Question must be at least {self.min_question_len} characters",
                "enabled": True,
                "min_length": self.min_question_len
            },
            {
                "name": "answer_leakage",
                "description": "Question must not contain the seed answer",
                "enabled": self.forbid_answer_leak
            },
            {
                "name": "evidence_usage",
                "description": f"Must have at least {self.min_evidence_count} evidence snippets",
                "enabled": True,
                "min_count": self.min_evidence_count
            },
            {
                "name": "tool_usage",
                "description": "Evidence must come from search tool usage",
                "enabled": True
            },
            {
                "name": "format",
                "description": "Question must be properly formatted",
                "enabled": True
            }
        ]
    
    def get_verification_threshold(self) -> float:
        """Return the current verification score threshold."""
        return self.pass_threshold