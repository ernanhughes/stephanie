"""
Rule-Based Filters for SSP

This module implements the rule-based filters required before
RAG-gated verification, as specified in the SSP paper.
"""

import re
from typing import List


def check_question_length(question: str, min_length: int = 20) -> bool:
    """Check if question meets minimum length requirement."""
    return len(question.strip()) >= min_length


def check_answer_leakage(question: str, seed_answer: str) -> bool:
    """
    Check if the question contains the seed answer (answer leakage).
    
    Returns True if leakage is detected (which means the filter fails).
    """
    # Normalize strings
    q_norm = re.sub(r'[^\w\s]', '', question.lower())
    ans_norm = re.sub(r'[^\w\s]', '', seed_answer.lower())
    
    # Check for direct inclusion
    if ans_norm in q_norm:
        return True
    
    # Check for word-by-word inclusion (more robust)
    ans_words = set(ans_norm.split())
    q_words = set(q_norm.split())
    
    # If more than 50% of answer words appear in question, consider it leakage
    if ans_words and (len(ans_words & q_words) / len(ans_words)) > 0.5:
        return True
    
    return False


def check_evidence_usage(evidence_snippets: List[str], min_count: int = 1) -> bool:
    """Check if sufficient evidence snippets were gathered."""
    return len(evidence_snippets) >= min_count


def check_tool_usage(evidence_snippets: List[str]) -> bool:
    """Check if evidence comes from actual tool usage (not empty)."""
    return bool(evidence_snippets) and any(snippet.strip() for snippet in evidence_snippets)


def check_format(question: str) -> bool:
    """Check if question is properly formatted."""
    # Must end with question mark
    if not question.strip().endswith('?'):
        return False
    
    # Must not be all caps
    if question.isupper():
        return False
    
    # Must not contain prompt injection patterns
    injection_patterns = [
        r"ignore\s+previous",
        r"forget\s+all",
        r"system\s+prompt",
        r"role\s+play",
        r"disregard\s+instructions"
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return False
    
    return True