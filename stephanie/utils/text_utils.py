# stephanie/utils/text_utils.py
"""
Text Processing Utilities

This module provides functions for text segmentation, word extraction, 
and text similarity analysis using lexical overlap metrics.

Features:
- Sentence segmentation with configurable limits
- Word tokenization with Unicode support
- Lexical overlap (Jaccard similarity) calculation
"""
from __future__ import annotations

import re
from typing import List, Optional

# Regular expressions compiled at module level for efficiency
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")  # Split on sentence boundaries
WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)  # Match words with Unicode support
ALPHA_WORD_PATTERN = re.compile(r"[A-Za-z]+", re.UNICODE)  # Strict alphabetic words

def sentences(text: str, max_sents: Optional[int] = None) -> List[str]:
    """
    Split text into sentences with optional length limit.
    
    Args:
        text: Input text to split into sentences
        max_sents: Maximum number of sentences to return. If None, returns all sentences.
    
    Returns:
        List of sentence strings, stripped of whitespace
        
    Examples:
        >>> sentences("Hello world! How are you? I'm fine.")
        ['Hello world!', 'How are you?', "I'm fine."]
        
        >>> sentences("Short. A", max_sents=1)
        ['Short.']
    """
    if not text or not text.strip():
        return []
    
    # Split and clean sentences
    raw_sentences = SENTENCE_SPLIT.split(text)
    cleaned_sentences = [sent.strip() for sent in raw_sentences if sent.strip()]
    
    # Filter out very short sentences (likely false positives)
    meaningful_sentences = [sent for sent in cleaned_sentences if len(sent) > 2]
    
    # Apply length limit if specified
    if max_sents is not None and max_sents > 0:
        return meaningful_sentences[:max_sents]
    
    return meaningful_sentences


def words(text: str) -> List[str]:
    """
    Extract all words from text, converted to lowercase.
    
    Args:
        text: Input text to extract words from
        
    Returns:
        List of lowercase words
        
    Examples:
        >>> words("Hello World! How are you?")
        ['hello', 'world', 'how', 'are', 'you']
        
        >>> words("Email: test@example.com")
        ['email', 'test', 'example', 'com']
    """
    if not text:
        return []
    
    return WORD_PATTERN.findall(text.lower())


def lexical_overlap(a: str, b: str) -> float:
    """
    Calculate lexical overlap (Jaccard similarity) between two texts.
    
    The similarity is computed as the size of the intersection of word sets
    divided by the size of the first text's word set.
    
    Args:
        a: First text for comparison
        b: Second text for comparison
        
    Returns:
        Float between 0.0 and 1.0 representing the overlap ratio
        
    Examples:
        >>> lexical_overlap("hello world", "world peace")
        0.5  # 1 common word / 2 words in first text
        
        >>> lexical_overlap("hello", "goodbye")
        0.0  # No common words
    """
    words_a, words_b = words(a), words(b)
    
    # Early return for empty cases
    if not words_a:
        return 0.0
    
    # Convert to sets for efficient intersection
    set_a, set_b = set(words_a), set(words_b)
    
    # Calculate Jaccard similarity based on set A
    common_words = set_a & set_b
    return len(common_words) / len(set_a)


def jaccard_similarity(a: str, b: str) -> float:
    """
    Calculate symmetric Jaccard similarity between two texts.
    
    This is an alternative to lexical_overlap that considers both texts
    equally: |A ∩ B| / |A ∪ B|
    
    Args:
        a: First text for comparison
        b: Second text for comparison
        
    Returns:
        Float between 0.0 and 1.0 representing the similarity
        
    Examples:
        >>> jaccard_similarity("hello world", "world peace")
        0.333  # 1 common word / 3 unique words total
    """
    set_a, set_b = set(words(a)), set(words(b))
    
    # Handle case where both sets are empty
    if not set_a and not set_b:
        return 1.0  # Both are empty, consider them identical
    if not set_a or not set_b:
        return 0.0  # One is empty, no similarity
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union


# Alternative name for lexical_overlap for clarity
text_coverage = lexical_overlap


# ---- Section / quality helpers ----------------------------------------


def alpha_words(text: str) -> List[str]:
    """
    Extract alphabetic-only words from text (A–Z), lowercased.

    This intentionally ignores numbers / weird tokens like 'cid89', so that
    section quality isn't inflated by LaTeX artifacts or encoding noise.
    """
    if not text:
        return []
    return ALPHA_WORD_PATTERN.findall(text.lower())


def count_alpha_words(text: str) -> int:
    """Convenience wrapper around alpha_words."""
    return len(alpha_words(text))


def section_quality(text: str, min_words: int = 30) -> float:
    """
    Compute a simple section quality score in [0.0, 1.0].

    - Strip to alphabetic words
    - Score = min(1.0, word_count / min_words)

    So:
      - 0 words      -> 0.0
      - 15 words     -> 0.5  (if min_words=30)
      - 30+ words    -> 1.0
    """
    n = count_alpha_words(text)
    if n <= 0:
        return 0.0
    return min(1.0, n / float(min_words))


def is_high_quality_section(text: str, min_words: int = 30) -> bool:
    """
    Return True if this looks like a 'real' section for our purposes.

    Right now the rule is:
      - After stripping to alphabetic words, we require at least min_words.

    You can evolve this later (e.g. add sentence count, stopword ratio, etc.)
    without touching all the callers.
    """
    return section_quality(text, min_words=min_words) >= 1.0

def safe_slice(t: str, start: int, end: int) -> str:
    start = max(0, min(start, len(t)))
    end = max(start, min(end, len(t)))
    return t[start:end]


def safe_snip(text: Optional[str], n: int) -> Optional[str]:
    if not text:
        return None
    t = re.sub(r"\s+", " ", text).strip()
    return t[:n] + ("…" if len(t) > n else "")


if __name__ == "__main__":
    # Simple test examples
    test_text = "This is a test. This is only a test! Do you understand?"
    
    print("Sentences:", sentences(test_text))
    print("Words:", words(test_text))
    print("Lexical overlap:", lexical_overlap("hello world", "world peace"))
    print("Jaccard similarity:", jaccard_similarity("hello world", "world peace"))