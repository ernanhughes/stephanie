# stephanie/tools/text_tool.py
from __future__ import annotations

import logging
import math
from collections import Counter

from stephanie.tools.base_tool import BaseTool

log = logging.getLogger(__name__)

class TextTool(BaseTool):
    """
    Compute basic text statistics for a scorable.
    Fast, pure-Python, no dependencies.
    """

    name = "text"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.include_entropy = bool(cfg.get("include_entropy", True))
        self.include_sentences = bool(cfg.get("include_sentences", True))

    # ---------------------------------------------------------
    async def apply(self, scorable, context: dict):
        text = scorable.text or ""
        if not text.strip():
            return scorable

        stats = self._compute_stats(text)

        # attach to scorable.meta
        scorable.meta.setdefault("text_stats", {})
        scorable.meta["text_stats"].update(stats)

        return scorable

    # ---------------------------------------------------------
    def _compute_stats(self, text: str) -> dict:
        stripped = text.strip()

        tokens = stripped.split()
        num_tokens = len(tokens)
        num_chars = len(stripped)
        num_digits = sum(c.isdigit() for c in stripped)
        num_upper = sum(c.isupper() for c in stripped)
        num_alpha = sum(c.isalpha() for c in stripped)
        num_spaces = stripped.count(" ")
        punct = sum(c in ".,?!;:()[]{}\"'" for c in stripped)

        word_lengths = [len(t) for t in tokens]
        avg_word_len = (sum(word_lengths) / len(word_lengths)) if word_lengths else 0.0

        # lexical diversity
        unique_tokens = len(set(tokens))
        lex_div = (unique_tokens / num_tokens) if num_tokens > 0 else 0.0

        stats = {
            "chars": num_chars,
            "tokens": num_tokens,
            "digits": num_digits,
            "upper_ratio": (num_upper / num_chars) if num_chars else 0.0,
            "punct_ratio": (punct / num_chars) if num_chars else 0.0,
            "spaces": num_spaces,
            "num_alpha": num_alpha,
            "avg_word_len": avg_word_len,
            "lexical_diversity": lex_div,
        }

        # optional entropy
        if self.include_entropy:
            stats["entropy"] = self._entropy(stripped)

        # optional sentence metrics
        if self.include_sentences:
            sents = [s.strip() for s in stripped.replace("!", ".").replace("?", ".").split(".") if s.strip()]
            num_sents = len(sents)
            avg_sent_len = (sum(len(s) for s in sents) / num_sents) if num_sents else 0.0
            stats["sentences"] = num_sents
            stats["avg_sentence_len"] = avg_sent_len

        return stats

    # ---------------------------------------------------------
    def _entropy(self, text: str) -> float:
        counts = Counter(text)
        total = sum(counts.values())
        if total == 0:
            return 0.0
        p = [c / total for c in counts.values()]
        return float(-sum(pi * math.log(pi + 1e-12) for pi in p))
