# stephanie/models/capability.py
"""Model capability facts (new build — §5 of the Stage 1 spec).

Capabilities describe facts about a model. Policy decides whether
Stephanie should use them. Never mix the two.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelCapabilities:
    text_generation: bool = True

    reasoning: bool = False
    tool_use: bool = False
    structured_output: bool = False

    vision: bool = False
    audio_input: bool = False
    audio_output: bool = False

    embeddings: bool = False

    streaming: bool = False
    async_supported: bool = True

    max_context_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None

    def satisfies(self, required: set[str]) -> bool:
        """Check that every capability name in ``required`` is truthy."""
        for name in required:
            if not getattr(self, name, False):
                return False
        return True
