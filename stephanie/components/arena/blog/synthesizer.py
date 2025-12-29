# stephanie/components/arena/blog/synthesizer.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class SectionSynthesizer:
    """
    Single responsibility:
      (problem_text + support snippets) -> final section text

    Inject any callable:
      synth_fn(problem_text, supports, ctx) -> str
    """

    def __init__(self, *, synth_fn: Callable[[str, List[Dict[str, Any]], Dict[str, Any]], Any]):
        self.synth_fn = synth_fn

    async def synthesize(
        self,
        *,
        problem_text: str,
        supports: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        # synth_fn may be sync or async; reuse arena's pattern if you want
        out = self.synth_fn(problem_text, supports, dict(context or {}))
        if hasattr(out, "__await__"):
            out = await out
        return (out or "").strip()
