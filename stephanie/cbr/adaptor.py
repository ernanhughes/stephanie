# stephanie/cbr/adaptor.py
from __future__ import annotations

from typing import Any, Dict

from dspy import InputField, OutputField, Predict, Signature


class AdaptSignature(Signature):
    goal      = InputField(desc="Goal text (what we are trying to achieve)")
    candidate = InputField(desc="Base candidate text to improve")
    retrieved = InputField(desc="Relevant retrieved case text to adapt from")
    revised   = OutputField(desc="Revised/improved candidate that better fits the goal")
    rationale = OutputField(desc="Brief explanation of what was changed and why")

_ADAPT_SYSTEM_PROMPT = (
    "You are an expert editor. Improve the candidate so it better satisfies the goal. "
    "Use helpful ideas from the retrieved case, but keep things concise, testable, and actionable. "
    "Do NOT add preambles; return only the revised content."
)

class DefaultAdaptor:
    """
    Simple LLM-based adaptor/reviser.
    Uses the globally configured DSPy LM (no separate client needed).
    """

    def __init__(self, cfg: Dict[str, Any], logger=None):
        self.cfg = cfg or {}
        self.logger = logger
        # one-shot predictor; rely on upstream dspy.configure(lm=...)
        self._predict = Predict(AdaptSignature)

    def revise(self, *, goal_text: str, candidate_text: str, retrieved_text: str) -> Dict[str, str]:
        try:
            # Some LLMs benefit from a system prompt; DSPy can include it via messages if configured.
            out = self._predict(goal=goal_text, candidate=candidate_text, retrieved=retrieved_text)
            revised = (getattr(out, "revised", "") or "").strip()
            rationale = (getattr(out, "rationale", "") or "").strip()
            if not revised:
                # Fallback: if model returns nothing, just return the candidate
                revised = candidate_text
                rationale = rationale or "No change; adaptor fallback."
            return {"revised": revised, "rationale": rationale}
        except Exception as e:
            self.logger and self.logger.log("AdaptError", {"error": str(e)})
            return {"revised": candidate_text, "rationale": "Adaptor error; returning original."}
