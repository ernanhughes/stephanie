from __future__ import annotations
import asyncio, logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

@runtime_checkable
class PairScorer(Protocol):
    async def score_text_pair(self, goal: str, reply: str, *, model_alias: str = "chat",
                              monitor_alias: str = "cove", context: Optional[Dict[str, Any]] = None) -> Dict[str, float]: ...

@dataclass
class CoVeConfig:
    questions: int = 5
    timeout_s: float = 6.0

COVE_PLAN_PROMPT = """You are verifying the following draft answer.
Question/Goal: {goal}
Draft answer:
\"\"\"{reply}\"\"\"

Generate {k} short verification questions that, if answered correctly, would confirm key claims in the draft. Keep them atomic.
"""

COVE_VERIFY_PROMPT = """Answer the following verification question concisely. If unknown, say 'UNKNOWN'.
Question: {q}
"""

class CoVeScorer(PairScorer):
    """
    Chain-of-Verification style monitor:
      - Plan K verify-questions about the draft reply.
      - Answer each independently.
      - Score pass-rate by checking if answers agree with the draft (string match / optional NLI).
    """
    def __init__(self, container: Any, logger: Optional[logging.Logger] = None, cfg: Optional[CoVeConfig] = None):
        self.container = container
        self.logger = logger or logging.getLogger(__name__)
        self.cfg = cfg or CoVeConfig()
        self.llm = getattr(container, "chat_model", None) or getattr(container, "llm", None)
        self.tiny_nli = getattr(container, "tiny_nli", None)

    async def score_text_pair(self, goal: str, reply: str, *, model_alias: str = "chat",
                              monitor_alias: str = "cove", context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        if not callable(self.llm):
            return dict(confidence01=0.5, faithfulness_risk01=0.5, ood_hat01=0.5, delta_gap01=0.5)

        try:
            plan = await asyncio.wait_for(self.llm(COVE_PLAN_PROMPT.format(goal=goal, reply=reply, k=self.cfg.questions)), timeout=self.cfg.timeout_s)
            qs = [q.strip("- ").strip() for q in plan.split("\n") if q.strip()]
            qs = qs[: self.cfg.questions]
            if not qs:
                return dict(confidence01=0.5, faithfulness_risk01=0.5, ood_hat01=0.5, delta_gap01=0.5)

            # answer independently
            answers = []
            for q in qs:
                ans = await asyncio.wait_for(self.llm(COVE_VERIFY_PROMPT.format(q=q)), timeout=self.cfg.timeout_s)
                answers.append(ans.strip())

            # crude pass: answer text occurs in reply (or not UNKNOWN)
            passes = []
            for q, a in zip(qs, answers):
                if a.upper().startswith("UNKNOWN"):
                    passes.append(0.0)
                elif a and a.lower() in reply.lower():
                    passes.append(1.0)
                elif callable(self.tiny_nli):
                    # use NLI: does reply entail the (q,a) atom?
                    try:
                        res = await self.tiny_nli([reply], [f"{q} -> {a}"])
                        entail = float(res[0].get("entail", 0.0))
                        passes.append(1.0 if entail >= 0.5 else 0.0)
                    except Exception:
                        passes.append(0.0)
                else:
                    passes.append(0.0)

            verified = sum(passes) / max(1, len(passes))  # 0..1
            return dict(
                confidence01=verified,
                faithfulness_risk01=1.0 - verified,
                ood_hat01=float(context.get("ood_hat01", 0.5) if context else 0.5),
                delta_gap01=1.0 - verified,
                verify_questions=len(qs),
            )
        except Exception:
            self.logger.exception("CoVeScorer failed; returning neutral metrics")
            return dict(confidence01=0.5, faithfulness_risk01=0.5, ood_hat01=0.5, delta_gap01=0.5)
