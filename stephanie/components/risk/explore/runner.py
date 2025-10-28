# stephanie/components/risk/explore/runner.py I
from __future__ import annotations

from typing import Any, Dict, List

from stephanie.components.gap.risk.orchestrator import GapRiskOrchestrator
from stephanie.components.gap.risk.explore.triggers import apply_triggers


class ExplorationRunner:
    def __init__(self, container: Any, profile: str = "chat.standard"):
        self.container = container
        self.orch = GapRiskOrchestrator(container, policy_profile=profile)

        # sampler: prefer container.chat_sampler(goal, temperature, top_p, max_tokens, n)
        self.sampler = getattr(container, "chat_sampler", None)
        self.chat = getattr(container, "chat_model", None) or getattr(container, "llm", None)

    async def _sample(self, prompt: str, *, temperature=1.1, top_p=0.98, max_tokens=512) -> str:
        if callable(self.sampler):
            out = await self.sampler(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=1)
            return out[0] if isinstance(out, list) else out
        # simple fallback
        return await self.chat(prompt) if callable(self.chat) else ""

    async def explore_goal(self, goal: str, *, k_triggers=4, divergence=0.8) -> List[Dict[str, Any]]:
        # map divergence → sampling knobs
        temperature = 0.7 + 0.6 * divergence
        top_p = 0.9 + 0.08 * divergence

        prompts = apply_triggers(goal, k=k_triggers)
        results: List[Dict[str, Any]] = []
        for p in prompts:
            reply = await self._sample(p, temperature=temperature, top_p=top_p)
            if not reply.strip():
                continue
            rec = await self.orch.evaluate(goal=p, reply=reply, model_alias="chat-hrm", monitor_alias="tiny")
            # compute a crude novelty score vs original goal (Jaccard as placeholder)
            novelty = _novelty(reply, goal)
            results.append({**rec, "novelty": novelty, "seed_goal": goal, "prompt_used": p})
        return results

def _novelty(text: str, ref: str) -> float:
    import re
    A = set(re.findall(r"[A-Za-z0-9]+", text.lower()))
    B = set(re.findall(r"[A-Za-z0-9]+", ref.lower()))
    if not A: return 0.0
    inter = len(A & B); union = len(A | B)
    jaccard = inter / union if union else 0.0
    return 1.0 - jaccard  # higher = more novel
