# stephanie/components/ssp/impl/proposers/searching_proposer.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import asyncio
import time
import re
import logging

from stephanie.components.ssp.core.protocols import EpisodeContext
from stephanie.components.ssp.impl.solvers.solution_search import SolutionSearch
from stephanie.components.ssp.utils.parser import parse_proposer_lines
from stephanie.prompts.prompt_loader import PromptLoader

_logger = logging.getLogger(__name__)

PROPOSER_PROMPT_TMPL = """You are building an SSP dataset. Given the canonical mechanism (SEED_ANSWER), write ONE precise, verifiable question whose correct answer is that mechanism.

SEED_ANSWER:
{seed_answer}

CONSTRAINTS:
- Ask for the mechanism directly (no trivia, no multi-part).
- Be specific and test factual understanding.
- No explanations or extra lines.

OUTPUT FORMAT — WRITE EXACTLY FOUR LINES, IN THIS ORDER, NO CODE FENCES:
rationale: <1–2 sentences on why this question targets the mechanism>
difficulty: <integer 0-100>
verifiability: <integer 0-100>
question: <the single best question>
"""

class SearchingProposer:
    def __init__(self, cfg, memory, container, logger, solution_search):
        if not solution_search:
            raise ValueError("SolutionSearch is required for evidence gathering")

        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger
        self.solution_search: SolutionSearch = solution_search

        # Services
        self.vpm_control = container.get("vpm_control")
        self.prompt_loader = PromptLoader(memory=self.memory, logger=self.logger)
        self.prompt_service = container.get("prompt")

        # Config (with safe defaults)
        p = (self.cfg.get("proposer") or {})
        self.rewrites: int = int(p.get("rewrites", 3))
        self.max_snippets: int = int(p.get("max_snippets", 6))
        self.min_question_len: int = int(p.get("min_question_len", 12))
        self.forbid_answer_leak: bool = bool(p.get("forbid_answer_leak", True))
        self.retries: int = int(p.get("retries", 2))
        self._backoff: float = float(p.get("backoff_sec", 0.5))

    async def propose(
        self,
        seed_answer: str,
        context: Optional[EpisodeContext] = None
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Generate a question from a seed answer WITH EVIDENCE GATHERING.
        Returns: (question, evidence_snippets, meta)
        """
        t0 = time.time()
        ctx = dict(context or {})

        # 1) Generate lightweight rewrites
        rewrites = self._generate_query_rewrites(seed_answer)

        # 2) Gather evidence using whichever API your SolutionSearch exposes
        all_evidence: List[str] = []
        for rewrite in rewrites:
            snippets = await self.solution_search.find_snippets(rewrite, top_k=max(1, self.max_snippets - len(all_evidence)))

            if snippets:
                for s in snippets:
                    if s and s not in all_evidence:
                        all_evidence.append(s)
            if len(all_evidence) >= self.max_snippets:
                break

        # 3) Craft question via prompt
        question, meta = await self._craft_question(seed_answer, all_evidence, ctx)

        # 4) Basic validation + safe fallback
        if not question or len(question) < self.min_question_len:
            # Minimal safe fallback so downstream never breaks
            fallback_q = f"What is {seed_answer}?"
            meta.update({
                "rationale": meta.get("rationale") or "Auto-fallback: generated question too short/empty",
                "difficulty": int(meta.get("difficulty") or 0),
                "verifiability": int(meta.get("verifiability") or 0),
                "raw_ok": False,
                "fallback_used": True,
            })
            question = fallback_q

        # Optional: forbid exact answer leak in the surface form
        if self.forbid_answer_leak and seed_answer and question:
            if seed_answer.lower() in question.lower():
                # Keep it pointed but avoid literal echo
                question = self._anonymize_mechanism(question, seed_answer)

        # 5) Emit a tiny VPM frame for visibility
        dt = round(time.time() - t0, 3)
        if self.vpm_control:
            try:
                self.vpm_control.decide(
                    unit=f"proposer:{(hash(seed_answer) & 0xffff):04x}",
                    kind="text",
                    dims={
                        "evidence_quality": min(1.0, len(all_evidence) / max(1, self.max_snippets)),
                        "question_length": min(1.0, len(question) / 100.0),
                    },
                    step_idx=ctx.get("step_idx", 0),
                    meta={
                        "seed_answer": seed_answer,
                        "evidence_count": len(all_evidence),
                        "latency_s": dt,
                    },
                )
            except Exception as e:
                _logger.info("VPM decide failed (non-fatal): %s", e)

        return question, all_evidence, meta

    def _generate_query_rewrites(self, seed_answer: str) -> List[str]:
        rewrites = [
            f"What is {seed_answer}?",
            f"Explain {seed_answer} in detail",
            f"How does {seed_answer} work?",
        ]
        additional = (self.cfg.get("proposer") or {}).get("additional_rewrites", [])
        for pattern in additional:
            try:
                rewrites.append(pattern.format(seed_answer=seed_answer))
            except Exception:
                pass
        return rewrites[: max(1, self.rewrites)]

    async def _craft_question(
        self,
        seed_answer: str,
        evidence: List[str],
        context: Optional[EpisodeContext] = None
    ) -> Tuple[str, Dict[str, Any]]:
        merged_context = {
            "seed_answer": seed_answer,
            "evidence": "\n".join(evidence or []),
            **(context or {}),
        }
        prompt = self.prompt_loader.from_text(PROPOSER_PROMPT_TMPL, merged_context)

        response = ""
        attempt = 0
        while attempt <= self.retries:
            try:
                response = await self.prompt_service.run_prompt(prompt_text=prompt, context=merged_context)
                break
            except Exception as e:
                attempt += 1
                _logger.warning("Proposer prompt failed (attempt %d/%d): %s", attempt, self.retries, e)
                if attempt > self.retries:
                    parsed = {"rationale": "prompt failure", "difficulty": 0, "verifiability": 0, "ok": False}
                    return "", parsed
                await asyncio.sleep(self._backoff * attempt)

        parsed = parse_proposer_lines(response or "")
        question = self._normalize_question(parsed.get("question", ""))

        meta = {
            "rationale": parsed.get("rationale", ""),
            "difficulty": int(parsed.get("difficulty", 0) or 0),
            "verifiability": int(parsed.get("verifiability", 0) or 0),
            "raw_ok": bool(parsed.get("ok", False)),
        }
        return question, meta

    def _normalize_question(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\?+", "?", text).strip()
        if text and not text.endswith("?"):
            text += "?"
        return text

    def _anonymize_mechanism(self, q: str, seed_answer: str) -> str:
        # Very light-touch anonymization to avoid literal leakage
        pattern = re.compile(re.escape(seed_answer), re.IGNORECASE)
        q2 = pattern.sub("this mechanism", q)
        # Keep it a question
        return self._normalize_question(q2 or q)

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "supports_search_during_proposal": True,
            "max_evidence_snippets": self.max_snippets,
            "min_question_length": self.min_question_len,
        }
