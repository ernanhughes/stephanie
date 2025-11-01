"""
Searching Proposer Implementation

This implementation adds the critical "search while proposing" capability
described in the SSP paper. The proposer actively gathers evidence while
crafting the question, which is then used in the RAG-gated verification.
"""

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple
import time

from stephanie.components.ssp.core.roles.proposer import Proposer
from stephanie.components.ssp.core.protocols import EpisodeContext
from stephanie.components.ssp.utils.parser import parse_proposer_lines
from stephanie.prompts.prompt_loader import PromptLoader
import logging


SEARCHING_PROPOSER_PROMPT = """
SYSTEM:
You are building an SSP dataset. Given this result the (SEED_ANSWER), 
write ONE precise, verifiable question whose correct answer is that result.

SEED_ANSWER:
{{ answer }}

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

_logger = logging.getLogger(__name__)

class SearchingProposer(Proposer):
    """
    Proposer implementation that gathers evidence WHILE crafting questions.
    
    This implements the "searching proposer" from the SSP paper:
    1. Takes a seed answer
    2. Generates query rewrites
    3. Performs retrieval for each rewrite
    4. Crafts a question using the gathered evidence
    5. Returns both question and evidence
    """
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        memory: Any,
        container: Any,
        logger: Any,
        solution_search
    ):
        """
        Initialize the SearchingProposer.
        
        Args:
            cfg: Configuration dictionary
            memory: Memory tool
            container: Dependency container
            logger: Logger instance
            solution_search: Optional pre-configured SolutionSearch instance
        """
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self._prompt_name = cfg.get("proposer", {}).get("prompt_name", "proposer")
        self._prompt_text = cfg.get("proposer", {}).get("prompt_text")
        self._backoff = cfg.get("proposer", {}).get("backoff_sec", 0.5)
        self.retries = cfg.get("proposer", {}).get("retries", 2)
        
        # Get solution search for evidence gathering
        self.solution_search = solution_search
        if not self.solution_search:
            raise ValueError("SolutionSearch is required for evidence gathering")
            
        # Get VPM control service for decision making
        self.vpm_control = container.get("vpm_control")
        self.prompt_loader = PromptLoader(memory=self.memory, logger=self.logger)
        self.prompt_service = container.get("prompt")

        # Configuration parameters
        self.rewrites = cfg.get("proposer", {}).get("rewrites", 3)
        self.max_snippets = cfg.get("proposer", {}).get("max_snippets", 6)
        self.min_question_len = cfg.get("proposer", {}).get("min_question_len", 20)
        self.forbid_answer_leak = cfg.get("proposer", {}).get("forbid_answer_leak", True)
    
    async def propose(
        self,
        seed_answer: str,
        context: Optional[EpisodeContext] = None
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Generate a question from a seed answer WITH EVIDENCE GATHERING.
        
        Args:
            seed_answer: Ground truth answer to build a question around
            context: Additional context for the proposal
            
        Returns:
            Tuple of (question, evidence_snippets, metadata)
        """
        t0 = time.time()
        
        # 1. Generate query rewrites for evidence gathering
        rewrites = self._generate_query_rewrites(seed_answer)
        
        # 2. Gather evidence using each rewrite
        all_evidence = []
        for rewrite in rewrites:
            # Search for evidence related to this rewrite
            snippets = await self.solution_search.search(
                rewrite, 
                seed_answer=seed_answer,
                context=context
            )
            all_evidence.extend(snippets)
            
            # Deduplicate evidence
            all_evidence = list(dict.fromkeys(all_evidence))
            
            # Limit total evidence
            if len(all_evidence) >= self.max_snippets:
                break
        
        # 3. Craft question using the gathered evidence
        question, meta = await self._craft_question(
            seed_answer, 
            all_evidence,
            context
        )
        
        # 4. Apply basic validation
        if not question or len(question) < self.min_question_len:
            return "", all_evidence, {
                "rationale": "Question too short after validation",
                "difficulty": 0,
                "verifiability": 0,
                "raw_ok": False
            }
        
        # 5. Record metrics for VPM
        dt = round(time.time() - t0, 3)
        if self.vpm_control:
            self.vpm_control.decide(
                unit=f"proposer:{hash(seed_answer) & 0xffff:04x}",
                kind="text",
                dims={
                    "evidence_quality": len(all_evidence) / self.max_snippets,
                    "question_length": min(1.0, len(question) / 100),
                },
                step_idx=context.get("step_idx", 0) if context else 0,
                meta={
                    "seed_answer": seed_answer,
                    "evidence_count": len(all_evidence),
                    "question_length": len(question),
                }
            )
        
        return question, all_evidence, meta
    
    def _generate_query_rewrites(self, seed_answer: str) -> List[str]:
        """
        Generate query rewrites for evidence gathering.
        
        Args:
            seed_answer: Ground truth answer
            
        Returns:
            List of query rewrites
        """
        # Base rewrites - could be enhanced with LLM-based rewrites
        rewrites = [
            f"What is {seed_answer}?",
            f"Explain {seed_answer} in detail",
            f"How does {seed_answer} work?"
        ]
        
        # Add more rewrites based on configuration
        additional = self.cfg.get("proposer", {}).get("additional_rewrites", [])
        for pattern in additional:
            rewrites.append(pattern.format(seed_answer=seed_answer))
            
        return rewrites[:self.rewrites]  # Limit to configured number
    
    async def _craft_question(
        self,
        seed_answer: str,
        evidence: List[str],
        context: Optional[EpisodeContext] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Craft a question using the gathered evidence.
        
        Args:
            seed_answer: Ground truth answer
            evidence: Evidence snippets gathered
            context: Additional context
            
        Returns:
            Tuple of (question, metadata)
        """
        # Prepare context for prompt
        merged_context = {
            "seed_answer": seed_answer,
            "evidence": "\n".join(evidence),
            **(context or {})
        }
        
        prompt = self.prompt_loader.from_text(SEARCHING_PROPOSER_PROMPT, merged_context)

        # # Load prompt
        # if self._prompt_name:
        #     prompt = self.prompt_loader.from_file(f"{self._prompt_name}.txt", self.cfg, merged_context)
        #     psrc = f"file:{self._prompt_name}.txt"
        # else:
        #     prompt = self.prompt_loader.from_text(self._prompt_text or PROPOSER_PROMPT, merged_context)
        #     psrc = "inline"
        
        _logger.debug("Proposer: loaded prompt (%s)", prompt)
        
        # Call model with retry logic
        response = ""
        attempt = 0
        while attempt <= self.retries:
            try:
                response = await self.prompt_service.run_prompt(prompt, merged_context)
                break
            except Exception as e:
                attempt += 1
                _logger.warning("Proposer prompt call failed (attempt %d): %s", attempt, e)
                if attempt > self.retries:
                    return "", {
                        "rationale": "Failed to generate question after retries",
                        "difficulty": 0,
                        "verifiability": 0,
                        "raw_ok": False
                    }
                await asyncio.sleep(self._backoff * attempt)
        
        # Parse response
        _logger.debug("Proposer LLM response (first 160 chars): %s", 
                         (response or " ").replace("\n", " ")[:160])
        parsed = parse_proposer_lines(response)
        
        # Normalize and validate question
        question = self._normalize_question(parsed.get("question", ""))
        meta = {
            "rationale": parsed.get("rationale", ""),
            "difficulty": int(parsed.get("difficulty", 0) or 0),
            "verifiability": int(parsed.get("verifiability", 0) or 0),
            "raw_ok": bool(parsed.get("ok", False)),
        }
        
        return question, meta
    
    def _normalize_question(self, text: str) -> str:
        """Clean and normalize the generated question."""
        if not text:
            return ""
        
        # Remove trailing question marks if duplicated
        text = re.sub(r"\?+", "?", text)
        text = text.strip()
        
        # Ensure it ends with a question mark
        if text and not text.endswith("?"):
            text += "?"
            
        return text
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "supports_search_during_proposal": True,
            "max_evidence_snippets": self.max_snippets,
            "min_question_length": self.min_question_len
        }