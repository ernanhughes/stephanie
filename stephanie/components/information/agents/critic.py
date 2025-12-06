# stephanie/components/information/agents/critic.py
import asyncio
import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from stephanie.agents.base_agent import BaseAgent
from stephanie.services.prompt_service import LLMRole
from stephanie.types.idea import Idea

log = logging.getLogger(__name__)

class IdeaCriticHead(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(
            cfg=cfg, memory=memory, container=container, logger=logger
        )
        self.kg = self.container.get("knowledge_graph")
        self.embedding_store = self.memory.embedding
        self.prompt_service = self.container.get("prompt")
        self.feasibility_min_score = self.cfg.get("feasibility_min_score", 0.2)
        self.novelty_distance_threshold = self.cfg.get(
            "novelty_distance_threshold", 0.85
        )
        self.risk_keywords = self.cfg.get(
            "risk_keywords", ["weapons", "biohazard", "exploit", "unethical"]
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ideas: List[Idea] = context.get(self.input_key, [])
        scored_ideas: List[Idea] = []
        for idea in ideas:
            scored = await self.evaluate(idea)
            scored_ideas.append(scored)
        context[self.output_key] = scored_ideas
        return context

    async def evaluate(self, idea: Idea) -> Idea:
        """Async parallel scoring (4x speedup)"""
        scores = await asyncio.gather(
            self._score_novelty(idea),
            self._score_feasibility(idea),
            self._score_utility(idea),
            self._score_risk(idea),
        )
        (
            idea.novelty_score,
            idea.feasibility_score,
            idea.utility_score,
            idea.risk_score,
        ) = scores
        idea.r_final = self._combine_scores(idea)
        return idea

    async def _score_novelty(self, idea: Idea) -> float:
        """
        Novelty scoring via embedding distance.
        Higher distance from existing ideas => higher novelty.
        """
        try:
            idea_emb = await self.embedding_store.get_or_create(
                idea.hypothesis
            )

            # Collect existing idea embeddings (excluding current idea if present)
            existing_embs = await self.embedding_store.get_all_idea_embeddings()
            if not existing_embs:
                # No baseline, treat as moderately novel
                return 0.5

            # Compute pairwise cosine similarity
            similarities = cosine_similarity(
                [idea_emb], existing_embs
            )[0]  # 1 x N -> N
            max_similarity = float(np.max(similarities))
            novelty = 1.0 - max_similarity  # distance

            # Clip into [0,1]
            novelty = max(0.0, min(1.0, novelty))
            return novelty
        except Exception as e:
            log.warning(f"Novelty scoring failed: {e}")
            # Fallback: neutral novelty
            return 0.5

    async def _score_feasibility(self, idea: Idea) -> float:
        """
        Feasibility scoring using LLM critic.
        Returns a float in [0,1], conservative default if parsing fails.
        """
        prompt = f"""
You are an expert AI research critic.

Hypothesis:
{idea.hypothesis}

Proposed method:
{idea.method}

Score the overall feasibility of this idea on a scale from 0.0 to 1.0.
Consider:
- Technical plausibility with current AI tools
- Testability with available data/resources
- Clear success/failure criteria

Be conservative. Output ONLY a single float number between 0.0 and 1.0.
"""

        try:
            # UPDATED for new PromptService interface:
            # run_prompt(prompt_text, context, *, model=None, role=None, sys_preamble=None, params=None, timeout=None)
            score = await self.prompt_service.run_prompt(
                prompt_text=prompt,
                context=None,
                role=LLMRole.CRITIC_FEASIBILITY,
                params={"max_tokens": 10},
            )
            try:
                val = float(score.strip())
            except (TypeError, ValueError):
                log.warning(
                    f"Feasibility parse failed for idea {idea.id}: {score!r}"
                )
                return self.feasibility_min_score

            # Clamp into [0,1]
            val = max(0.0, min(1.0, val))
            return val
        except Exception as e:
            log.warning(f"Feasibility scoring failed: {e}")
            return self.feasibility_min_score

    async def _score_utility(self, idea: Idea) -> float:
        """
        Utility = how well this idea addresses known gaps.
        Uses KG gap severity + semantic similarity.
        """
        try:
            # If idea has explicit gap_ids, use them; otherwise search.
            gap_ids = list(idea.gap_ids or [])
            if not gap_ids:
                gap_ids = await self.kg.find_relevant_gaps(
                    query=idea.hypothesis, severity_threshold=0.5, limit=3
                )

            if not gap_ids:
                return 0.0

            idea_emb = await self.embedding_store.get_or_create(
                idea.hypothesis
            )

            max_utility = 0.0
            for gap_id in gap_ids:
                gap = await self.kg.get_gap(gap_id)
                if not gap or not getattr(gap, "description", None):
                    continue

                gap_emb = await self.embedding_store.get_or_create(
                    gap.description
                )
                sim = float(
                    cosine_similarity([idea_emb], [gap_emb])[0][0]
                )  # in [-1,1], but usually [0,1]
                sim = max(0.0, sim)

                severity = float(getattr(gap, "severity", 0.5))
                util = sim * severity
                max_utility = max(max_utility, util)

            # Clip into [0,1]
            max_utility = max(0.0, min(1.0, max_utility))
            return max_utility
        except Exception as e:
            log.warning(f"Utility scoring failed: {e}")
            return 0.0

    async def _score_risk(self, idea: Idea) -> float:
        """
        Crude risk scoring:
        - 0.0 = no obvious issues
        - 1.0 = clearly problematic keywords present
        """
        try:
            text = (
                f"{idea.hypothesis}\n{idea.method}\n{idea.impact_summary}"
            ).lower()
            for kw in self.risk_keywords:
                if kw.lower() in text:
                    return 1.0
            return 0.0
        except Exception as e:
            log.warning(f"Risk scoring failed: {e}")
            return 0.0

    def _combine_scores(self, idea: Idea) -> float:
        """
        Final reward function:
        R = 0.4 * novelty + 0.3 * feasibility + 0.3 * utility - 0.2 * risk
        """
        novelty_w = 0.4
        feasibility_w = 0.3
        utility_w = 0.3
        risk_w = -0.2

        r = (
            novelty_w * (idea.novelty_score or 0.0)
            + feasibility_w * (idea.feasibility_score or 0.0)
            + utility_w * (idea.utility_score or 0.0)
            + risk_w * (idea.risk_score or 0.0)
        )
        # Clip into [0,1]
        r = max(0.0, min(1.0, r))
        return r

    # -------------------- RL export helpers (async-safe) --------------------

    async def export_rl_training_data(
        self,
        ideas: List[Idea],
        min_final_score: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Export high-quality ideas as RL training samples for RL4LMs/PPO.
        Async-safe: pre-fetches all concepts in one go.
        """
        # Collect all concept IDs
        all_concept_ids = {cid for idea in ideas for cid in (idea.concept_ids or [])}
        concepts_map: Dict[str, Any] = {}

        # Pre-fetch concepts to avoid N+1 calls
        if self.kg and all_concept_ids:
            tasks = [self.kg.get_concept(cid) for cid in all_concept_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for cid, res in zip(all_concept_ids, results):
                if isinstance(res, Exception):
                    log.warning(f"RL export: failed to fetch concept {cid}")
                    concepts_map[cid] = {"id": cid, "name": cid, "summary": ""}
                else:
                    concepts_map[cid] = {
                        "id": res.id,
                        "name": getattr(res, "name", cid),
                        "summary": getattr(res, "summary", "") or "",
                    }

        samples: List[Dict[str, Any]] = []
        for idea in ideas:
            if idea.r_final < min_final_score:
                continue

            try:
                prompt = self._build_rl_prompt(idea, concepts_map)
                response = self._build_rl_response(idea)
                samples.append(
                    {
                        "prompt": prompt,
                        "response": response,
                        "reward": float(idea.r_final),
                        "meta": {
                            "idea_id": idea.id,
                            "concept_ids": idea.concept_ids,
                            "gap_ids": idea.gap_ids,
                            "scores": {
                                "novelty": idea.novelty_score,
                                "feasibility": idea.feasibility_score,
                                "utility": idea.utility_score,
                                "risk": idea.risk_score,
                            },
                        },
                    }
                )
            except Exception as e:
                log.error(
                    f"RL export failed for idea {getattr(idea, 'id', 'unknown')}: {e}"
                )

        return samples

    def _build_rl_prompt(
        self,
        idea: Idea,
        concepts_map: Dict[str, Dict[str, Any]],
    ) -> str:
        """Reconstruct prompt using pre-fetched concepts (no async calls)."""
        concept_lines: List[str] = []
        for cid in idea.concept_ids or []:
            c = concepts_map.get(cid, {"id": cid, "name": cid, "summary": ""})
            name = c.get("name", cid)
            summary = c.get("summary", "").strip()
            if summary:
                concept_lines.append(f"- {name}: {summary}")
            else:
                concept_lines.append(f"- {name}")

        concepts_block = "\n".join(concept_lines) or "No concepts provided"

        return (
            "You are an AI research ideator.\n\n"
            "Given the following concepts:\n\n"
            f"{concepts_block}\n\n"
            "Propose a concrete, novel research idea that combines them.\n"
            "Your answer should include a hypothesis and a method.\n"
        )

    def _build_rl_response(self, idea: Idea) -> str:
        """Flatten idea for RL trainers (standardized format)."""
        return f"Hypothesis: {idea.hypothesis}\n\nMethod: {idea.method}\n"
