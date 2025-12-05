# stephanie/components/information/agents/idea.py
from __future__ import annotations

import asyncio
import inspect
import json
import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.services.prompt_service import LLMRole
from stephanie.types.idea import Idea
from stephanie.services.knowledge_graph_service import KnowledgeGraphService
import logging

log = logging.getLogger(__name__)

@dataclass
class IdeaGenerationSettings:
    """Tunable hyperparameters for ideation."""
    top_k_pairs: int = 10          # how many concept pairs to sample
    ideas_per_pair: int = 3        # how many ideas to request per pair
    min_creative_dist: float = 0.0 # reserved for future embedding-based filtering
    max_creative_dist: float = 1.0
    temperature: float = 0.7
    max_tokens: int = 600


class IdeaGenerationHead(BaseAgent):
    """
    Head that proposes research ideas by connecting pairs of frontier concepts.

    Responsibilities
    ----------------
    * Sample "frontier" concepts from the KnowledgeGraph (struggle / mid-band zone).
    * Form concept pairs with high creative potential.
    * Call the IDEATOR LLM head to generate structured idea candidates.
    * Parse raw LLM output into `Idea` objects for downstream critics & storage.

    This class is intentionally light on policy logic — the heavy lifting
    (novelty, feasibility, utility, risk) is done in `IdeaCriticHead`.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        memory,
        container,
        logger,
    ) -> None:
        super().__init__(cfg=cfg, memory=memory, container=container, logger=logger)

        self.kg: KnowledgeGraphService = self.container.get("knowledge_graph")
        self.emb = self.memory.embedding
        self.prompt = self.container.get("prompt")

        # Hydrate settings from cfg, but be robust to extra keys.
        self.settings = IdeaGenerationSettings()
        for key in (
            "top_k_pairs",
            "ideas_per_pair",
            "min_creative_dist",
            "max_creative_dist",
            "temperature",
            "max_tokens",
        ):
            if key in cfg:
                setattr(self.settings, key, cfg[key])

        # Optional: which generator model this head is notionally using
        self.generator_model = cfg.get("generator_model", "ideator:default")


    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entrypoint: generate frontier ideas.

        Populates context["ideas_generated"] with a list of `Idea` objects.
        """
        ideas: List[Idea] = await self.generate_frontier_ideas()
        context["ideas_generated"] = ideas
        return context

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate_frontier_ideas(
        self,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[Idea]:
        """
        Main entrypoint: sample frontier concept pairs and generate ideas.

        Returns a flat list of `Idea` objects (unscored).
        """
        constraints = constraints or {}
        pairs = await self._sample_frontier_pairs()

        if not pairs:
            log.warning("IdeaGenerationHead: no frontier pairs available")
            return []

        tasks = [
            self.propose_from_pair(a_id, b_id, constraints=constraints)
            for (a_id, b_id) in pairs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        ideas: List[Idea] = []
        for res in results:
            if isinstance(res, Exception):
                log.error(
                    "IdeaGenerationHead: error generating from pair",
                    exc_info=res,
                )
                continue
            ideas.extend(res)

        return ideas

    async def propose_from_pair(
        self,
        concept_a_id: str,
        concept_b_id: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[Idea]:
        """
        Generate one or more ideas that bridge two concepts.

        Returns a list of `Idea` objects; may be empty if parsing fails.
        """
        constraints = constraints or {}

        # Support both sync and async KG implementations.
        a = await self._maybe_await(self.kg.get_concept(concept_a_id))
        b = await self._maybe_await(self.kg.get_concept(concept_b_id))

        prompt = self._build_idea_prompt(a, b, constraints)

        # Ask for multiple variants in parallel for diversity.
        tasks = [
            self.prompt.run_prompt(prompt=prompt, role=LLMRole.IDEATOR)
            for _ in range(self.settings.ideas_per_pair)
        ]
        raw_texts = await asyncio.gather(*tasks, return_exceptions=True)

        ideas: List[Idea] = []
        for raw in raw_texts:
            if isinstance(raw, Exception):
                log.error(
                    "IdeaGenerationHead: LLM error for pair (%s, %s)",
                    concept_a_id,
                    concept_b_id,
                    exc_info=raw,
                )
                continue

            try:
                idea = self._parse_idea(
                    raw,
                    concept_ids=[concept_a_id, concept_b_id],
                    prompt=prompt,
                )
                ideas.append(idea)
            except Exception as e:
                log.warning(
                    "IdeaGenerationHead: failed to parse idea text: %s", str(e)
                )

        return ideas

    # ------------------------------------------------------------------
    # Frontier sampling
    # ------------------------------------------------------------------
    async def _sample_frontier_pairs(self) -> List[Tuple[str, str]]:
        """
        Sample concept pairs from the "frontier" band.

        This uses KnowledgeGraphService.query_frontier_concepts if available.
        If not present, it will fall back to an empty list (caller will log).
        """
        concepts = []

        # Prefer dedicated frontier API if available.
        query_fn = getattr(self.kg, "query_frontier_concepts", None)
        if query_fn is not None:
            try:
                concepts = await self._maybe_await(
                    query_fn(limit=self.settings.top_k_pairs * 2)
                )
            except Exception as e:
                log.error(
                    "IdeaGenerationHead: query_frontier_concepts failed: %s", str(e)
                )
                concepts = []

        if not concepts:
            return []

        # Extract IDs robustly (supports dicts or ORM/pydantic objects).
        ids: List[str] = []
        for c in concepts:
            if isinstance(c, dict):
                cid = c.get("id")
            else:
                cid = getattr(c, "id", None)
            if cid:
                ids.append(cid)

        if len(ids) < 2:
            return []

        # Simple pair sampling for now; can be replaced with embedding-based
        # "productive distance" once embedding plumbing is finalized.
        pairs: List[Tuple[str, str]] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((ids[i], ids[j]))

        random.shuffle(pairs)
        return pairs[: self.settings.top_k_pairs]

    # ------------------------------------------------------------------
    # Prompt + parsing
    # ------------------------------------------------------------------
    def _build_idea_prompt(
        self,
        concept_a: Any,
        concept_b: Any,
        constraints: Dict[str, Any],
    ) -> str:
        """
        Build a structured prompt asking the IDEATOR head to propose
        a novel-but-feasible research idea that bridges two concepts.

        We ask for strict JSON so parsing is reliable.
        """
        def _name(c: Any) -> str:
            if isinstance(c, dict):
                return c.get("name") or c.get("id") or "Unknown"
            return getattr(c, "name", None) or getattr(c, "id", "Unknown")

        def _summary(c: Any) -> str:
            if isinstance(c, dict):
                return (c.get("summary") or "").strip()
            return getattr(c, "summary", "") or ""

        goal = constraints.get(
            "goal",
            "Propose a single, concrete, novel-but-feasible research idea in AI/ML.",
        )
        novelty_target = constraints.get("novelty", "high")
        feasibility_target = constraints.get("feasibility", "medium")

        return f"""
You are an AI research ideator.

Your task: {goal}

You are given two concepts:

[Concept A]
Name: {_name(concept_a)}
Summary: {_summary(concept_a)}

[Concept B]
Name: {_name(concept_b)}
Summary: {_summary(concept_b)}

Requirements:
- The idea MUST meaningfully combine BOTH Concept A and Concept B.
- Target novelty: {novelty_target} (avoid trivial restatements of known work).
- Target feasibility: {feasibility_target} (must be testable with current or near-term tools).
- Be specific: describe a hypothesis and an experiment / method that could falsify it.

Respond ONLY with valid JSON matching this schema:

{{
  "title": "Short, punchy title for the idea",
  "hypothesis": "One-paragraph statement of the core claim being tested.",
  "method": "Concrete description of how to test / validate this hypothesis.",
  "impact": "Why this matters if it works; who benefits; what it unlocks."
}}
""".strip()

    def _parse_idea(
        self,
        raw_text: str,
        concept_ids: List[str],
        prompt: str,
    ) -> Idea:
        """
        Parse raw LLM output into an Idea.

        We expect JSON per the schema in _build_idea_prompt, but are
        defensive against wrappers or minor deviations.
        """
        data: Dict[str, Any] = {}

        # Try to locate a JSON object in the response.
        try:
            start = raw_text.index("{")
            end = raw_text.rindex("}") + 1
            json_str = raw_text[start:end]
            data = json.loads(json_str)
        except Exception:
            # Fallback: treat whole text as hypothesis.
            title = raw_text.strip().splitlines()[0][:120] or "Untitled idea"
            data = {
                "title": title,
                "hypothesis": raw_text.strip(),
                "method": "",
                "impact": "",
            }

        title = (data.get("title") or "").strip() or "Untitled idea"
        hypothesis = (data.get("hypothesis") or "").strip()
        method = (data.get("method") or "").strip()
        impact = (data.get("impact") or "").strip()

        prompt_hash = self._hash_prompt(prompt)

        idea = Idea(
            title=title,
            hypothesis=hypothesis,
            method=method,
            impact_summary=impact,
            concept_ids=list(concept_ids),
            gap_ids=[],
            generator_model=self.generator_model,
            prompt_hash=prompt_hash,
        )
        return idea

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    async def _maybe_await(self, value: Any) -> Any:
        """Helper to transparently handle sync vs async service methods."""
        if inspect.isawaitable(value):
            return await value
        return value

    def _hash_prompt(self, prompt: str) -> str:
        """Short, stable hash for provenance tracking."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
