# stephanie/agents/icl_reasoning.py
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sqlalchemy.orm import joinedload

from stephanie.agents.base_agent import BaseAgent
from stephanie.orm.cartridge_domain import CartridgeDomainORM
from stephanie.orm.cartridge_triple import CartridgeTripleORM
from stephanie.orm.theorem import TheoremORM
from stephanie.tools.scorable_classifier import ScorableClassifier


class ICLReasoningAgent(BaseAgent):
    """
    In-Context Learning Reasoning Agent.
    Retrieves high-value triplets and theorems,
    formats them as structured facts, and generates reasoning with an LLM.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.top_k_triplets = cfg.get("top_k_triplets", 5)
        self.min_value_threshold = cfg.get("min_triplet_score", 0.6)
        self.use_embeddings = cfg.get("use_triplet_embeddings", False)
        self.score_weights = cfg.get(
            "score_weights",
            {
                "similarity": 0.5,
                "domain_relevance": 0.3,
                "specificity": 0.1,
                "confidence": 0.1,
            },
        )
        self.prompt_template = cfg.get("icl_prompt_template", "icl_reasoning_prompt.txt")

        self.domain_classifier = ScorableClassifier(
            memory,
            logger,
            cfg.get("domain_seed_config_path", "config/domain/seeds.yaml"),
        )

        # Embedding cache to avoid recomputation
        self._embedding_cache: Dict[str, np.ndarray] = {}

    async def run(self, context: dict) -> dict:
        goal = context.get("goal")
        if not goal:
            self.report({
                "event": "error",
                "step": "ICLReasoning",
                "error": "No goal provided in context"
            })
            return context

        goal_text = goal.get("goal_text", "")

        self.report({
            "event": "start",
            "step": "ICLReasoning",
            "goal": goal_text
        })

        try:
            # --- Step 1: Gather inputs
            top_triplets = self.retrieve_top_triplets(goal)
            top_theorems = self.retrieve_top_theorems(goal)

            self.report({
                "event": "inputs_collected",
                "step": "ICLReasoning",
                "triplet_count": len(top_triplets),
                "theorem_count": len(top_theorems),
                "triplet_examples": [
                    f"({t.subject}, {t.predicate}, {t.object})" for t in top_triplets[:3]
                ],
                "theorem_examples": [t.statement[:100] for t in top_theorems[:2]]
            })

            # --- Step 2: Format facts + theorems
            learned_facts = self.format_triplets_as_facts(top_triplets)
            learned_theorems = self.format_theorems(top_theorems)

            merged_context = {
                "goal": goal,
                "learned_facts": learned_facts,
                "learned_theorems": learned_theorems,
                **context,
            }

            # --- Step 3: Generate reasoning
            prompt = self.prompt_loader.load_prompt(self.cfg, merged_context)
            response = self.call_llm(prompt, context=context)

            self.report({
                "event": "reasoning_generated",
                "step": "ICLReasoning",
                "prompt_excerpt": prompt[:300],
                "response_excerpt": response[:300],
            })

            context[self.output_key] = response

            self.report({
                "event": "completed",
                "step": "ICLReasoning",
                "output_length": len(response),
            })

        except Exception as e:
            self.report({
                "event": "error",
                "step": "ICLReasoning",
                "error": str(e),
                "goal": goal_text or "Unknown"
            })
            context[self.output_key] = "Unable to generate reasoning due to an error."

        return context

    # ---------------- Retrieval ---------------- #

    def retrieve_top_theorems(self, goal: dict, top_k: int = 3) -> List[TheoremORM]:
        """Retrieve top theorems by domain + similarity"""
        session = self.memory.session
        goal_text = goal.get("goal_text", "")
        if not goal_text:
            return []

        goal_vec = self.memory.embedding.get_or_create(goal_text)

        all_theorems = session.query(TheoremORM).all()

        # Optional: filter by goal domains if available
        goal_domains = self.domain_classifier.classify(goal_text)
        goal_domain_names = [d[0] for d in goal_domains]

        if goal_domain_names:
            all_theorems = [
                thm for thm in all_theorems
                if any(domain in goal_domain_names for domain in getattr(thm, "domains", []))
            ] or all_theorems

        # Score + rank
        scored = [(thm, self.similarity(goal_vec, thm.statement)) for thm in all_theorems]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [thm for thm, _ in scored[:top_k]]

    def retrieve_top_triplets(self, goal: dict) -> List[CartridgeTripleORM]:
        """Retrieve and score candidate triplets"""
        session = self.memory.session
        goal_text = goal.get("goal_text", "")
        if not goal_text:
            return []

        goal_domains = self.domain_classifier.classify(goal_text)
        goal_domain_names = [d[0] for d in goal_domains]
        if not goal_domain_names:
            self.logger.log("NoDomainsFound", {"goal": goal_text}, level="warning")
            return []

        # Query triplets from relevant cartridges/domains
        query = (
            session.query(CartridgeTripleORM)
            .join(CartridgeTripleORM.cartridge)
            .join(CartridgeDomainORM, CartridgeDomainORM.cartridge_id == CartridgeTripleORM.cartridge_id)
            .filter(CartridgeDomainORM.domain.in_(goal_domain_names))
            .options(joinedload(CartridgeTripleORM.cartridge))
        )
        triplets = query.distinct().all()

        # Score + filter
        goal_vec = self.memory.embedding.get_or_create(goal_text) if self.use_embeddings else None
        scored = []
        for t in triplets:
            score = self.score_triplet(t, goal_text, goal_vec, goal_domain_names)
            if score >= self.min_value_threshold:
                scored.append((t, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[:self.top_k_triplets]]

    # ---------------- Scoring ---------------- #

    def score_triplet(
        self, triplet: CartridgeTripleORM, goal_text: str,
        goal_vec: Optional[np.ndarray], goal_domains: List[str]
    ) -> float:
        """Weighted multi-criteria score for triplets"""
        scores = {}

        # Embedding similarity
        if self.use_embeddings and goal_vec is not None:
            triplet_text = f"({triplet.subject}, {triplet.predicate}, {triplet.object})"
            scores["similarity"] = self.similarity(goal_vec, triplet_text)
        else:
            scores["similarity"] = 0.5

        # Domain match
        cartridge_domains = [d.domain for d in getattr(triplet.cartridge, "domains", [])]
        scores["domain_relevance"] = 1.0 if any(d in goal_domains for d in cartridge_domains) else 0.3

        # Specificity (proxy: string length)
        scores["specificity"] = min(1.0, (len(triplet.subject) + len(triplet.predicate) + len(triplet.object)) / 100)

        # Confidence
        scores["confidence"] = getattr(triplet, "confidence", 0.7) or 0.7

        # Weighted sum
        total = sum(scores[k] * self.score_weights.get(k, 0) for k in scores)
        return total

    # ---------------- Formatting ---------------- #

    def format_theorems(self, theorems: List[TheoremORM]) -> str:
        """Format theorems for LLM consumption"""
        if not theorems:
            return "No relevant theorems found."

        lines = []
        for i, thm in enumerate(theorems, 1):
            proof = (thm.proof or "No proof available").strip()
            proof_text = (proof[:200] + "...") if len(proof) > 200 else proof
            lines.append(f"{i}. {thm.statement}\n   Proof: {proof_text}")
        return "\n\n".join(lines)

    def format_triplets_as_facts(self, triplets: List[CartridgeTripleORM]) -> str:
        """Format triplets as numbered facts with provenance + score"""
        if not triplets:
            return "No relevant facts found."

        lines = []
        for i, t in enumerate(triplets, 1):
            subj, pred, obj = t.subject, t.predicate, t.object
            source = "Unknown"
            if getattr(t, "cartridge", None):
                source = f"{t.cartridge.source_type}:{t.cartridge.source_uri}"

            score = getattr(t, "confidence", None)
            fact_line = f"{i}. ({subj}, {pred}, {obj}) [Source: {source}]"
            if score:
                fact_line += f" [Conf: {round(score, 2)}]"
            lines.append(fact_line)

        return "\n".join(lines)

    # ---------------- Utils ---------------- #

    def similarity(self, goal_vec: np.ndarray, text: str) -> float:
        """Cosine similarity with caching"""
        if text in self._embedding_cache:
            text_vec = self._embedding_cache[text]
        else:
            text_vec = self.memory.embedding.get_or_create(text)
            self._embedding_cache[text] = text_vec

        # Use numpy dot (faster than sklearn)
        return float(
            np.dot(goal_vec, text_vec) /
            (np.linalg.norm(goal_vec) * np.linalg.norm(text_vec) + 1e-9)
        )
