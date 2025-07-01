# stephanie/agents/evolution.py
import itertools

from sklearn.metrics.pairwise import cosine_similarity

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import EVOLVED, GOAL, HYPOTHESES, RANKING
from stephanie.parsers import extract_hypotheses


class IdeaEvolutionAgent(BaseAgent):
    """
    The Evolution Agent refines hypotheses iteratively using several strategies:

    - Grafting similar hypotheses into unified statements
    - Feasibility improvement through LLM reasoning
    - Out-of-the-box hypothesis generation
    - Inspiration from top-ranked ideas
    - Simplification and clarity enhancement

    These improvements are based on the paper_score:
    "NOVELSEEK: When Agent Becomes the Scientist – Building Closed-Loop System from Hypothesis to Verification"
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.use_grafting = cfg.get("use_grafting", False)
        self.max_variants_per_idea = cfg.get("max_variants", 3)
        self.max_evolution_rounds = cfg.get("evolution_rounds", 4)
        self.selection_top_k = cfg.get("select_top_k", 5)
        self.preferences = cfg.get("preferences", ["novelty", "feasibility"])

    async def run(self, context: dict) -> dict:
        """
        Evolve top-ranked hypotheses across multiple rounds.
        """
        # Get input hypotheses
        ranked_hypotheses = context.get(RANKING, [])
        fallback_hypotheses = context.get(HYPOTHESES, [])
        preferences = context.get("preferences", self.preferences)
        current_round = context.get("evolution_round", 0)

        if not ranked_hypotheses and not fallback_hypotheses:
            self.logger.log("NoHypothesesToEvolve", {"reason": "no_ranked_or_unranked_input"})
            context[EVOLVED] = []
            return context

        # Decide which hypotheses to evolve
        top_texts = [h.get("text") for h, _ in ranked_hypotheses[:3]] if ranked_hypotheses else fallback_hypotheses

        # Run evolution strategies
        all_variants = await self._mutate_all(top_texts, context, preferences)

        # Optionally use grafting
        if self.use_grafting:
            all_variants += await self.graft_similar(context)

        # Score and select top K
        scored_variants = self._score_variants(all_variants, context)
        top_variants = scored_variants[:self.selection_top_k]

        # Save to DB
        self._save_evolved(top_variants, context)

        # Update context
        context["evolved"] = top_variants
        context["evolution_round"] = current_round + 1
        context["evolved_count"] = len(top_variants)

        self.logger.log(
            "EvolutionCompleted",
            {
                "evolved_count": len(top_variants),
                "preferences": preferences,
                "round": current_round + 1
            }
        )

        return context

    async def _mutate_all(self, hypotheses: list, context: dict, preferences: list) -> list:
        """Generate multiple variants for each hypothesis"""
        all_mutants = []

        for h in hypotheses:
            prompt_context = {
                "hypothesis": h,
                "literature_summary": context.get("knowledge_base_summaries", []),
                "critique": context.get("scores", {}),
                "focus_area": context.get(GOAL, {}).get("focus_area"),
                "preferences": ", ".join(preferences)
            }

            prompt = self.prompt_loader.load_prompt(self.cfg, prompt_context)
            raw_output = self.call_llm(prompt, context)

            mutants = extract_hypotheses(raw_output)
            self.logger.log("HypothesisMutated", {
                "original": h[:60],
                "mutations": mutants[:2]
            })

            all_mutants.extend(mutants)

        return all_mutants

    async def graft_similar(self, context: dict, threshold: float = 0.85) -> list:
        """
        Graft pairs of highly similar hypotheses into unified versions.
        """
        hypotheses = self.get_hypotheses(context)
        embeddings = [await self.memory.embedding.get_or_create(h.get("text")) for h in hypotheses]
        used = set()
        grafted = []

        for (i, h1), (j, h2) in itertools.combinations(enumerate(hypotheses), 2):
            if i in used or j in used:
                continue

            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                self.logger.log("GraftingPair", {
                    "similarity": sim,
                    "h1": h1[:60] + "...",
                    "h2": h2[:60] + "..."
                })
                prompt = (
                    f"Combine the following hypotheses into a clearer, more innovative statement:\n\n"
                    f'A: {h1.get("text")}\nB: {h2.get("text")}'
                )
                try:
                    response = self.call_llm(prompt, context)
                    combined = extract_hypotheses(response)
                    grafted.extend(combined)
                    used.update([i, j])
                except Exception as e:
                    self.logger.log("GraftingFailed", {"error": str(e)})
                    continue

        # Add ungrafted hypotheses back
        hypotheses = context.get(HYPOTHESES, [])
        for i, h in enumerate(hypotheses):
            if i not in used:
                grafted.append(h)

        return grafted

    def _score_variants(self, variants: list, context: dict) -> list:
        """
        Score variants using ScorerAgent logic and sort by total score
        """
        scorer = self.memory.scores
        scored = []

        for v in variants:
            score_data = scorer.score(v, context)
            score_data["text"] = v
            scored.append(score_data)

        # Sort by composite score
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def _save_evolved(self, variants: list, context: dict):
        """
        Save evolved hypotheses to database with lineage info using save_hypothesis().
        """
        results = []
        for v in variants:
            hyp = self.save_hypothesis(
                {
                    "text": v["text"],
                    "parent_id": context.get("current_hypothesis", {}).get("id"),
                    "evolution_level": context.get("evolution_round", 0),
                },
                context=context
            )
            results.append(hyp)
        return results

