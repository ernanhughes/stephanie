# co_ai/agents/evolution.py
import itertools
import re

import numpy as np

from co_ai.agents.base import BaseAgent
from co_ai.tools.embedding_tool import get_embedding
from co_ai.constants import RANKING, HYPOTHESES, EVOLVED, GOAL
from co_ai.models import Hypothesis


class EvolutionAgent(BaseAgent):
    """
    The Evolution Agent refines hypotheses iteratively using several strategies:
    
    - Grafting similar hypotheses into unified statements
    - Feasibility improvement through LLM reasoning
    - Out-of-the-box hypothesis generation
    - Inspiration from top-ranked ideas
    - Simplification and clarity enhancement
    
    These improvements are based on the paper:
    "The Evolution agent continuously refines and improves existing hypotheses..."
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.use_grafting = cfg.get("use_grafting", False)
        self.preferences = cfg.get("preferences", ["novelty", "feasibility"])

    async def run(self, context: dict) -> dict:
        """
        Evolve top-ranked hypotheses individually.
       
        Args:
            context: Dictionary with keys:
                - ranked: list of (hypotheses, score) tuples
                - hypotheses: list of unranked hypotheses (fallback)
                - preferences: override criteria for refinement
        """
        ranked = context.get(RANKING, [])
        fallback_hypotheses = context.get(HYPOTHESES, [])
        preferences = context.get("preferences", self.preferences)

        # Decide which hypotheses to evolve
        if ranked:
            top_texts = [hyp for hyp, _ in ranked[:3]]
        elif fallback_hypotheses:
            top_texts = fallback_hypotheses
        else:
            self.logger.log(
                "NoHypothesesToEvolve", {"reason": "no_ranked_or_unranked_input"}
            )
            context[EVOLVED] = []
            return context

        evolved = []
        for h in top_texts:
            try:
                prompt = self.prompt_loader.load_prompt(
                    {**self.cfg, **{HYPOTHESES: h}}, context
                )
                raw_output = self.call_llm(prompt, context)
                refined_list = self.extract_hypothesis(raw_output)
                self.logger.log("EvolvedParsedHypotheses", {
                    "raw_response_snippet": raw_output[:300],
                    "parsed": refined_list
                })

                if refined_list:
                    for r in refined_list:
                        goal = self.extract_goal_text(context.get(GOAL))
                        evolved_goal = f"Evolved from top-ranked {goal}"
                        hyp = Hypothesis(goal=evolved_goal, text=h)
                        self.memory.hypotheses.store(hyp)
                        evolved.append(r)
                else:
                    self.logger.log(
                        "EvolutionFailed",
                        {"original": h[:100], "response_snippet": raw_output[:200]},
                    )

            except Exception as e:
                print(f"❌ Exception: {type(e).__name__}: {e}")
                self.logger.log(
                    "EvolutionError", {"error": str(e), "hypotheses": h[:100]}
                )

        context["evolved"] = evolved
        self.logger.log(
            "EvolutionCompleted",
            {"evolved_count": len(evolved), "preferences": preferences},
        )

        return context

    async def graft_similar(self, context: dict, threshold: float = 0.90) -> list[str]:
        """
        Graft pairs of highly similar hypotheses into unified versions.
        """
        hypotheses =self.get_hypotheses(context)
        embeddings = [get_embedding(h, self.cfg) for h in hypotheses]
        used = set()
        grafted = []

        for (i, h1), (j, h2) in itertools.combinations(enumerate(hypotheses), 2):
            if i in used or j in used:
                continue
            sim = self.cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                self.logger.log(
                    "GraftingPair",
                    {"similarity": sim, "h1": h1[:60] + "...", "h2": h2[:60] + "..."},
                )
                prompt = (
                    f"Combine the following hypotheses into a clearer, unified statement:\n\n"
                    f"A: {h1}\nB: {h2}"
                )
                graft = self.call_llm(prompt, context)
                grafted.append(graft)
                used.update([i, j])

        # Add ungrafted hypotheses back
        hypotheses = context.get(HYPOTHESES, [])
        for i, h in enumerate(hypotheses):
            if i not in used:
                grafted.append(h)

        return grafted

    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def extract_hypothesis(self, text: str) -> list[str]:
        """
        Extracts hypothesis from markdown-style output with '**Hypothesis:**' section.
        Returns a list of simplified hypotheses.
        """
        pattern = re.compile(r"\*\*Hypothesis:\*\*\s*(.*?)\n", re.IGNORECASE)
        matches = pattern.findall(text)

        if len(matches) == 0:
            return [text]
        return [match.strip() for match in matches if match.strip()]