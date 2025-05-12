# co_ai/agents/evolution.py
import itertools
from typing import Any, Dict, List, Optional

import numpy as np
import re

from co_ai.agents.base import BaseAgent
from co_ai.memory.embedding_tool import get_embedding
from co_ai.memory.hypothesis_model import Hypothesis

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

    PROMPT_MAP = {
        "goal_aligned": "evolution_goal_aligned.txt",
        "inspiration": "evolution_inspiration.txt",
        "feasibility": "evolution_feasibility.txt",
        "simplification": "evolution_simplification.txt",
        "out_of_the_box": "evolution_out_of_the_box.txt",
        # "grafting": "evolution_grafting.txt" causes too much trouble
    }

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        print(f"Initializing Evolution Agent: {cfg}")
        self.strategy = cfg.get("strategy", "grafting")
        self.use_grafting = cfg.get("use_grafting", False)
        self.preferences = cfg.get("preferences", ["novelty", "feasibility"])
        print(f"Evolution Preferences: {self.preferences}")
        self.cfg = cfg

    async def run(self, context: dict) -> dict:
        """
        Evolve top-ranked hypotheses individually.
        
        Args:
            context: Dictionary with keys:
                - ranked: list of (hypothesis, score) tuples
                - hypotheses: list of unranked hypotheses (fallback)
                - preferences: override criteria for refinement
        """
        ranked = context.get("ranked", [])
        fallback_hypotheses = context.get("hypotheses", [])
        preferences = context.get("preferences", self.preferences)

        # Decide which hypotheses to evolve
        if ranked:
            top_texts = [hyp for hyp, _ in ranked[:3]]
        elif fallback_hypotheses:
            top_texts = fallback_hypotheses
        else:
            self.logger.log("NoHypothesesToEvolve", {
                "reason": "no_ranked_or_unranked_input"
            })
            return {"evolved": []}

        evolved = []
        for h in top_texts:
            try:
                prompt = self.prompt_loader.load_prompt({**self.cfg, **{"hypothesis":h}}, context)
                raw_output = self.call_llm(prompt).strip()
                refined_list = self.extract_list_items(raw_output)

                if refined_list:
                    for r in refined_list:
                        hypothesis = Hypothesis(
                            goal="Evolved from top-ranked",
                            text=r.strip(),
                            source="evolution"
                        )
                        self.memory.store_hypothesis(hypothesis)
                        evolved.append(hypothesis)
                else:
                    self.logger.log("EvolutionFailed", {
                        "original": h[:100],
                        "response_snippet": raw_output[:200]
                    })

            except Exception as e:
                self.logger.log("EvolutionError", {
                    "error": str(e),
                    "hypothesis": h[:100]
                })

        context["evolved"] = evolved
        self.logger.log("EvolutionCompleted", {
            "evolved_count": len(evolved),
            "preferences": preferences
        })
        return context
   
    async def graft_similar(self, hypotheses: list[str], threshold: float = 0.90) -> list[str]:
        """
        Graft pairs of highly similar hypotheses into unified versions.
        """
        embeddings = [get_embedding(h, self.cfg) for h in hypotheses]
        used = set()
        grafted = []

        for (i, h1), (j, h2) in itertools.combinations(enumerate(hypotheses), 2):
            if i in used or j in used:
                continue
            sim = self.cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                self.logger.log("GraftingPair", {
                    "similarity": sim,
                    "h1": h1[:60] + "...",
                    "h2": h2[:60] + "..."
                })
                prompt = (
                    f"Combine the following hypotheses into a clearer, unified statement:\n\n"
                    f"A: {h1}\nB: {h2}"
                )
                graft = self.call_llm(prompt).strip()
                grafted.append(graft)
                used.update([i, j])

        # Add ungrafted hypotheses back
        for i, h in enumerate(hypotheses):
            if i not in used:
                grafted.append(h)

        return grafted

    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def extract_list_items(self, text: str) -> list[str]:
        """
        Extract items from a numbered list in the LLM response.
        """
        lines = text.strip().splitlines()
        items = []

        for line in lines:
            if line.lstrip().startswith(tuple(str(i) + "." for i in range(1, 10))):
                content = line.lstrip()[2:].strip()
                if content:
                    items.append(content)
        return items
