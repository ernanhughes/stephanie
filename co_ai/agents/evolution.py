# co_ai/agents/evolution.py
import itertools
from typing import Any, Dict, List, Optional

import numpy as np
import re

from co_ai.agents.base import BaseAgent
from co_ai.memory.embedding_tool import get_embedding
from co_ai.memory.hypothesis_model import Hypothesis
from co_ai.utils import load_prompt_from_file

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

    async def run(self, input_data: dict) -> dict:
        """
        Evolve top-ranked hypotheses individually.
        
        Args:
            input_data: Dictionary with keys:
                - ranked: list of (hypothesis, score) tuples
                - hypotheses: list of unranked hypotheses (fallback)
                - preferences: override criteria for refinement
        """
        ranked = input_data.get("ranked", [])
        fallback_hypotheses = input_data.get("hypotheses", [])
        preferences = input_data.get("preferences", self.preferences)

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
                prompt = self._build_evolution_prompt(h, preferences)
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

        return {"evolved": evolved} 
   
    def get_prompt_template(self, input_data: dict) -> str:
        """
        Load the prompt template based on the strategy.
        """
        strategy = input_data.get("strategy", "grafting") # called by base
        prompt_file = self.PROMPT_MAP.get(strategy)
        if not prompt_file:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        try:
            return load_prompt_from_file(prompt_file)
        except Exception as e:
            self.logger.log("PromptLoadFailed", {"error": str(e)})
            raise

    def _build_evolution_prompt(self, goal: str, hypotheses: str, preferences: str = "") -> str:
        """
        Build prompt by injecting goal and preferences into the loaded template.
        """
        print(f"Prompt template: {self.prompt_template} goal:{goal} ")
        return self.prompt_template.format(
            goal=goal,
            preferences=", ".join(preferences or self.preferences),
            hypotheses=hypotheses)

    async def graft_similar(self, hypotheses: list[str], threshold: float = 0.90) -> list[str]:
        """
        Graft pairs of highly similar hypotheses into unified versions.
        """
        embeddings = [get_embedding(h) for h in hypotheses]
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

    async def _fallback_rank_hypotheses(self, hypotheses: List[str], goal: str) -> list:
        """
        Use the Evolution model to simulate ranking when no real ranking exists.
        """
        if not self.use_fallback_ranking:
            self.logger.log("FallbackRankingDisabled", {"reason": "user_config"})
            return []

        if len(hypotheses) < 2:
            self.logger.log("FallbackRankingUsedOnSingle", {"count": len(hypotheses)})
            return [(h, 5) for h in hypotheses]

        # Load ranking fallback prompt once
        try:
            rank_prompt = load_prompt_from_file("evolution_ranking_fallback.txt")
        except Exception as e:
            self.logger.log("FallbackPromptLoadFailed", {"error": str(e)})
            return [(h, 0) for h in hypotheses]

        scores = [0] * len(hypotheses)

        for i, h1 in enumerate(hypotheses):
            for j, h2 in enumerate(hypotheses):
                if i == j:
                    continue

                prompt = rank_prompt.format(
                    goal=goal,
                    preferences=", ".join(self.preferences),
                    hypothesis_1=h1,
                    hypothesis_2=h2
                )

                response = self.call_llm(prompt)
                match = re.search(r"better hypothesis:<(\d+)>", response)

                if match:
                    winner_idx = int(match.group(1)) - 1
                    scores[i if winner_idx == 1 else j] += 1
                else:
                    self.logger.log("FallbackRankingFailedToParse", {
                        "prompt": prompt[:200],
                        "response": response[:300]
                    })

        scored_pairs = sorted(zip(hypotheses, scores), key=lambda x: x[1], reverse=True)
        return scored_pairs

    def _build_rank_prompt(self, goal, h1, h2):
        with open("prompts/evolution_ranking_fallback.txt", "r") as f:
            prompt_template = f.read()
        return prompt_template.format(
            goal=goal,
            preferences=", ".join(self.preferences),
            hypothesis_1=h1,
            hypothesis_2=h2
        )