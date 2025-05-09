# co_ai/agents/evolution.py
import itertools
from typing import Any, Dict, List, Optional

import numpy as np

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

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        print(f"Initializing Evolution Agent: {cfg}")
        self.strategy = cfg.get("strategy", "grafting")
        self.use_grafting = cfg.get("use_grafting", False)
        self.preferences = cfg.get("preferences", ["novelty", "feasibility"])

    async def run(self, input_data: dict) -> dict:
        """
        Evolve top-ranked hypotheses by combining, refining, or improving them.
        
        Args:
            input_data: Dictionary with keys:
                - ranked: list of (hypothesis, score) tuples
                - use_grafting: whether to combine similar hypotheses
                - preferences: optional criteria for quality improvement
        
        Returns:
            dict with key 'evolved' containing evolved hypotheses
        """
        ranked = input_data.get("ranked", [])
        use_grafting = input_data.get("use_grafting", True)
        preferences = input_data.get("preferences", "")

        top_texts = [hyp for hyp, score in ranked[:3]]

        self.logger.log("EvolvingTopHypotheses", {
            "top_hypotheses": top_texts,
            "grafting_enabled": use_grafting,
            "preferences": preferences
        })

        if use_grafting:
            self.log("Grafting similar hypotheses before evolution...")
            top_texts = await self.graft_similar(top_texts)

        self.log("Evolving hypotheses via LLM...")

        # Use a structured prompt that guides the LLM toward specific improvements
        prompt = self._build_evolution_prompt(top_texts, preferences)
        raw_output = self.call_llm(prompt)
        evolved_lines = self.extract_list_items(raw_output)

        evolved = []
        for h in evolved_lines:
            if h.strip():
                hypothesis = Hypothesis(
                    goal="Evolved from top-ranked",
                    text=h.strip(),
                    source="evolution"
                )
                self.memory.store_hypothesis(hypothesis)
                evolved.append(hypothesis)

        return {"evolved": evolved}

    def _build_evolution_prompt(self, hypotheses: List[str], preferences: str = "") -> str:
        """
        Build a structured prompt for evolving hypotheses.
        """
        prompt = (
            "You are an expert researcher tasked with refining and evolving the following research hypotheses.\n\n"
            "Goal: Improve the hypotheses based on the following criteria:\n"
            f"{preferences or 'novelty, feasibility, testability'}\n\n"
            "Instructions:\n"
            "1. Review each hypothesis carefully and identify weaknesses or gaps.\n"
            "2. Propose clear improvements focusing on practical implementation.\n"
            "3. Combine elements from multiple hypotheses if beneficial.\n"
            "4. Ensure the revised hypotheses retain novelty while being more testable.\n\n"
            "Hypotheses to evolve:\n"
        )

        for i, h in enumerate(hypotheses, 1):
            prompt += f"{i}. {h}\n"

        prompt += "\nPlease output only the improved hypotheses as a numbered list."
        return prompt

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
    
    async def _fallback_rank_hypotheses(self, hypotheses, goal):
        """
        Use the Evolution LLM to compare pairs of hypotheses and return a ranked list.
        Simulates a basic tournament when real ranking isn't available.
        """
        if len(hypotheses) < 2:
            # Not enough to rank
            return [(h, 5) for h in hypotheses]

        ranked = []
        scores = [0] * len(hypotheses)

        for i, h1 in enumerate(hypotheses):
            for j, h2 in enumerate(hypotheses):
                if i == j:
                    continue

                prompt = self._build_rank_prompt(goal, h1, h2)
                response = self.call_llm(prompt)
                match = re.search(r"better hypothesis:<(\d+)>", response)

                if match:
                    winner_idx = int(match.group(1)) - 1
                    winner = h1 if winner_idx == 1 else h2
                    scores[i if winner == h1 else j] += 1
                else:
                    self.logger.log("FallbackRankingFailedToParse", {
                        "prompt": prompt[:200],
                        "response": response[:300]
                    })

        # Pair each hypothesis with its score
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