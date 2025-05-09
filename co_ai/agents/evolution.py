# co_ai/agents/evolution.py
import itertools

import numpy as np

from co_ai.agents.base import BaseAgent
from co_ai.memory.embedding_tool import get_embedding
from co_ai.memory.hypothesis_model import Hypothesis


class EvolutionAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, input_data: dict) -> dict:
        ranked = input_data.get("ranked", [])
        use_grafting = input_data.get("use_grafting", True)
        top_texts = [hyp for hyp, score in ranked[:3]]

        if use_grafting:
            self.log("Grafting similar hypotheses before evolution...")
            top_texts = await self.graft_similar(top_texts)

        self.log("Evolving hypotheses via LLM...")
        prompt = "Improve the following hypotheses:\n" + "\n".join(f"- {t}" for t in top_texts)
        raw_output = self.call_llm(prompt)
        evolved_lines = self.extract_list_items(raw_output)

        evolved = [
            Hypothesis(goal="Evolved from top-ranked", text=h.strip())
            for h in evolved_lines if h.strip()
        ]

        for h in evolved:
            self.memory.store_hypothesis(h)

        return {"evolved": evolved}

    async def graft_similar(self, hypotheses: list[str], threshold: float = 0.90) -> list[str]:
        embeddings = [get_embedding(h) for h in hypotheses]
        used = set()
        grafted = []

        for (i, h1), (j, h2) in itertools.combinations(enumerate(hypotheses), 2):
            if i in used or j in used:
                continue
            sim = self.cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                self.log(f"Grafting pair with similarity {sim:.2f}")
                prompt = (
                    f"Combine the following hypotheses into a clearer, unified statement:\n\n"
                    f"A: {h1}\nB: {h2}"
                )
                graft = self.call_llm(prompt).strip()
                grafted.append(graft)
                used.update([i, j])

        for i, h in enumerate(hypotheses):
            if i not in used:
                grafted.append(h)

        return grafted

    def cosine_similarity(self, vec1, vec2):
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
