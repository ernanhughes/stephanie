# co_ai/agents/evolution.py
import itertools

import numpy as np
from dspy import InputField, Module, OutputField, Predict, Signature

from co_ai.agents.base import BaseAgent
from co_ai.memory.embedding_tool import get_embedding
from co_ai.memory.hypothesis_model import Hypothesis


class EvolveSignature(Signature):
    top_hypotheses = InputField()
    improved_hypotheses = OutputField()

class GraftSignature(Signature):
    hypothesis_a = InputField()
    hypothesis_b = InputField()
    grafted_hypothesis = OutputField()

class EvolveHypotheses(Module):
    def __init__(self):
        super().__init__()
        self.generator = Predict(EvolveSignature)

    def forward(self, top_hypotheses: list[str]) -> dict:
        joined = "\n".join(f"- {h}" for h in top_hypotheses)
        return self.generator(top_hypotheses=joined)

class GraftHypotheses(Module):
    def __init__(self):
        super().__init__()
        self.grafter = Predict(GraftSignature)

    def forward(self, a: str, b: str) -> str:
        return self.grafter(hypothesis_a=a, hypothesis_b=b).grafted_hypothesis

class EvolutionAgent(BaseAgent):
    def __init__(self, memory):
        super().__init__(memory)
        self.evolver = EvolveHypotheses()
        self.grafter = GraftHypotheses()

    async def run(self, input_data: dict) -> dict:
        ranked = input_data.get("ranked", [])
        use_grafting = input_data.get("use_grafting", True)
        top_texts = [hyp for hyp, score in ranked[:3]]

        if use_grafting:
            self.log("Grafting similar hypotheses before evolution...")
            top_texts = await self.graft_similar(top_texts)

        self.log("Evolving hypotheses via DSPy...")
        result = self.evolver.forward(top_texts)

        evolved = result.improved_hypotheses.split("\n")
        evolved_hypotheses = [
            Hypothesis(goal="Evolved from top-ranked", text=h.strip())
            for h in evolved if h.strip()
        ]

        for h in evolved_hypotheses:
            self.memory.store_hypothesis(h)

        return {"evolved": evolved_hypotheses}

    async def graft_similar(self, hypotheses: list[str], threshold: float = 0.90) -> list[str]:
        embeddings = [get_embedding(h) for h in hypotheses]
        used = set()
        grafted = []

        for (i, h1), (j, h2) in itertools.combinations(enumerate(hypotheses), 2):
            if i in used or j in used:
                continue
            sim = self.cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                self.log(f"Grafting pair with similarity {sim:.2f}:")
                graft = self.grafter.forward(h1, h2)
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
