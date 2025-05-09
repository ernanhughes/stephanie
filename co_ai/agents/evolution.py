# co_ai/agents/evolution.py
import itertools
import numpy as np

from dspy import Predict, Signature, InputField, OutputField
from dspy import LM

from co_ai.agents.base import BaseAgent
from co_ai.memory.hypothesis_model import Hypothesis
from co_ai.memory.embedding_tool import get_embedding
from typing import List

class EvolveSignature(Signature):
    top_hypotheses = InputField()  # joined string of bullet points
    improved_hypotheses = OutputField()  # joined string of bullet points

class GraftSignature(Signature):
    hypothesis_a = InputField()
    hypothesis_b = InputField()
    grafted_hypothesis = OutputField()

class EvolutionAgent(BaseAgent):
    def __init__(self, memory=None, logger=None, model_config=None):
        super().__init__(memory=memory, logger=logger)
        self.model_config = model_config or {
            "name": "ollama_chat/qwen3",
            "api_base": "http://localhost:11434",
            "api_key": None,
        }
        lm = LM(
            self.model_config["name"],
            api_base=self.model_config["api_base"],
            api_key=self.model_config.get("api_key")
        )
        self.evolver = Predict(EvolveSignature, lm=lm)
        self.grafter = Predict(GraftSignature, lm=lm)

    async def run(self, input_data: dict) -> dict:
        ranked = input_data.get("ranked", [])
        use_grafting = input_data.get("use_grafting", True)
        top_texts = [item["hypothesis"] for item in ranked[:3]]

        if use_grafting:
            self.log("Grafting similar hypotheses before evolution...")
            top_texts = await self.graft_similar(top_texts)

        self.log("Evolving hypotheses via DSPy...")
        result = self.evolver(top_hypotheses="\n".join(f"- {h}" for h in top_texts))

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
                self.log(f"Grafting pair with similarity {sim:.2f}")
                graft = self.grafter(hypothesis_a=h1, hypothesis_b=h2).grafted_hypothesis
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
