from haystack import component
from dspy import LM

@component
class OptimistDebater:
    def __init__(self):
        self.lm = LM("ollama_chat/mistral", api_base="http://localhost:11434")

    def run(self, hypotheses: list[str]) -> dict:
        print("[OptimistDebater] Critiquing with optimism...")
        return {"optimist_reviews": [f"Optimistically analyzing: {h}" for h in hypotheses]}

@component
class SkepticDebater:
    def __init__(self):
        self.lm = LM("openai/gpt-4", api_base="https://api.openai.com/v1")

    def run(self, hypotheses: list[str]) -> dict:
        print("[SkepticDebater] Critiquing with skepticism...")
        return {"skeptic_reviews": [f"Skeptically analyzing: {h}" for h in hypotheses]}
