from __future__ import annotations
import re
from typing import Any

class DummyModel:
    """Very small stand‑in for your actual model router.
    Call with text → returns text. Deterministic and offline.
    """
    def __call__(self, prompt: str) -> str:
        # Look for <answer>..</answer> and produce a simple question.
        m = re.search(r"<answer>(.*?)</answer>", prompt, re.DOTALL | re.I)
        answer = (m.group(1).strip() if m else "the provided answer")
        return f"<question>What real‑world mechanism most directly explains: {answer}?</question>"

class Proposer:
    def __init__(self, model: Any | None = None, prompt_template: str | None = None):
        self.model = model or DummyModel()
        self.prompt_tmpl = (
            prompt_template
            or "Given <answer>{answer}</answer>, generate ONE challenging, verifiable question wrapped in <question> tags."
        )

    def propose(self, seed_answer: str) -> str:
        out = self.model(self.prompt_tmpl.format(answer=seed_answer))
        m = re.search(r"<question>(.*?)</question>", out, re.DOTALL | re.I)
        return (m.group(1).strip() if m else out.strip())

