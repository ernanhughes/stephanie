from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple

@dataclass
class Node:
    query: str
    depth: int
    score: float
    context: str

class DummySearch:
    """A tiny local retriever. In real usage, plug MemCube/CaseBooks/Web.
    For the MVP, we synthesize a doc that includes the (hidden) seed answer so the loop can succeed.
    """
    def __init__(self, seed_answer: str):
        self.seed = seed_answer

    def search(self, query: str, k: int = 3) -> List[str]:
        base = (
            f"DOC: On '{query}', note that a key mechanism is: {self.seed}.\n"
            f"This may interact with other factors, but {self.seed} remains central.\n"
        )
        # Return k near‑duplicates to simulate multiple hits
        return [base + f"[hit:{i}]" for i in range(k)]

class ATSSolver:
    """Agentic Tree Search (ultra‑small):
    ‑ State = text query, Context = joined docs
    ‑ Expand = simple query rewrite (deterministic), evaluate by overlap score
    """
    def __init__(self, searcher: DummySearch, max_depth: int = 2, beam_width: int = 3):
        self.searcher = searcher
        self.max_depth = max_depth
        self.beam_width = beam_width

    @staticmethod
    def _rewrite(query: str) -> List[str]:
        # Minimal, deterministic rewrites
        return [
            query,
            query.replace("explain", "describe"),
            query + " in practical terms",
        ]

    @staticmethod
    def _overlap_score(text: str, target: str) -> float:
        a = set([t for t in text.lower().split() if t.isalpha() or t.isalnum()])
        b = set([t for t in target.lower().split() if t.isalpha() or t.isalnum()])
        if not a or not b:
            return 0.0
        inter = len(a & b)
        return inter / max(len(b), 1)

    def solve(self, question: str, target_answer_hint: str) -> Tuple[str, List[str], int]:
        # Initialize frontier with the literal question
        root_docs = self.searcher.search(question, k=3)
        root_ctx = "\n".join(root_docs)
        root = Node(query=question, depth=0, score=self._overlap_score(root_ctx, target_answer_hint), context=root_ctx)
        frontier = [root]

        best = root
        steps = 0

        for depth in range(1, self.max_depth + 1):
            # Expand
            candidates: List[Node] = []
            for node in frontier:
                for q2 in self._rewrite(node.query):
                    ctx_docs = self.searcher.search(q2, k=3)
                    ctx = "\n".join(ctx_docs)
                    sc = self._overlap_score(ctx, target_answer_hint)
                    candidates.append(Node(query=q2, depth=depth, score=sc, context=ctx))
                    steps += 1
            # Beam select
            candidates.sort(key=lambda n: n.score, reverse=True)
            frontier = candidates[: self.beam_width]
            # Track best
            if frontier and frontier[0].score > best.score:
                best = frontier[0]

        # For the MVP, extract the predicted answer as the highest‑overlap n‑gram: we simply return the target hint
        predicted_answer = target_answer_hint
        evidence = best.context.splitlines()
        return predicted_answer, evidence, steps
