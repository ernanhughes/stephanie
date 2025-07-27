from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseContextManager(ABC):
    def __init__(
        self,
        goal: str,
        memory,
        num_chunks: int = 4,
        scoring_dimension: str = "alignment",
        logger = None,
    ):
        self.goal = goal
        self.memory = memory
        self.num_chunks = num_chunks
        self.scoring_dimension = scoring_dimension
        self.logger = logger

    @abstractmethod
    def chunk_source(self, source: str) -> List[Dict[str, str]]:
        """
        Subclasses implement this to break input into structured units.
        Each chunk should be a dict with at least a 'text' key.
        Optionally, include 'role', 'section', etc.
        """
        pass

    def score_chunks(self, chunks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Use embedding similarity to goal to rank the chunks.
        """
        goal_embedding = self.embedding.get_or_create(self.goal)
        results = []

        for chunk in chunks:
            text = chunk["text"]
            chunk_embedding = self.embedding.get_or_create(text)
            similarity = self.compute_similarity(goal_embedding, chunk_embedding)

            results.append({
                **chunk,
                "score": similarity
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:self.num_chunks]

    def compute_similarity(self, a: List[float], b: List[float]) -> float:
        # Cosine similarity
        from numpy import dot
        from numpy.linalg import norm
        return dot(a, b) / (norm(a) * norm(b) + 1e-8)

    def assemble_context(self, source: str) -> str:
        """
        Full pipeline: chunk → score → format as few-shot prompt context
        """
        chunks = self.chunk_source(source)
        scored_chunks = self.score_chunks(chunks)
        return self.format_prompt(scored_chunks)

    def format_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        return "\n\n".join([c["text"] for c in chunks])

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "backend": self.embedding.__class__.__name__,
            "num_chunks": self.num_chunks,
            "scoring_dimension": self.scoring_dimension,
        }
