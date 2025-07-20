# stephanie/embedding/protocols.py
from typing import List, Protocol, Tuple


class EmbedderProtocol(Protocol):
    def get_or_create(self, text: str) -> list[float]:
        """Generate or retrieve embedding for text"""
        ...
    
    def find_similar(self, embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar texts by embedding"""
        ...