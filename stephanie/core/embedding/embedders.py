# stephanie/embedding/embedders.py
from sentence_transformers import SentenceTransformer

from stephanie.core.embedding.protocols import EmbedderProtocol
from stephanie.memory.embedding_store import EmbeddingStore


class StephanieEmbedder(EmbedderProtocol):
    def __init__(self, embedding_store: EmbeddingStore):
        self.store = embedding_store
    
    def get_or_create(self, text: str) -> list[float]:
        return self.store.get_or_create(text)
    
    def find_similar(self, embedding: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        return self.store.find_similar(embedding, top_k)
    


class HuggingFaceEmbedder(EmbedderProtocol):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.cache = {}  # Simple in-memory cache
    
    def get_or_create(self, text: str) -> list[float]:
        if text in self.cache:
            return self.cache[text]
        
        embedding = self.model.encode(text).tolist()
        self.cache[text] = embedding
        return embedding
    
    def find_similar(self, embedding: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        import numpy as np
        from scipy.spatial import distance

        # Convert input to numpy array
        emb_array = np.array(embedding)
        results = []
        
        for text, stored_emb in self.cache.items():
            sim = 1 - distance.cosine(emb_array, np.array(stored_emb))
            results.append((text, sim))
        
        # Sort by similarity
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    

class HNetEmbedder(EmbedderProtocol):
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
        self.cache = {}
    
    def _load_model(self, path: str):
        """Load HNet model (stubbed for now)"""
        # Replace with actual HNet loading logic
        return lambda text: [0.1, 0.2, 0.3, 0.4]  # Mock embedding
    
    def get_or_create(self, text: str) -> list[float]:
        if text in self.cache:
            return self.cache[text]
        
        embedding = self.model(text)
        self.cache[text] = embedding
        return embedding
    
    def find_similar(self, embedding: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        """Find similar texts using HNet logic"""
        # Replace with HNet similarity search
        return [(f"mock_{i}", 1.0 - i*0.1) for i in range(top_k)]