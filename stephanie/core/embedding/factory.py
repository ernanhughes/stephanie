# stephanie/embedding/factory.py
from stephanie.core.embedding.protocols import EmbedderProtocol
from stephanie.registry.component_registry import ComponentRegistry


class EmbedderFactory:
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
    
    def get_embedder(self, name: str = None) -> EmbedderProtocol:
        """
        Get embedder by name or best-performing one
        
        Args:
            name: Optional embedder name (e.g., "huggingface")
        """
        if name:
            embedder = self.registry.get(name, EmbedderProtocol)
            if not embedder:
                raise ValueError(f"Embedder '{name}' not registered")
            return embedder
        
        # Get best-performing embedder
        best = self.registry.get_best_by_performance(EmbedderProtocol)
        if best:
            return best
        return self._get_default()
    
    def _get_default(self) -> EmbedderProtocol:
        """Fallback embedder if nothing else works"""
        # Use Stephanie's basic embedder
        return self.registry.get("stephanie", EmbedderProtocol)
    
    def list_available(self) -> list[str]:
        """List all registered embedders"""
        return [
            comp["name"] for comp in 
            self.registry.get_components_with_metadata(EmbedderProtocol)
        ]
    
    def validate_embedder(self, embedder: EmbedderProtocol):
        """Ensure embedder meets requirements"""
        if not hasattr(embedder, "get_or_create"):
            raise InterfaceValidationError("Missing required method: get_or_create")
        if not hasattr(embedder, "find_similar"):
            raise InterfaceValidationError("Missing required method: find_similar")
    
    def get_embedding(self, text: str, embedder_name: str = None) -> list[float]:
        """Get embedding from selected embedder"""
        embedder = self.get_embedder(embedder_name)
        return embedder.get_or_create(text)
    
    def find_similar(self, embedding: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        """Find similar texts using best embedder"""
        embedder = self.get_embedder()
        return embedder.find_similar(embedding, top_k)