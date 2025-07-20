from stephanie.core.embedding.embedders import (HNetEmbedder,
                                                HuggingFaceEmbedder,
                                                StephanieEmbedder)
from stephanie.core.embedding.protocols import EmbedderProtocol


# In your system initialization
def initialize_embedding_components(self):
    # Stephanie's current embedder
    self.registry.register(
        name="stephanie",
        component=StephanieEmbedder(self.memory.embedding_store),
        protocol_type=EmbedderProtocol,
        meta={"version": "v1", "sensitivity": "public"}
    )
    
    # HuggingFace embedder
    self.registry.register(
        name="huggingface",
        component=HuggingFaceEmbedder(),
        protocol_type=EmbedderProtocol,
        meta={"version": "v1", "sensitivity": "internal"}
    )
    
    # HNet embedder (future)
    self.registry.register(
        name="hnet",
        component=HNetEmbedder(model_path="models/hnet"),
        protocol_type=EmbedderProtocol,
        meta={"version": "experimental", "sensitivity": "confidential"}
    )