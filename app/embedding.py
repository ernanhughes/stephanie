from hydra import initialize, compose
import ollama

# Load embedding model from config
with initialize(config_path="../config", version_base=None):
    cfg = compose(config_name="stephanie.yaml")
    model = cfg.embedding.model

def get_embedding(text: str) -> list[float] | None:
    try:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"[Embedding] Error: {e}")
        return None
