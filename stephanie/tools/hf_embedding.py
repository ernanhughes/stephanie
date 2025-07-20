# stephanie/embeddings/huggingface_embedder.py

import time

import torch
from sentence_transformers import SentenceTransformer


class HuggingFaceEmbedder:
    _model_instance = None  # class-level singleton

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_name = cfg.get("hf_model_name", "BAAI/bge-large-en")

        if HuggingFaceEmbedder._model_instance is None:
            HuggingFaceEmbedder._model_instance = SentenceTransformer(self.model_name)

        self.model = HuggingFaceEmbedder._model_instance
        self.dim = 1024
        self.hdim = self.dim / 2

    def embed(self, text: str) -> list[float]:
        start_time = time.time()

        if not text or not text.strip():
            print("Empty text provided for embedding.")
            return [0.0] * self.dim

        chunks = self.chunker.chunk(text)
        if not chunks:
            return [0.0] * self.dim

        # Log number of chunks
        print(f"[HNet] Processing {len(chunks)} chunks...")

        # Monitor GPU memory before embedding
        if torch.cuda.is_available():
            mem_reserved = torch.cuda.memory_reserved() / 1e6
            mem_allocated = torch.cuda.memory_allocated() / 1e6
            print(f"[GPU] Reserved: {mem_reserved:.1f}MB, Allocated: {mem_allocated:.1f}MB")

        # Embedding
        chunk_embeddings = self.embedder.batch_embed(chunks)

        end_time = time.time()
        print(f"[HNet] Embedding took {end_time - start_time:.2f}s")

        return self.pooler.mean_pool(chunk_embeddings)

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        processed = [f"passage: {t.strip()}" if "e5" in self.model_name.lower() else t for t in texts]
        embeddings = self.model.encode(processed, convert_to_numpy=True, normalize_embeddings=True, device="cuda")
        return embeddings.tolist()



_model_instance = None


def load_model(model_name="BAAI/bge-large-en"):
    global _model_instance
    if _model_instance is None:
        _model_instance = SentenceTransformer(model_name)
        _model_instance = _model_instance.half()
    return _model_instance


def get_embedding(text: str, cfg: dict) -> list[float]:
    """
    Embed a single piece of text using HuggingFace model.
    """
    model_name = cfg.get("hf_model_name", "BAAI/bge-large-en")
    model = load_model(model_name)

    if not text.strip():
        return []

    # Some E5 models expect prefixes
    if "e5" in model_name.lower():
        text = f"passage: {text.strip()}"

    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True, device="cuda")
    return embedding.tolist()


def batch_embed(texts: list[str], cfg: dict) -> list[list[float]]:
    model_name = cfg.get("hf_model_name", "intfloat/e5-large-v2")
    model = load_model(model_name)

    texts = [f"passage: {t.strip()}" if "e5" in model_name.lower() else t for t in texts]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, device="cuda")
    return embeddings.tolist()
