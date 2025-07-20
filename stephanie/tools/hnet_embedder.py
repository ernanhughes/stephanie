import numpy as np
import torch
import torch.nn as nn

from stephanie.tools.embedding_tool import MXBAIEmbedder


class ByteLevelTokenizer:
    def tokenize(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))  # Raw byte tokenization

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")

class ChunkBoundaryPredictor(nn.Module):
    def __init__(self, vocab_size=256, hidden_dim=128, device="cpu"):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, hidden_dim).to(self.device)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=False).to(self.device)
        self.boundary_scorer = nn.Linear(hidden_dim * 2, 1).to(self.device)

    def forward(self, tokens: list[int]) -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)
        else:
            tokens = tokens.detach().clone().long()

        tokens = tokens.to(self.device)
        x = self.embedding(tokens).float()
        
        # Handle sequences of length 1 separately
        if x.size(0) == 1:
            # Directly use embedding for single-token sequences
            x = torch.cat([x, torch.zeros_like(x)], dim=-1)  # Simulate bidirectional output
        else:
            x = x.unsqueeze(1)  # [seq_len, 1, hidden_dim]
            # Ensure memory layout is contiguous
            if not x.is_contiguous():
                x = x.contiguous()
            # Disable cuDNN for sequences of length > 1 to avoid compatibility issues
            with torch.backends.cudnn.flags(enabled=False):
                x, _ = self.lstm(x)
            x = x.squeeze(1)  # Remove batch dimension
        
        scores = self.boundary_scorer(x)
        return scores.sigmoid().flatten()

class StephanieHNetChunker:
    def __init__(self, boundary_predictor=None, threshold=0.7):
        self.tokenizer = ByteLevelTokenizer()
        self.boundary_predictor = boundary_predictor or ChunkBoundaryPredictor(device="cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

    def chunk(self, text: str) -> list[str]:
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return []

        tokens_tensor = torch.tensor(tokens).long()
        with torch.no_grad():
            scores = self.boundary_predictor(tokens_tensor)

        boundaries = (scores > self.threshold).nonzero(as_tuple=True)[0].tolist()
        chunks = []
        prev = 0
        for b in boundaries:
            chunk_tokens = tokens[prev:b + 1]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            prev = b + 1

        if prev < len(tokens):
            final_chunk = self.tokenizer.decode(tokens[prev:])
            chunks.append(final_chunk)

        return chunks


class PoolingStrategy:
    @staticmethod
    def mean_pool(embeddings: list[list[float]]) -> list[float]:
        return np.mean(embeddings, axis=0).tolist() if embeddings else []

    @staticmethod
    def weighted_mean_pool(embeddings: list[list[float]], weights: list[float]) -> list[float]:
        return np.average(embeddings, weights=weights, axis=0).tolist() if embeddings else []


class StephanieHNetEmbedder:
    def __init__(self, embedder):
        self.chunker = StephanieHNetChunker()
        self.embedder = embedder
        self.dim = self.embedder.dim
        self.hdim = self.embedder.dim / 2
        self.pooler = PoolingStrategy()

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            print("Empty text provided for embedding.")
            return [0.0] * self.dim  # fallback vector

        chunks = self.chunker.chunk(text)
        if not chunks:
            return [0.0] * self.dim  # fallback vector if chunking failed

        chunk_embeddings = self.embedder.batch_embed(chunks)
        return self.pooler.mean_pool(chunk_embeddings)

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


# Singleton instance for reuse
_hnet_instance = None


def get_embedding(text: str, cfg: dict) -> list[float]:
    global _hnet_instance
    if _hnet_instance is None:
        base_embedder = MXBAIEmbedder(cfg)  # Direct init
        _hnet_instance = StephanieHNetEmbedder(embedder=base_embedder)

    return _hnet_instance.embed(text)