import torch
from typing import Any
from .base import SignalProvider, SignalResult
from ..axes import RewardAxis


class EmbeddingProvider(SignalProvider):
    def __init__(self, embedder: Any):
        self.embedder = embedder

    def compute(self, context_pack: Any, plan_trace: Any, output: Any, **kwargs) -> SignalResult:
        goal_emb = kwargs.get("goal_embedding")
        output_emb = self.embedder.encode(output)

        margin = torch.nn.functional.cosine_similarity(output_emb, goal_emb, dim=0).item()

        return SignalResult(
            axis_values={RewardAxis.EMBEDDING_MARGIN: margin},
            diagnostics={"margin": margin},
            confidence=0.95,
        )
