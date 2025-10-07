# stephanie/metrics/providers/embed.py
from .base import MetricProvider, MetricVector
from typing import Dict, Any, Tuple
import asyncio

class EmbeddingProvider:
    name = "embed"
    version = "1.0.0"

    def __init__(self, backend):
        self.backend = backend

    async def compute(self, *, goal: str, text: str, context: Dict[str, Any]) -> MetricVector:
        if not self.backend:
            return MetricVector((), ())
        try:
            if hasattr(self.backend, "embed_text_async"):
                v = await self.backend.embed_text_async(text)
            else:
                m = self.backend.embed_text(text)
                v = await m if asyncio.iscoroutine(m) else m
            if hasattr(v, "tolist"): v = v.tolist()
            vals = tuple(float(x) for x in v)
            names = tuple(f"embed.d{idx:04d}" for idx in range(len(vals)))
            return MetricVector(names, vals)
        except Exception:
            return MetricVector((), ())
