# stephanie/components/ssp/retrievers/base.py
from __future__ import annotations
from typing import Any, Dict, List

class BaseRetriever:
    async def retrieve(self, query: str, seed_answer: str, context: Dict[str, Any], k: int) -> List[str]:
        raise NotImplementedError
