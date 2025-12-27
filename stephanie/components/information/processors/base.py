# stephanie/components/information/processors/base.py
from __future__ import annotations
from typing import Any, Dict, Protocol

class DocumentProcessor(Protocol):
    name: str

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ...
