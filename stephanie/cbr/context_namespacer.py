# stephanie/cbr/context_namespacer.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict

CTX_NS = "_MEMENTO"
VARIANTS = "variants"

class DefaultContextNamespacer:
    def ns(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return ctx.setdefault(CTX_NS, {VARIANTS: {}})

    def variant_bucket(self, ctx: Dict[str, Any], variant: str) -> Dict[str, Any]:
        return self.ns(ctx)[VARIANTS].setdefault(variant, {})

    def variant_output_key(self, variant: str) -> str:
        return f"{CTX_NS}.{variant}.output"

    @contextmanager
    def temp_key(self, ctx: Dict[str, Any], key: str, value):
        _sentinel = object()
        old = ctx.get(key, _sentinel)
        ctx[key] = value
        try:
            yield
        finally:
            if old is _sentinel:
                ctx.pop(key, None)
            else:
                ctx[key] = old
