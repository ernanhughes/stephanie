# stephanie/models/pricing.py
"""Price knowledge, separated from execution (§10 — primarily new build)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Mapping, Optional

from stephanie.models.model import Model
from stephanie.models.usage import ModelUsage


@dataclass(frozen=True)
class PriceEntry:
    input_per_1k_usd: Decimal
    output_per_1k_usd: Decimal
    effective_from: Optional[date] = None


class PricingService(ABC):
    @abstractmethod
    def estimate(self, model: Model, usage: ModelUsage) -> Optional[Decimal]:
        """Return estimated cost in USD, or None when pricing is unknown."""


class NullPricingService(PricingService):
    """Default: unknown pricing stays unknown (never zero)."""

    def estimate(self, model: Model, usage: ModelUsage) -> Optional[Decimal]:
        return None


class StaticPricingService(PricingService):
    """Versioned, provider-specific, date-sensitive price table."""

    def __init__(self, table: Mapping[str, PriceEntry], version: str = "v1"):
        self.table = dict(table)
        self.version = version

    def estimate(self, model: Model, usage: ModelUsage) -> Optional[Decimal]:
        entry = self.table.get(model.id) or self.table.get(model.name)
        if entry is None or usage is None:
            return None
        total = Decimal("0")
        if usage.input_tokens:
            total += entry.input_per_1k_usd * (Decimal(usage.input_tokens) / Decimal(1000))
        if usage.output_tokens:
            total += entry.output_per_1k_usd * (Decimal(usage.output_tokens) / Decimal(1000))
        return total
