# stephanie/metrics/registry.py
from typing import Dict, Any, List, Type

from stephanie.scoring.calculations.mars_calculator import MARSCalculator
from .providers.base import MetricProvider
from .providers.basic import BasicProvider
from .providers.embed import EmbeddingProvider
from .providers.scorer import ScorerProvider
from .providers.mars import MARSProvider

DEFAULT_PROVIDER_MAP = {
    "basic": BasicProvider,
    "embed": EmbeddingProvider,
    "scorer": ScorerProvider,
    "mars": MARSProvider,
}

def build_providers(cfg: Dict[str, Any], container, memory, logger) -> List[MetricProvider]:
    mcfg = (cfg or {}).get("metrics", {}) or {}
    names: List[str] = mcfg.get("providers", ["basic","embed","scorer","mars"])
    out: List[MetricProvider] = []
    for name in names:
        Cls = DEFAULT_PROVIDER_MAP.get(name)
        if not Cls: continue
        if name == "embed":
            backend = memory.embedding
            out.append(Cls(backend))
        elif name == "scorer":
            scoring = container.get("scoring")
            out.append(Cls(scoring)) 
        elif name == "mars":
            mars_calculator = MARSCalculator(cfg.get("mars", {}), memory, container, logger)
            # pass scorer provider instance if present later (service will wire context)
            out.append(Cls(mars_calculator))
        else:
            out.append(Cls())
    return out
