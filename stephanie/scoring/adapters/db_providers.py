from __future__ import annotations
import logging
from typing import Dict, Any

from stephanie.scoring.feature_io import FeatureProvider
from stephanie.scoring.scorable import Scorable

log = logging.getLogger(__name__)

class DomainDBProvider:
    def __init__(self, memory):
        self.memory = memory

    async def hydrate(self, scorable: Scorable) -> Dict[str, Any]:
        try:
            rows = self.memory.scorable_domains.find(
                scorable_id=str(scorable.id),
                scorable_type=scorable.target_type,
            )
            domains = [{
                "name": r["domain"],
                "score": float(r["score"]),
                "source": "db",
            } for r in rows]
            return {"domains": domains}
        except Exception as e:
            log.warning(f"DomainDBProvider failed for {scorable.id}: {e}")
            return {}


class EntityDBProvider:
    def __init__(self, memory):
        self.memory = memory

    async def hydrate(self, scorable: Scorable) -> Dict[str, Any]:
        try:
            rows = self.memory.scorable_entities.find(
                scorable_id=str(scorable.id),
                scorable_type=scorable.target_type,
            )
            ner = [{
                "text": r["entity_text"],
                "type": r.get("entity_type"),
                "span": [r.get("start"), r.get("end")],
                "source": "db",
            } for r in rows]
            return {"ner": ner}
        except Exception as e:
            log.warning(f"EntityDBProvider failed for {scorable.id}: {e}")
            return {}
