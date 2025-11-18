from __future__ import annotations

import logging
from typing import Any, Dict

from stephanie.scoring.scorable import Scorable

log = logging.getLogger(__name__)

class DomainDBWriter:
    def __init__(self, memory):
        self.memory = memory

    async def persist(self, scorable: Scorable, features: Dict[str, Any]) -> None:
        new_domains = features.get("domains_new", [])
        if not new_domains:
            return
        for d in new_domains:
            try:
                self.memory.scorable_domains.insert({
                    "scorable_id": str(scorable.id),
                    "scorable_type": scorable.target_type,
                    "domain": d["name"],
                    "score": float(d.get("score", 1.0)),
                })
            except Exception as e:
                log.warning(f"DomainDBWriter insert failed: {e}")


class EntityDBWriter:
    def __init__(self, memory):
        self.memory = memory

    async def persist(self, scorable: Scorable, features: Dict[str, Any]) -> None:
        new_ner = features.get("ner_new", [])
        if not new_ner:
            return
        for e in new_ner:
            try:
                span = e.get("span") or [None, None]
                self.memory.scorable_entities.insert({
                    "scorable_id": str(scorable.id),
                    "scorable_type": scorable.target_type,
                    "entity_text": e["text"],
                    "entity_text_norm": e["text"].lower(),
                    "entity_type": e.get("type"),
                    "start": span[0],
                    "end": span[1],
                    "similarity": float(e.get("similarity", 1.0)),
                    "source_text": (scorable.text or "")[:100] + "...",
                })
            except Exception as ex:
                log.warning(f"EntityDBWriter insert failed: {ex}")
