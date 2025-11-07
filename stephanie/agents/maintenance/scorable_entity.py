# stephanie/agents/maintenance/scorable_entity.py
from __future__ import annotations

import traceback
from typing import Dict, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable


class ScorableEntityAgent(BaseAgent):
    """
    Enriches scorables with named entities using the NER Retriever.

    - Detects entity spans (people, orgs, concepts, methods, etc.)
    - Embeds them with the NER projection (if enabled)
    - Stores them in the scorable_entities table
    - Provides metadata for cross-CaseBook linking
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.min_length = cfg.get("entity_min_length", 2)
        self.max_entities = cfg.get("max_entities_per_scorable", 20)
        self.ner_enabled = memory.embedding.is_ner_enabled()

    async def run(self, context: dict) -> dict:
        """
        Enrich all scorables in context with entities.
        """
        try:
            scorables = self.get_scorables(context)
            enriched = []

            for sc in scorables:
                if not isinstance(sc, Scorable):
                    sc = Scorable(
                        id=str(sc.get("id")),
                        text=sc.get("text"),
                        target_type=sc.get("target_type", "document"),
                        meta=sc.get("metadata", {}),
                    )

                if not self.ner_enabled:
                    continue

                # Detect + embed entities
                entities = self._extract_entities(sc)
                if not entities:
                    continue

                # Save entities to memory
                self.memory.scorable_entities.set_entities(sc.id, sc.target_type, entities)

                record = {
                    "id": sc.id,
                    "target_type": sc.target_type,
                    "entities": entities,
                }
                enriched.append(record)

                self.logger.log("ScorableEntitiesEnriched", record)

            context["enriched_entities"] = enriched
            if enriched:
                self.report({
                    "event": "entity_enrichment",
                    "step": "ScorableEntityAgent",
                    "details": f"{len(enriched)} scorables enriched with entities.",
                    "total_entities": sum(len(r["entities"]) for r in enriched),
                    "scorable_ids": [r["id"] for r in enriched],
                })

            return context

        except Exception as e:
            self.logger.log(
                "ScorableEntityAgentError",
                {"error": str(e), "trace": traceback.format_exc()},
            )
            return context

    def _extract_entities(self, sc: Scorable) -> List[Dict]:
        """
        Use NER Retriever to extract and embed entities from scorable text.
        """
        entities = []
        try:
            detected = self.memory.embedding.ner_retriever.entity_detector.detect_entities(sc.text)
            for start, end, etype in detected[: self.max_entities]:
                entity_text = sc.text[start:end].strip()
                if len(entity_text) < self.min_length:
                    continue

                # Embed entity (projection applied if enabled)
                embedding = self.memory.embedding.ner_retriever.embed_entity(sc.text, (start, end))

                entities.append({
                    "entity_text": entity_text,
                    "entity_type": etype,
                    "start": start,
                    "end": end,
                    "embedding": embedding.cpu().numpy().tolist(),
                })
        except Exception as e:
            self.logger.log("EntityExtractionFailed", {"error": str(e), "scorable_id": sc.id})

        return entities
