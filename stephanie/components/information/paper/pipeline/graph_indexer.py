# stephanie/components/information/paper/pipeline/graph_indexer.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from stephanie.components.information.data import PaperSection
from stephanie.components.information.paper.pipeline.ids import make_section_scorable_id
from stephanie.constants import PIPELINE_RUN_ID

log = logging.getLogger(__name__)


@dataclass
class PaperSectionGraphIndexerConfig:
    enabled: bool = True


class PaperSectionGraphIndexer:
    """
    Index sections into your graph substrates:
      - Nexus graph nodes/edges
      - KG indexing events
    Mirrors PaperPipelineAgent._index_sections_into_graphs + _publish_kg_index_event.
    """

    def __init__(self, *, cfg: Dict[str, Any], memory: Any, logger: Any):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        gcfg = (cfg.get("graph_index", {}) if isinstance(cfg, dict) else {}) or {}
        self.enabled = bool(gcfg.get("enabled", True))

        self.publish_kg_events = bool(gcfg.get("publish_kg_events", True))
        self.nexus_index = bool(gcfg.get("nexus_index", True))

    async def index(self, *, arxiv_id: str, sections: List[PaperSection], context: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        if self.nexus_index:
            self._index_into_nexus(arxiv_id=arxiv_id, sections=sections, context=context)

        if self.publish_kg_events:
            await self._publish_kg_events(arxiv_id=arxiv_id, sections=sections, context=context)

    # ---------------------------- Nexus ---------------------------------


    def _index_into_nexus(self, *, arxiv_id: str, sections: List[PaperSection], context: Dict[str, Any]) -> None:
        nexus = getattr(self.memory, "nexus", None)
        if nexus is None:
            return

        run_id = context.get(PIPELINE_RUN_ID)

        paper_node_id = f"paper:{arxiv_id}"

        # 1) Ensure the paper node exists (as a scorable)
        nexus.upsert_scorable(
            {
                "id": paper_node_id,
                "chat_id": None,
                "turn_index": None,
                "target_type": "paper",
                "text": str(arxiv_id),
                "domains": [],
                "entities": [],
                "meta": {"arxiv_id": str(arxiv_id)},
            }
        ) 

        # 2) Upsert section nodes as scorables + collect edges
        edges = []
        added = 0

        for sec in sections:
            scorable_id = str(make_section_scorable_id(arxiv_id=arxiv_id, section=sec))

            title = getattr(sec, "title", None) or "section"
            summary = getattr(sec, "summary", None) or ""
            text = getattr(sec, "text", None) or ""

            # Choose what Nexus stores as the "display text"
            display = str(title).strip() or "section"

            # Put everything else in meta (including summary + raw text offsets)
            meta = dict(getattr(sec, "meta", None) or {})
            meta.update(
                {
                    "arxiv_id": str(arxiv_id),
                    "section_index": int(getattr(sec, "section_index", 0) or 0),
                    "section_id": getattr(sec, "section_id", None),
                    "role": getattr(sec, "role", None),
                    "summary": str(summary),
                    "start_char": getattr(sec, "start_char", None),
                    "end_char": getattr(sec, "end_char", None),
                }
            )

            # If you have domains/entities already computed, keep them surfaced
            domains = meta.get("domains") or []
            entities = meta.get("entities") or meta.get("ner") or []

            nexus.upsert_scorable(
                {
                    "id": scorable_id,
                    "chat_id": None,
                    "turn_index": None,
                    "target_type": "paper_section",
                    "text": display,
                    "domains": domains,
                    "entities": entities,
                    "meta": meta,
                }
            )  # :contentReference[oaicite:2]{index=2}

            edges.append(
                {
                    "src": paper_node_id,
                    "dst": scorable_id,
                    "type": "paper_has_section",
                    "weight": 1.0,
                    "channels": {"arxiv_id": str(arxiv_id)},
                }
            )
            added += 1

        # 3) Write edges in one batch (fast)
        nexus.write_edges(run_id, edges)  # :contentReference[oaicite:3]{index=3}

        log.info("PaperSectionGraphIndexer: indexed %d section scorables into Nexus", added)


    async def _publish_kg_events(self, *, arxiv_id: str, sections: List[PaperSection], context: Dict[str, Any]) -> None:
        bus = getattr(self.memory, "bus", None) or getattr(self.memory, "event_bus", None)
        if bus is None:
            return

        run_id = context.get(PIPELINE_RUN_ID)

        published = 0
        for sec in sections:
            scorable_id = make_section_scorable_id(arxiv_id=arxiv_id, section=sec)
            payload = {
                "run_id": run_id,
                "arxiv_id": str(arxiv_id),
                "section_id": getattr(sec, "section_id", None) or getattr(sec, "id", None),
                "scorable_id": str(scorable_id),
            }
            try: 
                await bus.publish("kg.index.section", payload)
                published += 1
            except Exception as e:
                log.warning("PaperSectionGraphIndexer: kg publish failed scorable_id=%s error: %s", scorable_id, str(e))

        log.info("PaperSectionGraphIndexer: published %d KG index events", published)

