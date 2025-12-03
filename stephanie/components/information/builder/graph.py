# stephanie/components/information/graph_builder.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from stephanie.services.knowledge_graph_service import KnowledgeGraphService
from stephanie.utils.date_utils import iso_now

from ..models import InformationRequest, InformationResult, InformationSource

log = logging.getLogger(__name__)


class InformationGraphBuilder:
    """
    Take an InformationRequest + InformationResult
    and emit nodes/edges into the KnowledgeGraphService.

    This is intentionally simple and conservative:
      - One Topic node per information MemCube
      - One MemCube node
      - One primary Document node
      - Nodes for each related source
      - A handful of edges connecting them
    """

    def __init__(self, knowledge_graph_service: KnowledgeGraphService, logger=None) -> None:
        self.kg = knowledge_graph_service
        self.logger = logger

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def build_from_information(
        self,
        request: InformationRequest,
        result: InformationResult,
    ) -> None:
        """
        Main entrypoint.

        Safe to call in the same pipeline run or offline batch.
        """
        topic = result.topic
        memcube_id = result.memcube_id

        # If we didn't create a MemCube, we can still create a graph, but
        # it's usually most useful when bound to a MemCube node.
        log.info(
            "InformationGraphBuilder_Start topic=%s memcube_id=%s goal_id=%s casebook_id=%s",
            topic,
            memcube_id,
            result.goal_id,
            result.casebook_id,
        )

        # 1) Resolve all sources: primary + extras
        primary_source = request.sources[0]
        related_sources: List[InformationSource] = []

        # We stored all information sources in extra_data when writing MemCube.
        info_sources = result.extra.get("information_sources", [])
        if not info_sources:
            # Fallback: only primary
            related_sources = []
        else:
            # Rehydrate as InformationSource-like objects (they came from asdict)
            for s in info_sources[1:]:  # skip primary (index 0)
                related_sources.append(
                    InformationSource(
                        kind=s["kind"],
                        id=s["id"],
                        title=s.get("title", ""),
                        text=s.get("text", ""),
                        meta=s.get("meta", {}),
                    )
                )

        # 2) Build / upsert nodes
        topic_node_id = self._ensure_topic_node(topic, request, result)
        memcube_node_id = self._ensure_memcube_node(memcube_id, result)
        primary_doc_node_id = self._ensure_primary_doc_node(primary_source, request)

        # 3) Connect Topic ↔ MemCube ↔ Primary Document
        self._ensure_edge(
            src=topic_node_id,
            dst=memcube_node_id,
            rel_type="REPRESENTED_BY",
            properties={"created_at": iso_now()},
        )
        self._ensure_edge(
            src=memcube_node_id,
            dst=primary_doc_node_id,
            rel_type="SUMMARIZES",
            properties={"created_at": iso_now()},
        )

        # 4) Add related sources as nodes + edges
        for src in related_sources:
            node_id = self._ensure_related_node(src)
            self._ensure_edge(
                src=memcube_node_id,
                dst=node_id,
                rel_type="ENRICHED_BY",
                properties={
                    "source_kind": src.kind,
                    "created_at": iso_now(),
                },
            )

            log.info(
                "InformationGraphBuilder_Done topic=%s memcube_node_id=%s primary_doc_node_id=%s related_count=%d",
                topic,
                memcube_node_id,
                primary_doc_node_id,
                len(related_sources),
            )

    # ------------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------------

    def _ensure_topic_node(
        self,
        topic: str,
        request: InformationRequest,
        result: InformationResult,
    ) -> str:
        slug = self._slugify(topic)
        node_id = f"topic:{slug}"
        props: Dict[str, Any] = {
            "type": "Topic",
            "title": topic,
            "slug": slug,
            "created_at": iso_now(),
        }

        # Optional: domains / tags from target meta
        domains = request.target.meta.get("domains", [])
        if domains:
            props["domains"] = domains

        # This assumes your KnowledgeGraphService exposes an "upsert_node"-style API.
        self.kg.upsert_node(node_id=node_id, properties=props)
        return node_id

    def _ensure_memcube_node(
        self,
        memcube_id: Optional[str],
        result: InformationResult,
    ) -> str:
        node_id = f"memcube:{memcube_id}" if memcube_id else "memcube:unknown"
        props: Dict[str, Any] = {
            "type": "MemCube",
            "memcube_id": memcube_id,
            "topic": result.topic,
            "created_at": iso_now(),
        }
        self.kg.upsert_node(node_id=node_id, properties=props)
        return node_id

    def _ensure_primary_doc_node(
        self,
        primary: InformationSource,
        request: InformationRequest,
    ) -> str:
        # Use doc_id if available, otherwise fall back to source id.
        doc = (request.context or {}).get("document") or {}
        doc_id = str(doc.get("id") or doc.get("doc_id") or primary.id)
        node_id = f"doc:{doc_id}"

        props: Dict[str, Any] = {
            "type": "Document",
            "doc_id": doc_id,
            "title": primary.title or doc.get("title", ""),
            "source_kind": primary.kind,
            "created_at": iso_now(),
        }
        self.kg.upsert_node(node_id=node_id, properties=props)
        return node_id

    def _ensure_related_node(self, s: InformationSource) -> str:
        if s.kind == "arxiv":
            node_id = f"paper:{self._safe_id(s.id)}"
            node_type = "Paper"
        elif s.kind == "wiki":
            node_id = f"wiki:{self._slugify(s.title or s.id)}"
            node_type = "Concept"
        elif s.kind == "web":
            node_id = f"web:{self._safe_id(s.id)}"
            node_type = "WebPage"
        else:
            node_id = f"src:{self._safe_id(s.id)}"
            node_type = "Source"

        props: Dict[str, Any] = {
            "type": node_type,
            "title": s.title,
            "source_kind": s.kind,
            "id": s.id,
            "created_at": iso_now(),
        }
        # Attach a short summary / snippet
        if s.text:
            props["summary"] = s.text[:512]

        # Include some meta fields if present
        for k in ("url", "authors", "published", "score"):
            if k in (s.meta or {}):
                props[k] = s.meta[k]

        self.kg.upsert_node(node_id=node_id, properties=props)
        return node_id

    # ------------------------------------------------------------------
    # Edge helper
    # ------------------------------------------------------------------

    def _ensure_edge(
        self,
        src: str,
        dst: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Thin wrapper. Adjust to match your KnowledgeGraphService API.
        """
        properties = properties or {}
        self.kg.upsert_edge(
            source_id=src,
            target_id=dst,
            rel_type=rel_type,
            properties=properties,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _slugify(self, text: str) -> str:
        import re

        text = (text or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "-", text)
        text = re.sub(r"-+", "-", text).strip("-")
        return text or "topic"

    def _safe_id(self, text: str) -> str:
        # Turn URLs / IDs into safe node ids
        return self._slugify(text)

