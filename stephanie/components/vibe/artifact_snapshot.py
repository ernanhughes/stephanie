#stephanie/components/vibe/artifact_snapshot.py
from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Artifact types
# ---------------------------------------------------------------------------

class ArtifactType(str, Enum):
    """
    High-level type of a cognitive artifact.

    This is intentionally broad: we care more about how to turn the artifact
    into text (for scoring / LLMs) than about its exact internal schema.
    """

    TEXT = "text"              # plain or lightly formatted text
    MARKDOWN = "markdown"      # blog sections, research notes
    CODE = "code"              # source code (Python, etc.)
    PLAN_TRACE = "plan_trace"  # reasoning traces / ExecutionStep sequences
    GRAPH = "graph"            # Nexus / knowledge graphs
    JSON = "json"              # JSON-like structured data
    VPM_IMAGE = "vpm_image"    # visual policy maps / PNG paths
    OTHER = "other"            # fallback


# ---------------------------------------------------------------------------
# Provenance / location metadata
# ---------------------------------------------------------------------------

@dataclass
class ArtifactLocation:
    """
    Optional provenance for where this artifact came from.

    This is deliberately generic so you can attach:
      - conversation + turn ids
      - file paths
      - Nexus node ids
      - MemCube ids
    """

    source: str = ""              # e.g. "chat_history", "blog_repo", "nexus"
    conversation_id: Optional[str] = None
    turn_index: Optional[int] = None
    file_path: Optional[str] = None
    node_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Artifact snapshot
# ---------------------------------------------------------------------------

@dataclass
class ArtifactSnapshot:
    """
    A normalized snapshot of a cognitive artifact at a point in time.

    Key design goals:
      - Generic enough to wrap text, code, PlanTraces, graphs, etc.
      - Always able to produce a textual representation via `to_text()`.
      - Carry enough metadata to support diffing, scoring, and provenance.

    Fields:
      - artifact_id: stable id within your system (optional but recommended)
      - type: broad artifact type (see ArtifactType)
      - content: raw / primary representation (string, dict, object, etc.)
      - meta: arbitrary metadata (dimensions, domains, tags, etc.)
      - location: provenance information
      - created_at: timestamp for this snapshot
    """

    artifact_id: str
    type: ArtifactType
    content: Any

    meta: Dict[str, Any] = field(default_factory=dict)
    location: Optional[ArtifactLocation] = None
    created_at: _dt.datetime = field(default_factory=lambda: _dt.datetime.utcnow())

    # Optional cached text representation (to avoid recomputing)
    _text_cache: Optional[str] = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def to_text(self, max_len: Optional[int] = None) -> str:
        """
        Return a textual representation of this artifact suitable for:
          - LLM prompts
          - Vibe / rubric scoring
          - logging / debugging

        `max_len` (if provided) soft-truncates the result.
        """

        if self._text_cache is not None:
            text = self._text_cache
        else:
            text = self._materialize_text()
            self._text_cache = text

        if max_len is not None and max_len > 0 and len(text) > max_len:
            return text[: max_len - 3] + "..."
        return text

    def short_label(self, max_len: int = 80) -> str:
        """
        Human-friendly one-line description for logs / UIs.
        """
        base = self.meta.get("title") or self.meta.get("name") or ""
        if not base:
            base = self.to_text(max_len=max_len).replace("\n", " ")
        prefix = f"[{self.type.value}] "
        s = prefix + base
        if len(s) > max_len:
            return s[: max_len - 3] + "..."
        return s

    def with_meta(self, **extra: Any) -> ArtifactSnapshot:
        """
        Return a shallow copy with updated meta.
        """
        meta = dict(self.meta)
        meta.update(extra)
        return ArtifactSnapshot(
            artifact_id=self.artifact_id,
            type=self.type,
            content=self.content,
            meta=meta,
            location=self.location,
            created_at=self.created_at,
        )

    def to_dict(self, include_content: bool = True) -> Dict[str, Any]:
        """
        Serialize snapshot to a JSON-serializable dict.

        By default includes `content` as-is (may need special handling if the
        content is a complex object).
        """
        d: Dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "type": self.type.value,
            "meta": self.meta,
            "location": self.location.to_dict() if self.location else None,
            "created_at": self.created_at.isoformat() + "Z",
        }
        if include_content:
            d["content"] = self._serialize_content()
        return d

    # ------------------------------------------------------------------
    # Internal helpers for text materialization / serialization
    # ------------------------------------------------------------------

    def _materialize_text(self) -> str:
        """
        Convert `content` into a string, depending on type.

        This is intentionally conservative: if we don't recognize the type,
        we fall back to `str(content)` / JSON pretty-printing where possible.
        """

        c = self.content

        try:
            if self.type in (ArtifactType.TEXT, ArtifactType.MARKDOWN, ArtifactType.CODE):
                if isinstance(c, str):
                    return c
                return str(c)

            if self.type == ArtifactType.JSON:
                if isinstance(c, (dict, list)):
                    return json.dumps(c, indent=2, ensure_ascii=False)
                return str(c)

            if self.type == ArtifactType.PLAN_TRACE:
                # Best effort: use custom render methods if present
                if hasattr(c, "to_markdown") and callable(c.to_markdown):
                    return c.to_markdown()
                if hasattr(c, "to_text") and callable(c.to_text):
                    return c.to_text()
                if hasattr(c, "__dict__"):
                    return json.dumps(c.__dict__, indent=2, default=str, ensure_ascii=False)
                return str(c)

            if self.type == ArtifactType.GRAPH:
                # Expect something like a networkx graph or custom Nexus graph.
                # We avoid importing heavy libs here; just summarize.
                return self._summarize_graph(c)

            if self.type == ArtifactType.VPM_IMAGE:
                # For VPMs, the "text" is a short description + path.
                path = None
                if isinstance(c, dict):
                    path = c.get("path") or c.get("file")
                elif isinstance(c, str):
                    path = c
                desc = self.meta.get("description") or "VPM image"
                return f"{desc}\n\nVPM file: {path or '<unknown>'}"

            # Fallback for OTHER or unknown types
            if isinstance(c, (dict, list)):
                try:
                    return json.dumps(c, indent=2, ensure_ascii=False, default=str)
                except Exception:
                    return str(c)

            return str(c)

        except Exception:
            log.exception("Failed to materialize text for ArtifactSnapshot %s", self.artifact_id)
            return str(c)

    def _serialize_content(self) -> Any:
        """
        Best-effort serialization of content for JSON logging / storage.
        """
        c = self.content
        if isinstance(c, (str, int, float, bool)) or c is None:
            return c
        if isinstance(c, (dict, list)):
            try:
                # Validate it's JSON-serializable
                json.dumps(c, default=str)
                return c
            except Exception:
                pass
        # Fallback: string representation
        return str(c)

    @staticmethod
    def _summarize_graph(graph: Any) -> str:
        """
        Very lightweight graph summarization to text.
        Avoids importing any specific graph libraries.
        """
        try:
            # Heuristics for common graph-like structures
            if hasattr(graph, "number_of_nodes") and hasattr(graph, "number_of_edges"):
                n = graph.number_of_nodes()
                m = graph.number_of_edges()
                return f"Graph with {n} nodes and {m} edges."

            # adjacency dict: {node: [neighbors]}
            if isinstance(graph, dict):
                nodes = list(graph.keys())
                n = len(nodes)
                edges = sum(len(v) for v in graph.values())
                preview_nodes = ", ".join(str(x) for x in nodes[:5])
                return (
                    f"Graph-like dict with {n} nodes and ~{edges} edges.\n"
                    f"Example nodes: {preview_nodes}"
                )

            return f"Graph-like object: {repr(graph)[:200]}"

        except Exception:
            log.exception("Failed to summarize graph artifact")
            return f"Graph-like object: {repr(graph)[:200]}"


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def new_artifact_id(prefix: str = "artifact") -> str:
    """
    Simple id generator. Replace with your own id service if desired.
    """
    # You can swap this for a ULID/UUID generator centrally later.
    ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    return f"{prefix}_{ts}"


def snapshot_text(
    text: str,
    *,
    artifact_id: Optional[str] = None,
    markdown: bool = False,
    meta: Optional[Dict[str, Any]] = None,
    location: Optional[ArtifactLocation] = None,
) -> ArtifactSnapshot:
    """
    Convenience factory for text / markdown snapshots.
    """
    return ArtifactSnapshot(
        artifact_id=artifact_id or new_artifact_id("text"),
        type=ArtifactType.MARKDOWN if markdown else ArtifactType.TEXT,
        content=text,
        meta=meta or {},
        location=location,
    )


def snapshot_code(
    code: str,
    *,
    artifact_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    location: Optional[ArtifactLocation] = None,
) -> ArtifactSnapshot:
    """
    Convenience factory for code snapshots.
    """
    return ArtifactSnapshot(
        artifact_id=artifact_id or new_artifact_id("code"),
        type=ArtifactType.CODE,
        content=code,
        meta=meta or {},
        location=location,
    )


def snapshot_json(
    data: Union[Dict[str, Any], List[Any]],
    *,
    artifact_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    location: Optional[ArtifactLocation] = None,
) -> ArtifactSnapshot:
    """
    Convenience factory for JSON-like snapshots.
    """
    return ArtifactSnapshot(
        artifact_id=artifact_id or new_artifact_id("json"),
        type=ArtifactType.JSON,
        content=data,
        meta=meta or {},
        location=location,
    )


def snapshot_plan_trace(
    trace: Any,
    *,
    artifact_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    location: Optional[ArtifactLocation] = None,
) -> ArtifactSnapshot:
    """
    Convenience factory for PlanTrace snapshots.

    `trace` can be your PlanTrace object or any structure with `to_markdown()`
    / `to_text()` methods; ArtifactSnapshot will try to use those when
    materializing text.
    """
    return ArtifactSnapshot(
        artifact_id=artifact_id or new_artifact_id("plan"),
        type=ArtifactType.PLAN_TRACE,
        content=trace,
        meta=meta or {},
        location=location,
    )


def snapshot_graph(
    graph: Any,
    *,
    artifact_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    location: Optional[ArtifactLocation] = None,
) -> ArtifactSnapshot:
    """
    Convenience factory for graph snapshots (e.g., Nexus graphs).
    """
    return ArtifactSnapshot(
        artifact_id=artifact_id or new_artifact_id("graph"),
        type=ArtifactType.GRAPH,
        content=graph,
        meta=meta or {},
        location=location,
    )


def snapshot_vpm_image(
    path: str,
    *,
    artifact_id: Optional[str] = None,
    description: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    location: Optional[ArtifactLocation] = None,
) -> ArtifactSnapshot:
    """
    Convenience factory for VPM image snapshots.
    """
    m = dict(meta or {})
    if description:
        m.setdefault("description", description)
    return ArtifactSnapshot(
        artifact_id=artifact_id or new_artifact_id("vpm"),
        type=ArtifactType.VPM_IMAGE,
        content={"path": path},
        meta=m,
        location=location,
    )
