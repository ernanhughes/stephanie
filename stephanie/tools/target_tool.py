# stephanie/tools/target_tool.py
from __future__ import annotations

import os
import json
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from stephanie.utils.hash_utils import hash_text


def _default_ext(fmt: Optional[str]) -> str:
    fmt = (fmt or "").lower().strip()
    return {
        "markdown": "md",
        "md": "md",
        "html": "html",
        "json": "json",
        "pdf": "pdf",
        "docx": "docx",
        "txt": "txt",
    }.get(fmt, "txt")


def _slugify(text: str) -> str:
    s = (text or "").strip().lower()
    s = "".join(ch if ch.isalnum() else "-" for ch in s)
    s = "-".join([p for p in s.split("-") if p])
    return s[:80] or "target"


@dataclass
class TargetRecord:
    target_id: int
    target_uri: str
    target_type: str
    target_format: Optional[str]
    title: Optional[str]
    content_hash: Optional[str]


class TargetTool:
    """
    A provenance-aware artifact creator/registrar.

    Requires memory:
      - memory.targets: TargetStore
      - (optional) memory.target_inputs: TargetInputStore
      - (optional) memory.sources: SourceStore (for validating source ids)
    """

    def __init__(self, memory: Any, logger: Any, *, run_dir_tpl: str = "runs/paper_blogs/${run_id}") -> None:
        self.memory = memory
        self.logger = logger
        self.run_dir_tpl = run_dir_tpl

    def resolve_run_dir(self, *, run_id: int) -> str:
        return (self.run_dir_tpl or "runs/paper_blogs/${run_id}").replace("${run_id}", str(run_id))

    def create_target_from_content(
        self,
        *,
        pipeline_run_id: int,
        target_type: str,
        title: str,
        content: str,
        target_format: str = "markdown",
        run_dir: Optional[str] = None,
        root_node_type: Optional[str] = None,
        root_node_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        input_source_ids: Optional[Sequence[int]] = None,
        input_relation_type: str = "derived_from",
    ) -> TargetRecord:
        meta = meta or {}
        ext = _default_ext(target_format)
        slug = _slugify(title)
        content_hash = hash_text(content)

        run_dir = run_dir or self.resolve_run_dir(run_id=pipeline_run_id)
        out_dir = os.path.join(run_dir, "targets", target_type)
        os.makedirs(out_dir, exist_ok=True)

        # deterministic filename (stable across retries)
        fname = f"{slug}_{content_hash[:10]}.{ext}"
        fpath = os.path.join(out_dir, fname)

        # write artifact
        if target_format.lower() in ("json",):
            try:
                obj = json.loads(content)
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=2)
            except Exception:
                # fallback: raw
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)
        else:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)

        target_uri = fpath  # keep as local path (you can add file:// later if you want)

        target_id = self.memory.targets.insert_target(
            pipeline_run_id=int(pipeline_run_id),
            target_type=target_type,
            target_format=target_format,
            title=title,
            target_uri=target_uri,
            canonical_uri=target_uri,
            status="created",
            content_hash=content_hash,
            root_node_type=root_node_type,
            root_node_id=root_node_id,
            meta={
                **meta,
                "slug": slug,
                "ext": ext,
            },
        )

        # link inputs (optional)
        self._link_inputs(
            target_id=target_id,
            source_ids=list(input_source_ids or []),
            relation_type=input_relation_type,
        )

        return TargetRecord(
            target_id=target_id,
            target_uri=target_uri,
            target_type=target_type,
            target_format=target_format,
            title=title,
            content_hash=content_hash,
        )

    def register_existing_file(
        self,
        *,
        pipeline_run_id: int,
        target_type: str,
        target_uri: str,
        title: Optional[str] = None,
        target_format: Optional[str] = None,
        root_node_type: Optional[str] = None,
        root_node_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        input_source_ids: Optional[Sequence[int]] = None,
        input_relation_type: str = "derived_from",
    ) -> TargetRecord:
        meta = meta or {}

        # best effort content hash (if file exists)
        content_hash = None
        try:
            if os.path.exists(target_uri) and os.path.isfile(target_uri):
                with open(target_uri, "rb") as f:
                    content_hash = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            content_hash = None

        if target_format is None:
            # infer from file ext
            _, ext = os.path.splitext(target_uri)
            ext = (ext or "").lower().lstrip(".")
            target_format = {
                "md": "markdown",
                "html": "html",
                "json": "json",
                "pdf": "pdf",
                "docx": "docx",
                "txt": "txt",
            }.get(ext, "unknown")

        target_id = self.memory.targets.insert_target(
            pipeline_run_id=int(pipeline_run_id),
            target_type=target_type,
            target_format=target_format,
            title=title,
            target_uri=target_uri,
            canonical_uri=target_uri,
            status="created",
            content_hash=content_hash,
            root_node_type=root_node_type,
            root_node_id=root_node_id,
            meta=meta,
        )

        self._link_inputs(
            target_id=target_id,
            source_ids=list(input_source_ids or []),
            relation_type=input_relation_type,
        )

        return TargetRecord(
            target_id=target_id,
            target_uri=target_uri,
            target_type=target_type,
            target_format=target_format,
            title=title,
            content_hash=content_hash,
        )

    # -------------------------
    # internals
    # -------------------------

    def _link_inputs(self, *, target_id: int, source_ids: List[int], relation_type: str) -> None:
        if not source_ids:
            return
        for sid in source_ids:
            try:
                self.memory.target.link(
                    target_id=int(target_id),
                    source_id=int(sid),
                    relation_type=relation_type,
                )
            except Exception as e:
                self.logger.log("TargetInputLinkFailed", {"target_id": target_id, "source_id": sid, "error": str(e)})
