# stephanie/components/information/processor.py
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional

from stephanie.utils.date_utils import iso_now
import os

from .models import (
    InformationSource,
    InformationTargetConfig,
    InformationRequest,
    InformationResult,
)
import re
from pathlib import Path

class InformationProcessor:
    """
    Core 'information builder' for the current iteration.

    Responsibilities (current minimal version):
      - Take an InformationRequest (sources + target + context).
      - Build a simple blog-style markdown page from the primary source.
      - Persist that page as a MemCube (if memory.memcubes is available).
      - Return an InformationResult with memcube_id + markdown.

    This is intentionally conservative:
      - No external web calls.
      - No hard dependency on scoring or buckets.
    We can gradually swap in richer behaviour (buckets, scoring, LLMs)
    without breaking the agent interface.
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger) -> None:
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger

        # Try to locate the MemCube store if one exists on memory.
        # In your current system this is usually memory.memcubes.
        self.memcubes = getattr(memory, "memcubes", None) or getattr(
            memory, "memcube_store", None
        )

        # How many characters of source text to include by default in the page.
        self.max_chars: int = int(self.cfg.get("max_chars", 8000))

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    async def process(self, request: InformationRequest) -> InformationResult:
        """
        Main entrypoint used by InformationIngestAgent.

        Steps:
          1. Choose a primary source and topic.
          2. Build blog-style markdown from that source.
          3. Optionally create/update a MemCube row.
          4. Return InformationResult.
        """
        if not request.sources:
            raise ValueError("InformationProcessor.process: no sources provided")

        primary = request.sources[0]
        target = request.target

        topic = target.name or primary.title or primary.id or "Untitled"
        goal_id = target.goal_id
        casebook_id = target.casebook_id

        if self.logger:
            self.logger.log(
                "InformationProcessor_Start",
                {
                    "topic": topic,
                    "source_kind": primary.kind,
                    "source_id": primary.id,
                    "goal_id": goal_id,
                    "casebook_id": casebook_id,
                },
            )

        # 1) Build the markdown page (no LLM yet; pure deterministic transform)
        blog_markdown = self._build_markdown(topic, primary, request)

        # 2) Persist as MemCube (if store is available)
        memcube_id = None
        if self.memcubes is not None and target.kind.lower() == "memcube":
            memcube_id = self._upsert_memcube(topic, primary, blog_markdown, request)

        # 3) Export markdown to file (optional)
        markdown_path = self._export_markdown_file(
            topic=topic,
            blog_markdown=blog_markdown,
            primary=primary,
            request=request,
        )

        # 4) Construct result
        result = InformationResult(
            memcube_id=memcube_id,
            bucket_id=None,  # placeholder for future bucket / Nexus integration
            blog_markdown=blog_markdown,
            topic=topic,
            goal_id=goal_id,
            casebook_id=casebook_id,
            markdown_path=markdown_path,
            extra={
                "source_meta": primary.meta,
                "target_meta": target.meta,
                "created_at":  iso_now(),
            },
        )

        if self.logger:
            self.logger.log(
                "InformationProcessor_Done",
                {
                    "topic": topic,
                    "memcube_id": memcube_id,
                    "markdown_len": len(blog_markdown),
                },
            )

        return result

    # ------------------------------------------------------------------
    # Markdown builder
    # ------------------------------------------------------------------

    def _slugify(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"[^a-z0-9]+", "-", text)
        text = re.sub(r"-+", "-", text).strip("-")
        return text or "post"

    def _build_front_matter(
        self,
        topic: str,
        primary: InformationSource,
        request: InformationRequest,
    ) -> str:
        """
        Build Hugo-style TOML front matter, matching your existing posts.

        Example:

        +++
        date = '2025-08-11T22:41:43+01:00'
        draft = false
        title = "ZeroModel: Visual AI you can scrutinize"
        thumbnail = '/img/zeromodel.png'
        categories = [...]
        tags = [...]
        +++
        """
        now_iso = datetime.utcnow().isoformat() + "Z"
        target = request.target
        meta = target.meta or {}

        # fallback fields
        title = topic
        thumbnail = meta.get("thumbnail", "/img/default.png")
        categories = meta.get("categories", ["Information", "Stephanie"])
        tags = meta.get("tags", meta.get("domains", [])) or ["information_ingest"]

        # TOML front matter
        lines = [
            "+++",
            f"date = '{now_iso}'",
            "draft = true",  # first version is always a draft
            f"title = '{title}'",
            f"thumbnail = '{thumbnail}'",
            "+++",
        ]

        if categories:
            cat_items = ", ".join(f'"{c}"' for c in categories)
            lines.append(f"categories = [{cat_items}]")

        if tags:
            tag_items = ", ".join(f'"{t}"' for t in tags)
            lines.append(f"tags = [{tag_items}]")

        lines.append("+++")
        return "\n".join(lines)

    def _export_markdown_file(
        self,
        topic: str,
        blog_markdown: str,
        primary: InformationSource,
        request: InformationRequest,
    ) -> Optional[str]:
        """
        If enabled in cfg, write a Hugo-friendly .md file to disk and return its path.

        Config keys used:
          - export_markdown: bool
          - export_dir: str (directory for generated posts)
        """
        if not self.cfg.get("export_markdown", False):
            return None

        export_dir = self.cfg.get("export_dir", "blog_drafts")
        out_dir = Path(export_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        slug = self._slugify(topic)
        date_prefix = datetime.utcnow().strftime("%Y-%m-%d")
        filename = f"{date_prefix}-{slug}.md"
        path = out_dir / filename

        front_matter = self._build_front_matter(topic, primary, request)
        full_text = f"{front_matter}\n\n{blog_markdown}"

        os.makedirs(out_dir, exist_ok=True)
        path.write_text(full_text, encoding="utf-8")
        print(f"Exported markdown to {path}")

        if self.logger:
            self.logger.log(
                "InformationProcessor_MarkdownExported",
                {"topic": topic, "path": str(path)},
            )

        return str(path)

    def _build_markdown(
        self,
        topic: str,
        primary: InformationSource,
        request: InformationRequest,
    ) -> str:
        """
        Build a first-pass blog-style page from the primary source.

        This is deliberately simple and deterministic so that:
          - You can run it without any LLM backing.
          - We get something usable immediately.
        Later, we can replace this body generation with a call into your
        prompt engine / DSPy / Supervisor stack.
        """
        # Header
        lines = [f"# {topic}", ""]

        # Optional metadata block
        meta = primary.meta or {}
        if meta:
            lines.append("> **Metadata**")
            for k, v in meta.items():
                lines.append(f"> - **{k}**: {v}")
            lines.append("")

        # Original source hint
        lines.append(f"> _Source kind_: **{primary.kind}**  ")
        lines.append(f"> _Source id_: `{primary.id}`")
        lines.append("")

        # Body text (truncated if necessary)
        text = primary.text or ""
        if self.max_chars and len(text) > self.max_chars:
            body = text[: self.max_chars] + "\n\n*(truncated)*"
        else:
            body = text

        # Simple separator
        lines.append("---")
        lines.append("")
        lines.append(body)

        # Optionally include some context footer
        ctx = request.context or {}
        footer_bits = []
        goal = ctx.get("goal") or {}
        if goal.get("goal_text"):
            footer_bits.append(f"- Goal: {goal['goal_text']}")
        if "pipeline_run_id" in ctx:
            footer_bits.append(f"- Pipeline run: {ctx['pipeline_run_id']}")

        if footer_bits:
            lines.append("")
            lines.append("---")
            lines.append("**Pipeline context**")
            for line in footer_bits:
                lines.append(line)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # MemCube writer
    # ------------------------------------------------------------------

    def _upsert_memcube(
        self,
        topic: str,
        primary: InformationSource,
        blog_markdown: str,
        request: InformationRequest,
    ) -> Optional[str]:
        """
        Create or update a MemCube row representing this information object.

        We try to be conservative and compatible with your existing MemCubeORM:
          - scorable_id: best-effort integer (or 0 if unknown)
          - scorable_type: 'document' by default
          - content: original source text
          - refined_content: blog_markdown
          - dimension: 'information_page' (configurable via cfg["dimension"])
        """
        if self.memcubes is None:
            return None

        # Try to extract an integer-ish scorable_id from context or meta.
        scorable_id = 0
        doc = (request.context or {}).get("document") or {}
        doc_id = doc.get("id") or doc.get("doc_id") or primary.meta.get("doc_id")
        try:
            if doc_id is not None:
                scorable_id = int(doc_id)
        except (TypeError, ValueError):
            scorable_id = 0  # safe fallback

        dimension = self.cfg.get("dimension", "information_page")
        source = request.target.meta.get("source", "information_ingest") if request.target.meta else "information_ingest"

        data: Dict[str, Any] = {
            "scorable_id": scorable_id,
            "scorable_type": "document",
            "content": primary.text or "",
            "dimension": dimension,
            "original_score": None,
            "refined_score": None,
            "refined_content": blog_markdown,
            "version": "v1",
            "source": source,
            "model": None,
            "priority": int(self.cfg.get("priority", 5)),
            "sensitivity": self.cfg.get("sensitivity", "public"),
            "ttl": self.cfg.get("ttl"),
            "usage_count": 0,
            "extra_data": {
                "topic": topic,
                "information_target": asdict(request.target),
                "information_sources": [asdict(s) for s in request.sources],
            },
        }

        cube = self.memcubes.upsert(data)
        return cube.id if cube is not None else None
