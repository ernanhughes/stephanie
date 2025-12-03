# stephanie/components/information/processor.py
from __future__ import annotations

import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from stephanie.tools.arxiv_tool import search_arxiv
from stephanie.tools.web_search import WebSearchTool
from stephanie.tools.wikipedia_tool import WikipediaTool
from stephanie.utils.time_utils import now_iso

from .models import (InformationRequest, InformationResult, InformationSource,
                     InformationTargetConfig)


class InformationProcessor:
    """
    Core 'information builder' for the current iteration.

    Responsibilities:
      - Take an InformationRequest (sources + target + context).
      - Treat the first source as the PRIMARY document.
      - Optionally fetch **similar / complementary** information
        from arXiv, Wikipedia, and the web.
      - Build a multi-section blog-style markdown page.
      - Persist that page as a MemCube (if memory.memcubes is available).
      - Optionally export a Hugo-ready .md file on disk.
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger) -> None:
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger

        # Try to locate the MemCube store if one exists on memory.
        self.memcubes = getattr(memory, "memcubes", None) or getattr(
            memory, "memcube_store", None
        )

        # How many characters of primary text to include by default in the page.
        self.max_chars: int = int(self.cfg.get("max_chars", 8000))

        # Enrichment config (what "similar data" to pull in).
        enrich = self.cfg.get("enrich", {})
        self.enrich_arxiv: bool = bool(enrich.get("enable_arxiv", True))
        # Wikipedia enrichment is off by default to reduce API load and generally poor results.
        self.enrich_wiki: bool = bool(enrich.get("enable_wikipedia", False))
        self.enrich_web: bool = bool(enrich.get("enable_web", True))

        self.max_results_arxiv: int = int(enrich.get("max_results_arxiv", 3))
        self.max_results_wiki: int = int(enrich.get("max_results_wiki", 2))
        self.max_results_web: int = int(enrich.get("max_results_web", 3))

        # Markdown export config
        self.export_markdown: bool = bool(self.cfg.get("export_markdown", False))
        self.export_dir: str = self.cfg.get("export_dir", "blog_drafts")

        # Optional extra fields for MemCube
        self.dimension: str = self.cfg.get("dimension", "information_page")
        self.default_priority: int = int(self.cfg.get("priority", 5))
        self.default_sensitivity: str = self.cfg.get("sensitivity", "public")

        # Tools
        self.wiki_tool = WikipediaTool(memory=memory, logger=logger, lang="en")
        self.web_tool = WebSearchTool(
            cfg=self.cfg.get("web_search", {}), logger=logger
        )

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    async def process(self, request: InformationRequest) -> InformationResult:
        """
        Main entrypoint used by InformationIngestAgent.

        Steps:
          1. Choose a primary source and topic.
          2. Fetch related sources (papers / wiki / web) as configured.
          3. Build a multi-source markdown page.
          4. Create/update a MemCube row.
          5. Optionally export a Hugo-ready .md file.
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

        # 1) Fetch complementary / similar data
        related_sources = await self._gather_related_sources(topic, primary)

        # All sources (primary + related) for extra_data, debugging, etc.
        all_sources: List[InformationSource] = [primary] + related_sources

        # 2) Build the markdown page from all sources
        blog_markdown = self._build_markdown(topic, primary, related_sources, request)

        # 3) Persist as MemCube (if store is available)
        memcube_id = None
        if self.memcubes is not None and target.kind.lower() == "memcube":
            memcube_id = self._upsert_memcube(
                topic=topic,
                primary=primary,
                sources=all_sources,
                blog_markdown=blog_markdown,
                request=request,
            )

        # 4) Export markdown to Hugo-friendly file (optional)
        markdown_path = self._export_markdown_file(
            topic=topic,
            blog_markdown=blog_markdown,
            primary=primary,
            sources=all_sources,
            request=request,
        )

        # 5) Construct result
        result = InformationResult(
            memcube_id=memcube_id,
            bucket_id=None,  # placeholder for future Nexus/bucket integration
            blog_markdown=blog_markdown,
            topic=topic,
            goal_id=goal_id,
            casebook_id=casebook_id,
            markdown_path=markdown_path,
            extra={
                "source_meta": primary.meta,
                "target_meta": target.meta,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "markdown_path": markdown_path,
                "related_counts": {
                    "total": len(related_sources),
                    "arxiv": sum(1 for s in related_sources if s.kind == "arxiv"),
                    "wiki": sum(1 for s in related_sources if s.kind == "wiki"),
                    "web": sum(1 for s in related_sources if s.kind == "web"),
                },
            },
        )

        if self.logger:
            self.logger.log(
                "InformationProcessor_Done",
                {
                    "topic": topic,
                    "memcube_id": memcube_id,
                    "markdown_len": len(blog_markdown),
                    "markdown_path": markdown_path,
                },
            )

        return result

    # ------------------------------------------------------------------
    # Enrichment: "complement this data with similar data"
    # ------------------------------------------------------------------

    async def _gather_related_sources(
        self,
        topic: str,
        primary: InformationSource,
    ) -> List[InformationSource]:
        """
        Use existing tools (arXiv, Wikipedia, web search) to fetch
        *similar* or *complementary* information to the primary source.

        This is deliberately shallow and safe:
          - small, configurable max results
          - short summaries, not full PDFs
        """
        related: List[InformationSource] = []

        # Use the topic / primary title as the query
        query = primary.title or topic

        # 1) arXiv related papers
        if self.enrich_arxiv:
            try:
                arxiv_results = search_arxiv([query], max_results=self.max_results_arxiv)
                for r in arxiv_results:
                    text = f"{r.get('summary', '')}\n\nAuthors: {', '.join(r.get('authors', []))}"
                    related.append(
                        InformationSource(
                            kind="arxiv",
                            id=r.get("url", ""),
                            title=r.get("title", "arxiv paper"),
                            text=text,
                            meta={
                                "authors": r.get("authors", []),
                                "published": r.get("published"),
                                "url": r.get("url"),
                            },
                        )
                    )
            except Exception as e:
                if self.logger:
                    self.logger.log(
                        "InformationProcessor_ArxivError",
                        {"topic": topic, "error": repr(e)},
                    )

        # 2) Wikipedia context
        if self.enrich_wiki:
            try:
                wiki_results = self.wiki_tool.search(query)
                for w in wiki_results[: self.max_results_wiki]:
                    related.append(
                        InformationSource(
                            kind="wiki",
                            id=w.get("url", w.get("title", "")),
                            title=w.get("title", "Wikipedia article"),
                            text=w.get("summary", ""),
                            meta={
                                "score": w.get("score"),
                                "url": w.get("url"),
                            },
                        )
                    )
            except Exception as e:
                if self.logger:
                    self.logger.log(
                        "InformationProcessor_WikiError",
                        {"topic": topic, "error": repr(e)},
                    )

        # 3) Web search (good blog posts / explainers)
        if self.enrich_web:
            try:
                web_results = await self.web_tool.search(query)
                for w in web_results[: self.max_results_web]:
                    url = w.get("url")
                    snippet = w.get("snippet", "")
                    title = w.get("title", url or "Web result")

                    # Optionally fetch full text when configured
                    text = snippet
                    if self.web_tool.fetch_page and url:
                        try:
                            page = self.web_tool.fetch_and_parse_readable(url)
                            text = page.get("text") or snippet or ""
                            title = page.get("title") or title
                        except Exception:
                            # fall back to snippet-only
                            pass

                    related.append(
                        InformationSource(
                            kind="web",
                            id=url or title,
                            title=title,
                            text=text,
                            meta={
                                "url": url,
                                "snippet": snippet,
                            },
                        )
                    )
            except Exception as e:
                if self.logger:
                    self.logger.log(
                        "InformationProcessor_WebError",
                        {"topic": topic, "error": repr(e)},
                    )

        return related

    # ------------------------------------------------------------------
    # Markdown builder
    # ------------------------------------------------------------------

    def _build_markdown(
        self,
        topic: str,
        primary: InformationSource,
        related_sources: List[InformationSource],
        request: InformationRequest,
    ) -> str:
        """
        Build a first-pass blog-style page from the primary + related sources.

        This stays deterministic for now (no LLM) so the pipeline is testable
        and cheap. Later we can swap the inner sections out for an LLM-backed
        "knowledge distillation" pass.
        """
        lines: List[str] = []

        # H1 title
        lines.append(f"# {topic}")
        lines.append("")

        # Simple metadata block based on primary
        meta = primary.meta or {}
        if meta:
            lines.append("> **Primary source metadata**")
            for k, v in meta.items():
                lines.append(f"> - **{k}**: {v}")
            lines.append("")

        lines.append(f"> _Primary kind_: **{primary.kind}**  ")
        lines.append(f"> _Primary id_: `{primary.id}`")
        lines.append("")

        # --- Primary content section ---
        lines.append("---")
        lines.append("")
        lines.append("## 1. Primary Paper / Document")
        lines.append("")

        text = primary.text or ""
        if self.max_chars and len(text) > self.max_chars:
            body = text[: self.max_chars] + "\n\n*(truncated)*"
        else:
            body = text

        lines.append(body)
        lines.append("")

        # --- Related sources grouped by kind ---
        if related_sources:
            lines.append("---")
            lines.append("")
            lines.append("## 2. Related Material (Auto-Discovered)")
            lines.append("")

            # Group by kind for readability
            by_kind: Dict[str, List[InformationSource]] = {}
            for s in related_sources:
                by_kind.setdefault(s.kind, []).append(s)

            # arXiv
            if by_kind.get("arxiv"):
                lines.append("### 2.1 Related Papers (arXiv)")
                lines.append("")
                for s in by_kind["arxiv"]:
                    url = s.meta.get("url", s.id)
                    authors = s.meta.get("authors", [])
                    published = s.meta.get("published")
                    lines.append(f"#### {s.title}")
                    if authors:
                        lines.append(f"- Authors: {', '.join(authors)}")
                    if published:
                        lines.append(f"- Published: {published}")
                    if url:
                        lines.append(f"- Link: {url}")
                    lines.append("")
                    lines.append(s.text)
                    lines.append("")

            # Wikipedia
            if by_kind.get("wiki"):
                lines.append("### 2.2 Background Concepts (Wikipedia)")
                lines.append("")
                for s in by_kind["wiki"]:
                    url = s.meta.get("url", s.id)
                    score = s.meta.get("score")
                    lines.append(f"#### {s.title}")
                    if score is not None:
                        lines.append(f"- Similarity score: {score}")
                    if url:
                        lines.append(f"- Link: {url}")
                    lines.append("")
                    lines.append(s.text)
                    lines.append("")

            # Web
            if by_kind.get("web"):
                lines.append("### 2.3 Web Articles & Blog Posts")
                lines.append("")
                for s in by_kind["web"]:
                    url = s.meta.get("url", s.id)
                    snippet = s.meta.get("snippet", "")
                    lines.append(f"#### {s.title}")
                    if url:
                        lines.append(f"- Link: {url}")
                    if snippet:
                        lines.append(f"- Search snippet: {snippet}")
                    lines.append("")
                    # Keep body short; snippets are usually enough.
                    body = s.text
                    if len(body) > 1500:
                        body = body[:1500] + "\n\n*(truncated)*"
                    lines.append(body)
                    lines.append("")

        # Optional pipeline footer
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
    # Markdown export (Hugo-ready .md file)
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
        """
        now_iso_str = now_iso()
        target = request.target
        meta = target.meta or {}

        title = topic
        thumbnail = meta.get("thumbnail", "/img/default.png")
        categories = meta.get("categories", ["Information", "Stephanie"])
        tags = meta.get("tags", meta.get("domains", [])) or ["information_ingest"]

        lines = [
            "+++",
            f"date = '{now_iso_str}'",
            "draft = true",  # first version always draft
            f'title = "{title}"',
            f"thumbnail = '{thumbnail}'",
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
        sources: List[InformationSource],
        request: InformationRequest,
    ) -> Optional[str]:
        """
        If enabled in cfg, write a Hugo-friendly .md file to disk and return its path.
        """
        if not self.export_markdown:
            return None

        out_dir = Path(self.export_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        slug = self._slugify(topic)
        date_prefix = datetime.utcnow().strftime("%Y-%m-%d")
        filename = f"{date_prefix}-{slug}.md"
        path = out_dir / filename

        front_matter = self._build_front_matter(topic, primary, request)
        full_text = f"{front_matter}\n\n{blog_markdown}"

        path.write_text(full_text, encoding="utf-8")

        if self.logger:
            self.logger.log(
                "InformationProcessor_MarkdownExported",
                {"topic": topic, "path": str(path)},
            )

        return str(path)

    # ------------------------------------------------------------------
    # MemCube writer
    # ------------------------------------------------------------------

    def _upsert_memcube(
        self,
        topic: str,
        primary: InformationSource,
        sources: List[InformationSource],
        blog_markdown: str,
        request: InformationRequest,
    ) -> Optional[str]:
        """
        Create or update a MemCube row representing this information object.
        """
        if self.memcubes is None:
            return None

        # Try to extract a scorable_id from context or meta.
        scorable_id = 0
        doc = (request.context or {}).get("document") or {}
        doc_id = doc.get("id") or doc.get("doc_id") or primary.meta.get("doc_id")
        try:
            if doc_id is not None:
                scorable_id = int(doc_id)
        except (TypeError, ValueError):
            scorable_id = 0

        target = request.target
        source = target.meta.get("source", "information_ingest") if target.meta else "information_ingest"

        data: Dict[str, Any] = {
            "scorable_id": scorable_id,
            "scorable_type": "document",
            "content": primary.text or "",
            "dimension": self.dimension,
            "original_score": None,
            "refined_score": None,
            "refined_content": blog_markdown,
            "version": "v1",
            "source": source,
            "model": None,
            "priority": self.default_priority,
            "sensitivity": self.default_sensitivity,
            "ttl": self.cfg.get("ttl"),
            "usage_count": 0,
            "extra_data": {
                "topic": topic,
                "information_target": asdict(target),
                "information_sources": [asdict(s) for s in sources],
            },
        }

        cube = self.memcubes.upsert(data)
        return cube.id if cube is not None else None
