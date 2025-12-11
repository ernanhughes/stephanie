from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.information.data import (
    ConceptCluster,
    DocumentSection,
    PaperReferenceGraph,
)
from stephanie.services.prompt_service import LLMRole
from stephanie.utils.file_utils import (
    save_to_timestamped_file,
)  # roles for PromptService :contentReference[oaicite:1]{index=1}

log = logging.getLogger(__name__)


class PaperBlogGeneratorAgent(BaseAgent):
    """
    Turn the paper graph + sections + clusters into a human blog post.

    Expects in context (from PaperPipelineAgent + friends):
      - "arxiv_id" or "paper_arxiv_id" or "root_arxiv_id"
      - "paper_graph": PaperReferenceGraph
      - "paper_sections": List[DocumentSection]
      - "concept_clusters": List[ConceptCluster]
      - optional "paper_documents": List[DocumentORM | dict]

    Writes:
      - context[cfg.output_key] (default: "paper_blog_markdown")
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(
            cfg=cfg, memory=memory, container=container, logger=logger
        )

        # We always go through your PromptService, not BaseAgent.async_call_llm
        self.prompt_service = self.container.get(
            "prompt"
        )  # same pattern as ExplainerJudgeAgent :contentReference[oaicite:2]{index=2}

        blog_cfg = cfg.get("blog", {}) or {}
        self.max_sections = int(blog_cfg.get("max_sections", 16))
        self.max_clusters = int(blog_cfg.get("max_clusters", 6))
        self.max_similar_papers = int(blog_cfg.get("max_similar_papers", 4))
        self.target_length_words = int(
            blog_cfg.get("target_length_words", 2200)
        )
        self.audience = blog_cfg.get("audience", "informed engineer")
        self.output_key = cfg.get("output_key", "paper_blog_markdown")

        self.blog_model = blog_cfg.get("model")  # dict | str | None
        # Call-level overrides (merged on top of model.params)
        self.blog_params = (
            blog_cfg.get("prompt_params")
            or blog_cfg.get("params")
            or {}
        )

        # Optional: where report agent stored markdown; not required here,
        # but we can reuse that summary in the future if you like.
        self.report_key = cfg.get("report_key", "paper_report_markdown")
        self.report_dir = cfg.get(
            "report_dir", f"runs/paper_reports/{self.run_id}"
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # ------------------- gather inputs -------------------
        arxiv_id = (
            context.get("arxiv_id")
            or context.get("paper_arxiv_id")
            or context.get("root_arxiv_id")
            or "unknown"
        )

        graph: Optional[PaperReferenceGraph] = context.get("paper_graph")
        sections: List[DocumentSection] = context.get("paper_sections") or []
        clusters: List[ConceptCluster] = context.get("concept_clusters") or []
        docs = context.get("paper_documents") or []

        root_title, root_summary = self._get_root_doc_meta(
            arxiv_id=arxiv_id,
            graph=graph,
            docs=docs,
            sections=sections,
        )

        graph_block = self._build_graph_block(graph)
        sections_block = self._build_sections_block(sections)
        clusters_block = self._build_clusters_block(clusters, sections)

        prompt_text = self._build_blog_prompt(
            arxiv_id=arxiv_id,
            title=root_title,
            paper_summary=root_summary,
            graph_block=graph_block,
            sections_block=sections_block,
            clusters_block=clusters_block,
        )

        # ------------------- call LLM via PromptService -------------------

        sys_preamble = (
            "You are writing a clear, technically precise blog post about an AI/ML paper. "
            f"The intended audience is an {self.audience}. "
            "You must be faithful to the provided summaries and graph context, "
            "avoid fabricating specific results, and explain things in an intuitive way."
        )

        try:
            blog_markdown = await self.prompt_service.run_prompt(
                prompt_text=prompt_text,
                context=None,
                model=self.blog_model,
                role=LLMRole.EXPLAINER,
                sys_preamble=sys_preamble,
                params=self.blog_params,
            )
        except Exception as e:
            log.exception("PaperBlogGeneratorAgent: prompt failed: %s", e)
            blog_markdown = f"ERROR: blog generation failed: {e}"

        context[self.output_key] = blog_markdown
        log.info(
            "PaperBlogGeneratorAgent: generated blog for %s (%d chars)",
            arxiv_id,
            len(blog_markdown or ""),
        )

        report_path = save_to_timestamped_file(
            data=blog_markdown,
            output_dir=self.report_dir,
            file_prefix=f"{arxiv_id}_blog",
            file_extension="md",
        )
        context["paper_blog_path"] = str(report_path)

        return context

    # ------------------------------------------------------------------ helpers

    def _get_root_doc_meta(
        self,
        *,
        arxiv_id: str,
        graph: Optional[PaperReferenceGraph],
        docs: List[Any],
        sections: List[DocumentSection],
    ) -> (str, str):
        """
        Try to get a good (title, summary) pair for the root paper.

        Priority:
          1. DocumentStore metadata (title + summary)
          2. Graph root node title
          3. Fall back to arxiv_id + section[0] summary
        """
        title = None
        summary = None

        # 1) DocumentStore
        for d in docs:
            ext_id = getattr(d, "external_id", None) or getattr(
                d, "paper_id", None
            )
            if not ext_id and isinstance(d, dict):
                ext_id = d.get("external_id") or d.get("paper_id")

            if str(ext_id) == str(arxiv_id):
                title = getattr(d, "title", None) or (
                    d.get("title") if isinstance(d, dict) else None
                )
                summary = getattr(d, "summary", None) or (
                    d.get("summary") if isinstance(d, dict) else None
                )
                break

        # 2) Graph root node title if missing
        if graph is not None and not title:
            nodes = getattr(graph, "nodes", {}) or {}
            root_id = getattr(graph, "root_id", None)
            if root_id and root_id in nodes:
                node = nodes[root_id]
                title = getattr(node, "title", None) or title

        # 3) Fallbacks
        if not title:
            title = f"arXiv {arxiv_id}"

        if not summary and sections:
            first = sections[0]
            summary = (
                getattr(first, "summary", None)
                or getattr(first, "text", None)
                or ""
            )
            summary = self._clean_text(summary, max_len=600)

        return title, summary or ""

    def _build_graph_block(self, graph: Optional[PaperReferenceGraph]) -> str:
        if graph is None:
            return "(no graph information available)"

        nodes = getattr(graph, "nodes", {}) or {}
        edges = list(getattr(graph, "edges", []) or [])

        roles: Dict[str, List[str]] = {}
        for pid, node in nodes.items():
            role = getattr(node, "role", None) or getattr(
                node, "kind", "unknown"
            )
            title = getattr(node, "title", None) or pid
            label = f"{pid} — {title}"
            roles.setdefault(role, []).append(label)

        # similar papers (for context) limited
        similar = roles.get("similar", []) + roles.get("candidate", [])
        similar = similar[: self.max_similar_papers]

        lines: List[str] = []
        lines.append("GRAPH CONTEXT")
        lines.append("")
        lines.append(f"- Total papers in graph: {len(nodes)}")
        lines.append(f"- Total edges: {len(edges)}")

        root_list = roles.get("root") or []
        if root_list:
            lines.append(f"- Root paper: {root_list[0]}")

        if similar:
            lines.append("- Notable similar papers:")
            for s in similar:
                lines.append(f"  - {s}")

        refs = roles.get("reference") or roles.get("ref") or []
        if refs:
            lines.append(f"- Number of direct references: {len(refs)}")

        return "\n".join(lines)

    def _build_sections_block(self, sections: List[DocumentSection]) -> str:
        if not sections:
            return "(no sections were generated)"

        lines: List[str] = []
        lines.append("SECTION SUMMARIES")
        lines.append("")

        for idx, sec in enumerate(sections[: self.max_sections]):
            title = (
                getattr(sec, "title", None)
                or getattr(sec, "section_name", None)
                or f"Section {idx + 1}"
            )
            summary = (
                getattr(sec, "summary", None)
                or getattr(sec, "text", None)
                or ""
            )
            summary = self._clean_text(summary, max_len=350)
            lines.append(f"{idx + 1}. {title}")
            lines.append(f"    {summary}")
            lines.append("")

        if len(sections) > self.max_sections:
            lines.append(
                f"... ({len(sections) - self.max_sections} more sections omitted for brevity)"
            )

        return "\n".join(lines)

    def _build_clusters_block(
        self,
        clusters: List[ConceptCluster],
        sections: List[DocumentSection],
    ) -> str:
        if not clusters:
            return "(no concept clusters available)"

        # Map section id -> short name
        section_name_by_id: Dict[str, str] = {}
        for idx, sec in enumerate(sections):
            sid = getattr(sec, "section_id", None) or getattr(sec, "id", None)
            if sid is None:
                continue
            title = (
                getattr(sec, "title", None)
                or getattr(sec, "section_name", None)
                or f"Section {idx + 1}"
            )
            section_name_by_id[str(sid)] = title

        lines: List[str] = []
        lines.append("CONCEPT CLUSTERS")
        lines.append("")

        for i, cl in enumerate(clusters[: self.max_clusters]):
            label = (
                getattr(cl, "label", None)
                or getattr(cl, "name", None)
                or f"Cluster {i + 1}"
            )

            member_ids = (
                getattr(cl, "section_ids", None)
                or getattr(cl, "sections", None)
                or getattr(cl, "members", None)
                or []
            )
            names: List[str] = []
            for mid in member_ids:
                key = str(getattr(mid, "id", mid))
                if key in section_name_by_id:
                    names.append(section_name_by_id[key])

            if names:
                preview = ", ".join(names[:4])
                if len(names) > 4:
                    preview += f" (+{len(names) - 4} more)"
                lines.append(f"- {label}: touches {preview}")
            else:
                lines.append(f"- {label}: (no resolvable sections)")

        if len(clusters) > self.max_clusters:
            lines.append(
                f"... ({len(clusters) - self.max_clusters} more clusters omitted)"
            )

        return "\n".join(lines)

    def _build_blog_prompt(
        self,
        *,
        arxiv_id: str,
        title: str,
        paper_summary: str,
        graph_block: str,
        sections_block: str,
        clusters_block: str,
    ) -> str:
        return f"""
You are writing a long-form technical blog post about a machine learning / AI paper.

The goal is to turn the paper into an intelligible, engaging explanation for an informed engineer
who wants to understand *what the paper does, why it matters, and how it fits into the landscape*.

Use ONLY the information provided in the PAPER SUMMARY, GRAPH CONTEXT, SECTION SUMMARIES,
and CONCEPT CLUSTERS below. Do NOT invent numerical results, experimental setups, or citations
that are not implied by the summaries.

=== PAPER ID ===
{arxiv_id}

=== TITLE ===
{title}

=== PAPER SUMMARY (if present) ===
{paper_summary or "(no separate summary; rely on sections instead)"}

=== GRAPH CONTEXT (root + references + similar papers) ===
{graph_block}

=== SECTION SUMMARIES ===
{sections_block}

=== CONCEPT CLUSTERS (themes across sections) ===
{clusters_block}

--- WRITING INSTRUCTIONS ---

Write a single cohesive blog post in **Markdown** with:

1. A short, hooky introduction:
   - What problem the paper addresses
   - Why this problem matters
   - A one-paragraph intuitive description of the core idea

2. A "Core Idea" section:
   - Explain the main method or contribution in intuitive terms
   - Use analogies and simple math language where helpful
   - Connect the explanation to the specific sections and clusters mentioned above

3. A "How It Works" section:
   - Walk through the key components or steps (roughly following the section summaries)
   - Highlight any important design choices or trade-offs
   - If you mention other papers, they MUST come from the GRAPH CONTEXT (references / similar papers)

4. A "Why It Matters" section:
   - Explain what this changes for practitioners or researchers
   - Mention how it compares (conceptually) to the similar papers in the graph, if possible

5. A concise conclusion:
   - Recap the core idea in one paragraph
   - Mention one or two open questions or possible extensions (high level, no speculation about experiments)

STYLE:
- Use clear headings (##, ###) and short paragraphs.
- Use bullet lists when enumerating ideas.
- Avoid marketing language; aim for an honest, insightful explainer.
- Keep jargon under control; briefly gloss technical terms when they first appear.

TARGET LENGTH:
- Roughly {self.target_length_words} words (it can be shorter if the material is sparse).

Now write the full blog post in Markdown, starting with a top-level title line.
""".strip()

    @staticmethod
    def _clean_text(text: str, max_len: int) -> str:
        text = " ".join(str(text).split())
        if len(text) > max_len:
            return text[: max_len - 3] + "..."
        return text
