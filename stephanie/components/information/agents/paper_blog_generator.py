# stephanie/components/information/agents/paper_blog_generator.py
from __future__ import annotations
import re
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.information.data import (
    ConceptCluster,
    DocumentSection,
    PaperReferenceGraph,
)
from stephanie.utils.file_utils import save_to_timestamped_file
from stephanie.services.prompt_service import LLMRole, PromptService
from dataclasses import field

log = logging.getLogger(__name__)


@dataclass
class PaperBlogGeneratorConfig:
    """
    Lightweight config wrapper so we don't have to dig through cfg dicts all the time.
    """

    # High-level knobs
    max_sections: int = 8
    max_reference_items: int = 12
    intro_words: int = 400
    section_words: int = 900
    conclusion_words: int = 400

    # Model keys used by PromptService
    intro_model: str = "blog.intro"
    section_model: str = "blog.section"
    conclusion_model: str = "blog.conclusion"

    # Where to drop debug markdown
    out_root: str = "runs/paper_blogs"

    model: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "ollama/llama3.1:8b",
            "api_base": "http://localhost:11434",
        }
    )

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "PaperBlogGeneratorConfig":
        default_model = {
            "name": "ollama/llama3.1:8b",
            "api_base": "http://localhost:11434",
        }
        return cls(
            max_sections=int(cfg.get("max_sections", cls.max_sections)),
            max_reference_items=int(
                cfg.get("max_reference_items", cls.max_reference_items)
            ),
            intro_words=int(cfg.get("intro_words", cls.intro_words)),
            section_words=int(cfg.get("section_words", cls.section_words)),
            conclusion_words=int(
                cfg.get("conclusion_words", cls.conclusion_words)
            ),
            intro_model=str(cfg.get("intro_model", cls.intro_model)),
            section_model=str(cfg.get("section_model", cls.section_model)),
            conclusion_model=str(
                cfg.get("conclusion_model", cls.conclusion_model)
            ),
            out_root=str(cfg.get("out_root", cls.out_root)),
            model=cfg.get("model", default_model),
        )


class PaperBlogGeneratorAgent(BaseAgent):
    """
    Sectional blog generator for a single paper.

    Expected context keys (coming from PaperPipelineAgent + friends):

      - "paper_graph": PaperReferenceGraph
      - "paper_sections": List[DocumentSection]
      - "concept_clusters": List[ConceptCluster]
      - "paper_spine": Optional[List[Any]]  # sections + attached visual elements
      - "paper_documents": Optional[List[dict]]  # raw paper metadata (title, summary)
      - "arxiv_id" / "paper_arxiv_id": root arxiv id (fallback if graph missing)

    Writes:

      - context["paper_blog_markdown"]: full blog markdown string
      - context["paper_blog_meta"]: small dict with debug info
    """

    name = "paper_blog_generator"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        self.cfg_obj = PaperBlogGeneratorConfig.from_cfg(cfg)
        self.prompt_service: PromptService = container.get("prompt")

        self.max_sections = self.cfg_obj.max_sections
        self.max_reference_items = self.cfg_obj.max_reference_items
        self.intro_words = self.cfg_obj.intro_words
        self.section_words = self.cfg_obj.section_words
        self.conclusion_words = self.cfg_obj.conclusion_words

        self.intro_model = self.cfg_obj.intro_model
        self.section_model = self.cfg_obj.section_model
        self.conclusion_model = self.cfg_obj.conclusion_model

        self.out_root = Path(f"{self.cfg_obj.out_root}")

        self.model = self.cfg_obj.model

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entrypoint. Builds a multi-section blog post for the *root* paper.
        """
        graph: Optional[PaperReferenceGraph] = context.get("paper_graph")
        sections: List[DocumentSection] = context.get("paper_sections") or []
        clusters: List[ConceptCluster] = context.get("concept_clusters") or []
        spine: List[Any] = context.get("paper_spine") or []
        docs: List[Dict[str, Any]] = context.get("paper_documents") or []

        if not sections:
            log.warning(
                "PaperBlogGeneratorAgent: no sections in context; skipping"
            )
            return context

        # Resolve root paper id + meta
        arxiv_id, paper_title, paper_summary = self._get_root_doc_meta(
            context=context,
            graph=graph,
            sections=sections,
            docs=docs,
        )

        log.info(
            "PaperBlogGeneratorAgent: generating SECTIONAL blog for %s (%d sections)",
            arxiv_id,
            len(sections),
        )

        # 1) Intro
        intro_md = await self._generate_intro(
            arxiv_id=arxiv_id,
            paper_title=paper_title,
            paper_summary=paper_summary,
            sections=sections,
            context=context,
        )

        # 2) Section blocks (root paper only, limited to max_sections)
        root_sections = [
            s
            for s in sections
            if getattr(s, "paper_arxiv_id", None) == arxiv_id
        ]
        if not root_sections:
            root_sections = sections

        selected_sections = root_sections[: self.max_sections]
        selected_spine = spine[: len(selected_sections)] if spine else []

        section_tasks = []
        for idx, sec in enumerate(selected_sections):
            neighbor_block = self._get_neighbor_sections(
                section=sec,
                sections=selected_sections,
                window=1,
            )
            section_spine = (
                selected_spine[idx] if idx < len(selected_spine) else None
            )
            section_tasks.append(
                self._generate_section_block(
                    arxiv_id=arxiv_id,
                    paper_title=paper_title,
                    paper_summary=paper_summary,
                    section=sec,
                    neighbor_block=neighbor_block,
                    clusters=clusters,
                    section_spine=section_spine,
                    context=context,
                )
            )

        section_blocks = await asyncio.gather(*section_tasks)

        # 3) Conclusion
        conclusion_md = await self._generate_conclusion(
            arxiv_id=arxiv_id,
            paper_title=paper_title,
            paper_summary=paper_summary,
            sections=selected_sections,
            context=context,
        )

        # 4) Assemble + save
        full_blog = "\n\n".join(
            [
                self._clean_text(intro_md),
                *map(self._clean_text, section_blocks),
                self._clean_text(conclusion_md),
            ]
        )

        context["paper_blog_markdown"] = full_blog
        context["paper_blog_meta"] = {
            "arxiv_id": arxiv_id,
            "title": paper_title,
            "n_sections": len(selected_sections),
            "used_spine": bool(spine),
        }

        save_to_timestamped_file(
            data=full_blog,
            file_prefix=f"{arxiv_id}_blog",
            file_extension="md",
            output_dir=f"{self.out_root}/{self.run_id}",
        )

        return context

    # ------------------------------------------------------------------
    # Intro / Outro
    # ------------------------------------------------------------------

    async def _generate_intro(
        self,
        *,
        arxiv_id: str,
        paper_title: str,
        paper_summary: str,
        sections: List[DocumentSection],
        context: Dict[str, Any],
    ) -> str:
        """
        Generate a blog-style introduction that orients the reader.

        We don't use the full GROWS loop here – just a single strong pass.
        """
        # Build a tiny "table of contents" of the sections we plan to cover.
        toc_lines: List[str] = []
        for s in sections[: self.max_sections]:
            name = getattr(s, "section_name", None) or getattr(s, "title", "")
            if not name:
                continue
            toc_lines.append(f"- {name.strip()}")

        toc_block = (
            "\n".join(toc_lines) if toc_lines else "(sections will follow)"
        )

        prompt = f"""
You are writing the opening section of a technical blog post about a machine learning / AI paper.

Paper:
- Title: {paper_title}
- arXiv ID: {arxiv_id}

Overall summary:
{paper_summary}

High-level structure you will cover:
{toc_block}

Goal for this intro:
- Hook an informed engineer who is curious about the paper.
- Explain *why* the paper matters in practical, engineering-relevant terms.
- Set expectations for what the rest of the blog will cover.
- Avoid hype; be clear, concrete, and trustworthy.

Write a Markdown section that:
- Starts with a level-1 heading with the paper title.
- Uses short paragraphs and maybe a small bullet list or two.
- Stays within roughly {self.intro_words} words.

Do NOT include a table of contents or section list; you've already used that just to reason about what to say.
""".strip()

        intro_md = await self._call_llm(
            prompt=prompt,
            context=context,
            max_tokens=1200,
        )
        return intro_md

    async def _generate_conclusion(
        self,
        *,
        arxiv_id: str,
        paper_title: str,
        paper_summary: str,
        sections: List[DocumentSection],
        context: Dict[str, Any],
    ) -> str:
        """
        Short wrap-up that ties everything together.
        """
        section_titles = []
        for s in sections[: self.max_sections]:
            name = getattr(s, "section_name", None) or getattr(s, "title", "")
            if name:
                section_titles.append(name.strip())

        sections_block = (
            "\n".join(f"- {t}" for t in section_titles) or "(see post above)"
        )

        prompt = f"""
You are writing the closing section of a technical blog post about a machine learning / AI paper.

Paper:
- Title: {paper_title}
- arXiv ID: {arxiv_id}

The blog has covered sections like:
{sections_block}

Goal for this conclusion:
- Briefly remind the reader what problem the paper tackles.
- Summarize the key ideas in 2-3 sentences.
- Offer 2-3 concrete takeaways for engineers or researchers.
- Optionally hint at open questions or future directions.
- End on a grounded, non-hype note.

Write a Markdown section that:
- Starts with a level-2 heading (e.g., "## Wrapping up" or similar).
- Stays within roughly {self.conclusion_words} words.
""".strip()

        conclusion_md = await self._call_llm(
            prompt=prompt,
            context=context,
            max_tokens=800,
        )
        return conclusion_md

    # ------------------------------------------------------------------
    # Section generator (with GROWS-style loop in the prompt)
    # ------------------------------------------------------------------

    async def _generate_section_block(
        self,
        *,
        arxiv_id: str,
        paper_title: str,
        paper_summary: str,
        section: DocumentSection,
        neighbor_block: str,
        clusters: List[ConceptCluster],
        section_spine: Optional[Any] = None,
        context: Dict[str, Any],
    ) -> str:
        """
        Generate a single blog section for a given DocumentSection.

        This uses a *prompt-internal* GROWS loop:
          - Generate
          - Review
          - Optimize
          - Work again
          - Stop (when it would score ≥ 8/10)

        We only see the final, best section.
        """
        section_title = getattr(section, "section_name", None) or getattr(
            section, "title", ""
        )
        section_summary = getattr(section, "summary", None) or getattr(
            section, "text", ""
        )

        section_text = section_summary or ""
        if not section_text:
            return ""

        # Optional: add any cluster labels that map to this section
        cluster_titles: List[str] = []
        for c in clusters or []:
            # We keep this defensive: ConceptCluster may or may not have section ids
            sec_ids = getattr(c, "section_ids", None) or getattr(
                c, "sections", None
            )
            if sec_ids and getattr(section, "section_id", None) in sec_ids:
                label = getattr(c, "label", None) or getattr(c, "topic", None)
                if label:
                    cluster_titles.append(str(label))

        clusters_block = (
            "\n".join(f"- {t}" for t in cluster_titles)
            if cluster_titles
            else ""
        )

        visuals_hint = self._build_visuals_markdown_for_spine(section_spine)

        prompt = f"""
You are an iterative writing assistant tasked with improving a technical blog section
using the **GROWS Loop**:

1. **Generate** a draft of the section.
2. **Review** it honestly, scoring 1–10 for clarity, structure, and usefulness to an
   informed engineer.
3. **Optimize** the section based on your own review.
4. **Work again** if the new version still feels weak.
5. **Stop** once the section would score at least 8/10 by your own judgement.

You will run this loop **internally** and then output only the final, best version.

Paper context:
- Title: {paper_title}
- arXiv ID: {arxiv_id}

High-level paper summary:
{paper_summary}

Target section:
- Section title: {section_title}

If helpful, you can keep in mind that this section is loosely associated with
concept clusters like:
{clusters_block or "(no specific clusters; just make it clear and accurate)."}

Current section (rough Markdown, from the paper processing pipeline):

```markdown
{section_text}
Neighbor sections (for context only, do NOT rewrite them):
{neighbor_block}

Visual elements attached to this section (figures / tables) will be rendered
automatically after your text. You do not need to write the Markdown image
tags yourself, but you may briefly reference visuals in natural language if it
helps the story.

Your task:

Write the final section in Markdown.

Start with a level-2 heading: "## {section_title}" (or a slightly improved variant).

Aim for roughly {self.section_words} words (±30%).

Use short paragraphs and bullets where it improves readability.

Do NOT include any commentary about scores, the GROWS loop, or your process.
""".strip()

        max_tokens = self._with_length_params(
            requested_words=self.section_words, default_tokens=1800
        )

        section_md = await self._call_llm(
            prompt=prompt,
            context=context,
            max_tokens=max_tokens,
        )

        section_md = self._clean_text(section_md)

        # Append a deterministic visuals block if we have attached elements
        visuals_block = visuals_hint
        if visuals_block:
            section_md = section_md.rstrip() + "\n\n" + visuals_block + "\n"

        return section_md

    def _get_root_doc_meta(
        self,
        *,
        context: Dict[str, Any],
        graph: Optional[PaperReferenceGraph],
        sections: List[DocumentSection],
        docs: List[Dict[str, Any]],
    ) -> tuple[str, str, str]:
        """
        Resolve (arxiv_id, title, summary) for the root paper.
        """
        arxiv_id = context.get("arxiv_id") or context.get("paper_arxiv_id")
        if not arxiv_id and graph is not None:
            arxiv_id = graph.root_id

        if not arxiv_id and sections:
            arxiv_id = getattr(sections[0], "paper_arxiv_id", None)

        if not arxiv_id:
            arxiv_id = "unknown"
        # Try to find title/summary from docs
        title = ""
        summary = ""

        for d in docs or []:
            did = d.get("paper_id") or d.get("id") or d.get("arxiv_id")
            if did == arxiv_id:
                title = d.get("title") or title
                summary = d.get("summary") or d.get("abstract") or summary
                break

        if not title and sections:
            # Fall back to first section's paper_title if present
            title = getattr(sections[0], "paper_title", "") or title

        if not summary:
            # As a last resort, stitch together 2–3 section summaries
            bits: List[str] = []
            for s in sections[:3]:
                s_sum = getattr(s, "summary", None) or getattr(s, "text", "")
                if s_sum:
                    bits.append(str(s_sum))
            summary = "\n\n".join(bits)

        return arxiv_id, title or f"Paper {arxiv_id}", summary

    async def _call_llm(
        self,
        *,
        prompt: str,
        context: Dict[str, Any],
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Thin wrapper over PromptService.run_prompt so we can tweak params in one place.
        Uses the BLOG role + its default system prompt.
        """
        params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        blog_markdown = await self.prompt_service.run_prompt(
            prompt_text=prompt,
            context=None,
            model=self.model,  # use default model from prompt_service.cfg
            role=LLMRole.BLOG,  # or EXPLAINER if you're still using that
            sys_preamble=None,
            params=params,  # if you wired params
        )

        arxiv_id = context.get("arxiv_id") or context.get("paper_arxiv_id")
        graph: Optional[PaperReferenceGraph] = context.get("paper_graph")
        sections: List[DocumentSection] = context.get("paper_sections") or []
        ref_block = self._build_references_markdown(
            arxiv_id=arxiv_id,
            graph=graph,
            sections=sections,
        )
        if ref_block:
            blog_markdown = (blog_markdown or "").rstrip() + "\n\n" + ref_block + "\n"

        context[self.output_key] = blog_markdown
        log.info(
            "PaperBlogGeneratorAgent: generated blog for %s (%d chars)",
            arxiv_id,
            len(blog_markdown or ""),
        )


    @staticmethod
    def _clean_text(text: str) -> str:
        return (text or "").strip()

    @staticmethod
    def _get_neighbor_sections(
        section: DocumentSection,
        sections: List[DocumentSection],
        window: int = 1,
    ) -> str:
        """
        Return a small Markdown block with the previous/next sections for context.
        """
        try:
            idx = sections.index(section)
        except ValueError:
            return "(no neighbor context available)"

        start = max(0, idx - window)
        end = min(len(sections), idx + window + 1)
        neighbors = sections[start:end]

        lines: List[str] = []
        for s in neighbors:
            name = getattr(s, "section_name", None) or getattr(s, "title", "")
            summary = getattr(s, "summary", None) or getattr(s, "text", "")
            if not name and not summary:
                continue
            lines.append(f"### {name or 'Section'}")
            if summary:
                lines.append(summary.strip())
            lines.append("")
        return "\n".join(lines).strip() or "(no neighbor context available)"

    @staticmethod
    def _build_visuals_markdown_for_spine(section_spine: Optional[Any]) -> str:
        """
        Turn any attached visual elements (figures/tables/images) into Markdown.
          This is deliberately defensive so it can handle either dicts or dataclasses
        coming out of attach_elements_to_sections().
        """
        if not section_spine:
            return ""

        elements = getattr(section_spine, "elements", None)
        if elements is None and isinstance(section_spine, dict):
            elements = section_spine.get("elements")
        if not elements:
            return ""

        lines: List[str] = []
        for elem in elements:
            etype = (
                getattr(elem, "element_type", None)
                or getattr(elem, "type", None)
                or getattr(elem, "kind", None)
                or (
                    elem.get("element_type")
                    if isinstance(elem, dict)
                    else None
                )
                or (elem.get("type") if isinstance(elem, dict) else None)
                or (elem.get("kind") if isinstance(elem, dict) else None)
            )
            etype = str(etype).lower() if etype is not None else ""

            caption = (
                getattr(elem, "caption", None)
                or getattr(elem, "title", None)
                or (elem.get("caption") if isinstance(elem, dict) else None)
                or (elem.get("title") if isinstance(elem, dict) else None)
                or ""
            )
            image_path = (
                getattr(elem, "image_path", None)
                or getattr(elem, "path", None)
                or (elem.get("image_path") if isinstance(elem, dict) else None)
                or (elem.get("path") if isinstance(elem, dict) else None)
            )

            if etype in {"figure", "image", "chart", "plot"}:
                if image_path:
                    alt = caption or etype.title()
                    lines.append(f"![{alt}]({image_path})")
                elif caption:
                    lines.append(f"*{etype.title()}:* {caption}")
            elif etype == "table":
                table_md = getattr(elem, "markdown", None) or (
                    elem.get("markdown") if isinstance(elem, dict) else None
                )
                header = f"**Table:** {caption or 'Extracted table'}"
                if table_md:
                    lines.append(header)
                    lines.append("")
                    lines.append(str(table_md))
                else:
                    lines.append(header)

        if not lines:
            return ""

        return "\n".join(["### Visuals"] + lines)

    @staticmethod
    def _with_length_params(
        requested_words: int,
        default_tokens: int,
    ) -> int:
        """
        Very rough word→token mapping so we don't blow past limits.
        """
        approx_tokens = int(requested_words * 1.5)
        return max(default_tokens, approx_tokens)

    def save_to_timestamped_file(
        self,
        *,
        arxiv_id: str,
        title: str,
        markdown: str,
        meta: Dict[str, Any],
    ) -> None:
        """
        Save blog + meta to disk for inspection / debugging.
        """
        try:
            self.out_root.mkdir(parents=True, exist_ok=True)
            safe_title = (
                "".join(
                    c for c in title if c.isalnum() or c in (" ", "-", "")
                ).strip()
                or arxiv_id
            )
            base = f"{arxiv_id.replace('/', '')}{safe_title[:80].replace(' ', '_')}"
            md_path = self.out_root / f"{base}.md"
            meta_path = self.out_root / f"{base}.meta.json"
            md_path.write_text(markdown, encoding="utf-8")
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            log.info("PaperBlogGeneratorAgent wrote blog to %s", md_path)
        except Exception as e:
            log.warning(
                "PaperBlogGeneratorAgent failed to save debug files: %s", e
            )

    def _build_references_markdown(
        self,
        *,
        arxiv_id: str,
        graph: Optional[PaperReferenceGraph],
        sections: List[DocumentSection],
    ) -> str:
        """
        Build a Markdown 'References & Further Reading' block.

        v1 strategy:
          - Use graph nodes with role in {"reference", "ref", "similar", "candidate"}
          - Plus anything we can mine from a 'References' / 'Bibliography' section
            in DocumentSection.
        """
        items: List[Dict[str, str]] = []

        # ---------- 1) Graph-based references / similar ----------
        if graph is not None:
            nodes = getattr(graph, "nodes", {}) or {}
            for pid, node in nodes.items():
                role = getattr(node, "role", None) or getattr(
                    node, "kind", None
                )
                if role not in {"reference", "ref", "similar", "candidate"}:
                    continue

                title = getattr(node, "title", None) or pid
                url = getattr(node, "url", None) or f"https://arxiv.org/abs/{pid}"
                label = f"{pid} — {title}"
                items.append({"label": label, "url": url})

        # ---------- 2) Section-based reference lines ----------
        section_lines = self._extract_reference_lines_from_sections(sections)
        for line in section_lines:
            # try to detect an arXiv ID in the line
            m = re.search(r"\b(\d{4}\.\d{4,5}(v\d+)?)\b", line)
            url = f"https://arxiv.org/abs/{m.group(1)}" if m else ""
            items.append({"label": line, "url": url})

        # ---------- 3) Deduplicate ----------
        deduped: List[Dict[str, str]] = []
        seen = set()
        for item in items:
            key = item["label"]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        if not deduped:
            return ""

        deduped = deduped[: self.max_reference_items]

        # ---------- 4) Build Markdown ----------
        lines: List[str] = []
        lines.append("## References & Further Reading")
        lines.append("")

        for item in deduped:
            label = item["label"].strip()
            url = (item.get("url") or "").strip()
            if url:
                lines.append(f"- [{label}]({url})")
            else:
                lines.append(f"- {label}")

        return "\n".join(lines)


    def _extract_reference_lines_from_sections(
        self,
        sections: List[DocumentSection],
    ) -> List[str]:
        """
        Look for sections whose title looks like 'References', 'Bibliography', etc.
        Then heuristically split their text into per-reference lines.

        This is intentionally simple v1: good enough to get a useful block,
        and we can swap in GROBID / a dedicated parser later.
        """
        ref_texts: List[str] = []

        for sec in sections:
            title = (
                getattr(sec, "title", None)
                or getattr(sec, "section_name", None)
                or ""
            )
            title_l = str(title).lower()

            if any(
                key in title_l
                for key in ["references", "bibliography", "citations", "reference"]
            ):
                text = (
                    getattr(sec, "text", None)
                    or getattr(sec, "summary", None)
                    or ""
                )
                if text:
                    ref_texts.append(str(text))

        if not ref_texts:
            return []

        # Merge all reference-text fragments and split into lines
        raw_lines = []
        for block in ref_texts:
            raw_lines.extend(str(block).splitlines())

        cleaned: List[str] = []
        for ln in raw_lines:
            ln = ln.strip()
            if not ln:
                continue

            # Heuristics: keep lines that look like real references:
            #   - contain a year, or
            #   - start with [n], or "n." / "n)"
            if re.search(r"\b(19|20)\d{2}\b", ln) or re.match(
                r"^(\[\d+\]|\d+\.\s+|\d+\)\s+)", ln
            ):
                cleaned.append(ln)

        return cleaned
