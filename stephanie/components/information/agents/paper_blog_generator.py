# stephanie/components/information/agents/paper_blog_generator.py
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.information.data import (BlogConfig, ConceptCluster,
                                                   PaperReferenceGraph,
                                                   PaperSection)
from stephanie.services.prompt_service import LLMRole, PromptService
from stephanie.utils.file_utils import save_to_timestamped_file

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# arXiv id / heading / de-duplication helpers (pure, unit-testable)
# ----------------------------------------------------------------------

_ARXIV_PREFIX_RE = re.compile(r"^(arxiv:|arxiv/|https?://arxiv\.org/abs/)", re.IGNORECASE)
_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
_HEADING_STRIP_RE = re.compile(r"[#*_`]+")


def normalize_arxiv_id(value: Any) -> str:
    """Canonicalise an arXiv id so root/reference ids compare reliably.

    Handles the variants seen in practice: ``2506.21734``, ``2506.21734v2``,
    ``arXiv:2506.21734``, and full ``https://arxiv.org/abs/...`` URLs.
    """
    s = str(value or "").strip()
    if not s:
        return ""
    s = _ARXIV_PREFIX_RE.sub("", s)
    s = s.strip().strip("/")
    s = _ARXIV_VERSION_RE.sub("", s)
    return s.lower()


def sanitize_heading(text: Any, *, max_chars: int = 90, fallback: str = "") -> str:
    """Turn a possibly-messy section label into a clean, single-line heading.

    Section ``title``/``summary`` values are frequently whole sentences (and
    sometimes truncated mid-word). A blog heading must be short, on one line,
    and must not end mid-sentence, so we cut at the first sentence boundary and
    then trim to a word boundary.
    """
    s = str(text or "").strip()
    if not s:
        return fallback
    s = s.splitlines()[0].strip()
    s = _HEADING_STRIP_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    # If it is still sentence-like, keep only the first sentence.
    sentence = re.split(r"(?<=[.!?])\s+", s, maxsplit=1)[0].strip()
    if sentence:
        s = sentence
    if len(s) > max_chars:
        s = s[:max_chars].rsplit(" ", 1)[0].strip()
    s = s.strip(" \t-:;,.")
    if not s:
        return fallback
    return s[0].upper() + s[1:]


def _heading_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{4,}", (text or "").lower())}


def partition_sections(sections: List[Any], root_id: str) -> tuple[List[Any], List[Any]]:
    """Split sections into (root paper, other papers) using normalised ids.

    Returns ``(root_sections, related_sections)``. Sections with no usable id
    are treated as related so they can still be labelled, never silently
    presented as the root paper.
    """
    canonical_root = normalize_arxiv_id(root_id)
    root: List[Any] = []
    related: List[Any] = []
    for sec in sections or []:
        sec_pid = normalize_arxiv_id(
            getattr(sec, "paper_arxiv_id", None) or getattr(sec, "paper_id", None)
        )
        if canonical_root and sec_pid and sec_pid == canonical_root:
            root.append(sec)
        else:
            related.append(sec)
    return root, related


def dedupe_sections(sections: List[Any]) -> List[Any]:
    """Drop sections that repeat an earlier one.

    Two sections collide when their normalised heading matches exactly, when
    their headings overlap heavily (Jaccard >= 0.8), or when one heading's
    keywords are a subset of the other's (the same topic, reworded or expanded).
    This removes the repeated "chain of thought" blocks seen in the same blog.
    """
    kept: List[Any] = []
    kept_keys: set[str] = set()
    kept_token_sets: List[set[str]] = []
    seen_text: set[str] = set()

    def _is_duplicate(tokens: set[str]) -> bool:
        if not tokens:
            return False
        for other in kept_token_sets:
            if not other:
                continue
            union = tokens | other
            if len(tokens & other) / max(1, len(union)) >= 0.8:
                return True
            smaller, larger = (tokens, other) if len(tokens) <= len(other) else (other, tokens)
            if len(smaller) >= 2 and smaller.issubset(larger):
                return True
        return False

    for sec in sections or []:
        heading = str(getattr(sec, "title", None) or getattr(sec, "section_name", None) or "")
        text = str(getattr(sec, "text", None) or getattr(sec, "summary", None) or "")
        key = sanitize_heading(heading, fallback="").lower()
        text_key = text.strip()[:400].lower()
        if key and key in kept_keys:
            continue
        if text_key and text_key in seen_text:
            continue
        tokens = _heading_tokens(heading)
        if _is_duplicate(tokens):
            continue
        kept.append(sec)
        if key:
            kept_keys.add(key)
        if text_key:
            seen_text.add(text_key)
        kept_token_sets.append(tokens)
    return kept



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
    save_sections: bool = False

    # Cross-paper synthesis: keep related-paper sections, but label them.
    include_related: bool = True
    max_related_sections: int = 4

    model: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "ollama/llama3.1:8b",
            "api_base": "http://localhost:11434",
        }
    )

    max_ticks: int = 3
    quality_threshold: float = 0.80     # 0..1 or 0..100 depending on your judge output
    tick_min_delta: float = 0.01        # stop if improvement is tiny

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
            include_related=bool(
                cfg.get("include_related", cls.include_related)
            ),
            max_related_sections=int(
                cfg.get("max_related_sections", cls.max_related_sections)
            ),
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
        self.blog_cfg = BlogConfig.from_any(cfg, default=BlogConfig())
        self.save_sections = self.cfg_obj.save_sections

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entrypoint. Builds a multi-section blog post for the *root* paper.
        """
        self.blog_cfg = BlogConfig.from_any(context.get("blog_config"), default=self.blog_cfg)

        # Always write it back so downstream stages + DB logging have the exact payload used
        context["blog_config"] = self.blog_cfg.to_dict()
        context["blog_config_id"] = self.blog_cfg.fingerprint()
        graph: Optional[PaperReferenceGraph] = context.get("paper_graph")
        sections: List[PaperSection] = context.get("paper_sections") or []
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

        # 2) Partition sections: root paper vs related papers.
        #    This is explicit: we never silently relabel another paper's sections
        #    as the root paper. Related-paper material is kept (cross-paper
        #    synthesis) but is labelled as coming from another paper.
        root_id = normalize_arxiv_id(arxiv_id)
        root_sections, related_sections = partition_sections(sections, root_id)

        if not root_sections:
            log.warning(
                "PaperBlogGeneratorAgent: no sections matched root id %r "
                "(got ids like %r); treating all sections as related material",
                arxiv_id,
                [getattr(s, "paper_arxiv_id", None) for s in sections[:3]],
            )
            related_sections = list(sections)

        root_sections = dedupe_sections(root_sections)
        related_sections = dedupe_sections(related_sections)

        selected_sections = root_sections[: self.blog_cfg.max_sections]
        selected_spine = spine[: len(selected_sections)] if spine else []

        section_blocks = await self._generate_section_blocks(
            sections=selected_sections,
            spine=selected_spine,
            arxiv_id=arxiv_id,
            paper_title=paper_title,
            paper_summary=paper_summary,
            clusters=clusters,
            context=context,
        )

        related_blocks: List[str] = []
        if self.cfg_obj.include_related and related_sections:
            picked_related = related_sections[: self.cfg_obj.max_related_sections]
            log.info(
                "PaperBlogGeneratorAgent: adding %d labelled related-paper sections",
                len(picked_related),
            )
            related_blocks = await self._generate_section_blocks(
                sections=picked_related,
                spine=[],
                arxiv_id=arxiv_id,
                paper_title=paper_title,
                paper_summary=paper_summary,
                clusters=clusters,
                context=context,
                heading_level=3,
                attribute=True,
            )

        # 3) Conclusion
        conclusion_md = await self._generate_conclusion(
            arxiv_id=arxiv_id,
            paper_title=paper_title,
            paper_summary=paper_summary,
            sections=selected_sections,
            context=context,
        )

        # 4) Assemble + save
        parts: List[str] = [self._clean_text(intro_md)]
        parts.extend(self._clean_text(block) for block in section_blocks)
        if related_blocks:
            parts.append(
                "## Related work from other papers\n\n"
                "The sections below summarise *other* papers. They are labelled with "
                "their source and are supporting context, not part of the paper under discussion."
            )
            parts.extend(self._clean_text(block) for block in related_blocks)
        parts.append(self._clean_text(conclusion_md))
        body = "\n\n".join(part for part in parts if part)

        # References are built once, at the end, instead of being appended to
        # every section and then collapsed.
        references_md = self._build_references_markdown(
            arxiv_id=arxiv_id,
            graph=graph,
            sections=sections,
        )
        full_blog = body
        if references_md:
            full_blog = body.rstrip() + "\n\n" + references_md + "\n"

        context["paper_blog_markdown_with_refs"] = full_blog

        save_to_timestamped_file(
            data=full_blog,
            file_prefix=f"{arxiv_id}_blog_with_refs",
            file_extension="md",
            output_dir=self._output_dir(),
        )

        context["paper_blog_markdown"] = full_blog

        context["paper_blog_meta"] = {
            "arxiv_id": arxiv_id,
            "title": paper_title,
            "n_sections": len(selected_sections),
            "n_related_sections": len(related_blocks),
            "related_sections_labelled": bool(related_blocks),
            "used_spine": bool(spine),
            "root_id_normalised": root_id,
        }

        save_to_timestamped_file(
            data=full_blog,
            file_prefix=f"{arxiv_id}_blog",
            file_extension="md",
            output_dir=f"{self.out_root}/{self.run_id}",
        )

        return context

    # ------------------------------------------------------------------
    # Selection / assembly helpers
    # ------------------------------------------------------------------

    def _output_dir(self) -> str:
        """Output directory for this run.

        ``run_id`` can itself be a path in some configs, which previously
        produced nested ``runs/paper_blogs/<id>/runs/paper_blogs/...`` folders;
        only the final path segment is used.
        """
        run_key = Path(str(self.run_id or "")).name or "run"
        return str(Path(self.out_root) / run_key)

    def _section_heading(self, section: Any) -> str:
        """Best available human heading for a section (never a truncated sentence)."""
        meta = getattr(section, "meta", None) or {}
        candidates: List[Any] = []
        if isinstance(meta, dict):
            candidates.extend(
                [meta.get("heading"), meta.get("section_heading"), meta.get("section_name")]
            )
        candidates.append(getattr(section, "section_name", None))
        candidates.append(getattr(section, "title", None))
        for candidate in candidates:
            heading = sanitize_heading(candidate)
            if heading:
                return heading
        idx = getattr(section, "section_index", None)
        return f"Section {idx + 1}" if isinstance(idx, int) else "Section"

    def _paper_label(self, section: Any, context: Dict[str, Any]) -> str:
        """Human label for the paper a section came from."""
        pid = normalize_arxiv_id(
            getattr(section, "paper_arxiv_id", None) or getattr(section, "paper_id", None)
        ) or "unknown"
        title = ""
        for doc in context.get("paper_documents") or []:
            did = normalize_arxiv_id(doc.get("paper_id") or doc.get("id") or doc.get("arxiv_id"))
            if did and did == pid:
                title = str(doc.get("title") or "")
                break
        if not title:
            meta = getattr(section, "meta", None) or {}
            if isinstance(meta, dict):
                title = str(meta.get("paper_title") or "")
        return f"{title} ({pid})" if title else f"paper {pid}"

    @staticmethod
    def _force_heading(markdown: str, *, title: str, level: int, source_note: str = "") -> str:
        """Deterministically set the section heading and optional source line.

        The model is free to write the body, but the heading must be the
        sanitised title we computed — so a stray summary sentence (or a
        truncated fragment) can never end up in the heading.
        """
        prefix = "#" * max(2, min(6, level))
        lines = (markdown or "").strip().splitlines()
        while lines and lines[0].lstrip().startswith("#"):
            lines.pop(0)
        body = "\n".join(lines).strip()
        header = f"{prefix} {title}"
        if source_note:
            header = f"{header}\n\n*Source: {source_note}.*"
        return f"{header}\n\n{body}".strip() if body else header

    async def _generate_section_blocks(
        self,
        *,
        sections: List[PaperSection],
        spine: List[Any],
        arxiv_id: str,
        paper_title: str,
        paper_summary: str,
        clusters: List[ConceptCluster],
        context: Dict[str, Any],
        heading_level: int = 2,
        attribute: bool = False,
    ) -> List[str]:
        """Generate one blog block per section, concurrently."""
        if not sections:
            return []
        matches = context.get("section_matches") or []
        all_sections = context.get("paper_sections") or sections
        tasks = []
        for idx, sec in enumerate(sections):
            tasks.append(
                self._generate_section_block(
                    arxiv_id=arxiv_id,
                    paper_title=paper_title,
                    paper_summary=paper_summary,
                    section=sec,
                    neighbor_block=self._get_neighbor_sections(
                        section=sec, sections=sections, window=1
                    ),
                    clusters=clusters,
                    section_spine=spine[idx] if idx < len(spine) else None,
                    context=context,
                    all_sections=all_sections,
                    section_matches=matches,
                    heading_level=heading_level,
                    attribute=attribute,
                )
            )
        return list(await asyncio.gather(*tasks))

    # ------------------------------------------------------------------
    # Intro / Outro
    # ------------------------------------------------------------------

    async def _generate_intro(
        self,
        *,
        arxiv_id: str,
        paper_title: str,
        paper_summary: str,
        sections: List[PaperSection],
        context: Dict[str, Any],
    ) -> str:
        """
        Generate a blog-style introduction that orients the reader.

        We don't use the full GROWS loop here – just a single strong pass.
        """
        # Build a tiny "table of contents" of the sections we plan to cover.
        toc_lines: List[str] = []
        for s in sections[: self.blog_cfg.max_sections]:
            name = self._section_heading(s)
            if not name:
                continue
            toc_lines.append(f"- {name}")

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
- Stays within roughly {self.blog_cfg.intro_words} words.

Do NOT include a table of contents or section list; you've already used that just to reason about what to say.
""".strip()

        intro_md = await self._call_llm(
            section="Introduction",
            prompt=prompt,
            context=context,
            max_tokens=1200,
        )
        context["intro"] = intro_md
        return intro_md

    async def _generate_conclusion(
        self,
        *,
        arxiv_id: str,
        paper_title: str,
        paper_summary: str,
        sections: List[PaperSection],
        context: Dict[str, Any],
    ) -> str:
        """
        Short wrap-up that ties everything together.
        """
        section_titles = []
        for s in sections[: self.blog_cfg.max_sections]:
            name = self._section_heading(s)
            if name:
                section_titles.append(name)

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
- Stays within roughly {self.blog_cfg.conclusion_words} words.
""".strip()

        conclusion_md = await self._call_llm(
            section="Conclusion",
            prompt=prompt,
            context=context,
            max_tokens=800,
        )
        context["conclusion"] = conclusion_md
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
        section: PaperSection,
        neighbor_block: str,
        clusters: List[ConceptCluster],
        section_spine: Optional[Any] = None,
        context: Dict[str, Any],
        all_sections: List[PaperSection],
        section_matches: List[Any],
        heading_level: int = 2,
        attribute: bool = False,
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

        ``heading_level``/``attribute`` support the labelled related-paper
        block: related sections are rendered as sub-headings carrying a source
        line naming the paper they came from.
        """
        section_title = self._section_heading(section)
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

        supports = self.select_support_sections(
            root_section=section,
            all_sections=all_sections,
            section_matches=section_matches,
            max_supports=8,
        )

        support_blocks = []
        for s in supports:
            title = getattr(s, "title", None) or getattr(s, "section_name", None) or "Section"
            pid = getattr(s, "paper_arxiv_id", None) or getattr(s, "paper_id", None) or "unknown"
            txt = (getattr(s, "summary", None) or getattr(s, "text", "") or "").strip()
            if not txt:
                continue
            support_blocks.append(f"### {pid} — {title}\n{txt[:900]}")

        support_context = "\n\n".join(support_blocks) if support_blocks else "(none)"

        log.info("found %d support sections for section %s", len(supports), getattr(section, "section_id", "unknown"))  

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

---
Supporting excerpts from related papers (use these to strengthen accuracy & context; do not copy verbatim):
{support_context}
---

##  Final task

Your task:

Write the final section in Markdown.

Start with a level-{heading_level} heading; the exact heading text is applied
automatically after you write, so focus on writing a strong body.

{("This section summarises a *related* paper. Frame it clearly as supporting "
  "context from another paper, not as the paper under discussion.") if attribute else ""}

Aim for roughly {self.blog_cfg.section_words} words (±30%).

Use short paragraphs and bullets where it improves readability.

Do NOT include any commentary about scores, the GROWS loop, or your process.
""".strip()

        max_tokens = self._with_length_params(
            requested_words=self.blog_cfg.section_words, default_tokens=1800
        )

        section_md = await self._call_llm(
            section=section_title,
            prompt=prompt,
            context=context,
            max_tokens=max_tokens,
        )
        context["last_section_generated"] = section_md

        section_md = self._clean_text(section_md)
        section_md = self._force_heading(
            section_md,
            title=section_title,
            level=heading_level,
            source_note=self._paper_label(section, context) if attribute else "",
        )

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
        sections: List[PaperSection],
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
        section: str,
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

        if self.save_sections:
            save_to_timestamped_file(blog_markdown, file_prefix=f"{arxiv_id}_section_", file_extension="md", output_dir=f"{self.out_root}/{self.run_id}")

        log.info(
            "PaperBlogGeneratorAgent: generated blog section '%s...' for %s (%d chars)",
            section[:20],
            arxiv_id,
            len(blog_markdown or ""),
        )
        return blog_markdown or ""

    @staticmethod
    def _section_id(sec: Any) -> str:
        return str(getattr(sec, "id", None) or getattr(sec, "section_id", None) or "")

    @staticmethod
    def _paper_id(sec: Any) -> str:
        return str(getattr(sec, "paper_arxiv_id", None) or getattr(sec, "paper_id", None) or "")

    @staticmethod
    def _match_fields(m: Any) -> dict:
        if isinstance(m, dict):
            return m
        # dataclass / object fallback
        return {k: getattr(m, k) for k in dir(m) if not k.startswith("_")}

    @staticmethod
    def _extract_pair(m: dict) -> tuple[str, str, float]:
        # try common patterns
        a = m.get("a") or m.get("a_id") or m.get("left") or m.get("source") or m.get("source_id") or m.get("section_id")
        b = m.get("b") or m.get("b_id") or m.get("right") or m.get("target") or m.get("target_id") or m.get("match_section_id")
        s = m.get("score") or m.get("sim") or m.get("similarity") or m.get("cosine") or 0.0
        return (str(a or ""), str(b or ""), float(s or 0.0))

    @staticmethod
    def select_support_sections(
        *,
        root_section: Any,
        all_sections: list[Any],
        section_matches: list[Any],
        max_supports: int = 12,
    ) -> list[Any]:
        root_id = PaperBlogGeneratorAgent._section_id(root_section)
        root_paper = PaperBlogGeneratorAgent._paper_id(root_section)
        if not root_id:
            return []

        by_id = {PaperBlogGeneratorAgent._section_id(s): s for s in all_sections if PaperBlogGeneratorAgent._section_id(s)}
        scored: list[tuple[float, Any]] = []

        for m in section_matches or []:
            d = PaperBlogGeneratorAgent._match_fields(m)
            a, b, score = PaperBlogGeneratorAgent._extract_pair(d)
            if not a or not b:
                continue

            other_id = None
            if a == root_id:
                other_id = b
            elif b == root_id:
                other_id = a
            else:
                continue

            sec = by_id.get(other_id)
            if not sec:
                continue
            # only pull from other papers
            if PaperBlogGeneratorAgent._paper_id(sec) == root_paper:
                continue

            scored.append((score, sec))

        scored.sort(key=lambda x: x[0], reverse=True)

        out = []
        seen = set()
        for _, sec in scored:
            sid = PaperBlogGeneratorAgent._section_id(sec)
            if sid in seen:
                continue
            seen.add(sid)
            out.append(sec)
            if len(out) >= max_supports:
                break
        return out


    @staticmethod
    def _clean_text(text: str) -> str:
        return (text or "").strip()

    @staticmethod
    def _get_neighbor_sections(
        section: PaperSection,
        sections: List[PaperSection],
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
        sections: List[PaperSection],
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

        deduped = deduped[: self.blog_cfg.max_reference_items]

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


    _REF_ENTRY_START_RE = re.compile(r"^(\[\d+\]|\d+[.)]\s*)")
    _REF_AUTHOR_START_RE = re.compile(r"^[A-Z][A-Za-z'’\-]+,\s")
    _REF_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

    def _extract_reference_lines_from_sections(
        self,
        sections: List[PaperSection],
    ) -> List[str]:
        """
        Extract reference entries from any 'References' / 'Bibliography' section.

        Reference blocks are usually hard-wrapped, so a single citation can span
        several physical lines. We first group continuation lines into one entry
        (starting a new entry at ``[n]`` / ``n.`` markers or an ``Author,`` line),
        then keep entries that look like real citations. The previous version kept
        individual physical lines, which merged and mangled citations.
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

        raw_lines: List[str] = []
        for block in ref_texts:
            raw_lines.extend(str(block).splitlines())

        entries: List[str] = []
        current: List[str] = []
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            is_start = bool(
                self._REF_ENTRY_START_RE.match(line)
                or self._REF_AUTHOR_START_RE.match(line)
            )
            if is_start and current:
                entries.append(" ".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            entries.append(" ".join(current))

        cleaned: List[str] = []
        for entry in entries:
            entry = re.sub(r"\s+", " ", entry).strip()
            if not entry:
                continue
            if self._REF_ENTRY_START_RE.match(entry) or self._REF_YEAR_RE.search(entry):
                cleaned.append(entry)
        return cleaned

    def _consolidate_references_at_end(self, markdown: str) -> str:
        """
        Consolidate repeated '## References & Further Reading' blocks
        into a single block at the end of the document.

        Deterministic, idempotent, no content rewriting.
        """
        if not markdown.strip():
            return markdown

        heading = "## References & Further Reading"

        # Match a references block until next level-2 heading or EOF
        pattern = re.compile(
            rf"{re.escape(heading)}\n+(.*?)(?=\n## |\Z)",
            re.DOTALL,
        )

        all_refs = []
        for match in pattern.finditer(markdown):
            block = match.group(1)
            for line in block.splitlines():
                line = line.strip()
                if line.startswith("-"):
                    all_refs.append(line)

        # Remove all reference blocks from body
        cleaned = pattern.sub("", markdown).strip()

        if not all_refs:
            return cleaned

        # Deduplicate references
        seen = set()
        deduped = []
        for ref in all_refs:
            key = self._normalize_reference_line(ref)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ref)

        # Rebuild final references block
        lines = []
        lines.append(heading)
        lines.append("")
        lines.extend(deduped)

        return cleaned.rstrip() + "\n\n" + "\n".join(lines) + "\n"

    @staticmethod
    def _normalize_reference_line(line: str) -> str:
        """
        Normalize reference lines for deduplication.
        """
        line = line.lower()
        line = re.sub(r"https?://\S+", "", line)  # remove URLs
        line = re.sub(r"\W+", " ", line)          # collapse punctuation
        return line.strip()
