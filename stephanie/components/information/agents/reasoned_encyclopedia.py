# stephanie/components/encyclopedia/agents/reasoned_encyclopedia.py
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
# MementoAgent: case-based + MCTS reasoning
# Adjust import if your path differs.
from stephanie.agents.dspy.memento import MementoAgent
from stephanie.core.context.context_manager import ContextManager

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Data models for clarity
# ---------------------------------------------------------------------


@dataclass
class ReasonedSection:
    section_id: str
    title: str
    text: str
    candidates: List[Dict[str, Any]]
    casebook_ref: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class ReasonedBlogResult:
    outline: List[Dict[str, Any]]
    sections: List[ReasonedSection]
    full_text: str
    meta: Dict[str, Any]


# ---------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------


class ReasonedEncyclopediaAgent(BaseAgent):
    """
    Stage 2: Reasoned enhancement for the AI Encyclopedia.

    Runs after the InformationIngestAgent. For each ingested paper, this agent:
      - Builds per-section "blog goals" (what each section should explain)
      - Calls MementoAgent (case-based + MCTS reasoning) to generate and rank
        candidate section texts
      - Assembles a 'reasoned_blog' artifact in the context:
            context["reasoned_blog"] = { ... }

    This does NOT do final polishing / Hugo layout.
    It just produces a better, more structured V1.5 blog draft.
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger) -> None:
        super().__init__(cfg, memory, container, logger)

        # Basic knobs
        self.max_sections: int = int(cfg.get("max_sections", 4))
        self.section_selection: str = cfg.get(
            "section_selection", "top_level"
        )  # "top_level" | "all" | "subset"

        # How many candidate drafts per section to keep
        self.candidates_per_section: int = int(
            cfg.get("candidates_per_section", 3)
        )

        # Memento configuration sub-tree
        self.memento_cfg: Dict[str, Any] = cfg.get("memento", {}) or {}
        self.memento_casebook_tag: str = self.memento_cfg.get(
            "casebook_tag", "encyclopedia"
        )

        self.memento_agent: MementoAgent = MementoAgent(
            cfg=self.memento_cfg, memory=memory, container=container, logger=logger
        )

    # ------------------------------------------------------------------
    # Pipeline entrypoint
    # ------------------------------------------------------------------

    async def run(self, context: ContextManager) -> ContextManager:
        """
        Main pipeline entrypoint.

        Expected context input (after ingest stage):
          - context["information_ingest"]["documents"] : list of per-paper dicts
            each containing at minimum:
              {
                "document_id": <str>,
                "title": <str>,
                "casebook_id": <int>,
                "goal_id": <int or str>,
                "memcube_id": <int>,
                "blog_markdown": <str>,
                "bucket_id": <int or None>,
                "section_count": <int>,
                # ... anything else ingest wants to add
              }

          - Optionally:
                context["paper"]           # richer paper metadata
                context["draft_blog"]      # initial draft from ingest
        """

        ctx = context  # just a shorter alias

        ingest_block = ctx.get("information_ingest") or {}
        if ingest_block.get("status") != "ok":
            log.warning(
                "ReasonedEncyclopediaAgent: information_ingest.status != 'ok', skipping."
            )
            ctx["reasoned_blog"] = {
                "status": "skipped",
                "reason": "information_ingest not ready",
            }
            return ctx

        documents = ingest_block.get("documents") or []
        if not documents:
            log.warning(
                "ReasonedEncyclopediaAgent: no documents found in information_ingest."
            )
            ctx["reasoned_blog"] = {
                "status": "no_documents",
                "reason": "no documents to enhance",
            }
            return ctx

        # For now, handle the first document only.
        # It's straightforward to generalize if you want multi-doc runs.
        doc_meta = documents[0]
        reasoned_result = await self._process_single_document(doc_meta, ctx)

        ctx["reasoned_blog"] = {
            "status": "ok",
            "result": asdict(reasoned_result),
        }
        return ctx

    # ------------------------------------------------------------------
    # Core per-document logic
    # ------------------------------------------------------------------

    async def _process_single_document(
        self, doc_meta: Dict[str, Any], ctx: ContextManager
    ) -> ReasonedBlogResult:
        """
        Given per-document metadata from ingest, build a reasoned blog.
        """
        paper_id = str(doc_meta.get("document_id") or doc_meta.get("doc_id"))
        paper_title = doc_meta.get("title") or f"Paper {paper_id}"
        memcube_id = doc_meta.get("memcube_id")
        casebook_id = doc_meta.get("casebook_id")

        log.info(
            "ReasonedEncyclopediaAgent: processing paper '%s' (id=%s, memcube=%s, casebook=%s)",
            paper_title,
            paper_id,
            memcube_id,
            casebook_id,
        )

        # TODO: If you have a "paper" object in context with richer meta (authors, venue),
        # pull it here for better prompts:
        paper_obj = (ctx.get("paper") or {}).copy()
        sections = self._resolve_sections_for_paper(doc_meta, paper_obj, ctx)

        # Optionally: restrict how many sections we process
        selected_sections = self._select_sections(sections)

        reasoned_sections: List[ReasonedSection] = []

        for sec in selected_sections:
            sec_id = sec.get("id") or sec.get("section_id") or sec.get("name")
            sec_title = sec.get("title") or sec.get("section_name") or "Section"

            # Build a goal text for this section
            goal_dict = self._build_section_goal(
                paper_id=paper_id,
                paper_title=paper_title,
                section_id=str(sec_id),
                section_title=sec_title,
            )

            # Build a section-specific context for Memento
            section_context = self._build_section_context(
                base_ctx=ctx,
                paper=paper_obj,
                doc_meta=doc_meta,
                section=sec,
                goal=goal_dict,
            )

            # Run Memento to get candidate texts
            best_text, candidates, case_ref = await self._run_memento_for_section(
                section_context, goal_dict
            )

            reasoned_sections.append(
                ReasonedSection(
                    section_id=str(sec_id),
                    title=sec_title,
                    text=best_text,
                    candidates=candidates,
                    casebook_ref=case_ref,
                    meta={
                        "goal_id": goal_dict.get("id"),
                        "paper_id": paper_id,
                    },
                )
            )

        # Assemble outline & full text
        outline = self._build_outline(reasoned_sections)
        full_text = self._build_full_text(outline, reasoned_sections)

        meta = {
            "source_paper_id": paper_id,
            "memcube_id": memcube_id,
            "casebook_id": casebook_id,
            "agent": "ReasonedEncyclopediaAgent",
            "memento_casebook_tag": self.memento_casebook_tag,
        }

        return ReasonedBlogResult(
            outline=outline,
            sections=reasoned_sections,
            full_text=full_text,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Helpers: section resolution & selection
    # ------------------------------------------------------------------

    def _resolve_sections_for_paper(
        self,
        doc_meta: Dict[str, Any],
        paper_obj: Dict[str, Any],
        ctx: ContextManager,
    ) -> List[Dict[str, Any]]:
        """
        Try to resolve the list of sections for this paper from context.

        Priority:
          1) context["paper"]["sections"]
          2) doc_meta["sections"]
          3) context["draft_blog"]["sections"] as a fallback scaffold

        Each section should at least have:
          - id or section_id
          - title or section_name
          - text or content (optional but desirable)
        """
        # 1) `paper` object
        doc_id = str(doc_meta.get("document_id") or doc_meta.get("doc_id"))
        sections = sections = self.memory.document_sections.get_by_document(int(doc_id))
        if sections:
            return [s.to_dict() for s in sections]

        # 2) doc_meta from ingest
        sections = doc_meta.get("sections")
        if sections:
            return list(sections)

        # 3) fallback: the draft blog's sections (if set)
        draft_blog = ctx.get("draft_blog") or {}
        sections = draft_blog.get("sections")
        if sections:
            return list(sections)

        # Worst-case: return a single pseudo-section using the whole draft/full text
        full_text = draft_blog.get("full_text") or doc_meta.get("blog_markdown", "")
        if not full_text:
            return []

        return [
            {
                "id": "s0",
                "title": "Overview",
                "text": full_text,
            }
        ]

    def _select_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Select which sections to reason over.

        For now:
          - "top_level": take the first N sections
          - "all": all of them
          - "subset": also first N (but you can later add smarter policies)
        """
        if not sections:
            return []

        sel = self.section_selection
        if sel == "all":
            return sections
        # For now "top_level" and "subset" behave the same
        return sections[: self.max_sections]

    # ------------------------------------------------------------------
    # Helpers: goal & context for Memento
    # ------------------------------------------------------------------

    def _build_section_goal(
        self,
        paper_id: str,
        paper_title: str,
        section_id: str,
        section_title: str,
    ) -> Dict[str, Any]:
        """
        Build a per-section goal dict for Memento.

        This is what tells Memento what we are trying to achieve.
        """
        goal_id = f"{paper_id}:{section_id}"
        goal_text = (
            f"Write a clear, engaging encyclopedia/blog section explaining the "
            f"'{section_title}' part of the paper '{paper_title}' for a technically "
            f"literate engineer. Focus on intuition, core ideas, and how this fits in "
            f"the bigger picture of self-improving AI systems. Avoid restating the "
            f"abstract; instead, explain why this section matters."
        )

        return {
            "id": goal_id,
            "goal_text": goal_text,
            "paper_id": paper_id,
            "section_id": section_id,
            "section_title": section_title,
        }

    def _build_section_context(
        self,
        base_ctx: ContextManager,
        paper: Dict[str, Any],
        doc_meta: Dict[str, Any],
        section: Dict[str, Any],
        goal: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a self-contained context dict for this section run.

        IMPORTANT:
        - MementoAgent expects to find:
            - "goal": the goal dict
            - any fields it needs for tool use / scoring / etc.
        - We should avoid mutating the base_ctx in-place; instead, use a shallow copy.
        """
        ctx_dict: Dict[str, Any] = dict(base_ctx.to_dict() if hasattr(base_ctx, "to_dict") else dict(base_ctx))

        ctx_dict["goal"] = goal

        # Attach the section as the primary source
        ctx_dict["source_section"] = {
            "id": section.get("id") or section.get("section_id"),
            "title": section.get("title") or section.get("section_name"),
            "text": section.get("text") or section.get("content") or "",
        }

        # Also attach some minimal paper meta if available
        ctx_dict.setdefault("paper", {})
        ctx_dict["paper"].setdefault("id", doc_meta.get("document_id"))
        ctx_dict["paper"].setdefault("title", doc_meta.get("title"))
        ctx_dict["paper"].setdefault("abstract", paper.get("abstract", ""))

        # If there's an initial section draft from ingest, provide it as a candidate hint
        draft_blog = ctx_dict.get("draft_blog") or {}
        draft_sections = draft_blog.get("sections") or []
        for ds in draft_sections:
            ds_id = ds.get("section_id") or ds.get("id")
            if str(ds_id) == str(section.get("id") or section.get("section_id")):
                ctx_dict["initial_draft_candidate"] = ds.get("draft_text") or ds.get(
                    "text", ""
                )
                break

        return ctx_dict

    # ------------------------------------------------------------------
    # Helpers: calling Memento per section
    # ------------------------------------------------------------------

    async def _run_memento_for_section(
        self,
        section_context: Dict[str, Any],
        goal: Dict[str, Any],
    ) -> Tuple[str, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Call MementoAgent for a single section and normalize its outputs.

        Returns:
            best_text: str
            candidates: list of {id, text, score, rank, source}
            case_ref: optional {case_id, variant}
        """
        # MementoAgent.run returns a context-like dict with ranked hypotheses
        m_ctx = await self.memento_agent.run(section_context)

        # Memento projects hypotheses under self.output_key; by default in the
        # implementation you shared, that's often "hypotheses" or similar.
        # We try a few common keys and fall back gracefully.
        ranked = (
            m_ctx.get("hypotheses")
            or m_ctx.get("memento_hypotheses")
            or m_ctx.get("outputs")
            or []
        )

        if not ranked:
            # Fallback: if Memento produced nothing, either reuse initial draft or
            # synthesize a trivial stub.
            initial = section_context.get("initial_draft_candidate") or ""
            if initial:
                return initial, [
                    {
                        "id": "initial_draft",
                        "text": initial,
                        "score": 0.0,
                        "rank": 0,
                        "source": "initial_draft",
                    }
                ], None

            stub = f"(TODO: write section for goal: {goal.get('goal_text')[:160]}...)"
            return stub, [
                {
                    "id": "stub",
                    "text": stub,
                    "score": 0.0,
                    "rank": 0,
                    "source": "stub",
                }
            ], None

        # Normalize candidates
        candidates: List[Dict[str, Any]] = []
        for i, c in enumerate(ranked):
            candidates.append(
                {
                    "id": c.get("id") or f"cand_{i}",
                    "text": c.get("text") or c.get("content") or "",
                    "score": c.get("mars_confidence")
                    or c.get("score")
                    or 0.0,
                    "rank": c.get("rank", i),
                    "source": c.get("source", "memento"),
                }
            )

        # Best text = top-ranked candidate
        best = min(candidates, key=lambda x: x["rank"])
        best_text = best["text"]

        # Try to extract casebook info if Memento wrote it into context
        case_ref = None
        m_meta = m_ctx.get("_MEMENTO_META") or {}
        last_case = m_meta.get("retained_case")
        if last_case:
            case_ref = {
                "case_id": last_case.get("id"),
                "variant": last_case.get("variant", "cbr"),
            }

        return best_text, candidates, case_ref

    # ------------------------------------------------------------------
    # Helpers: assemble outline & full text
    # ------------------------------------------------------------------

    def _build_outline(self, sections: List[ReasonedSection]) -> List[Dict[str, Any]]:
        """
        Basic outline: just use sections in order with their titles.
        Later we can add a small LLM step to refine headings.
        """
        outline: List[Dict[str, Any]] = []
        for sec in sections:
            outline.append(
                {
                    "id": sec.section_id,
                    "title": sec.title,
                    "source": "reasoned",
                }
            )
        return outline

    def _build_full_text(
        self,
        outline: List[Dict[str, Any]],
        sections: List[ReasonedSection],
    ) -> str:
        """
        Build a simple markdown document from the chosen sections.
        """
        sec_by_id = {s.section_id: s for s in sections}
        chunks: List[str] = []

        for item in outline:
            sec_id = item["id"]
            sec = sec_by_id.get(sec_id)
            if not sec:
                continue
            title = item["title"]
            chunks.append(f"## {title}\n\n{sec.text}")

        return "\n\n".join(chunks)
