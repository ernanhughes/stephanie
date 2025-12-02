# stephanie/agents/information_ingest.py
from __future__ import annotations
from stephanie.components.information.graph_builder import InformationGraphBuilder
from stephanie.components.information.quality import InformationQualityPass

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.models.casebook import CaseBookORM, CaseORM
from stephanie.scoring.scorable import ScorableType
from stephanie.utils.casebook_utils import generate_casebook_name
from stephanie.utils.paper_utils import (
    build_paper_goal_text,
    build_paper_goal_meta,
) 

# These are the new Information component types we designed earlier.
# Adjust the import paths if you put them somewhere else.
from stephanie.components.information.models import (
    InformationSource,
    InformationTargetConfig,
    InformationRequest,
    InformationResult,
)
from stephanie.components.information.processor import InformationProcessor
import logging

log = logging.getLogger(__name__)

@dataclass
class InformationIngestConfig:
    """
    Thin cfg wrapper so we don't sprinkle raw dicts everywhere.
    """
    input_key: str = "documents"
    casebook_action: str = "information_ingest"
    min_section_length: int = 120
    single_random_doc: bool = True

    # Sub-config: handed to InformationProcessor
    information: Dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "InformationIngestConfig":
        return cls(
            input_key=cfg.get("input_key", "documents"),
            casebook_action=cfg.get("casebook_action", "information_ingest"),
            min_section_length=int(cfg.get("min_section_length", 120)),
            single_random_doc=bool(cfg.get("single_random_doc", True)),
            information=cfg.get("information", {}) or {},
        )


class InformationIngestAgent(BaseAgent):
    """
    Take a single document and turn it into:
      - A CaseBook with per-section cases
      - A MemCube representing the 'information object'
      - A blog-style markdown draft for that MemCube

    This is the "doc → Info MemCube + blog draft" entry point.
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger) -> None:
        super().__init__(cfg, memory, container, logger)
        self.cfg_struct = InformationIngestConfig.from_dict(cfg)
        self.input_key = self.cfg_struct.input_key
        self.casebook_action = self.cfg_struct.casebook_action
        self.min_section_length = self.cfg_struct.min_section_length
        self.single_random_doc = self.cfg_struct.single_random_doc

        # Information processor (bucket → MemCube + blog)
        self.information_processor = InformationProcessor(
            cfg=self.cfg_struct.information or {},
            memory=memory,
            container=container,
            logger=logger,
        )

        kg = self.container.get("knowledge_graph") 
        self.graph_builder = InformationGraphBuilder(
            knowledge_graph_service=kg,  # or however you access it
            logger=logger,
        )


    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point.

        1. Pick a document (from context or random from memory).
        2. Create a CaseBook + section cases.
        3. Build an InformationRequest from that document.
        4. Ask InformationProcessor to:
           - build bucket
           - build MemCube
           - generate blog-style markdown draft.
        5. Attach a small summary into context under `information_ingest`.
        """
        documents = context.get(self.input_key, []) or []

        # If configured, ignore provided docs and just pick one at random
        if self.single_random_doc and not documents:
            doc = self.memory.documents.get_random()
            if doc is not None:
                documents = [doc.to_dict()]
        elif self.single_random_doc and documents:
            # Just take the first one
            documents = [documents[0]]

        self.report(
            {
                "event": "InformationIngest:input",
                "agent": self.name,
                "docs_count": len(documents),
            }
        )

        if not documents:
            # Nothing to do
            context["information_ingest"] = {
                "status": "no_documents",
                "message": "No documents available for ingestion.",
            }
            return context

        pipeline_run_id = context.get("pipeline_run_id")
        all_results: List[Dict[str, Any]] = []

        for paper in documents:
            res = await self._process_single_document(paper, pipeline_run_id, context)
            all_results.append(res)

        # For now, return all doc-results together
        context["information_ingest"] = {
            "status": "ok",
            "documents": all_results,
        }
        return context

    # ------------------------------------------------------------------
    # Core per-document logic
    # ------------------------------------------------------------------

    async def _process_single_document(
        self,
        paper: Dict[str, Any],
        pipeline_run_id: Optional[int],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process one document:
          - CaseBook + goal
          - section cases
          - InformationProcessor call
        """
        doc_id = paper.get("id") or paper.get("doc_id")
        title = paper.get("title", "") or f"Doc {doc_id}"

        # 1) CaseBook + Goal (same pattern as LfL but new description)
        casebook_name = generate_casebook_name(self.casebook_action, title)
        casebook = self.memory.casebooks.ensure_casebook(
            name=casebook_name,
            pipeline_run_id=pipeline_run_id,
            description=f"Information ingest runs for document {title}",
            tags=[self.casebook_action],
            meta={"source": "information_ingest"},
        )

        goal = self.memory.goals.get_or_create(
            {
                "goal_text": build_paper_goal_text(title),
                "description": "Information ingest: build MemCube + blog-style summary.",
                "meta": build_paper_goal_meta(
                    title, doc_id, domains=self.cfg.get("domains", [])
                ),
            }
        ).to_dict()

        # 2) Resolve sections (reuse LfL logic)
        sections = self._resolve_sections_with_attributes(paper, context)

        # Optional: progress hooks if you want
        self.logger.log(
            "InformationIngest_StartPaper",
            {
                "doc_id": doc_id,
                "title": title,
                "sections": len(sections),
                "casebook_id": casebook.id,
            },
        )

        # 3) Create per-section cases (for UI & future learning)
        section_cases: List[CaseORM] = []
        for section in sections:
            section_text = section.get("section_text") or ""
            if len(section_text.strip()) < self.min_section_length:
                continue

            case = self._create_section_case(
                casebook=casebook,
                paper=paper,
                section=section,
                context={"goal": goal},
            )
            section_cases.append(case)

        # 4) Build InformationRequest for this document
        info_request = self._build_information_request(
            paper=paper,
            sections=sections,
            casebook=casebook,
            goal=goal,
            context=context,
        )

        # 5) Invoke InformationProcessor (this is where the magic happens)
        info_result: InformationResult = await self.information_processor.process(
            info_request
        )

        print(info_result.markdown_path)
        
        # 6) Build knowledge graph entries
        try:
            self.graph_builder.build_from_information(info_request, info_result)
        except Exception as e:
            self.logger.log(
                "InformationIngest_GraphBuilderError",
                {"error": repr(e)},
            )

        quality_pass = InformationQualityPass(memcube_store=self.memory.memcubes, logger=self.logger)

        # For a single cube
        quality_pass.run_for_memcube_id(info_result.memcube_id)

 


        # 7) Assemble compact summary for the caller
        result_summary = {
            "document_id": doc_id,
            "title": title,
            "casebook_id": casebook.id,
            "goal_id": goal["id"],
            "memcube_id": info_result.memcube_id,
            "blog_markdown": info_result.blog_markdown,
            "bucket_id": info_result.bucket_id,
            "section_count": len(section_cases),
            "markdown_path": info_result.markdown_path,
        }

        self.logger.log(
            "InformationIngest_DonePaper",
            {
                **result_summary,
                "section_cases": [c.id for c in section_cases],
            },
        )

        return result_summary

    # ------------------------------------------------------------------
    # InformationRequest builder
    # ------------------------------------------------------------------

    def _build_information_request(
        self,
        paper: Dict[str, Any],
        sections: List[Dict[str, Any]],
        casebook: CaseBookORM,
        goal: Dict[str, Any],
        context: Dict[str, Any],
    ) -> InformationRequest:
        """
        Turn the document + sections into an InformationRequest that the
        InformationProcessor can consume.
        """
        doc_id = paper.get("id") or paper.get("doc_id")
        title = paper.get("title", "") or f"Doc {doc_id}"

        # Primary source text (fallback if we don't have sections)
        main_text = paper.get("text") or paper.get("abstract") or ""
        if not main_text and sections:
            # Concatenate sections as a fallback
            main_text = "\n\n".join(
                f"# {s['section_name']}\n\n{s['section_text']}"
                for s in sections
                if s.get("section_text")
            )

        source = InformationSource(
            kind="document",
            id=str(doc_id),
            title=title,
            text=main_text,
            meta={
                "paper_id": str(doc_id),
                "title": title,
                "section_names": [s["section_name"] for s in sections],
                "source": "information_ingest",
            },
        )

        # Target: MemCube (blog-view enabled)
        target_cfg = InformationTargetConfig(
            kind="memcube",
            name=title,
            description=f"Information MemCube for document: {title}",
            goal_id=goal["id"],
            casebook_id=casebook.id,
            enable_blog_view=True,
            # Optional: you can add defaults here for attributes, tags, etc.
            meta={
                "paper_id": str(doc_id),
                "source": "information_ingest",
                "domains": self.cfg.get("domains", []),
            },
        )

        return InformationRequest(
            sources=[source],
            target=target_cfg,
            context={
                **context,
                "goal": goal,
                "document": paper,
                "casebook_id": casebook.id,
            },
        )

    # ------------------------------------------------------------------
    # Section resolving / case creation (lifted from LfL with tiny tweaks)
    # ------------------------------------------------------------------

    def _resolve_sections_with_attributes(
        self, paper: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Normalize sections to dicts with attributes.

        Reuses the LfL pattern:
          - Try document_sections table
          - Fallback to a single 'Abstract' pseudo-section
        """
        doc_id = paper.get("id") or paper.get("doc_id")
        sections = self.memory.document_sections.get_by_document(doc_id) or []

        if sections:
            out: List[Dict[str, Any]] = []
            for sec in sections:
                out.append(
                    {
                        "section_name": sec.section_name,
                        "section_text": sec.section_text or "",
                        "section_id": sec.id,
                        "order_index": getattr(sec, "order_index", None),
                        "attributes": {
                            "paper_id": str(doc_id),
                            "section_name": sec.section_name,
                            "section_index": getattr(sec, "order_index", 0),
                            "case_kind": "summary",
                        },
                    }
                )
            return out

        # Fallback single "Abstract" section
        return [
            {
                "section_name": "Abstract",
                "section_text": f"{paper.get('title', '').strip()}\n\n{paper.get('abstract', '').strip()}",
                "section_id": None,
                "order_index": 0,
                "attributes": {
                    "paper_id": str(doc_id),
                    "section_name": "Abstract",
                    "section_index": 0,
                    "case_kind": "summary",
                },
            }
        ]

    def _create_section_case(
        self,
        casebook: CaseBookORM,
        paper: Dict[str, Any],
        section: Dict[str, Any],
        context: Dict[str, Any],
    ) -> CaseORM:
        """
        Create a case and attach attributes for this section (universal casebook pattern).

        This is basically the LfL version, with a couple of extra attrs
        (case_name / case_description) to make later UI / querying easier.
        """
        section_name = section["section_name"]
        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id,
            goal_id=context.get("goal").get("id"),
            prompt_text=f"Section: {section_name}",
            agent_name=self.name,
            meta={"type": "information_section_case"},
        )

        doc_id = paper.get("id") or paper.get("doc_id")

        # Core attributes
        self.memory.casebooks.set_case_attr(
            case.id,
            "paper_id",
            value_text=str(doc_id),
        )
        self.memory.casebooks.set_case_attr(
            case.id, "section_name", value_text=str(section_name)
        )
        if section.get("section_id") is not None:
            self.memory.casebooks.set_case_attr(
                case.id, "section_id", value_text=str(section["section_id"])
            )
        if section.get("order_index") is not None:
            self.memory.casebooks.set_case_attr(
                case.id,
                "section_index",
                value_num=float(section.get("order_index") or 0),
            )
        self.memory.casebooks.set_case_attr(
            case.id, "case_kind", value_text="summary"
        )
        self.memory.casebooks.set_case_attr(
            case.id,
            "scorable_id",
            value_text=str(section.get("section_id") or ""),
        )
        self.memory.casebooks.set_case_attr(
            case.id,
            "scorable_type",
            value_text=str(ScorableType.DOCUMENT_SECTION),
        )

        # Extra: "name" and "description" style attrs for easier querying
        self.memory.casebooks.set_case_attr(
            case.id,
            "case_name",
            value_text=f"{section_name} ({paper.get('title','')})",
        )
        self.memory.casebooks.set_case_attr(
            case.id,
            "case_description",
            value_text=f"Information ingest section '{section_name}' for paper '{paper.get('title','')}'",
        )

        return case
