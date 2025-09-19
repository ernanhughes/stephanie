# stephanie/agents/thought/paper_section_processor.py
from __future__ import annotations

import json
import time
import traceback
import uuid
from typing import Any, Dict

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.thought.paper_blog import SimplePaperBlogAgent
from stephanie.models.casebook import CaseBookORM
from stephanie.scoring.scorable_factory import TargetType


class PaperSectionProcessorAgent(BaseAgent):
    """
    Processes each section of a document individually, generates summaries,
    and logs the process to a case book for future reference.
    
    This agent:
    - Takes structured document data from DocumentProfilerAgent
    - Processes each section individually with the PaperSummarizer
    - Saves each section's input, summary, and metrics to a case book
    - Creates a comprehensive case book for the entire paper
    - Tracks section-by-section processing for future analysis
    
    # PRODUCTION-READY CASEBOOK MANAGEMENT
    #
    # This agent implements a structured casebook system that:
    # 1. Models a "blog casebook" explicitly with a structured naming convention
    # 2. Defines a tight case taxonomy (roles)
    # 3. Uses consistent scoring schema
    # 4. Implements global indexing strategy (not per-casebook)
    # 5. Manages entities and claims pipeline
    # 6. Tracks lineage between cases for provenance
    # 7. Supports governance & publishing gates
    #
    # Key principles:
    # - One casebook per blog post (not per paper)
    # - Cases are immutable; new iterations create new cases
    # - Text stored in CaseScorable rows, not in case meta
    # - Global index with metadata filters (not per-casebook indices)
    # - Clear role taxonomy for cases
    # - Lineage tracking between cases
    """
    
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.summarizer = SimplePaperBlogAgent(cfg, memory, container, logger)
        self.goal_template = cfg.get("goal_template", "academic_summary")
        self.min_section_length = cfg.get("min_section_length", 100)
        self.max_sections = cfg.get("max_sections", 10)
        
        # [CASEBOOK MANAGEMENT] Define casebook naming convention
        # Format: "blog::<paper_id>::<post_slug>"
        # Paper ID is the arXiv ID or document ID
        # Post slug is a normalized version of the post title
        self.casebook_name_template = cfg.get("casebook_name_template", "blog::{paper_id}::{post_slug}")
        
        # [CASEBOOK MANAGEMENT] Define case taxonomy (roles)
        # These are standardized roles for different types of cases
        self.case_roles = {
            "input_section": "raw section text from document",
            "summary_baseline": "Track A baseline summary",
            "summary_sharpened": "Track B sharpened summary",
            "summary_verified": "Track C verified summary",
            "critique": "hallucination/coverage notes",
            "citation_check": "citation verification results",
            "figure_grounding": "figure/table grounding results",
            "edit_patch": "diffs between draft versions",
            "final_section": "final approved section",
            "seo_meta": "SEO metadata for section",
            "social_snippet": "social media snippet for section",
            "claim": "extracted atomic claim",
            "entity": "linked entity information",
            "chat_turn": "conversation turn related to this section"
        }
        
        # [SCORING] Define consistent scoring schema
        # All metrics should be in [0,1] range with standardized names
        self.metrics_schema = [
            "overall", "coverage", "faithfulness", "coherence", "structure",
            "hallucination_rate", "knowledge_verification", "figure_grounding",
            "readability", "style_fit", "citation_support", "novelty", "stickiness"
        ]
        
        self.logger.info("PaperSectionProcessorAgent initialized", {
            "goal_template": self.goal_template,
            "min_section_length": self.min_section_length,
            "max_sections": self.max_sections,
            "case_roles": self.case_roles,
            "metrics_schema": self.metrics_schema
        })
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.report({
            "event": "start",
            "step": "PaperSectionProcessor",
            "details": "Processing document sections"
        })
        
        # Get structured document data from document_profiler
        documents = context.get(self.input_key, [])
        processed_sections = []
        # [CASEBOOK MANAGEMENT] Create casebook with structured naming
        casebook = self._get_casebook(context)
        
        for doc in documents:
            doc_id = doc.get("id")
            structured_data = doc.get("structured_data", {})
            title = doc.get("title", "")
            paper_id = doc.get("paper_id", doc_id)
            arxiv_id = doc.get("arxiv_id", "")
            
            if not structured_data:
                self.logger.warning(f"No structured data for document {doc_id}")
                continue
                
            self.logger.info(f"Processing document {doc_id} with {len(structured_data)} sections")
            
            # Process sections in order of importance
            section_order = ["abstract", "methods", "results", "conclusions", "introduction", "title"]
            sorted_sections = sorted(
                structured_data.items(),
                key=lambda x: section_order.index(x[0]) if x[0] in section_order else len(section_order)
            )
            
            for section_name, section_text in sorted_sections[:self.max_sections]:
                # Skip very short sections
                if len(section_text) < self.min_section_length:
                    self.logger.debug(f"Skipping short section '{section_name}' for doc {doc_id}")
                    continue
                    
                try:
                    # [CASEBOOK MANAGEMENT] Create context with proper casebook reference
                    section_context = {
                        "id": f"{doc_id}_{section_name}",
                        "title": title,
                        "summary": section_text,  # This is the section text
                        "section_name": section_name,
                        "goal_template": self.goal_template,
                        "paper_id": paper_id,
                        "arxiv_id": arxiv_id,
                        "pipeline_run_id": context.get("pipeline_run_id"),
                        "casebook_id": casebook.id,
                        "source": "document_profiler"
                    }
                    
                    # [CASEBOOK MANAGEMENT] Run summarizer on this section
                    section_summary = await self.summarizer.run(section_context)
                    
                    # [CASEBOOK MANAGEMENT] Save to case book with proper roles
                    self._save_section_to_casebook(
                        casebook, 
                        doc_id, 
                        section_name, 
                        section_text, 
                        section_summary,
                        context
                    )
                    
                    # [CASEBOOK MANAGEMENT] Track lineage and versioning
                    processed_sections.append({
                        "doc_id": doc_id,
                        "section_name": section_name,
                        "summary": section_summary.get("summary", ""),
                        "metrics": section_summary.get("metrics", {}),
                        "valid": section_summary.get("valid", False),
                        "case_id": section_summary.get("case_id"),  # Track the case ID
                        "version": section_summary.get("version", 1)  # Track version
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error processing section {section_name} for doc {doc_id}: {str(e)}")
                    traceback.print_exc()
                    
        context[self.output_key] = {
            "processed_sections": processed_sections,
            "casebook_id": casebook.id,
            "casebook_name": casebook.name  # Track the actual casebook name
        }
        
        self.report({
            "event": "end",
            "step": "PaperSectionProcessor",
            "details": f"Processed {len(processed_sections)} sections across {len(documents)} documents"
        })
        
        return context
        
    def _get_casebook(self, context: Dict[str, Any]) -> CaseBookORM:
        """Get or create a casebook for this processing run with structured naming
        
        # [CASEBOOK MANAGEMENT] Casebook naming convention:
        # Format: "blog::<paper_id>::<post_slug>"
        # Where:
        # - paper_id: arXiv ID or document ID
        # - post_slug: normalized version of the post title
        #
        # This ensures each blog post has its own dedicated casebook
        # with clear boundaries and provenance.
        """
        # Get paper metadata from context
        paper_id = context.get("paper_id")
        post_title = context.get("post_title", "blog")
        
        # Create normalized post slug (e.g., "my-paper-title" from "My Paper Title")
        post_slug = self._normalize_slug(post_title)
        
        # Build casebook name using template
        casebook_name = self.casebook_name_template.replace("{paper_id}", paper_id).replace("{post_slug}", post_slug)
        
        # [CASEBOOK MANAGEMENT] Create casebook with metadata
        casebook = self.memory.casebooks.get_casebook_by_name(casebook_name)
        if not casebook:
            casebook = self.memory.casebooks.create_casebook(
                name=casebook_name,
                description=f"Blog post casebook for paper {paper_id}",
                tag="blog_post",
                meta={
                    "paper_id": paper_id,
                    "post_title": post_title,
                    "post_slug": post_slug,
                    "created_at": time.time(),
                    "status": "draft",
                    "version": 1
                }
            )
        return casebook
    
    def _normalize_slug(self, title: str) -> str:
        """Convert title to URL-friendly slug"""
        return re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
    
    def _save_section_to_casebook(
        self,
        casebook: CaseBookORM,
        doc_id: str,
        section_name: str,
        section_text: str,
        section_summary: Dict[str, Any],
        context: Dict[str, Any]
    ):
        """Save a section and its summary to the case book with proper taxonomy
        
        # [CASEBOOK MANAGEMENT] Case taxonomy & storage layout:
        # - Text stored in CaseScorable rows (not in case meta)
        # - Different roles for different types of content
        # - Metrics stored as separate scorables
        # - Lineage tracking between cases
        # - Immutable cases (new iterations create new cases)
        #
        # Roles:
        # - "input_section": raw section text
        # - "summary_baseline": Track A summary
        # - "summary_sharpened": Track B summary
        # - "summary_verified": Track C summary
        # - "metrics": structured metrics
        # - "critique": critique notes
        # - "edit_patch": diffs between versions
        # - "final_section": final approved version
        # - "claim": extracted claims
        # - "entity": linked entities
        #
        # All cases are immutable - new iterations create new cases
        """
        # [CASEBOOK MANAGEMENT] Create case with proper role and lineage
        # Get current version from context if available
        version = section_summary.get("version", 1)
        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id,
            goal_id=context.get("goal", {}).get("id"),
            prompt_text=json.dumps({
                "doc_id": doc_id,
                "section_name": section_name,
                "goal_template": self.goal_template,
                "version": version
            }),
            agent_name=self.name,
            role="input_section",  # This is the input section case
            meta={
                "type": "section_processing",
                "doc_id": doc_id,
                "section_name": section_name,
                "timestamp": time.time()
            }
        )
        
        # Save section text as scorable
        section_scorable = self.memory.casebooks.add_scorable(
            case_id=case.id,
            scorable_id=str(uuid.uuid4()),
            scorable_type=TargetType.DOCUMENT_SECTION,
            text=section_text,
            role="input_section",
            meta={
                "doc_id": doc_id,
                "section_name": section_name,
                "length": len(section_text)
            }
        )
        
        # Save summary as scorable
        summary_text = section_summary.get("summary", "")
        if not summary_text:
            # Try to get summary from the summary object
            if "summary_v0" in section_summary and doc_id in section_summary["summary_v0"]:
                summary_text = section_summary["summary_v0"][doc_id].get("summary", "")
        
        summary_scorable = self.memory.casebooks.add_scorable(
            case_id=case.id,
            scorable_id=str(uuid.uuid4()),
            scorable_type=TargetType.DYNAMIC,
            text=summary_text,
            role="summary",
            meta={
                "doc_id": doc_id,
                "section_name": section_name,
                "metrics": section_summary.get("metrics", {})
            }
        )
        
        # Save metrics as scorable
        metrics_text = json.dumps(section_summary.get("metrics", {}), indent=2)
        metrics_scorable = self.memory.casebooks.add_scorable(
            case_id=case.id,
            scorable_id=str(uuid.uuid4()),
            scorable_type=TargetType.METRICS,
            text=metrics_text,
            role="metrics",
            meta={
                "doc_id": doc_id,
                "section_name": section_name
            }
        )
        
        # Log to case book
        self.logger.log("SectionProcessed", {
            "case_id": case.id,
            "doc_id": doc_id,
            "section_name": section_name,
            "summary_length": len(summary_text),
            "metrics": section_summary.get("metrics", {})
        })
        
        # Log to KnowledgeBus for tracking
        if hasattr(self.memory, "bus") and self.memory.bus:
            self.memory.bus.publish("section.processed", {
                "case_id": case.id,
                "doc_id": doc_id,
                "section_name": section_name,
                "summary_length": len(summary_text),
                "metrics": section_summary.get("metrics", {})
            })