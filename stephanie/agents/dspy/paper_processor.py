# stephanie/agents/dspy/paper_processor.py
import dspy
from typing import Dict, Any, List, Optional
import json
import re
import uuid
import time
from stephanie.agents.base_agent import BaseAgent
from stephanie.models.casebook import CaseBookORM
from stephanie.scoring.scorable_factory import TargetType
from stephanie.utils.json_sanitize import sanitize_for_json

# DSPy Signatures for each step in the processing pipeline
class ClaimExtractionSignature(dspy.Signature):
    """Extract key claims from a paper section with evidence grounding"""
    section_text = dspy.InputField(desc="Text of the paper section")
    claims = dspy.OutputField(desc="List of key claims with evidence grounding", 
                             format=lambda x: json.dumps(x, indent=2))

class ContextFusionSignature(dspy.Signature):
    """Fuse paper claims with relevant conversation history"""
    claims = dspy.InputField(desc="Key claims extracted from paper section")
    conversation_history = dspy.InputField(desc="Relevant conversation history")
    fused_context = dspy.OutputField(desc="Fused context with paper claims and conversation insights",
                                    format=lambda x: json.dumps(x, indent=2))

class DraftGenerationSignature(dspy.Signature):
    """Generate a blog section draft from fused context"""
    fused_context = dspy.InputField(desc="Fused context with paper claims and conversation insights")
    blog_section = dspy.OutputField(desc="Well-structured blog section draft",
                                   format=lambda x: json.dumps(x, indent=2))

class VerificationSignature(dspy.Signature):
    """Verify draft against knowledge base and identify gaps"""
    draft = dspy.InputField(desc="Blog section draft")
    knowledge_base = dspy.InputField(desc="Knowledge base with key facts and claims")
    verification_report = dspy.OutputField(desc="Verification report identifying gaps and issues",
                                          format=lambda x: json.dumps(x, indent=2))

class RefinementSignature(dspy.Signature):
    """Refine draft based on verification report"""
    draft = dspy.InputField(desc="Original blog section draft")
    verification_report = dspy.InputField(desc="Verification report identifying gaps and issues")
    refined_draft = dspy.OutputField(desc="Refined blog section draft",
                                    format=lambda x: json.dumps(x, indent=2))

class FinalValidationSignature(dspy.Signature):
    """Final validation of refined draft against quality standards"""
    refined_draft = dspy.InputField(desc="Refined blog section draft")
    quality_standards = dspy.InputField(desc="Quality standards for blog sections")
    final_validation = dspy.OutputField(desc="Final validation report with quality scores",
                                       format=lambda x: json.dumps(x, indent=2))

class DSPyPaperSectionProcessor(BaseAgent):
    """
    DSPy-based processor for transforming paper sections into high-quality blog posts
    Uses structured reasoning, iterative refinement, and verification against knowledge base
    """
    
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        
        # DSPy modules
        self.claim_extractor = dspy.ChainOfThought(ClaimExtractionSignature)
        self.context_fuser = dspy.ChainOfThought(ContextFusionSignature)
        self.draft_generator = dspy.ChainOfThought(DraftGenerationSignature)
        self.verification_module = dspy.ChainOfThought(VerificationSignature)
        self.refinement_module = dspy.ChainOfThought(RefinementSignature)
        self.final_validator = dspy.ChainOfThought(FinalValidationSignature)
        
        # Configuration
        self.max_refinements = cfg.get("max_refinements", 3)
        self.min_quality_threshold = cfg.get("min_quality_threshold", 0.85)
        self.casebook_name = cfg.get("casebook_name", "PaperSectionProcessing")
        self.goal_template = cfg.get("goal_template", "academic_summary")
        self.min_section_length = cfg.get("min_section_length", 100)
        
        # Initialize DSPy
        self._init_dspy()
        
        self.logger.info("DSPyPaperSectionProcessor initialized", {
            "max_refinements": self.max_refinements,
            "min_quality_threshold": self.min_quality_threshold,
            "casebook_name": self.casebook_name
        })
    
    def _init_dspy(self):
        """Initialize DSPy with appropriate configuration"""
        # Configure DSPy for best performance
        dspy.settings.configure(
            lm=self.container.get("lm"),  # Language model from container
            rm=self.container.get("rm"),  # Retrieval model from container
            max_tokens=2000,
            temperature=0.7
        )
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main entrypoint for processing paper sections"""
        self.report({"event": "start", "step": "DSPyPaperSectionProcessor", "details": "Processing paper sections"})
        
        # Get document data
        documents = context.get(self.input_key, [])
        processed_sections = []
        casebook = self._get_casebook()
        
        for doc in documents:
            doc_id = doc.get("id")
            structured_data = doc.get("structured_data", {})
            title = doc.get("title", "")
            
            if not structured_data:
                self.logger.warning(f"No structured data for document {doc_id}")
                continue
            
            self.logger.info(f"Processing document {doc_id} with {len(structured_data)} sections")
            
            # Process sections in order of importance
            for section in structured_data:
                section_name = section.get("section_name", "section")
                section_text = section.get("section_text", "")
                
                if len(section_text) < self.min_section_length:
                    continue
                
                try:
                    # Process section through DSPy pipeline
                    section_context = {
                        "section_text": section_text,
                        "section_name": section_name,
                        "goal_template": self.goal_template,
                        "paper_id": doc_id,
                        "pipeline_run_id": context.get("pipeline_run_id"),
                        "casebook_id": casebook.id,
                        "source": "document_profiler"
                    }
                    
                    # Run DSPy pipeline
                    result = await self._process_section(section_context)
                    
                    # Save to case book
                    self._save_section_to_casebook(casebook, doc_id, section_name, section_text, result, context)
                    
                    processed_sections.append({
                        "doc_id": doc_id,
                        "section_name": section_name,
                        "summary": result.get("final_draft", ""),
                        "metrics": result.get("validation_report", {}),
                        "valid": result.get("passed", False)
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error processing section {section_name} for doc {doc_id}: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        context[self.output_key] = {
            "processed_sections": processed_sections,
            "casebook_id": casebook.id
        }
        
        self.report({
            "event": "end",
            "step": "DSPyPaperSectionProcessor",
            "details": f"Processed {len(processed_sections)} sections across {len(documents)} documents"
        })
        
        return context
    
    async def _process_section(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single section through DSPy pipeline"""
        section_text = context.get("section_text", "")
        section_name = context.get("section_name", "section")
        paper_id = context.get("paper_id", "unknown")
        
        # Step 1: Extract claims from paper section
        claim_extraction = self.claim_extractor(section_text=section_text)
        claims = json.loads(claim_extraction.claims)
        
        # Step 2: Fuse with conversation history
        conversation_history = self._get_relevant_conversations(paper_id, claims)
        context_fusion = self.context_fuser(
            claims=json.dumps(claims),
            conversation_history=json.dumps(conversation_history)
        )
        fused_context = json.loads(context_fusion.fused_context)
        
        # Step 3: Generate initial draft
        draft_generation = self.draft_generator(fused_context=json.dumps(fused_context))
        draft = json.loads(draft_generation.blog_section)
        
        # Step 4: Verify draft against knowledge base
        knowledge_base = self._get_knowledge_base(paper_id)
        verification = self.verification_module(
            draft=json.dumps(draft),
            knowledge_base=json.dumps(knowledge_base)
        )
        verification_report = json.loads(verification.verification_report)
        
        # Step 5: Refine draft iteratively
        refined_draft = draft
        for i in range(self.max_refinements):
            refinement = self.refinement_module(
                draft=json.dumps(refined_draft),
                verification_report=json.dumps(verification_report)
            )
            refined_draft = json.loads(refinement.refined_draft)
            
            # Re-verify after refinement
            verification = self.verification_module(
                draft=json.dumps(refined_draft),
                knowledge_base=json.dumps(knowledge_base)
            )
            verification_report = json.loads(verification.verification_report)
            
            # Check if we've reached quality threshold
            if self._is_quality_threshold_met(verification_report):
                break
        
        # Step 6: Final validation
        quality_standards = {
            "coverage": 0.8,
            "correctness": 0.85,
            "coherence": 0.8,
            "citation_support": 0.9,
            "readability": 0.85,
            "novelty": 0.75
        }
        final_validation = self.final_validator(
            refined_draft=json.dumps(refined_draft),
            quality_standards=json.dumps(quality_standards)
        )
        validation_report = json.loads(final_validation.final_validation)
        
        return {
            "initial_draft": draft,
            "refined_draft": refined_draft,
            "verification_report": verification_report,
            "final_validation": validation_report,
            "passed": self._is_quality_threshold_met(validation_report),
            "refinement_iterations": i + 1
        }
    
    def _get_relevant_conversations(self, paper_id: str, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Retrieve relevant conversations based on paper claims"""
        # In a real implementation, this would query the conversation history
        # using the claims as search terms
        relevant_conversations = []
        
        # Example implementation (would be replaced with actual retrieval)
        for claim in claims:
            claim_text = claim.get("claim", "")
            # Query conversation history for relevant snippets
            # (This would be implemented with actual retrieval model)
            snippets = self._search_conversations(claim_text)
            relevant_conversations.extend(snippets)
        
        # Deduplicate and sort by relevance
        unique_conversations = []
        seen = set()
        for conv in relevant_conversations:
            if conv["text"] not in seen:
                seen.add(conv["text"])
                unique_conversations.append(conv)
        
        # Sort by relevance score (would be calculated in real implementation)
        unique_conversations.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Return top 10 relevant conversations
        return unique_conversations[:10]
    
    def _search_conversations(self, query: str) -> List[Dict[str, Any]]:
        """Search conversation history for relevant snippets"""
        # In a real implementation, this would use the retrieval model
        # to find relevant conversations based on the query
        return [{
            "role": "expert",
            "text": f"Relevant insight about {query[:20]}...",
            "relevance_score": 0.9
        }]
    
    def _get_knowledge_base(self, paper_id: str) -> Dict[str, Any]:
        """Retrieve knowledge base for the paper"""
        # In a real implementation, this would retrieve structured knowledge
        # from the knowledge graph or other sources
        return {
            "paper_id": paper_id,
            "key_facts": [
                "Key fact 1 from knowledge base",
                "Key fact 2 from knowledge base"
            ],
            "related_papers": [
                "Paper ID 1",
                "Paper ID 2"
            ]
        }
    
    def _is_quality_threshold_met(self, validation_report: Dict[str, Any]) -> bool:
        """Check if quality thresholds are met"""
        # Extract quality scores from validation report
        scores = validation_report.get("scores", {})
        
        # Check each dimension against threshold
        thresholds = {
            "coverage": 0.8,
            "correctness": 0.85,
            "coherence": 0.8,
            "citation_support": 0.9,
            "readability": 0.85,
            "novelty": 0.75
        }
        
        for dim, threshold in thresholds.items():
            if scores.get(dim, 0) < threshold:
                return False
        
        return True
    
    def _get_casebook(self) -> CaseBookORM:
        """Get or create a casebook for this processing run"""
        casebook = self.memory.casebooks.get_casebook_by_name(self.casebook_name)
        if not casebook:
            casebook = self.memory.casebooks.create_casebook(
                name=self.casebook_name,
                description="Casebook for DSPy paper section processing",
                tag="paper_processing"
            )
        return casebook
    
    def _save_section_to_casebook(self, casebook: CaseBookORM, doc_id: str, section_name: str, 
                                 section_text: str, result: Dict[str, Any], context: Dict[str, Any]):
        """Save a section and its results to the case book"""
        # Create case for this section
        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id,
            goal_id=context.get("goal_id"),
            prompt_text=json.dumps({
                "section_name": section_name,
                "paper_id": context.get("paper_id")
            }),
            agent_name=self.name,
            meta={
                "type": "draft_trajectory",
                "section_name": section_name,
                "paper_id": context.get("paper_id"),
                "timestamp": time.time()
            }
        )
        
        # Save raw section text
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            scorable_id=str(uuid.uuid4()),
            text=section_text,
            role="section_text",
            meta={
                "section_name": section_name,
                "paper_id": context.get("paper_id")
            }
        )
        
        # Save initial draft
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            scorable_id=str(uuid.uuid4()),
            text=json.dumps(result.get("initial_draft", {})),
            role="initial_draft",
            meta={
                "section_name": section_name,
                "paper_id": context.get("paper_id")
            }
        )
        
        # Save refined draft
        refined_draft = result.get("refined_draft", {})
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            scorable_id=str(uuid.uuid4()),
            text=json.dumps(refined_draft),
            role="refined_draft",
            meta={
                "section_name": section_name,
                "paper_id": context.get("paper_id"),
                "refinement_iterations": result.get("refinement_iterations", 0)
            }
        )
        
        # Save verification report
        verification_report = result.get("verification_report", {})
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            scorable_id=str(uuid.uuid4()),
            text=json.dumps(verification_report),
            role="verification_report",
            meta={
                "section_name": section_name,
                "paper_id": context.get("paper_id")
            }
        )
        
        # Save final validation report
        final_validation = result.get("final_validation", {})
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            scorable_id=str(uuid.uuid4()),
            text=json.dumps(final_validation),
            role="final_validation",
            meta={
                "section_name": section_name,
                "paper_id": context.get("paper_id")
            }
        )
        
        # Save metrics
        metrics = {
            "passed": result.get("passed", False),
            "refinement_iterations": result.get("refinement_iterations", 0),
            "verification_report": verification_report,
            "final_validation": final_validation
        }
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            scorable_id=str(uuid.uuid4()),
            text=json.dumps(metrics),
            role="metrics",
            meta={
                "section_name": section_name,
                "paper_id": context.get("paper_id")
            }
        )