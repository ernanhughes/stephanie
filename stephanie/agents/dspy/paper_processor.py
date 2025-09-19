# stephanie/agents/dspy/paper_processor.py
import dspy
from typing import Dict, Any, List, Optional, Callable
import json
import traceback
import uuid
import time
from stephanie.agents.base_agent import BaseAgent
from stephanie.models.casebook import CaseBookORM
from stephanie.utils.casebook_utils import generate_casebook_name
from stephanie.utils.paper_utils import (
    build_paper_goal_text,
    build_paper_goal_meta,
    system_guidance_from_goal,
    section_goal_text,
    section_quality,
)


# DSPy Signatures for each step in the processing pipeline
class ClaimExtractionSignature(dspy.Signature):
    """Extract key claims from a paper section with evidence grounding"""

    section_text = dspy.InputField(desc="Text of the paper section")
    claims = dspy.OutputField(
        desc="List of key claims with evidence grounding",
        format=lambda x: json.dumps(x, indent=2),
    )


class ContextFusionSignature(dspy.Signature):
    """Fuse paper claims with relevant conversation history"""

    claims = dspy.InputField(desc="Key claims extracted from paper section")
    conversation_history = dspy.InputField(
        desc="Relevant conversation history"
    )
    fused_context = dspy.OutputField(
        desc="Fused context with paper claims and conversation insights",
        format=lambda x: json.dumps(x, indent=2),
    )


class IntroSynthesisSignature(dspy.Signature):
    """Synthesize a compelling blog introduction from abstract and existing summary."""

    title = dspy.InputField(desc="Paper title")
    abstract = dspy.InputField(desc="Paper abstract")
    prior_summary = dspy.InputField(
        desc="Existing summary (e.g., arXiv or earlier pass)"
    )
    audience = dspy.InputField(
        desc="Target audience description (e.g., 'technical blog readers')"
    )
    goals = dspy.InputField(
        desc="Author goals for the introduction (tone, emphasis, what to include)"
    )
    intro_draft = dspy.OutputField(
        desc="Structured blog introduction draft in JSON with keys: {hook, context, core_contributions, why_it_matters, preview}",
        format=lambda x: json.dumps(x, indent=2),
    )


class IntroVerificationSignature(dspy.Signature):
    """Check intro claims against abstract/summary and flag missing/unsupported items."""

    intro_draft = dspy.InputField(desc="Intro draft JSON")
    abstract = dspy.InputField(desc="Paper abstract")
    prior_summary = dspy.InputField(desc="Existing summary")
    verification_report = dspy.OutputField(
        desc="JSON with {unsupported_points[], missing_essentials[], redundancy[], scores{coverage, correctness, coherence}}",
        format=lambda x: json.dumps(x, indent=2),
    )


class IntroRefinementSignature(dspy.Signature):
    """Refine the intro based on verification feedback."""

    intro_draft = dspy.InputField(desc="Intro draft JSON")
    verification_report = dspy.InputField(desc="Verification report JSON")
    refined_intro = dspy.OutputField(
        desc="Refined intro draft JSON with same keys as intro_draft",
        format=lambda x: json.dumps(x, indent=2),
    )


class IntroFinalValidationSignature(dspy.Signature):
    """Final gate for intro quality."""

    refined_intro = dspy.InputField(desc="Refined intro draft JSON")
    quality_standards = dspy.InputField(
        desc="Thresholds JSON (coverage, correctness, coherence, readability, hook_strength)"
    )
    final_validation = dspy.OutputField(
        desc="Validation result JSON with {scores, passed, notes}",
        format=lambda x: json.dumps(x, indent=2),
    )


class DraftGenerationSignature(dspy.Signature):
    """Generate a blog section draft from fused context"""

    fused_context = dspy.InputField(
        desc="Fused context with paper claims and conversation insights"
    )
    blog_section = dspy.OutputField(
        desc="Well-structured blog section draft",
        format=lambda x: json.dumps(x, indent=2),
    )


class VerificationSignature(dspy.Signature):
    """Verify draft against knowledge base and identify gaps"""

    draft = dspy.InputField(desc="Blog section draft")
    knowledge_base = dspy.InputField(
        desc="Knowledge base with key facts and claims"
    )
    verification_report = dspy.OutputField(
        desc="Verification report identifying gaps and issues",
        format=lambda x: json.dumps(x, indent=2),
    )


class RefinementSignature(dspy.Signature):
    """Refine draft based on verification report"""

    draft = dspy.InputField(desc="Original blog section draft")
    verification_report = dspy.InputField(
        desc="Verification report identifying gaps and issues"
    )
    refined_draft = dspy.OutputField(
        desc="Refined blog section draft",
        format=lambda x: json.dumps(x, indent=2),
    )


class FinalValidationSignature(dspy.Signature):
    """Final validation of refined draft against quality standards"""

    refined_draft = dspy.InputField(desc="Refined blog section draft")
    quality_standards = dspy.InputField(
        desc="Quality standards for blog sections"
    )
    final_validation = dspy.OutputField(
        desc="Final validation report with quality scores",
        format=lambda x: json.dumps(x, indent=2),
    )


class DSPyPaperSectionProcessorAgent(BaseAgent):
    """
    DSPy-based processor for transforming paper sections into high-quality blog posts
    Uses structured reasoning, iterative refinement, and verification against knowledge base
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # DSPy modules
        self.intro_synth = dspy.ChainOfThought(IntroSynthesisSignature)
        self.intro_verify = dspy.ChainOfThought(IntroVerificationSignature)
        self.intro_refine = dspy.ChainOfThought(IntroRefinementSignature)
        self.intro_final = dspy.ChainOfThought(IntroFinalValidationSignature)

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
        self.casebook_action = cfg.get("casebook_action", "blog")

        # Initialize DSPy
        self._init_dspy()

        self.logger.info(
            "DSPyPaperSectionProcessor initialized",
            {
                "max_refinements": self.max_refinements,
                "min_quality_threshold": self.min_quality_threshold,
                "casebook_name": self.casebook_name,
            },
        )

    def _init_dspy(self):
        """Initialize DSPy with appropriate configuration"""
        # Configure DSPy for best

        lm = dspy.LM(
            "ollama_chat/qwen3",
            api_base="http://localhost:11434",
            api_key="",
        )
        dspy.configure(lm=lm)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main entrypoint for processing paper sections"""
        self.report(
            {
                "event": "start",
                "step": "DSPyPaperSectionProcessor",
                "details": "Processing paper sections",
            }
        )

        # Get document data
        documents = context.get(self.input_key, [])
        processed_sections = []

        for doc in documents:
            doc_id = doc.get("id")
            title = doc.get("title", "")
            casebook_name = generate_casebook_name(self.casebook_action, title)
            casebook = self.memory.casebooks.ensure_casebook(
                name=casebook_name,
                description=f"Agent generated blog for paper {title}",
                tag=self.casebook_action,
            )
            # Create (or reuse) a paper-level goal
            paper_goal = self.memory.goals.get_or_create(
                {
                    "goal_text": build_paper_goal_text(title),
                    "description": "Generate section-wise blog drafts with verification & refinement.",
                    "meta": build_paper_goal_meta(
                        title, doc_id, domains=self.cfg.get("domains", [])
                    ),
                }
            ).to_dict()

            # Optional: place on context so each step can see it

            structured_data = self.memory.document_sections.get_by_document(
                doc_id
            )

            if not structured_data:
                self.logger.warning(
                    f"No structured data for document {doc_id}"
                )
                continue

            self.logger.info(
                f"Processing document {doc_id} with {len(structured_data)} sections"
            )

            # Process sections in order of importance
            for section in structured_data:
                section_name = section.section_name
                section_text = section.section_text

                if len(section_text) < self.min_section_length:
                    continue

                try:
                    # Process section through DSPy pipeline
                    section_context = {
                        "section_text": section_text,
                        "section_name": section_name,
                        "goal_template": self.goal_template,
                        "paper_id": doc_id,
                        "section_id": section.id,
                        "paper_title": title,                # <-- use paper_title consistently
                        "document_text": doc.get("text", ""),
                        "document_summary": doc.get("summary", ""),
                        "pipeline_run_id": context.get("pipeline_run_id"),
                        "casebook_id": casebook.id,
                        "source": self.name,
                        "goal_id": paper_goal["id"],
                        "goal_text": paper_goal["goal_text"],
                        "section_order_index": getattr(section, "order_index", None),
                    }

                    if "abstract" == section_name.lower().strip():
                        prior_summary = doc.get("summary", "")
                        intro_result = await self._process_introduction(
                            title=title,
                            abstract=section_text,
                            prior_summary=prior_summary,
                            audience="technical blog readers",
                            goals=self._intro_goal(title),
                        )
                        self._save_introduction_to_casebook(
                            casebook=casebook,
                            doc_id=doc_id,
                            goal_id=paper_goal["id"],
                            title=title,
                            abstract=section_text,
                            prior_summary=prior_summary,
                            intro_result=intro_result,
                            context=context,
                        )

                    # Run DSPy pipeline
                    result = await self._process_section(section_context)

                    # Save to case book
                    self._save_section_to_casebook(
                        casebook,
                        paper_goal["id"],
                        doc_id,
                        section_name,
                        section_text,
                        result,
                        context,
                    )

                    processed_sections.append(
                        {
                            "doc_id": doc_id,
                            "section_name": section_name,
                            "summary": result.get("final_draft", ""),
                            "metrics": result.get("validation_report", {}),
                            "valid": result.get("passed", False),
                        }
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error processing section {section_name} for doc {doc_id}: {str(e)}"
                    )
                    import traceback

                    traceback.print_exc()

        self.report(
            {
                "event": "end",
                "step": "DSPyPaperSectionProcessor",
                "details": f"Processed {len(processed_sections)} sections across {len(documents)} documents",
            }
        )

        return context


    def _json_or_empty(self, s: Any, default: Any):
        """Best-effort JSON parse; returns default on failure."""
        if s is None:
            return default
        if isinstance(s, (dict, list)):
            return s
        try:
            return json.loads(s)
        except Exception:
            try:
                # sanitize common trailing commas / stray backticks if you like
                return json.loads(str(s).strip())
            except Exception:
                return default

    def _lm_fallback(self, prompt: str, max_tokens: int = 800) -> str:
        """
        Very simple LM fallback using the configured DSPy LM.
        Returns raw text (caller can json-parse if needed).
        """
        try:
            lm = dspy.settings.lm
            # DSPy LMs usually expose .__call__(prompt=..., max_new_tokens=...)
            return lm(prompt=prompt, max_new_tokens=max_tokens)
        except Exception:
            return ""

    def safe_infer(
        self,
        module: Any,
        /,
        *,
        output_field: Optional[str] = None,
        expect_json: bool = False,
        retries: int = 1,
        backoff_sec: float = 0.5,
        **inputs,
    ) -> dict:
        """
        Seatbelt wrapper for DSPy calls (sync).
        - module: a dspy module (e.g., ChainOfThought(Signature))
        - output_field: select a single field from the module output
        - expect_json: parse selected field as JSON (fallback to {} or "")
        """
        start = time.time()
        last_err, raw_value = None, None

        for attempt in range(1, retries + 2):  # initial try + retries
            try:
                t0 = time.time()
                out = module(**inputs)
                t1 = time.time()

                if hasattr(out, "__dict__"):
                    fields = {k: getattr(out, k) for k in out.__dict__.keys() if not k.startswith("_")}
                elif isinstance(out, dict):
                    fields = out
                else:
                    fields = {"output": out}

                raw_value = fields.get(output_field) if output_field else fields

                if expect_json:
                    data = self._loads_or_empty(raw_value, {} if output_field else {})
                else:
                    data = raw_value

                self.logger.info("DSPySafeInferSuccess", {
                    "module": getattr(module, "__class__", type(module)).__name__,
                    "attempt": attempt,
                    "latency_ms": round((t1 - t0) * 1000, 1),
                    "output_field": output_field,
                    "keys": list(fields.keys())
                })
                return {
                    "success": True,
                    "data": data,
                    "raw": raw_value,
                    "error": None,
                    "meta": {"attempts": attempt, "duration_ms": round((time.time()-start)*1000,1)}
                }
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                self.logger.error("DSPySafeInferError", {
                    "module": getattr(module, "__class__", type(module)).__name__,
                    "attempt": attempt,
                    "error": last_err,
                    "trace": traceback.format_exc()[:2000],
                })
                time.sleep(backoff_sec)

        return {
            "success": False,
            "data": {} if expect_json else "",
            "raw": raw_value,
            "error": last_err,
            "meta": {"attempts": retries + 1, "duration_ms": round((time.time()-start)*1000,1)}
        }

    def _intro_goal(self, title: str) -> str:
        return (
            f"You are an expert technical blog writer. Craft the introduction for the paper '{title}'. "
            "Fuse the abstract and existing summary into a compelling intro with: "
            "a strong hook, clear context, crisp list of core contributions, why-it-matters, and a preview. "
            "Keep it accurate, readable, and respectful to the source—no over-claims."
        )

    async def _process_introduction(
        self,
        *,
        title: str,
        abstract: str,
        prior_summary: str,
        audience: str = "technical blog readers",
        goals: str = "be accurate, concise, and compelling; highlight novelty and stakes",
    ) -> Dict[str, Any]:
        """
        Build the blog introduction by fusing abstract + prior summary.
        Pipeline: synth → verify → (loop refine+verify) → final validate.
        """
        # 1) synthesize
        synth = self.safe_infer(
            self.intro_synth,
            output_field="intro_draft",
            expect_json=True,
            title=title or "",
            abstract=abstract or "",
            prior_summary=prior_summary or "",
            audience=audience,
            goals=goals,
        )
        intro_draft = self._loads_or_empty(
            synth["data"],
            default={
                "hook": "",
                "context": "",
                "core_contributions": [],
                "why_it_matters": "",
                "preview": "",
            },
        )

        # 2) verify
        verify = self.safe_infer(
            self.intro_verify,
            output_field="verification_report",
            expect_json=True,
            intro_draft=json.dumps(intro_draft, ensure_ascii=False),
            abstract=abstract or "",
            prior_summary=prior_summary or "",
        )
        verification_report = self._loads_or_empty(
            verify["data"],
            default={
                "unsupported_points": [],
                "missing_essentials": [],
                "redundancy": [],
                "scores": {"coverage": 0.0, "correctness": 0.0, "coherence": 0.0},
            },
        )

        # 3) refine loop
        refined_intro = intro_draft
        iters = 0
        for iters in range(self.max_refinements):
            ref = self.safe_infer(
                self.intro_refine,
                output_field="refined_intro",
                expect_json=True,
                intro_draft=json.dumps(refined_intro, ensure_ascii=False),
                verification_report=json.dumps(verification_report, ensure_ascii=False),
            )
            refined_intro = self._loads_or_empty(ref["data"], default=refined_intro)

            verify = self.safe_infer(
                self.intro_verify,
                output_field="verification_report",
                expect_json=True,
                intro_draft=json.dumps(refined_intro, ensure_ascii=False),
                abstract=abstract or "",
                prior_summary=prior_summary or "",
            )
            verification_report = self._loads_or_empty(verify["data"], default=verification_report)

            scores = verification_report.get("scores", {})
            if scores.get("coverage", 0) >= 0.85 and scores.get("correctness", 0) >= 0.90:
                break

        # 4) final validation
        quality_standards = {
            "coverage": 0.85,
            "correctness": 0.90,
            "coherence": 0.85,
            "readability": 0.88,
            "hook_strength": 0.80,
        }
        final = self.safe_infer(
            self.intro_final,
            output_field="final_validation",
            expect_json=True,
            refined_intro=json.dumps(refined_intro, ensure_ascii=False),
            quality_standards=json.dumps(quality_standards, ensure_ascii=False),
        )
        final_validation = self._loads_or_empty(final["data"], default={"scores": {}, "passed": False})

        passed = bool(final_validation.get("passed")) or self._is_quality_threshold_met(
            {
                "scores": {
                    "coverage": final_validation.get("scores", {}).get("coverage", 0),
                    "correctness": final_validation.get("scores", {}).get("correctness", 0),
                    "coherence": final_validation.get("scores", {}).get("coherence", 0),
                    "citation_support": 0.9,  # intros usually light on citations
                    "readability": final_validation.get("scores", {}).get("readability", 0),
                    "novelty": final_validation.get("scores", {}).get("hook_strength", 0),
                }
            }
        )

        return {
            "intro_draft": intro_draft,
            "refined_intro": refined_intro,
            "verification_report": verification_report,
            "final_validation": final_validation,
            "passed": passed,
            "refinement_iterations": iters + 1,
        }

    async def _process_section(self, section_details: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single section through DSPy pipeline (section-aware)."""
        section_text = section_details.get("section_text", "") or ""
        section_name = section_details.get("section_name", "section") or "section"
        paper_id = section_details.get("paper_id", "unknown")
        paper_title = section_details.get("paper_title", "Unknown Paper")

        # Section-aware goal & quality
        goal_text = section_details.get("goal_text") or section_goal_text(section_name, paper_title)
        quality = section_details.get("quality_standards") or section_quality(section_name)
        sysg = system_guidance_from_goal(goal_text, quality)

        # 1) claims
        claims_res = self.safe_infer(
            self.claim_extractor,
            output_field="claims",
            expect_json=True,
            section_text=f"{sysg}\n[SECTION NAME] {section_name}\n[SECTION TEXT]\n{section_text}"
        )
        claims = self._loads_or_empty(claims_res["data"], default=[])

        # 2) fuse
        conversation_history = self._get_relevant_conversations(paper_id, claims)
        fusion_res = self.safe_infer(
            self.context_fuser,
            output_field="fused_context",
            expect_json=True,
            claims=json.dumps({"section_name": section_name, "claims": claims}, ensure_ascii=False),
            conversation_history=json.dumps(
                {"system_guidance": sysg, "section_name": section_name, "snippets": conversation_history},
                ensure_ascii=False
            ),
        )
        fused_context = self._loads_or_empty(fusion_res["data"], default={})

        # 3) draft
        draft_res = self.safe_infer(
            self.draft_generator,
            output_field="blog_section",
            expect_json=True,
            fused_context=json.dumps(
                {"system_guidance": sysg, "section_name": section_name, "context": fused_context},
                ensure_ascii=False
            ),
        )
        draft = self._loads_or_empty(draft_res["data"], default={"title": section_name, "body": ""})

        # 4) verify
        knowledge_base = self._get_knowledge_base(paper_id)
        verify_res = self.safe_infer(
            self.verification_module,
            output_field="verification_report",
            expect_json=True,
            draft=json.dumps({"section_name": section_name, "draft": draft}, ensure_ascii=False),
            knowledge_base=json.dumps(knowledge_base, ensure_ascii=False)
        )
        verification_report = self._loads_or_empty(verify_res["data"], default={"scores": {}})

        # 5) refine loop
        refined_draft = draft
        iterations = 0
        for iterations in range(self.max_refinements):
            refine_res = self.safe_infer(
                self.refinement_module,
                output_field="refined_draft",
                expect_json=True,
                draft=json.dumps({"section_name": section_name, "draft": refined_draft}, ensure_ascii=False),
                verification_report=json.dumps({"section_name": section_name, "report": verification_report}, ensure_ascii=False),
            )
            refined_draft = self._loads_or_empty(refine_res["data"], default=refined_draft)

            verify_res = self.safe_infer(
                self.verification_module,
                output_field="verification_report",
                expect_json=True,
                draft=json.dumps({"section_name": section_name, "draft": refined_draft}, ensure_ascii=False),
                knowledge_base=json.dumps(knowledge_base, ensure_ascii=False),
            )
            verification_report = self._loads_or_empty(verify_res["data"], default=verification_report)

            if self._is_quality_threshold_met(verification_report):
                break

        # 6) final validation
        final_res = self.safe_infer(
            self.final_validator,
            output_field="final_validation",
            expect_json=True,
            refined_draft=json.dumps({"section_name": section_name, "draft": refined_draft}, ensure_ascii=False),
            quality_standards=json.dumps(quality, ensure_ascii=False),
        )
        validation_report = self._loads_or_empty(final_res["data"], default={"scores": {}, "passed": False})

        return {
            "initial_draft": draft,
            "refined_draft": refined_draft,
            "verification_report": verification_report,
            "final_validation": validation_report,
            "passed": bool(validation_report.get("passed")) or self._is_quality_threshold_met(validation_report),
            "refinement_iterations": iterations + 1,
            "section_name": section_name,
        }

    def _get_relevant_conversations(
        self, paper_id: str, claims: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
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
        unique_conversations.sort(
            key=lambda x: x.get("relevance_score", 0), reverse=True
        )

        # Return top 10 relevant conversations
        return unique_conversations[:10]

    def _search_conversations(self, query: str) -> List[Dict[str, Any]]:
        """Search conversation history for relevant snippets"""
        # In a real implementation, this would use the retrieval model
        # to find relevant conversations based on the query
        return [
            {
                "role": "expert",
                "text": f"Relevant insight about {query[:20]}...",
                "relevance_score": 0.9,
            }
        ]

    def _get_knowledge_base(self, paper_id: str) -> Dict[str, Any]:
        """Retrieve knowledge base for the paper"""
        # In a real implementation, this would retrieve structured knowledge
        # from the knowledge graph or other sources
        return {
            "paper_id": paper_id,
            "key_facts": [
                "Key fact 1 from knowledge base",
                "Key fact 2 from knowledge base",
            ],
            "related_papers": ["Paper ID 1", "Paper ID 2"],
        }

    def _is_quality_threshold_met(
        self, validation_report: Dict[str, Any]
    ) -> bool:
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
            "novelty": 0.75,
        }

        for dim, threshold in thresholds.items():
            if scores.get(dim, 0) < threshold:
                return False

        return True

    def _save_introduction_to_casebook(
        self,
        casebook: CaseBookORM,
        doc_id: str,
        goal_id: int,
        title: str,
        abstract: str,
        prior_summary: str,
        intro_result: Dict[str, Any],
        context: Dict[str, Any],
    ):
        pipeline_run_id = context.get("pipeline_run_id")
        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id,
            goal_id=goal_id,
            prompt_text=json.dumps(
                {"type": "introduction", "paper_id": doc_id, "title": title}
            ),
            agent_name=self.name,
            meta={
                "type": "introduction_trajectory",
                "paper_id": doc_id,
                "title": title,
                "timestamp": time.time(),
            },
        )

        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            role="introduction_abstract",
            text=abstract or "",
            meta={"paper_id": doc_id, "title": title},
        )
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            role="introduction_prior_summary",
            text=prior_summary or "",
            meta={"paper_id": doc_id, "title": title},
        )
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            role="introduction_draft",
            text=json.dumps(intro_result.get("intro_draft", {})),
            meta={"paper_id": doc_id, "title": title},
        )
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            role="introduction_refined",
            text=json.dumps(intro_result.get("refined_intro", {})),
            meta={"paper_id": doc_id, "title": title},
        )
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            role="introduction_verification",
            text=json.dumps(intro_result.get("verification_report", {})),
            meta={"paper_id": doc_id, "title": title},
        )
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            role="introduction_final_validation",
            text=json.dumps(intro_result.get("final_validation", {})),
            meta={"paper_id": doc_id, "title": title},
        )
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            role="introduction_metrics",
            text=json.dumps(
                {
                    "passed": intro_result.get("passed", False),
                    "refinement_iterations": intro_result.get(
                        "refinement_iterations", 0
                    ),
                }
            ),
            meta={"paper_id": doc_id, "title": title},
        )

    def _save_section_to_casebook(
        self,
        casebook: CaseBookORM,
        goal_id: int,
        doc_id: str,
        section_name: str,
        section_text: str,
        result: Dict[str, Any],
        context: Dict[str, Any],
    ):
        """
        Save a section and its results to the case book, making section_name
        a first-class field across prompt, case meta, and all scorables.
        """
        # Optional extras if available in caller context
        paper_title = context.get("paper_title")
        order_index = context.get("section_order_index")  # int or None
        paper_id = context.get("paper_id")
        pipeline_run_id = context.get("pipeline_run_id")

        # 1) Create the case (section-aware prompt + meta)
        case_prompt = {
            "paper_id": paper_id or doc_id,
            "paper_title": paper_title,
            "section_name": section_name,
            "section_order_index": order_index,
        }

        case_meta = {
            "type": "draft_trajectory",
            "paper_id": paper_id or doc_id,
            "paper_title": paper_title,
            "section_name": section_name,
            "section_order_index": order_index,
            "timestamp": time.time(),
            "source": "dspy.paper_processor",
        }

        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id,
            goal_id=goal_id,
            prompt_text=json.dumps(case_prompt, ensure_ascii=False),
            agent_name=self.name,
            meta=case_meta,
        )

        # Common scorable meta to keep things uniform
        def _smeta(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
            base = {
                "paper_id": paper_id or doc_id,
                "paper_title": paper_title,
                "section_name": section_name,
                "section_order_index": order_index,
            }
            if extra:
                base.update(extra)
            return base

        # 2) Raw section text
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=section_text,
            role="section_text",
            meta=_smeta(),
        )

        # 3) Initial draft
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=json.dumps(result.get("initial_draft", {}), ensure_ascii=False),
            role="initial_draft",
            meta=_smeta(),
        )

        # 4) Refined draft
        refined_draft = result.get("refined_draft", {})
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=json.dumps(refined_draft, ensure_ascii=False),
            role="refined_draft",
            meta=_smeta({"refinement_iterations": result.get("refinement_iterations", 0)}),
        )

        # 5) Verification report
        verification_report = result.get("verification_report", {})
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=json.dumps(verification_report, ensure_ascii=False),
            role="verification_report",
            meta=_smeta(),
        )

        # 6) Final validation
        final_validation = result.get("final_validation", {})
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=json.dumps(final_validation, ensure_ascii=False),
            role="final_validation",
            meta=_smeta(),
        )

        # 7) Metrics (passed, iterations, etc.)
        metrics = {
            "passed": result.get("passed", False),
            "refinement_iterations": result.get("refinement_iterations", 0),
            "verification_report": verification_report,
            "final_validation": final_validation,
        }
        self.memory.casebooks.add_scorable(
            case_id=case.id,
            pipeline_run_id=pipeline_run_id,
            text=json.dumps(metrics, ensure_ascii=False),
            role="metrics",
            meta=_smeta(),
        )

    def _loads_or_empty(self, obj, default):
        """
        Best-effort JSON parser.
        - dict/list -> return as-is
        - str -> json.loads (with light cleanup)
        - else -> default
        """
        if obj is None:
            return default
        if isinstance(obj, (dict, list)):
            return obj
        if isinstance(obj, (bytes, bytearray)):
            obj = obj.decode("utf-8", errors="ignore")
        s = str(obj).strip()
        if not s:
            return default
        try:
            return json.loads(s)
        except Exception:
            s2 = s.strip().strip("`")
            try:
                return json.loads(s2)
            except Exception:
                return default

    def _dumps_safe(self, obj) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return json.dumps({"_warning": "failed_to_dump", "repr": repr(obj)})

    def _ensure_dict(self, x, default=None):
        if isinstance(x, dict):
            return x
        parsed = self._loads_or_empty(x, None)
        return parsed if isinstance(parsed, dict) else (default or {})

    def _ensure_list(self, x, default=None):
        if isinstance(x, list):
            return x
        parsed = self._loads_or_empty(x, None)
        return parsed if isinstance(parsed, list) else (default or [])

    def _ensure_str(self, x, default=""):
        if isinstance(x, str):
            return x
        if x is None:
            return default
        return str(x)
