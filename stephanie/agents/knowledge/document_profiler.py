# stephanie/agents/knowledge/document_profiler.py
"""
Document Profiler Agent Module

This module provides the DocumentProfilerAgent class for analyzing and structuring
research documents into standardized sections with domain classification. It transforms
unstructured document text into organized, categorized content for better analysis
and retrieval in the research pipeline.

Key Features:
    - Multi-method document parsing (unstructured parsing + LLM fallback)
    - Section-based document analysis (title, abstract, methods, results, etc.)
    - Domain classification for document sections
Any exercise again    - Content quality evaluation and selection
    - Persistent storage of structured document sections
    - Comprehensive error handling and reporting

Classes:
    DocumentProfilerAgent: Main agent class for document profiling and structuring

Configuration Options:
    - summary_prompt_file: Prompt file for document summarization
    - use_unstructured: Enable/disable unstructured parsing
    - fallback_to_llm: Enable LLM fallback when parsing fails
    - store_inline: Enable inline storage of parsed sections
    - output_sections: List of sections to extract from documents
    - required_sections: Minimum required sections for successful parsing
    - min_chars_per_section: Minimum character threshold for section quality
    - force_domain_update: Force re-classification of existing documents
    - top_k_domains: Number of top domains to assign per section
    - min_classification_score: Minimum confidence score for domain classification

Dependencies:
    - BaseAgent: Core agent functionality and LLM integration
    - ScorableClassifier: Domain classification and scoring
    - DocumentSectionParser: Section extraction from unstructured text

Usage:
    Typically used after document loading to structure and categorize documents
    for further analysis, hypothesis generation, or knowledge extraction.
"""
from __future__ import annotations

import re

from stephanie.agents.base_agent import BaseAgent
from stephanie.tools.scorable_classifier import ScorableClassifier
from stephanie.utils.document_section_parser import DocumentSectionParser

# Default sections to extract from documents
DEFAULT_SECTIONS = ["title", "abstract", "methods", "results", "contributions"]
# Minimum required sections for document processing
REQUIRED_SECTIONS = ["title", "summary"]


class DocumentProfilerAgent(BaseAgent):
    """Agent responsible for structuring documents into standardized sections with domain classification"""
    
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        # Configuration parameters
        self.summary_prompt_file = cfg.get("summary_prompt_file", "summarize.txt")
        self.use_unstructured = cfg.get("use_unstructured", True)
        self.fallback_to_llm = cfg.get("fallback_to_llm", False)
        self.store_inline = cfg.get("store_inline", True)
        self.output_sections = cfg.get("output_sections", DEFAULT_SECTIONS)
        self.required_sections = cfg.get("required_sections", REQUIRED_SECTIONS)
        self.min_chars_per_sec = cfg.get("min_chars_per_section", 120)  # quality threshold

        # Domain classification settings
        self.force_domain_update = cfg.get("force_domain_update", False)
        self.top_k_domains = cfg.get("top_k_domains", 3)
        self.min_classification_score = cfg.get("min_classification_score", 0.6)

        # Initialize classifiers and parsers
        self.domain_classifier = ScorableClassifier(
            memory,
            logger,
            cfg.get("domain_seed_config_path", "config/domain/seeds.yaml"),
        )
        self.section_parser = DocumentSectionParser(cfg, logger)

    async def run(self, context: dict) -> dict:
        """Main execution method for document profiling pipeline"""
        documents = context.get(self.input_key, [])
        profiled = []

        # Start profiling process
        self.report({
            "event": "start",
            "step": "DocumentProfiler",
            "details": f"Profiling {len(documents)} documents",
        })

        # Process each document
        for doc in documents:
            try:
                doc_id = doc["id"]
                title = doc.get("title", "")

                # Check if document already profiled
                existing_sections = self.memory.document_sections.get_by_document(doc_id)
                if existing_sections and not self.force_domain_update:
                    self.report({
                        "event": "skipped_existing",
                        "step": "DocumentProfiler",
                        "doc_id": doc_id,
                        "title": title[:80],
                        "details": "Already profiled, skipping.",
                    })
                    continue

                summary = doc.get("summary")
                text = doc.get("content", doc.get("text", ""))

                # STEP 1: Try unstructured parsing first
                unstruct_data = {}
                if self.use_unstructured:
                    unstruct_data = self.section_parser.parse(text)
                    self.report({
                        "event": "parsed_unstructured",
                        "step": "DocumentProfiler",
                        "doc_id": doc_id,
                        "title": title[:80],
                        "sections": list(unstruct_data.keys()),
                    })

                # STEP 2: Use LLM fallback if unstructured parsing is insufficient
                if self.fallback_to_llm and self.needs_fallback(unstruct_data):
                    llm_data = await self.extract_with_prompt(text, context)
                    chosen = self.merge_outputs(unstruct_data, llm_data)
                    self.report({
                        "event": "used_fallback",
                        "step": "DocumentProfiler",
                        "doc_id": doc_id,
                        "title": title[:80],
                        "sections": list(chosen.keys()),
                    })
                else:
                    chosen = unstruct_data

                # Ensure required sections are present
                if title:
                    chosen["title"] = title
                if summary:
                    chosen["summary"] = summary
                else:
                    # Generate summary if missing
                    prompt = self.prompt_loader.from_file(
                        self.summary_prompt_file, self.cfg, context
                    )
                    chosen["summary"] = self.call_llm(prompt, context)

                # STEP 3: Persist sections to memory
                section_summaries = []
                for section, text in chosen.items():
                    existing = self.memory.document_sections.upsert(
                        {
                            "document_id": doc_id,
                            "section_name": section,
                            "section_text": text,
                            "source": "unstructured+llm",
                            "summary": summary,
                        }
                    )

                    # STEP 4: Domain classification for each section
                    section_domains = self.domain_classifier.classify(
                        text, self.top_k_domains, self.min_classification_score
                    )

                    # Store domain classifications
                    for domain, score in section_domains:
                        self.memory.document_section_domains.insert(
                            {
                                "document_section_id": existing.id,
                                "domain": domain,
                                "score": float(score),
                            }
                        )
                    if section_domains:
                        section_summaries.append({
                            "section": section,
                            "domains": [
                                {"domain": d, "score": float(s)} for d, s in section_domains
                            ],
                        })

                # Add to results
                profiled.append(
                    {
                        "id": doc_id,
                        "title": title[:80],
                        "structured_data": chosen,
                    }
                )

                self.report({
                    "event": "profiled",
                    "step": "DocumentProfiler",
                    "doc_id": doc_id,
                    "title": title[:80],
                    "sections": list(chosen.keys()),
                    "classified_domains": section_summaries,
                })

            except Exception as e:
                self.report({
                    "event": "error",
                    "step": "DocumentProfiler",
                    "doc_id": doc.get("id"),
                    "title": doc.get("title", "")[:80],
                    "details": str(e),
                })

        context[self.output_key] = profiled

        # Completion report
        self.report({
            "event": "end",
            "step": "DocumentProfiler",
            "details": f"Profiled {len(profiled)} documents successfully",
        })
        return context

    async def extract_with_prompt(self, text: str, context: dict) -> dict:
        """Extract document sections using LLM prompt-based approach"""
        prompt_ctx = {
            "text": text[: self.cfg.get("llm_max_chars", 12000)],
            "sections": ", ".join(self.output_sections),
        }
        prompt = self.prompt_loader.load_prompt(self.cfg, prompt_ctx)
        raw = self.call_llm(prompt, context)
        headings = self.parse_headings_from_response(raw)

        # Split text into sections based on detected headings
        return self.split_text_by_headings(text, headings)

    def needs_fallback(self, data: dict) -> bool:
        """
        Determine if LLM fallback is needed based on parsing quality
        
        Returns:
            True if any required section is missing or too short
        """
        if not data:
            return True
        for sec in self.required_sections:
            if sec not in data:
                print(f"[FALLBACK NEEDED] Missing section: {sec}")
                return True
            if sec != "title" and len(data[sec]) < self.min_chars_per_sec:
                print(f"[FALLBACK NEEDED] section too small: {sec}")
                return True
        return False

    def evaluate_content_quality(self, text: str) -> float:
        """
        Evaluate content quality using heuristic measures
        
        Args:
            text: Text content to evaluate
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not text:
            return 0.0

        # Calculate basic text metrics
        sentences = text.split(".")
        avg_word_len = (
            sum(len(word) for word in text.split()) / len(text.split())
            if text.split()
            else 0
        )
        sentence_score = len([s for s in sentences if len(s.strip()) > 20]) / max(
            1, len(sentences)
        )

        # Combine metrics into quality score
        score = (
            0.4 * min(1.0, len(text) / 500)  # Normalize length
            + 0.4 * sentence_score
            + 0.2
            * min(1.0, avg_word_len / 8)  # Prefer more complex words up to a point
        )
        return round(score, 2)

    def merge_outputs(self, primary: dict, fallback: dict) -> dict:
        """
        Merge results from different parsing methods, selecting the best version
        
        Args:
            primary: Results from primary parsing method
            fallback: Results from fallback parsing method
            
        Returns:
            Merged results with best version of each section
        """
        merged = {}

        for sec in self.output_sections:
            p_txt = primary.get(sec, "")
            f_txt = fallback.get(sec, "")

            # Skip if neither method found this section
            if not p_txt and not f_txt:
                continue

            # Use available result if only one method found it
            if not p_txt:
                merged[sec] = f_txt
                continue
            if not f_txt:
                merged[sec] = p_txt
                continue

            # Both methods found this section - select the better one
            p_len = len(p_txt)

            # Check if primary meets minimum length requirement
            if p_len >= self.min_chars_per_sec:
                p_score = self.evaluate_content_quality(p_txt)
                f_score = self.evaluate_content_quality(f_txt)

                # Select version with higher quality score
                if p_score >= f_score:
                    merged[sec] = p_txt
                else:
                    merged[sec] = f_txt
                    print(
                        f"[QUALITY WIN] Fallback used for '{sec}' (P: {p_score}, F: {f_score})"
                    )
            else:
                # Primary doesn't meet threshold - use fallback
                merged[sec] = f_txt

        return merged

    def parse_headings_from_response(self, response: str) -> list[str]:
        """
        Extract headings from LLM response text
        
        Args:
            response: Raw LLM response text
            
        Returns:
            List of cleaned heading strings
        """
        lines = response.strip().splitlines()
        candidates = []

        for line in lines[-20:]:  # Limit to last 20 lines to avoid rambling
            line = line.strip()
            # Match lines that are likely headings
            if line and len(line) < 100:  # reasonable length
                line = re.sub(
                    r"^[\-\*\d\.\)]+\s*", "", line
                )  # remove leading bullets/numbers
                if re.match(r"^[A-Z][\w\s\-]+$", line):  # simple heading pattern
                    candidates.append(line)

        return candidates

    def split_text_by_headings(self, text: str, headings: list[str]) -> dict:
        """
        Split text into sections based on detected headings
        
        Args:
            text: Full document text
            headings: List of section headings
            
        Returns:
            Dictionary of section names to section text
        """
        sections = {}
        current = None
        lines = text.splitlines()

        for line in lines:
            line_stripped = line.strip()

            # Check if this line matches one of the headings
            matched_heading = next(
                (h for h in headings if h.lower() in line_stripped.lower()), None
            )

            if matched_heading:
                current = matched_heading
                sections[current] = []
            elif current:
                sections[current].append(line)

        # Join and trim each section
        return {
            k.lower(): "\n".join(v).strip()
            for k, v in sections.items()
            if len(v) >= 3  # must have at least a few lines
        }