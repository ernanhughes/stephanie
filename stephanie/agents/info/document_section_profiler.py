from __future__ import annotations

import re

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.utils.document_section_parser import DocumentSectionParser


class DocumentSectionProfilerAgent(BaseAgent):
    """
    DocumentSectionProfilerAgent

    A simpler, section-centric variant of DocumentProfilerAgent that:
      - does NOT assume any fixed set of section names (no DEFAULT_SECTIONS)
      - does NOT require specific sections like title/summary
      - treats every detected section as a first-class unit
      - persists each section to memory.document_sections
      - emits a flat list of section records to context[self.output_key]

    This is ideal for downstream consumers (e.g. Auto-Blog, VPM, HRM) that want
    to process sections independently rather than as a single structured document.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Parsing behaviour
        self.use_unstructured = cfg.get("use_unstructured", True)
        self.fallback_to_llm = cfg.get("fallback_to_llm", False) # Note: I have found this unreliable
        self.min_chars_per_section = cfg.get("min_chars_per_section", 120)

        # Domain classification settings
        self.force_domain_update = cfg.get("force_domain_update", False)
        self.top_k_domains = cfg.get("top_k_domains", 10)
        self.min_classification_score = cfg.get("min_classification_score", 0.6)

        # Optional: prompt file for LLM-based headings extraction (fallback)
        self.heading_prompt_file = cfg.get("heading_prompt_file", "extract_headings.txt")
        self.llm_max_chars = cfg.get("llm_max_chars", 12000)

        # Core helpers
        self.domain_classifier = ScorableClassifier(
            memory,
            logger,
            cfg.get("domain_seed_config_path", "config/domain/seeds.yaml"),
        )
        self.section_parser = DocumentSectionParser(cfg, logger)

    async def run(self, context: dict) -> dict:
        """
        Main execution method. All right I don't need this 9:00 AM OK

        Input:
            context[self.input_key]: list of documents with fields:
                - id
                - title (optional)
                - summary (optional)
                - content or text

        Output:
            context[self.output_key]: list of section records:
                {
                    "document_id": ...,
                    "document_title": ...,
                    "section_id": ...,
                    "section_name": ...,
                    "section_text": ...,
                    "domains": [{"domain": str, "score": float}, ...],
                }
        """
        documents = context.get(self.input_key, [])
        section_records = []

        self.report(
            {
                "event": "start",
                "step": "DocumentSectionProfiler",
                "details": f"Profiling sections for {len(documents)} documents",
            }
        )

        for doc in documents:
            try:
                doc_id = doc["id"]
                title = doc.get("title", "") or ""
                summary = doc.get("summary")
                text = doc.get("content", doc.get("text", "")) or ""

                # If we already have sections and not forcing, skip
                existing_sections = self.memory.document_sections.get_by_document(doc_id)
                if existing_sections and not self.force_domain_update:
                    self.report(
                        {
                            "event": "skipped_existing",
                            "step": "DocumentSectionProfiler",
                            "doc_id": doc_id,
                            "title": title[:80],
                            "details": "Sections already in memory; skipping.",
                        }
                    )
                    continue

                # STEP 1: unstructured parsing into arbitrary sections
                sections = {}
                if self.use_unstructured:
                    sections = self.section_parser.parse(text)
                    self.report(
                        {
                            "event": "parsed_unstructured",
                            "step": "DocumentSectionProfiler",
                            "doc_id": doc_id,
                            "title": title[:80],
                            "sections": list(sections.keys()),
                        }
                    )

                # STEP 2: Fallback to LLM-based heading extraction if needed
                if self.fallback_to_llm and self._needs_fallback(sections):
                    llm_sections = await self._extract_sections_with_llm(text, context)
                    sections = self._merge_sections(sections, llm_sections)
                    self.report(
                        {
                            "event": "used_fallback",
                            "step": "DocumentSectionProfiler",
                            "doc_id": doc_id,
                            "title": title[:80],
                            "sections": list(sections.keys()),
                        }
                    )

                # If still nothing, skip this document
                if not sections:
                    self.report(
                        {
                            "event": "no_sections",
                            "step": "DocumentSectionProfiler",
                            "doc_id": doc_id,
                            "title": title[:80],
                            "details": "No sections detected even after fallback.",
                        }
                    )
                    continue

                # STEP 3: Persist each section + classify domains
                for section_name, section_text in sections.items():
                    if not section_text:
                        continue
                    if len(section_text) < self.min_chars_per_section:
                        # Skip extremely short fragments
                        continue

                    record = self.memory.document_sections.upsert(
                        {
                            "document_id": doc_id,
                            "section_name": section_name,
                            "section_text": section_text,
                            "source": "unstructured+llm" if self.fallback_to_llm else "unstructured",
                            "summary": summary,
                        }
                    )

                    # Domain classification for this section
                    section_domains = self.domain_classifier.classify(
                        section_text, self.top_k_domains, self.min_classification_score
                    )

                    domains_payload = []
                    for domain, score in section_domains:
                        self.memory.document_section_domains.insert(
                            {
                                "document_section_id": record.id,
                                "domain": domain,
                                "score": float(score),
                            }
                        )
                        domains_payload.append(
                            {"domain": domain, "score": float(score)}
                        )

                    # Emit section record for downstream agents (e.g. blog generator)
                    section_records.append(
                        {
                            "document_id": doc_id,
                            "document_title": title,
                            "section_id": record.id,
                            "section_name": section_name,
                            "section_text": section_text,
                            "domains": domains_payload,
                        }
                    )

                self.report(
                    {
                        "event": "profiled_document",
                        "step": "DocumentSectionProfiler",
                        "doc_id": doc_id,
                        "title": title[:80],
                        "num_sections": len(
                            [
                                s
                                for s in section_records
                                if s["document_id"] == doc_id
                            ]
                        ),
                    }
                )

            except Exception as e:
                self.report(
                    {
                        "event": "error",
                        "step": "DocumentSectionProfiler",
                        "doc_id": doc.get("id"),
                        "title": doc.get("title", "")[:80],
                        "details": str(e),
                    }
                )

        # Attach flat list of sections to context
        context[self.output_key] = section_records

        self.report(
            {
                "event": "end",
                "step": "DocumentSectionProfiler",
                "details": f"Emitted {len(section_records)} sections in total",
            }
        )
        return context

    # ------------------------------------------------------------------
    # Helper methods (no DEFAULT_SECTIONS / REQUIRED_SECTIONS logic)
    # ------------------------------------------------------------------

    def _needs_fallback(self, sections: dict) -> bool:
        """
        Decide if LLM fallback is needed.

        Here we only check whether we got *any* non-trivial sections.
        """
        if not sections:
            return True
        # Optionally: require at least one section with enough content
        long_enough = any(
            len(text or "") >= self.min_chars_per_section
            for text in sections.values()
        )
        return not long_enough

    async def _extract_sections_with_llm(self, text: str, context: dict) -> dict:
        """
        Use LLM to propose headings, then split the text by those headings.

        Unlike DocumentProfilerAgent, this version does NOT assume any fixed
        set of section names. The prompt is expected to return a list of
        natural headings present in the paper (e.g. Introduction, Method, ...).
        """
        snippet = text[: self.llm_max_chars]
        prompt_ctx = {
            "text": snippet,
        }
        # You can design heading_prompt_file to say:
        # "List the main section headings found in the given paper text, one per line."
        prompt = self.prompt_loader.from_file(
            self.heading_prompt_file, self.cfg, prompt_ctx
        )
        raw = self.call_llm(prompt, context)
        headings = self._parse_headings_from_response(raw)
        if not headings:
            return {}

        return self._split_text_by_headings(text, headings)

    def _parse_headings_from_response(self, response: str) -> list[str]:
        """
        Extract heading candidates from LLM response text.
        """
        lines = (response or "").strip().splitlines()
        candidates: list[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Strip bullets / numbering
            line = re.sub(r"^[\-\*\d\.\)]+\s*", "", line)
            # Reasonable heading length
            if 2 <= len(line) <= 120:
                candidates.append(line)

        # De-duplicate while preserving order
        unique: list[str] = []
        seen = set()
        for h in candidates:
            key = h.lower()
            if key not in seen:
                seen.add(key)
                unique.append(h)
        return unique

    def _split_text_by_headings(self, text: str, headings: list[str]) -> dict:
        """
        Split full text into sections based on detected headings.
        """
        if not headings:
            return {}

        sections: dict[str, list[str]] = {}
        current = None
        lines = text.splitlines()

        lower_headings = [h.lower() for h in headings]

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current:
                    sections[current].append(line)
                continue

            # If this line looks like a heading, start a new section
            match = None
            for h in headings:
                if h.lower() in stripped.lower():
                    match = h
                    break

            if match:
                current = match
                if current not in sections:
                    sections[current] = []
            elif current:
                sections[current].append(line)

        # Convert lists to joined strings and filter out tiny sections
        result: dict[str, str] = {}
        for name, lines_list in sections.items():
            joined = "\n".join(lines_list).strip()
            if len(joined) >= 3:  # at least some content
                result[name.lower()] = joined

        return result

    def _merge_sections(self, primary: dict, fallback: dict) -> dict:
        """
        Merge sections from unstructured parsing and LLM-based parsing.

        For each section key present in either dict:
            - If only one source has it, use that.
            - If both have it, pick the text with higher heuristically
              evaluated quality (length + basic stats).
        """
        if not primary and not fallback:
            return {}

        merged: dict[str, str] = {}
        all_keys = set(primary.keys()) | set(fallback.keys())

        for key in all_keys:
            p_txt = primary.get(key, "") or ""
            f_txt = fallback.get(key, "") or ""

            if not p_txt and not f_txt:
                continue
            if not p_txt:
                merged[key] = f_txt
                continue
            if not f_txt:
                merged[key] = p_txt
                continue

            p_score = self._evaluate_content_quality(p_txt)
            f_score = self._evaluate_content_quality(f_txt)

            if p_score >= f_score:
                merged[key] = p_txt
            else:
                merged[key] = f_txt

        return merged

    def _evaluate_content_quality(self, text: str) -> float:
        """
        Simple heuristic quality estimator used in _merge_sections.
        """
        if not text:
            return 0.0

        words = text.split()
        if not words:
            return 0.0

        sentences = [s.strip() for s in text.split(".") if s.strip()]
        avg_word_len = sum(len(w) for w in words) / len(words)
        long_sentences = sum(1 for s in sentences if len(s) > 40)

        length_score = min(1.0, len(text) / 500.0)              # up to ~500 chars
        sentence_score = long_sentences / max(1, len(sentences))
        word_score = min(1.0, avg_word_len / 8.0)               # prefer moderate complexity

        score = 0.4 * length_score + 0.4 * sentence_score + 0.2 * word_score
        return round(score, 3)
