import re

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.domain_classifier import DomainClassifier
from stephanie.utils.document_section_parser import DocumentSectionParser

DEFAULT_SECTIONS = ["title", "abstract", "methods", "results", "contributions"]
REQUIRED_SECTIONS = ["title", "summary"]

class DocumentProfilerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.summary_prompt_file = cfg.get("summary_prompt_file", "summarize.txt")
        self.use_unstructured = cfg.get("use_unstructured", True)
        self.fallback_to_llm = cfg.get("fallback_to_llm", False)
        self.store_inline = cfg.get("store_inline", True)
        self.output_sections = cfg.get("output_sections", DEFAULT_SECTIONS)
        self.required_sections = cfg.get("required_sections", REQUIRED_SECTIONS)
        self.min_chars_per_sec = cfg.get("min_chars_per_section", 120)  # quality 

        self.force_domain_update = cfg.get("force_domain_update", False)
        self.top_k_domains = cfg.get("top_k_domains", 3)
        self.min_classification_score = cfg.get("min_classification_score", 0.6)

        self.domain_classifier = DomainClassifier(
            memory,
            logger,
            cfg.get("domain_seed_config_path", "config/domain/seeds.yaml"),
        )
        self.section_parser = DocumentSectionParser(cfg, logger)

    async def run(self, context: dict) -> dict:
        documents = context.get(self.input_key, [])
        profiled = []

        for doc in documents:
            try:
                doc_id = doc["id"]

                existing_sections = self.memory.document_section.get_by_document(doc_id)
                if existing_sections and not self.force_domain_update:
                    # If we already have sections and not forcing update, skip
                    self.logger.log(
                        "DocumentAlreadyProfiled",
                        {"doc_id": doc_id, "title": doc.get("title", "")},
                    )
                    continue

                title = doc.get("title")
                summary = doc.get("summary")
                text = doc.get("content", doc.get("text", ""))

                # -- STEP 1 : Unstructured pass ---------------------------------
                unstruct_data = {}
                if self.use_unstructured:
                    unstruct_data = self.section_parser.parse(text)

                # -- STEP 2 : Quality check & optional LLM fallback -------------
                if self.fallback_to_llm and self.needs_fallback(unstruct_data):
                    llm_data = await self.extract_with_prompt(text, context)
                    chosen = self.merge_outputs(unstruct_data, llm_data)
                else:
                    chosen = unstruct_data

                # no point in overwriting arxiv data
                if title:
                    chosen["title"] = title
                if summary:
                    chosen["summary"] = summary 
                else:
                    prompt = self.prompt_loader.from_file(
                        self.summary_prompt_file, self.cfg, context
                    )
                    chosen["summary"] = self.call_llm(prompt, context)


                # -- STEP 3 : Persist ------------------------------------------
                for section, text in chosen.items():
                    existing = self.memory.document_section.upsert(
                        {
                            "document_id": doc_id,
                            "section_name": section,
                            "section_text": text,
                            "source": "unstructured+llm",
                            "summary": summary,
                        }
                    )

                    # -- STEP 4 : Domain detection ---------------------------------
                    # Classify domain for the section
                    section_domains = self.domain_classifier.classify(text, self.top_k_domains, self.min_classification_score)

                    # Insert classified domains for this section
                    for domain, score in section_domains:
                        self.memory.document_section_domains.insert(
                            {
                                "document_section_id": existing.id,
                                "domain": domain,
                                "score": float(score),
                            }
                        )

                profiled.append(
                    {
                        "id": doc_id,
                        "title": doc.get("title", "")[:80],
                        "structured_data": chosen,
                    }
                )

                self.logger.log(
                    "DocumentProfiled",
                    {
                        "doc_id": doc_id,
                        "method": "unstructured+llm"
                        if self.needs_fallback(unstruct_data)
                        else "unstructured",
                        "sections": list(chosen.keys()),
                    },
                )

            except Exception as e:
                self.logger.log(
                    "DocumentProfileFailed",
                    {"error": str(e), "title": doc.get("title")},
                )

        context[self.output_key] = profiled
        return context

    async def extract_with_prompt(self, text: str, context: dict) -> dict:
        prompt_ctx = {
            "text": text[: self.cfg.get("llm_max_chars", 12000)],
            "sections": ", ".join(self.output_sections),
        }
        prompt = self.prompt_loader.load_prompt(self.cfg, prompt_ctx)
        raw = self.call_llm(prompt, context)
        headings = self.parse_headings_from_response(raw)

        # 🧠 Heuristic split of text into chunks between headings
        return self.split_text_by_headings(text, headings)

    def needs_fallback(self, data: dict) -> bool:
        """
        Simple heuristic:
            • Missing any requested section
            • OR any section (except 'title') shorter than min_chars
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
        Evaluate content quality using a simple heuristic:
            - Length
            - Sentence coherence (number of periods)
            - Word complexity (average word length)
        Can be replaced with an LLM-based scorer later.
        """
        if not text:
            return 0.0

        sentences = text.split(".")
        avg_word_len = (
            sum(len(word) for word in text.split()) / len(text.split())
            if text.split()
            else 0
        )
        sentence_score = len([s for s in sentences if len(s.strip()) > 20]) / max(
            1, len(sentences)
        )

        # Combine into a basic score
        score = (
            0.4 * min(1.0, len(text) / 500)  # Normalize length
            + 0.4 * sentence_score
            + 0.2
            * min(1.0, avg_word_len / 8)  # Prefer more complex words up to a point
        )
        return round(score, 2)

    def merge_outputs(self, primary: dict, fallback: dict) -> dict:
        """
        Merges primary and fallback outputs by comparing both:
            - Uses length threshold as a gate
            - Compares semantic quality if both exist
            - Picks best version, not just longest
        """
        merged = {}

        for sec in self.output_sections:
            p_txt = primary.get(sec, "")
            f_txt = fallback.get(sec, "")

            # If neither exists, skip
            if not p_txt and not f_txt:
                continue

            # If only one exists, take it
            if not p_txt:
                merged[sec] = f_txt
                continue
            if not f_txt:
                merged[sec] = p_txt
                continue

            # Both exist — decide which is better
            p_len = len(p_txt)
            f_len = len(f_txt)

            # First check: does primary meet minimum length?
            if p_len >= self.min_chars_per_sec:
                p_score = self.evaluate_content_quality(p_txt)
                f_score = self.evaluate_content_quality(f_txt)

                # Decide winner
                if p_score >= f_score:
                    merged[sec] = p_txt
                else:
                    merged[sec] = f_txt
                    print(
                        f"[QUALITY WIN] Fallback used for '{sec}' (P: {p_score}, F: {f_score})"
                    )
            else:
                # Primary doesn't meet threshold — always use fallback
                merged[sec] = f_txt

        return merged

    def parse_headings_from_response(self, response: str) -> list[str]:
        """
        Extract a list of clean headings from the LLM response.
        Strips bullets, numbers, markdown, etc.
        Focuses on the final lines in case of trailing blocks.
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
