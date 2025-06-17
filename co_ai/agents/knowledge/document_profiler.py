import json
import re

from co_ai.agents.base_agent import BaseAgent
from co_ai.analysis.domain_classifier import DomainClassifier

DEFAULT_SECTIONS = ["title", "abstract", "methods", "results", "contributions"]

class DocumentProfilerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.summary_prompt_file       = cfg.get("summary_prompt_file", "summarize.txt")
        self.use_unstructured   = cfg.get("use_unstructured", True)
        self.fallback_to_llm    = cfg.get("fallback_to_llm", True)
        self.store_inline       = cfg.get("store_inline", True)
        self.output_sections    = cfg.get("output_sections", DEFAULT_SECTIONS)
        self.min_chars_per_sec  = cfg.get("min_chars_per_section", 120)  # quality gate
        self.domain_classifier = DomainClassifier(memory, logger, cfg.get("domain_seed_config_path", "config/domain/seeds.yaml"))

    async def run(self, context: dict) -> dict:
        documents   = context.get(self.input_key, [])
        profiled    = []

        for doc in documents:
            try:
                doc_id = doc["id"]
                text   = doc.get("content", doc.get("text", ""))

                # -- STEP 1 : Unstructured pass ---------------------------------
                unstruct_data = {}
                if self.use_unstructured:
                    unstruct_data = self.extract_with_unstructured(text)

                # -- STEP 2 : Quality check & optional LLM fallback -------------
                if (self.fallback_to_llm and
                    self.needs_fallback(unstruct_data)):
                    llm_data = await self.extract_with_prompt(text, context)
                    chosen   = self.merge_outputs(unstruct_data, llm_data)
                else:
                    chosen = unstruct_data

                prompt = self.prompt_loader.from_file(self.summary_prompt_file, self.cfg, context)
                generated_summary = self.call_llm(prompt, context)

                # -- STEP 3 : Domain detection ---------------------------------
                detected_domain = self.domain_classifier.classify(text)


                # -- STEP 3 : Persist ------------------------------------------
                for section, text in chosen.items():
                    self.memory.document_section.upsert(
                        {
                            "document_id": doc_id,
                            "section_name": section,
                            "section_text": text,
                            "source": "unstructured+llm",
                            "domain": detected_domain,            # ← Add this dynamically
                            "summary": generated_summary,
                        }
                    )

                profiled.append({
                    "id": doc_id,
                    "title": doc.get("title", "")[:80],
                    "structured_data": chosen
                })

                self.logger.log("DocumentProfiled", {
                    "doc_id": doc_id,
                    "method": "unstructured+llm" if self.needs_fallback(unstruct_data)
                                                   else "unstructured",
                    "sections": list(chosen.keys())
                })

            except Exception as e:
                self.logger.log("DocumentProfileFailed",
                                {"error": str(e), "title": doc.get("title")})

        context[self.output_key] = profiled
        return context

    def extract_with_unstructured(self, text: str) -> dict:
        from unstructured.partition.text import partition_text
        from unstructured.staging.base import elements_to_json

        elements   = partition_text(text=text)
        json_elems = elements_to_json(elements)

        blob = {sec: "" for sec in self.output_sections}
        for el in json_elems:
            if not isinstance(el, dict):
                continue  # Skip non-dict elements like raw text or nulls
            content = el.get("text", "").strip()
            if not content:
                continue
            for sec in self.output_sections:
                if sec.lower() in content.lower()[:40]:
                    blob[sec] += content + "\n"
        return {k: v.strip() for k, v in blob.items() if v.strip()}

    async def extract_with_prompt(self, text: str, context: dict) -> dict:
        prompt_ctx = {
            "text": text[:self.cfg.get("llm_max_chars", 12000)],
            "sections": ", ".join(self.output_sections)
        }
        prompt = self.prompt_loader.load_prompt(self.cfg, prompt_ctx)
        raw = self.call_llm(prompt, context)
        headings = self.parse_headings_from_response(raw)

        # 🧠 Heuristic split of text into chunks between headings
        return self.split_text_by_headings(text, headings)

    # ------------------------------------------------------------------ #
    def needs_fallback(self, data: dict) -> bool:
        """
        Simple heuristic:
            • Missing any requested section
            • OR any section shorter than min_chars
        """
        if not data:
            return True
        for sec in self.output_sections:
            if sec not in data:
                return True
            if len(data[sec]) < self.min_chars_per_sec:
                return True
        return False

    # ------------------------------------------------------------------ #
    def merge_outputs(self, primary: dict, fallback: dict) -> dict:
        """
        Keep primary content when it exists and passes threshold,
        otherwise take fallback content.
        """
        merged = {}
        for sec in self.output_sections:
            p_txt = primary.get(sec, "")
            if len(p_txt) >= self.min_chars_per_sec:
                merged[sec] = p_txt
            else:
                merged[sec] = fallback.get(sec, p_txt)
        return merged

    # ------------------------------------------------------------------ #
    def safe_json_parse(self, raw: str) -> dict:
        try:
            return json.loads(raw)
        except Exception:
            # Attempt crude line-by-line parse for '# section: text' pattern
            data = {}
            for line in raw.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    if k.strip().lower() in self.output_sections:
                        data[k.strip().lower()] = v.strip()
            return data

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
                line = re.sub(r"^[\-\*\d\.\)]+\s*", "", line)  # remove leading bullets/numbers
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
