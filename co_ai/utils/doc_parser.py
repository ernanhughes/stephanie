# co_ai/utils/doc_parser.py
import re
from fuzzywuzzy import process

# Target schema for normalized paper sections
TARGET_SECTIONS = {
    "title": ["title"],
    "abstract": ["abstract", "summary"],
    "introduction": ["introduction", "intro"],
    "related_work": ["related work", "background", "prior work", "literature review"],
    "method": ["method", "methods", "methodology", "approach", "algorithm"],
    "implementation": ["implementation", "code", "technical details"],
    "results": ["results", "result", "evaluation", "performance"],
    "discussion": ["discussion", "analysis", "interpretation"],
    "conclusion": ["conclusion", "conclusions", "final remarks"],
    "limitations": ["limitations", "drawbacks", "challenges"],
    "future_work": ["future work", "next steps", "extensions"],
    "references": ["references", "bibliography", "works cited"],
}

SECTION_TO_CATEGORY = {}
for cat, synonyms in TARGET_SECTIONS.items():
    for synonym in synonyms:
        SECTION_TO_CATEGORY[synonym] = cat


class DocumentParserUtils:
    def __init__(self, cfg=None, logger=None):
        self.cfg = cfg or {}
        self.logger = logger or print  # Default to print if no logger provided
        self.min_chars_per_sec = self.cfg.get("min_chars_per_sec", 20)

    def parse_unstructured_elements(self, elements: list[dict]) -> dict[str, str]:
        """
        Parse unstructured.io JSON output into structured content.
        Groups text under detected headings.
        """
        current_section = None
        current_content = []
        structured = {}

        for el in elements:
            el_type = el.get("type")
            el_text = el.get("text", "").strip()

            if not el_text:
                continue

            if el_type in ("Title", "Heading"):
                if current_section and current_content:
                    structured[current_section] = "\n\n".join(current_content).strip()
                current_section = el_text.strip()
                current_content = []
            elif el_type in ("NarrativeText", "UncategorizedText", "ListItem"):
                if current_section:
                    current_content.append(el_text)

        if current_section and current_content:
            structured[current_section] = "\n\n".join(current_content).strip()

        return structured

    def clean_section_heading(self, heading: str) -> str:
        """
        Clean up heading strings by removing numbers, bullets, and special chars.
        """
        if not heading:
            return ""
        # Remove leading numbers and dots
        heading = re.sub(r"^\s*[\d\.\s]+\s*", " ", heading)
        # Remove common prefixes
        heading = re.sub(r"^(section|chapter|part)\s+\w+", "", heading, flags=re.IGNORECASE)
        # Strip special characters
        heading = re.sub(r"[^\w\s]", "", heading)
        heading = re.sub(r"\s+", " ", heading).strip()
        return heading

    def map_sections(self, parsed_sections: dict[str, str]) -> dict[str, str]:
        """
        Map raw section names to standardized categories.
        """
        mapped = {}

        for sec_name, content in parsed_sections.items():
            normalized = self._normalize(sec_name)
            if normalized in SECTION_TO_CATEGORY:
                category = SECTION_TO_CATEGORY[normalized]
                mapped[category] = content
            else:
                best_match, score = process.extractOne(normalized, SECTION_TO_CATEGORY.keys())
                if score > 75:
                    category = SECTION_TO_CATEGORY[best_match]
                    mapped[category] = content

        return mapped

    def _normalize(self, name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower().strip())

    def is_valid_section(self, text: str) -> bool:
        """
        Heuristic filter to determine if a section has meaningful content.
        """
        if not text or len(text.strip()) < 10:
            return False

        garbage_patterns = [
            r"^\d+$",
            r"^[a-zA-Z]$",
            r"^[A-Z][a-z]+\s\d+$",
            r"^[ivxlcdmIVXLCDM]+$",
            r"^[\W_]+$",
            r"^[^\w\s].{0,20}$"
        ]

        for pattern in garbage_patterns:
            if re.fullmatch(pattern, text.strip()):
                return False

        return True

    def trim_low_quality_sections(self, structured_data: dict[str, str]) -> dict[str, str]:
        cleaned = {}
        for key, text in structured_data.items():
            if self.is_valid_section(text):
                cleaned[key] = text
            else:
                self.logger.log("TrimmingSection", {"section": key, "data": text[:50]})
        return cleaned