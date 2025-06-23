import json
import re
from pathlib import Path

import yaml
from fuzzywuzzy import process


class DocumentSectionParser:
    def __init__(self, cfg=None, logger=None):
        self.cfg = cfg or {}
        self.logger = logger or print
        self.min_chars_per_sec = self.cfg.get("min_chars_per_sec", 20)

        # Load target sections from YAML
        self.config_path = self.cfg.get(
            "target_sections_config",
            "config/domain/target_sections.yaml"
        )
        self.TARGET_SECTIONS = self._load_target_sections()
        self.SECTION_TO_CATEGORY = self._build_section_to_category()

    def parse(self, text: str) -> dict:
        from unstructured.partition.text import partition_text
        from unstructured.staging.base import elements_to_json

        elements = partition_text(text=text)
        json_elems = elements_to_json(elements)
        structure = self.parse_unstructured_elements(json.loads(json_elems))
        cleaned = {self.clean_section_heading(k): v for k, v in structure.items()}
        mapped = self.map_sections(cleaned)
        final = self.trim_low_quality_sections(mapped)
        return final

    def _load_target_sections(self) -> dict:
        """Load TARGET_SECTIONS from a YAML file"""
        path = Path(self.config_path)
        if not path.exists():
            raise FileNotFoundError(f"Target sections config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_section_to_category(self) -> dict:
        """Build reverse lookup map from synonyms to categories"""
        mapping = {}
        for cat, synonyms in self.TARGET_SECTIONS.items():
            for synonym in synonyms:
                normalized = self._normalize(synonym)
                mapping[normalized] = cat
        return mapping

    def _normalize(self, name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower().strip())

    def parse_unstructured_elements(self, elements: list[dict]) -> dict[str, str]:
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
        if not heading:
            return ""
        heading = re.sub(r"^\s*[\d\.\s]+\s*", " ", heading)
        heading = re.sub(r"^(section|chapter|part)\s+\w+", "", heading, flags=re.IGNORECASE)
        heading = re.sub(r"[^\w\s]", "", heading)
        heading = re.sub(r"\s+", " ", heading).strip()
        return heading

    def map_sections(self, parsed_sections: dict[str, str]) -> dict[str, str]:
        mapped = {}

        for sec_name, content in parsed_sections.items():
            normalized = self._normalize(sec_name)
            if normalized in self.SECTION_TO_CATEGORY:
                category = self.SECTION_TO_CATEGORY[normalized]
                mapped[category] = content
            else:
                best_match, score = process.extractOne(normalized, self.SECTION_TO_CATEGORY.keys())
                if score > 75:
                    category = self.SECTION_TO_CATEGORY[best_match]
                    mapped[category] = content

        return mapped

    def is_valid_section(self, text: str) -> bool:
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