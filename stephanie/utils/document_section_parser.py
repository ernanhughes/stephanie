# stephanie/utils/document_section_parser.py
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from fuzzywuzzy import process

from stephanie.utils.string_utils import (
    clean_heading as util_clean_heading,
    normalize_key,
    trunc,
)
from stephanie.utils.text_utils import (
    is_high_quality_section,
    section_quality,
)


class DocumentSectionParser:
    def __init__(self, cfg=None, logger=None):
        self.cfg = cfg or {}
        self.logger = logger or print

        # Old: min_chars_per_sec
        # New: also allow words-based gating (default 30)
        self.min_chars_per_sec = self.cfg.get("min_chars_per_sec", 20)
        self.min_words_per_sec = self.cfg.get("min_words_per_sec", 30)

        # Load target sections from YAML (can be disabled via config)
        self.config_path = self.cfg.get(
            "target_sections_config", "config/domain/target_sections.yaml"
        )
        self.enable_target_sections = self.cfg.get("enable_target_sections", False)

        self.TARGET_SECTIONS = (
            self._load_target_sections() if self.enable_target_sections else {}
        )
        self.SECTION_TO_CATEGORY = (
            self._build_section_to_category() if self.TARGET_SECTIONS else {}
        )

    # ------------------------------------------------------------------ public

    def parse(self, text: str) -> dict:
        """
        Parse raw document text into a dict of {section_name: section_text}.

        - Uses `unstructured` to segment the document
        - Groups text under headings/titles
        - Cleans headings and maps them to configured categories (if enabled)
        - Trims obviously low-quality sections
        """
        from unstructured.partition.text import partition_text
        from unstructured.staging.base import elements_to_json

        elements = partition_text(text=text)
        json_elems = elements_to_json(elements)
        structure = self.parse_unstructured_elements(json.loads(json_elems))

        # Clean headings (via utils) before mapping/quality filtering
        cleaned = {self.clean_section_heading(k): v for k, v in structure.items()}

        if self.SECTION_TO_CATEGORY:
            mapped = self.map_sections(cleaned)
        else:
            # If no target sections configured, just pass headings through
            mapped = cleaned

        final = self.trim_low_quality_sections(mapped)
        return final

    def parse_with_scores(self, text: str) -> tuple[dict[str, str], dict[str, float]]:
        """
        Extended variant of parse() that also returns per-section quality scores
        normalized within the document.

        Returns:
            (sections, scores)

            - sections: {section_name: text}
            - scores:   {section_name: score_in_[0,1]} where 1.0 is the best
                        section *within this document*.
        """
        sections = self.parse(text)
        raw_scores = {
            name: section_quality(content, min_words=self.min_words_per_sec)
            for name, content in sections.items()
        }
        if not raw_scores:
            return sections, {}

        max_score = max(raw_scores.values())
        if max_score <= 0:
            # Avoid division by zero; everything was 0 quality
            return sections, {k: 0.0 for k in raw_scores.keys()}

        normalized = {k: v / max_score for k, v in raw_scores.items()}
        return sections, normalized

    # ---------------------------------------------------------- config / setup

    def _load_target_sections(self) -> dict:
        """Load TARGET_SECTIONS from a YAML file."""
        path = Path(self.config_path)
        if not path.exists():
            raise FileNotFoundError(f"Target sections config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_section_to_category(self) -> dict:
        """Build reverse lookup map from synonyms to categories."""
        mapping = {}
        for cat, synonyms in self.TARGET_SECTIONS.items():
            for synonym in synonyms:
                normalized = self._normalize(synonym)
                mapping[normalized] = cat
        return mapping

    # ---------------------------------------------------------- core parsing

    def parse_unstructured_elements(self, elements: list[dict]) -> dict[str, str]:
        """
        Convert `unstructured` elements into a heading → text map.

        We keep the logic here, but rely on utils for cleaning/normalizing
        where appropriate.
        """
        current_section = None
        current_content: list[str] = []
        structured: dict[str, str] = {}

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

    # ---------------------------------------------------------- heading helpers

    def _normalize(self, name: str) -> str:
        """
        Backwards-compatible wrapper around string_utils.normalize_key.

        Kept as an instance method so any existing callers still work,
        but the actual logic lives in the shared util.
        """
        return normalize_key(name)

    def clean_section_heading(self, heading: str) -> str:
        """
        Backwards-compatible wrapper around string_utils.clean_heading.
        """
        return util_clean_heading(heading)

    # ---------------------------------------------------------- mapping / fuzz

    def map_sections(self, parsed_sections: dict[str, str]) -> dict[str, str]:
        """
        Map raw headings to configured target categories using:

        - Exact normalized lookup (fast path)
        - Fuzzy match as a fallback (via fuzzywuzzy)
        """
        mapped: dict[str, str] = {}

        for sec_name, content in parsed_sections.items():
            normalized = self._normalize(sec_name)

            # Exact
            if normalized in self.SECTION_TO_CATEGORY:
                category = self.SECTION_TO_CATEGORY[normalized]
                mapped[category] = content
                continue

            # Fuzzy
            if not self.SECTION_TO_CATEGORY:
                continue

            best_match, score = process.extractOne(
                normalized, self.SECTION_TO_CATEGORY.keys()
            )
            if score > 75:
                category = self.SECTION_TO_CATEGORY[best_match]
                mapped[category] = content

        return mapped

    # ---------------------------------------------------------- quality filters

    def is_valid_section(self, text: str) -> bool:
        """
        Decide whether a parsed section is 'good enough' to keep.

        Rules (can evolve over time):
          1. Must have at least min_words_per_sec alphabetic words
          2. Must not be trivially short in characters
          3. Must not match obvious garbage patterns (page numbers, roman numerals, etc.)
        """
        if not text:
            return False

        stripped = text.strip()
        if len(stripped) < self.min_chars_per_sec:
            return False

        # 1) Words-based filter via shared util
        if not is_high_quality_section(stripped, min_words=self.min_words_per_sec):
            return False

        # 2) Keep your existing garbage regexes as a second pass
        garbage_patterns = [
            r"^\d+$",  # numeric only
            r"^[a-zA-Z]$",  # single letter
            r"^[A-Z][a-z]+\s\d+$",  # "Figure 1", "Table 3"
            r"^[ivxlcdmIVXLCDM]+$",  # roman numerals
            r"^[\W_]+$",  # only symbols
            r"^[^\w\s].{0,20}$",  # short weird symbol-leading strings
        ]

        for pattern in garbage_patterns:
            if re.fullmatch(pattern, stripped):
                return False

        return True

    def trim_low_quality_sections(
        self, structured_data: dict[str, str]
    ) -> dict[str, str]:
        """
        Drop sections that do not pass is_valid_section().

        We also log a truncated preview of what we are dropping.
        """
        cleaned: dict[str, str] = {}
        for key, text in structured_data.items():
            if self.is_valid_section(text):
                cleaned[key] = text
            else:
                preview = trunc(text, 200) or ""
                try:
                    self.logger.log(
                        "TrimmingSection",
                        {
                            "section": key,
                            "preview": preview,
                        },
                    )
                except AttributeError:
                    # Fallback if logger is a plain print-like callable
                    self.logger(f"[TrimmingSection] {key}: {preview}")
        return cleaned
