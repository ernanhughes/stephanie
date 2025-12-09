# stephanie/tools/section_parser_tool.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from stephanie.scoring.scorable import Scorable
from stephanie.tools.base_tool import BaseTool
from stephanie.utils.document_section_parser import DocumentSectionParser

log = logging.getLogger(__name__)


class SectionParserTool(BaseTool):
    """
    Tool wrapper around DocumentSectionParser.

    - Takes scorable.text
    - Uses unstructured via DocumentSectionParser to get semantic sections
    - Optionally computes per-section quality scores
    - Attaches results to scorable.meta["sections"][<meta_key>]

    The output is a list of dicts like:

        {
          "name":  "Introduction / Background",
          "text":  "<section body>",
          "score": 0.87,          # relative quality in [0,1]
          "index": 0              # document order
        }
    """

    name = "section_parser"

    def __init__(self, cfg: Dict[str, Any], memory, container, logger) -> None:
        super().__init__(cfg, memory, container, logger)

        # Where to store results inside scorable.meta["sections"]
        self.meta_key: str = cfg.get("meta_key", "doc_sections")

        # Optional post-filtering
        self.min_score: float = float(cfg.get("min_score", 0.0))
        self.max_sections: int = int(cfg.get("max_sections", 0))  # 0 = no limit

        # Config for the underlying DocumentSectionParser
        parser_cfg: Dict[str, Any] = cfg.get("parser", {})
        # Allow top-level overrides for convenience
        if "min_chars_per_sec" in cfg:
            parser_cfg.setdefault("min_chars_per_sec", cfg["min_chars_per_sec"])
        if "min_words_per_sec" in cfg:
            parser_cfg.setdefault("min_words_per_sec", cfg["min_words_per_sec"])
        if "target_sections_config" in cfg:
            parser_cfg.setdefault("target_sections_config", cfg["target_sections_config"])
        if "enable_target_sections" in cfg:
            parser_cfg.setdefault("enable_target_sections", cfg["enable_target_sections"])

        self.parser = DocumentSectionParser(cfg=parser_cfg, logger=logger or log)

    # ------------------------------------------------------------------ #
    async def run(self, scorable: Scorable, **kwargs) -> Scorable:
        """
        Run section parsing on the scorable's text and attach sections to meta.
        """
        text = getattr(scorable, "text", None) or getattr(scorable, "content", None)
        if not text or not text.strip():
            log.debug("[SectionParserTool] Empty text for scorable %r; skipping",
                      getattr(scorable, "id", None))
            return scorable

        try:
            sections, scores = self._parse_with_scores(text)
        except Exception as exc:
            log.exception(
                "[SectionParserTool] Failed to parse sections for %r: %s",
                getattr(scorable, "id", None),
                exc,
            )
            return scorable

        # Flatten into ordered list of section dicts
        ordered_sections: List[Dict[str, Any]] = []
        for idx, (name, sec_text) in enumerate(sections.items()):
            score = float(scores.get(name, 0.0))
            if score < self.min_score:
                continue

            ordered_sections.append(
                {
                    "name": name,
                    "text": sec_text,
                    "score": score,
                    "index": idx,
                }
            )

        # Apply max_sections, if set
        if self.max_sections > 0 and len(ordered_sections) > self.max_sections:
            ordered_sections = ordered_sections[: self.max_sections]

        # Attach to scorable.meta
        if not hasattr(scorable, "meta") or scorable.meta is None:
            scorable.meta = {}

        sections_meta = scorable.meta.setdefault("sections", {})
        sections_meta[self.meta_key] = ordered_sections

        log.debug(
            "[SectionParserTool] Parsed %d sections for scorable %r",
            len(ordered_sections),
            getattr(scorable, "id", None),
        )

        return scorable

    # ------------------------------------------------------------------ #
    def _parse_with_scores(self, text: str) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Thin wrapper so we can swap between parse() and parse_with_scores()
        if needed in the future.
        """
        # Today we always use the score-aware variant.
        return self.parser.parse_with_scores(text)
