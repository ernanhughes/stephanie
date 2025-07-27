# File: stephanie/utils/idea_parser.py

import re
from typing import Dict, List, Optional

from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.scoring.svm_scorer import SVMScorer
# Import the existing document section parser
from stephanie.utils.document_section_parser import DocumentSectionParser


class IdeaParser:
    """
    Extracts structured, learnable ideas from research papers — especially from methods/results sections.

    Designed to be used downstream of DocumentProfilerAgent or DocumentSectionParser.

    Key Features:
        - Parses method/results sections for learnable claims
        - Structurizes ideas into cognitive scaffolds
        - Tags ideas by domain/technique
        - Scores usefulness/novelty/alignment
    """

    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.dimensions = cfg.get("dimensions", ["usefulness", "novelty", "alignment", "epistemic_gain"])
        self.logger = logger
        self.prompt_loader = None  # Will be injected by agent
        self.call_llm = None       # Will be injected by agent
        self.memory = None         # Will be injected by agent

        # Initialize section parser
        self.section_parser = DocumentSectionParser(cfg=cfg, logger=logger)

        # Initialize scorer
        self.scorer = SVMScorer(
            cfg,
            logger=self.logger,
            memory=self.memory,
            dimensions=self.dimensions
        )

    def parse(self, paper_text: str, paper_title: str = "", context={}, override_dimensions: Optional[list] = None) -> List[Dict]:
        """
        Main entry point — takes raw paper text and returns a list of structured ideas.

        Uses DocumentSectionParser to extract sections first, then IdeaParser to mine learnables.
        """
        try:
            # Step 1: Use DocumentSectionParser to extract structured content
            paper_sections = self.section_parser.parse(paper_text)

            if not paper_sections:
                self.logger.log("NoSectionsParsed", {"title": paper_title})
                return []

            # Step 2: Identify relevant sections
            method_text = self._get_section(paper_sections, ["method", "methods"])
            results_text = self._get_section(paper_sections, ["results"])

            if not method_text:
                self.logger.log("NoMethodSection", {"paper": paper_title})
                return []

            # Step 3: Prompt LLM to extract learnable ideas
            prompt_context = {
                "method_section": method_text[:self.cfg.get("llm_max_chars", 12000)],
                "results_section": results_text[:self.cfg.get("llm_max_chars", 12000)] if results_text else "",
                "sections_available": list(paper_sections.keys()),
                **context
            }

            prompt = self.prompt_loader.from_file(
                self.cfg.get("idea_prompt_file", "extract_ideas.txt"),
                self.cfg, prompt_context
            )

            raw_output = self.call_llm(prompt, prompt_context)

            # Step 4: Parse and validate output
            ideas = self._parse_idea_response(raw_output)

            # Step 5: Enrich with metadata
            enriched = []
            for idea in ideas:
                enriched.append({
                    **idea,
                    "source_section": "methods",
                    "source_paper": paper_title,
                    "tags": self._tag_idea(idea),
                    "scoring": self._score_idea(context, idea, override_dimensions),
                    "dimensions": override_dimensions or self.dimensions
                })

            return enriched

        except Exception as e:
            self.logger.log("IdeaParsingFailed", {"error": str(e)})
            return []

    def _get_section(self, sections: dict, candidates: list[str]) -> Optional[str]:
        """Helper to get first matching section."""
        for sec in candidates:
            if sec in sections:
                return sections[sec]
        return None


    def _parse_idea_response(self, response: str) -> List[Dict]:
        """
        Parses LLM output into structured idea objects.
        Cleans up markdown artifacts (like '**Description:**') in keys and values.
        """
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        ideas = []
        current = {}

        for line in lines:
            if re.match(r"^\d+\.", line):  # New idea starts
                if current:
                    ideas.append(current)
                    current = {}
                current["title"] = line[2:].strip()
            elif ":" in line:
                key, val = line.split(":", 1)
                key = re.sub(r'^[\*\s]*|[\*\s:]*$', '', key).strip().lower()
                val = re.sub(r'^[\*\s]*|[\*\s]*$', '', val).strip()
                current[key] = val

        if current:
            ideas.append(current)

        return ideas

    def _tag_idea(self, idea: dict) -> List[str]:
        """
        Uses heuristics + embeddings to assign tags like 'rl', 'stability', etc.
        """
        text = f"{idea.get('title')} {idea.get('description')}"
        if not text:
            return []

        tags = set()

        # Heuristic-based tagging
        if any(kw in text.lower() for kw in ["q-value", "reinforce", "policy"]):
            tags.add("reinforcement")
        if any(kw in text.lower() for kw in ["penalty", "regularization", "smooth"]):
            tags.add("stability")
        if any(kw in text.lower() for kw in ["loss", "objective", "gradient"]):
            tags.add("loss_term")
        if any(kw in text.lower() for kw in ["network", "transform", "representation"]):
            tags.add("representation")

        return list(tags)

    def _score_idea(self, context: dict, idea: dict, override_dimensions: Optional[list] = None) -> dict:
        """
        Use SVM scorer or other models to score idea along multiple dimensions.
        """
        scorable = Scorable(text=idea.get("description", ""), target_type=TargetType.IDEA)
        self.scorer.logger=self.logger
        self.scorer.memory=self.memory
        
        score_bundle = self.scorer.score(context.get("goal"), scorable=scorable, 
                                         dimensions=override_dimensions or self.dimensions)

        return {
            dim: round(score, 2)
            for dim, score in score_bundle.to_dict().items()
        }