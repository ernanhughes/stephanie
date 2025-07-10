# stephanie/agents/builders/triplet_extractor.py

import re


class TripletExtractor:
    def __init__(self, cfg, prompt_loader, memory=None, logger=None, call_llm=None):
        self.cfg = cfg
        self.prompt_loader = prompt_loader
        self.memory = memory
        self.logger = logger
        self.call_llm = call_llm
        self.triplets_file = cfg.get("triplets_file", "triplet.txt")

    def extract(self, points: list, context: dict) -> list[tuple[str, str, str]]:
        """
        Extract triplets from a list of section points using an LLM prompt.
        """
        merged_context = {
            "points": points,
            "goal": context.get("goal"),
            **context
        }
        prompt = self.prompt_loader.from_file(self.triplets_file, self.cfg, merged_context)
        response = self.call_llm(prompt, context=merged_context)
        return self.parse_triplets(response)

    def parse_triplets(self, markdown_text: str) -> list[tuple[str, str, str]]:
        """
        Parse a markdown response containing triplets in the format:
        - (subject, predicate, object)
        """
        pattern = re.compile(r"-\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*(.+?)\s*\)")
        matches = pattern.findall(markdown_text)
        return [(subj.strip(), pred.strip(), obj.strip()) for subj, pred, obj in matches]
