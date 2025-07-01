# stephanie/scoring/proximity.py
import re

from stephanie.scoring.base_evaluator import BaseEvaluator


class ProximityHeuristicEvaluator(BaseEvaluator):
    def evaluate(self, prompt:str, response: str) -> dict:
        analysis = response
        if not analysis:
            return {
                "score": 0.0,
                "rationale": "No proximity analysis available.",
                "themes": [],
                "grafts": [],
                "directions": [],
            }

        try:
            themes = self._extract_block(analysis, "Common Themes Identified")
            grafts = self._extract_block(analysis, "Grafting Opportunities")
            directions = self._extract_block(analysis, "Strategic Directions")
            score = self._heuristic_score(themes, grafts, directions)
            justification = self._generate_justification(themes, grafts, directions)

            return {
                "score": score,
                "rationale": justification,
                "themes": themes,
                "grafts": grafts,
                "directions": directions,
            }

        except Exception as e:
            return {
                "score": 0.0,
                "rationale": f"Failed to parse proximity analysis: {str(e)}",
                "themes": [],
                "grafts": [],
                "directions": [],
            }

    def _extract_block(self, text: str, section_title: str) -> list:
        pattern = rf"# {re.escape(section_title)}\n((?:- .+\n?)*)"
        match = re.search(pattern, text)
        if not match:
            return []
        block = match.group(1).strip()
        return [line.strip("- ").strip() for line in block.splitlines() if line.strip()]

    def _generate_justification(self, themes, grafts, directions) -> str:
        return (
            f"Identified {len(themes)} themes, {len(grafts)} grafting suggestions, "
            f"and {len(directions)} strategic directions."
        )

    def _fallback(self, message: str):
        return {
            "score": 0.0,
            "rationale": message,
            "dimensions": {
                "proximity_themes": {"score": 0, "weight": 0.3, "rationale": message},
                "proximity_grafts": {"score": 0, "weight": 0.3, "rationale": message},
                "proximity_directions": {"score": 0, "weight": 0.4, "rationale": message},
            }
        }

    def _heuristic_score(self, themes, grafts, directions) -> float:
        """
        Simple scoring heuristic based on the number of insights generated.
        """
        return min(100.0, 10 * len(themes) + 10 * len(grafts) + 20 * len(directions))
