# co_ai/scoring/proximity.py
import re

from co_ai.scoring.base_score import BaseScore


class ProximityScore(BaseScore):
    name = "proximity"
    default_value = 0.0

    def compute(self, hypothesis: dict, context: dict) -> float:
        analysis = hypothesis.get("proximity_analysis")
        if not analysis:
            return self.default_value

        try:
            themes = self._extract_block(analysis, "Common Themes Identified")
            grafts = self._extract_block(analysis, "Grafting Opportunities")
            directions = self._extract_block(analysis, "Strategic Directions")
            score = self._heuristic_score(themes, grafts, directions)
            justification = self._generate_justification(themes, grafts, directions)

            structured = {
                "themes": themes,
                "graft_suggestions": grafts,
                "strategic_directions": directions,
                "score": score,
                "justification": justification,
            }

            hypothesis["proximity_structured"] = structured

            self._store_score(
                hypothesis,
                context,
                score_type="proximity_usefulness",
                score_data={
                    "score": score,
                    "rationale": justification,
                    "themes": themes,
                    "grafts": grafts,
                    "directions": directions,
                }
            )

            return score

        except Exception as e:
            self.logger.log("ProximityScoreParseFailed", {
                "error": str(e),
                "snippet": analysis[:300],
            })
            return self.default_value

    def _extract_block(self, text: str, section_title: str) -> list:
        pattern = rf"# {re.escape(section_title)}\n((?:- .+\n?)*)"
        match = re.search(pattern, text)
        if not match:
            return []
        block = match.group(1).strip()
        return [line.strip("- ").strip() for line in block.splitlines() if line.strip()]

    def _heuristic_score(self, themes, grafts, directions) -> float:
        """
        Simple scoring heuristic based on the number of insights generated.
        """
        return min(100.0, 10 * len(themes) + 10 * len(grafts) + 20 * len(directions))

    def _generate_justification(self, themes, grafts, directions) -> str:
        return (
            f"Identified {len(themes)} themes, {len(grafts)} grafting suggestions, "
            f"and {len(directions)} strategic directions."
        )
