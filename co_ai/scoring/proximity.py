# co_ai/scoring/proximity.py
import re
from co_ai.scoring.base_score import BaseScore
from co_ai.models import ScoreORM


class ProximityScore(BaseScore):
    name = "proximity"
    default_value = 0.0

    def compute(self, hypothesis: dict, context: dict) -> float:
        analysis = hypothesis.get("proximity_analysis")
        if not analysis:
            return self.default_value

        try:
            themes = self._extract_block(analysis, "# Common Themes")
            grafts = self._extract_block(analysis, "# Grafting Suggestions")
            directions = self._extract_block(analysis, "# Strategic Directions")
            score = self._extract_score(analysis)
            justification = self._extract_justification(analysis)

            hypothesis["proximity_structured"] = {
                "themes": themes,
                "grafts": grafts,
                "directions": directions,
                "score": score,
                "justification": justification,
            }

            self._store_score(hypothesis, context, "proximity_usefulness", {
                "score": score,
                "rationale": justification
            })

            return score

        except Exception as e:
            self.logger.log("ProximityScoreParseFailed", {
                "error": str(e),
                "snippet": analysis[:300]
            })
            return self.default_value

    def _extract_block(self, text: str, section: str) -> list:
        pattern = rf"{re.escape(section)}\n(?:- .+\n)+"
        match = re.search(pattern, text)
        if not match:
            return []
        return [line.strip("- ").strip() for line in match.group().splitlines()[1:]]

    def _extract_score(self, text: str) -> float:
        match = re.search(r"# Overall Score \[0[–-]100\]\n(\d+)", text)
        return float(match.group(1)) if match else self.default_value

    def _extract_justification(self, text: str) -> str:
        match = re.search(r"# Justification\n(.+?)(\n#|\Z)", text, re.DOTALL)
        return match.group(1).strip() if match else ""
