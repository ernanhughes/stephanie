# stephanie/agents/paper_improver/curriculum.py

# curriculum.py — Paper difficulty scorer + curriculum scheduler for progressive learning.
# Scores papers by "teachability" → routes easy ones first to bootstrap learning.

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

class CurriculumScheduler:
    """
    Ranks papers by estimated difficulty (low = easy, high = hard).
    Prioritizes papers with:
      - Explicit pseudocode or algorithms
      - Clear metrics/tables
      - Minimal proofs/theorems
    Outputs ordered queue for training or processing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        return {
            "bonus_keywords": {
                "algorithm": 1.5,
                "pseudocode": 2.0,
                "procedure": 1.0,
                "table": 0.8,
                "figure": 0.5,
                "equation": 0.5,
                "eq.": 0.5,
                "implementation": 1.0,
                "code": 0.8
            },
            "penalty_keywords": {
                "proof": -1.0,
                "theorem": -1.0,
                "lemma": -0.8,
                "corollary": -0.8,
                "induction": -1.2,
                "derivation": -0.7,
                "by inspection": -0.5,
                "trivially": -0.3,
                "without loss of generality": -0.8
            },
            "section_bias": {
                "method": 1.0,
                "approach": 1.0,
                "experimental setup": 0.5,
                "results": 0.3,
                "related work": -0.5,
                "introduction": 0.0,
                "conclusion": -0.3
            },
            "min_score_threshold": 0.0,   # filter out papers below this
            "max_difficulty": 10.0        # cap for normalization
        }

    def compute_teachability_score(self, paper: Dict[str, Any]) -> float:
        """
        Compute a normalized teachability score [0, 1] for a paper.
        Higher = easier to turn into code/text automatically.
        """
        text = self._extract_relevant_text(paper).lower()
        title = paper.get("title", "").lower()
        section = paper.get("section", "").lower()

        score = 0.0

        # Bonus: presence of implementation-friendly keywords
        for keyword, weight in self.config["bonus_keywords"].items():
            count = text.count(keyword) + title.count(keyword)
            score += count * weight

        # Penalty: presence of proofy/abstract keywords
        for keyword, weight in self.config["penalty_keywords"].items():
            count = text.count(keyword) + title.count(keyword)
            score += count * weight

        # Section bias: prioritize method/approach sections
        for sec_keyword, bias in self.config["section_bias"].items():
            if sec_keyword in section:
                score += bias

        # Normalize to [0, 1]
        max_possible = sum(w * 10 for w in self.config["bonus_keywords"].values())  # rough upper bound
        score = max(0.0, min(self.config["max_difficulty"], score))
        normalized = score / self.config["max_difficulty"]

        return round(normalized, 3)

    def _extract_relevant_text(self, paper: Dict[str, Any]) -> str:
        """Extract text fields to score — prioritize body, abstract, captions."""
        chunks = []
        for key in ["abstract", "body", "text", "content", "caption", "description"]:
            if key in paper and isinstance(paper[key], str):
                chunks.append(paper[key])
        return " ".join(chunks)

    def schedule_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort papers by teachability score (descending: easiest first).
        Optionally filter by min_score_threshold.
        """
        scored = []
        for paper in papers:
            score = self.compute_teachability_score(paper)
            if score >= self.config["min_score_threshold"]:
                scored.append((score, paper))

        # Sort descending: highest teachability (easiest) first
        sorted_papers = [paper for score, paper in sorted(scored, key=lambda x: x[0], reverse=True)]

        return sorted_papers

    def tag_and_rank(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Return papers with added 'teachability_score' and 'difficulty_rank'.
        """
        scored = [(self.compute_teachability_score(p), p) for p in papers]
        sorted_scored = sorted(scored, key=lambda x: x[0], reverse=True)

        for rank, (score, paper) in enumerate(sorted_scored):
            paper["teachability_score"] = score
            paper["difficulty_rank"] = rank + 1  # 1 = easiest

        return [paper for score, paper in sorted_scored]

    def load_from_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load papers from JSON files (e.g., scraped/arxiv)."""
        papers = []
        for path in file_paths:
            p = Path(path)
            if p.exists() and p.suffix == ".json":
                try:
                    paper = json.loads(p.read_text())
                    paper["source_path"] = str(p)
                    papers.append(paper)
                except Exception as e:
                    print(f"⚠️ Failed to load {p}: {e}")
        return papers

    def save_curriculum(self, papers: List[Dict[str, Any]], output_path: str):
        """Save ranked curriculum to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(papers, f, indent=2)
        print(f"✅ Curriculum saved to {output_path} ({len(papers)} papers)")

    def get_stats(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return summary stats for dashboard."""
        if not papers:
            return {"count": 0, "avg_score": 0.0, "min_score": 0.0, "max_score": 0.0}

        scores = [p.get("teachability_score", self.compute_teachability_score(p)) for p in papers]
        return {
            "count": len(papers),
            "avg_score": round(sum(scores) / len(scores), 3),
            "min_score": round(min(scores), 3),
            "max_score": round(max(scores), 3),
            "passing_count": len([s for s in scores if s >= self.config["min_score_threshold"]])
        }