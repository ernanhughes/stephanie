"""
SectionPrioritizer
------------------
Ranks sections to process first. Methods/Results-heavy sections first,
then Discussion/Conclusion, then Intro/Related Work.

Signals:
- Title heuristics (priority by kind)
- Density of evidence (Figures/Tables)
- Technique keywords (method/algorithm/ablation/experiment)
"""

from __future__ import annotations

import re
from typing import Dict, List

TITLE_PRI = [
    ("method", 1.0),
    ("approach", 0.95),
    ("model", 0.92),
    ("experiment", 0.90),
    ("results", 0.90),
    ("ablation", 0.88),
    ("evaluation", 0.86),
    ("analysis", 0.84),
    ("discussion", 0.80),
    ("conclusion", 0.78),
    ("introduction", 0.60),
    ("related work", 0.55),
    ("background", 0.50),
]

TECH_HINTS = ("transformer", "adapter", "loss", "optimization", "pipeline", "retrieval", "graph", "policy", "reward")

def score_section(sec: Dict) -> float:
    title = (sec.get("section_name") or "").lower()
    text  = sec.get("section_text") or ""

    # 1) title heuristic
    base = 0.5
    for key, w in TITLE_PRI:
        if key in title:
            base = max(base, w)

    # 2) evidence density
    ev = len(re.findall(r"\b(fig\.|figure|table|tbl\.)\b", text.lower()))
    ev_score = min(0.15, 0.03 * ev)

    # 3) technique keywords
    tech = sum(1 for k in TECH_HINTS if k in text.lower())
    tech_score = min(0.2, 0.02 * tech)

    # length sanity
    length_score = min(0.15, 0.00002 * len(text))

    return float(min(1.0, base + ev_score + tech_score + length_score))

def prioritize_sections(sections: List[Dict], top_n: int = None) -> List[Dict]:
    ranked = sorted(sections, key=score_section, reverse=True)
    return ranked[:top_n] if top_n else ranked
