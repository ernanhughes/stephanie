# stephanie/agents/learning/scoring.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List
import re

class Scoring:
    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        try:
            from stephanie.scoring.scorer.knowledge_scorer import KnowledgeScorer
            self.knowledge = KnowledgeScorer(cfg.get("knowledge_scorer", {}), memory, container, logger)
        except Exception:
            self.knowledge = None

    def rubric_dims(self, text: str, ref: str) -> Tuple[float, float]:
        sents = [s for s in re.split(r"[.!?]\s+", (text or "").strip()) if s]
        avg_len = sum(len(s.split()) for s in sents) / max(1, len(sents))
        clarity = max(0.0, min(1.0, 1.1 - (abs(avg_len - 22) / 22)))
        def toks(t): return set(re.findall(r"\b\w+\b", (t or "").lower()))
        inter = len(toks(text) & toks(ref))
        grounding = max(0.0, min(1.0, inter / max(30, len(toks(ref)) or 1)))
        return clarity, grounding

    def weaknesses(self, summary: str, ref: str) -> List[str]:
        out = []
        if len(summary or "") < 400: out.append("too short / thin detail")
        if "we propose" in (ref or "").lower() and "we propose" not in (summary or "").lower():
            out.append("misses core claim language")
        if (summary or "").count("(") != (summary or "").count(")"):
            out.append("formatting/parens issues")
        return out

    def score_summary(self, text: str, paper: Dict[str, Any], section: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        clarity, grounding = self.rubric_dims(text, section.get("section_text", ""))
        comps = {}
        if self.knowledge:
            goal_text = f"{paper.get('title','')}\n\n{paper.get('abstract','')}"
            meta = {"text_len_norm": min(1.0, len(text)/2000.0)}
            # Support either .predict or .model.predict
            if hasattr(self.knowledge, "predict"):
                p, comps = self.knowledge.predict(goal_text, text, meta=meta, return_components=True)
            elif hasattr(self.knowledge, "model") and hasattr(self.knowledge.model, "predict"):
                p, comps = self.knowledge.model.predict(goal_text, text, meta=meta, return_components=True)
            else:
                p, comps = 0.5, {"probability": 0.5}
            knowledge = float((comps or {}).get("probability", p))
        else:
            knowledge = 0.5*clarity + 0.5*grounding
        overall = 0.6*knowledge + 0.25*clarity + 0.15*grounding
        return {
            "overall": overall,
            "knowledge_score": knowledge,
            "clarity": clarity,
            "grounding": grounding,
            "weaknesses": self.weaknesses(text, section.get("section_text","")),
            **(comps or {}),
        }

    def score_candidate(self, text: str, section_text: str) -> Dict[str, float]:
        dims = self.score_summary(text, {"title": "", "abstract": ""}, {"section_text": section_text}, {})
        k, c, g = dims["knowledge_score"], dims["clarity"], dims["grounding"]
        overall = 0.6*k + 0.25*c + 0.15*g
        verified = (g >= 0.45) and (len(text) >= self.cfg.get("min_verified_len", 250))
        return {"k": k, "c": c, "g": g, "overall": overall, "verified": bool(verified)}
