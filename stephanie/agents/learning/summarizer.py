# stephanie/agents/learning/summarizer.py
from __future__ import annotations
from typing import Dict, Any, List
import json

class Summarizer:
    def __init__(self, cfg, memory, container, logger, strategy, scoring=None, prompt_loader=None, call_llm=None):
        self.cfg, self.memory, self.container, self.logger = cfg, memory, container, logger
        self.strategy = strategy
        # fallbacks
        self.scoring = scoring or (container.get("scoring") if hasattr(container, "get") else None)
        if self.scoring is None:
            from .scoring import Scoring
            self.scoring = Scoring(cfg, memory, container, logger)
        self.prompt_loader = prompt_loader or (container.get("prompt_loader") if hasattr(container, "get") else None)
        self.call_llm = call_llm or (container.get("call_llm") if hasattr(container, "get") else None)

    def baseline(self, paper: Dict[str, Any], section: Dict[str, Any], critical_msgs: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        merged = {
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "focus_section": section.get("section_name"),
            "focus_text": section.get("section_text", "")[:5000],
            "hints": "\n".join((m.get("assistant_text") or m.get("text") or "") 
                for m in (critical_msgs[:6] if critical_msgs else [])),
            **context,
        }
        prompt = self.prompt_loader.from_file("baseline_summary", self.cfg, merged)
        return self.call_llm(prompt, merged)

    def improve_once(self, paper: Dict[str, Any], section: Dict[str, Any], current_summary: str, context: Dict[str, Any]) -> str:
        # compute weaknesses for the improver
        metrics = self.scoring.score_summary(current_summary, paper, section, context)
        merged = {
            "title": paper.get("title", ""),
            "section_name": section.get("section_name"),
            "section_text": section.get("section_text", "")[:6000],
            "current_summary": current_summary,
            "skeptic_weight": self.strategy.state.skeptic_weight,
            "editor_weight": self.strategy.state.editor_weight,
            "risk_weight": self.strategy.state.risk_weight,
            "weaknesses": json.dumps(metrics.get("weaknesses", []), ensure_ascii=False),
            **context,
        }
        prompt = self.prompt_loader.from_file("improve_summary", self.cfg, merged)
        return self.call_llm(prompt, context)

    def verify_and_improve(self, baseline: str, paper: Dict[str, Any], section: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        current = baseline
        iters: List[Dict[str, Any]] = []
        max_iter = int(self.cfg.get("max_iterations", 3))
        for i in range(1, max_iter + 1):
            metrics = self.scoring.score_summary(current, paper, section, context)
            iters.append({"iteration": i, "score": metrics["overall"], "metrics": metrics})
            if metrics["overall"] >= self.strategy.state.verification_threshold:
                break
            current = self.improve_once(paper, section, current, context)
        self.strategy.evolve(iters, context)
        return {"summary": current, "metrics": metrics, "iterations": iters}
