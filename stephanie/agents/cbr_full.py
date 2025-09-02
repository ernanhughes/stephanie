# stephanie/agents/cbr_agent.py
import time
import uuid
from typing import Dict, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.proximity import ProximityAgent
from stephanie.agents.lats_dspy import LATSProgram
from stephanie.scoring.scorable import Scorable
from stephanie.data.score_corpus import ScoreCorpus
from stephanie.scoring.calculations.mars_calculator import MARSCalculator
from stephanie.scoring.scorer.scorable_ranker import ScorableRanker
from stephanie.scoring.scoring_manager import ScoringManager

class Casebook:
    """Persistent library of cases (problem, solution, evaluation)."""
    def __init__(self, memory):
        self.memory = memory

    def retrieve(self, goal_text: str, k=5):
        # Use embeddings/proximity search
        return self.memory.hypotheses.find_similar(goal_text, top_k=k)

    def retain(self, case: Dict):
        return self.memory.hypotheses.save(case)

    def all(self):
        return self.memory.hypotheses.list()


class CBRAgent(BaseAgent):
    """Case-Based Reasoning Agent"""
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.casebook = Casebook(memory)
        self.proximity = ProximityAgent(cfg, memory, logger)
        self.lats_program = LATSProgram(cfg, self)
        self.ranker = ScorableRanker(cfg, memory, logger)
        self.mars = MARSCalculator(cfg, memory, logger)
        self.scorer_name = cfg.get("scorer_name", "sicql")
        self.dimensions = cfg.get("dimensions",
            ["alignment", "clarity", "implementability", "novelty", "relevance"])

    async def run(self, context: Dict) -> Dict:
        goal = context.get("goal", {})
        goal_text = goal.get("goal_text", "")
        self.logger.log("CBRStart", {"goal": goal_text})

        # === 1. RETRIEVE ===
        retrieved = self.casebook.retrieve(goal_text, k=5)
        self.logger.log("CBRRetrieve", {"retrieved": [r["text"] for r in retrieved]})

        # === 2. REUSE ===
        completions, steps = self.lats_program(state=goal_text, trace=[], depth=0)
        self.logger.log("CBRReuse", {"candidates": completions})

        # === 3. REVISE ===
        scored_candidates = []
        for comp in completions:
            scorable = Scorable(text=comp, metadata={"agent": "CBRAgent"})
            score_result = self.scoring.score(
                self.scorer_name,
                scorable=scorable,
                context=context,
                dimensions=self.dimensions,
            )
            scored_candidates.append({
                "text": comp,
                "score": score_result.aggregate(),
                "dimensions": score_result.to_dict()
            })

        corpus = ScoreCorpus.from_list(scored_candidates)
        mars_results = self.mars.calculate(corpus, context=context)

        best = max(scored_candidates, key=lambda x: x["score"])
        self.logger.log("CBRRevise", {"best": best, "mars": mars_results})

        # === 4. RETAIN ===
        case = {
            "problem": goal_text,
            "context": context,
            "solution": best["text"],
            "evaluation": {
                "scores": best["dimensions"],
                "mars": mars_results,
            },
            "metadata": {"agent": "CBRAgent", "created_at": time.time()}
        }
        retained = self.casebook.retain(case)
        self.logger.log("CBRRetain", {"case_id": retained.id})

        context["cbr_result"] = case
        return context
