"""
agent_scorer.py
================

AgentScorerAgent is a *meta-agent* that evaluates the outputs of other agents in the
Stephanie system. It plugs into the pipeline via PlanTraceMonitor or similar supervisors
to provide **fine-grained agent-level feedback**.

Core responsibilities:
----------------------
1. **Activation condition**: Only runs if the producing agent marks itself `is_scorable`.
2. **Scorable wrapping**: Converts agent output into a `Scorable` entity with metadata.
3. **Multi-scorer evaluation**: Runs across configured backends (SICQL, MRQ, EBT, etc.).
4. **MARS analysis**: Measures agreement/divergence between scorers.
5. **Ranking vs past cases**: Uses ScorableRanker for case-based reasoning context.
6. **Alternative suggestions**: Proposes different agents that historically performed well.
7. **Persistence**: Stores results into `EvaluationStore` for traceability and self-tuning.

This agent enables:
- **Case-based reasoning** at the agent level
- **Delta tracking** when combined with ScoreDeltaCalculator
- **Self-improving agent selection** in future pipeline runs
"""

import traceback
from typing import Dict, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorer.scorable_ranker import ScorableRanker
from stephanie.scoring.calculations.mars_calculator import MARSCalculator

import logging
logger = logging.getLogger(__name__)

class AgentScorerAgent(BaseAgent):
    """
    Meta-agent that scores the outputs of other agents.

    Workflow:
    - Checks if agent is scorable.
    - Converts output to a Scorable object.
    - Runs configured scorers (SICQL, MRQ, EBT, etc.).
    - Performs MARS analysis across scorer outputs.
    - Ranks result against similar cases.
    - Suggests alternative agents.
    - Persists results for long-term learning.
    """

    def __init__(self, cfg, memory, logger):
        # Initialize as standard BaseAgent
        super().__init__(cfg.get("agents", {}).get("agent_scorer", {}), memory, logger)

        # Scorers to run on every agent output
        self.enabled_scorers = cfg.get("enabled_scorers", ["sicql", "mrq", "ebt"])
        # Enable MARS consensus analysis
        self.include_mars = cfg.get("include_mars", True)
        # Enable ranking of cases for CBR
        self.include_ranking = cfg.get("include_ranking", True)

        # Top-k similar cases to retrieve
        self.rank_top_k = cfg.get("rank_top_k", 5)

        # Core components
        self.mars_calc = MARSCalculator(cfg, memory, logger)
        self.ranker = ScorableRanker(cfg, memory, logger)

        self.logger.log("AgentScorerInitialized", {
            "enabled_scorers": self.enabled_scorers,
            "include_mars": self.include_mars,
            "include_ranking": self.include_ranking,
            "rank_top_k": self.rank_top_k
        })

        logger.info(f"AgentScorerAgent initialized with scorers: {self.enabled_scorers}, "
                    f"include_mars: {self.include_mars}, include_ranking: {self.include_ranking}, "
                    f"rank_top_k: {self.rank_top_k}")

    async def run(self, context: Dict) -> Dict:
        """
        Main entrypoint: scores the last agent's output if scorable.
        """
        try:
            agent_obj = context.get("agent_obj")
            goal = context.get("goal", {})

            # 1. Only proceed if agent exists and is marked scorable
            if not agent_obj or not getattr(agent_obj, "is_scorable", False):
                self.logger.log("AgentScorerSkipped", {
                    "reason": "No agent_obj or agent not scorable",
                    "goal": goal.get("goal_text", "")[:80]
                })
                return context

            # 2. Convert agent output to Scorable
            scorable = self._make_scorable(agent_obj, goal)
            if not scorable:
                self.logger.log("AgentScorerNoScorable", {
                    "agent": getattr(agent_obj, "name", "unknown"),
                    "goal": goal.get("goal_text", "")[:80]
                })
                return context

            self.logger.log("AgentScorableCreated", scorable.to_dict())

            # 3. Run scorers (SICQL, MRQ, EBT, etc.)
            scores = self._run_multi_scorers(scorable, goal)
            self.logger.log("AgentScoresComputed", {
                "scorable_id": scorable.id,
                "scores": scores
            })

            # 4. MARS analysis (model agreement & reasoning signal)
            mars_result = {}
            if self.include_mars:
                mars_result = self.mars_calc.calculate([scores], reference="llm")
                self.logger.log("AgentMARSAnalysis", {
                    "scorable_id": scorable.id,
                    "agreement": mars_result.get("agreement_score"),
                    "dimensions": list(mars_result.keys())
                })

            # 5. Rank against similar cases for context
            ranking = []
            if self.include_ranking:
                ranking = self.ranker.rank_similar(
                    scorable=scorable,
                    goal=goal,
                    top_k=self.rank_top_k
                )
                self.logger.log("AgentRankingComplete", {
                    "scorable_id": scorable.id,
                    "candidates": len(ranking),
                    "top": ranking[:2]
                })

            # 6. Suggest alternatives from ranked results
            alternatives = self._suggest_alternatives(ranking, agent_obj)
            if alternatives:
                self.logger.log("AgentAlternativesSuggested", {
                    "scorable_id": scorable.id,
                    "count": len(alternatives),
                    "examples": alternatives[:2]
                })

            # 7. Store structured results back into context
            context["agent_scoring"] = {
                "scorable_id": scorable.id,
                "scores": scores,
                "mars": mars_result,
                "ranking": ranking,
                "alternatives": alternatives
            }

            # 8. Persist into memory for future reuse and CBR
            try:
                self.memory.evaluations.store_scores(
                    scorable_id=scorable.id,
                    scorable_type=scorable.target_type,
                    scores=scores,
                    mars=mars_result,
                    context=context
                )
                self.logger.log("AgentScoresPersisted", {"scorable_id": scorable.id})
            except Exception as e:
                self.logger.log("AgentScorePersistenceError", {"error": str(e)})

            return context

        except Exception as e:
            self.logger.log("AgentScorerError", {
                "error": str(e),
                "trace": traceback.format_exc()
            })
            return context

    def _make_scorable(self, agent_obj, goal: Dict) -> Scorable:
        """
        Wrap agent output into a Scorable object.
        """
        try:
            details = getattr(agent_obj, "scorable_details", {})
            return Scorable(
                id=f"{goal.get('id')}_{agent_obj.name}",
                text=details.get("output_text", ""),
                target_type="agent_output",
                metadata={
                    "goal_id": goal.get("id"),
                    "goal_text": goal.get("goal_text"),
                    "agent_name": agent_obj.name,
                }
            )
        except Exception as e:
            self.logger.log("MakeScorableError", {"error": str(e)})
            return None

    def _run_multi_scorers(self, scorable: Scorable, goal: Dict) -> Dict[str, float]:
        """
        Run all configured scorers on the given scorable.
        """
        results = {}
        for scorer_name in self.enabled_scorers:
            try:
                scorer = self.memory.get_scorer(scorer_name)
                results[scorer_name] = scorer.score(scorable, goal)
                self.logger.log("AgentScorerRan", {
                    "scorable_id": scorable.id,
                    "scorer": scorer_name,
                    "score": results[scorer_name]
                })
            except Exception as e:
                self.logger.log("AgentScorerError", {
                    "scorable_id": scorable.id,
                    "scorer": scorer_name,
                    "error": str(e)
                })
        return results

    def _suggest_alternatives(self, ranking: List[Dict], agent_obj) -> List[Dict]:
        """
        Suggest alternative agents based on ranked similar cases.
        """
        alternatives = []
        for r in ranking:
            alt_agent = r.get("agent_name")
            if alt_agent and alt_agent != agent_obj.name:
                alternatives.append(r)
        return alternatives
