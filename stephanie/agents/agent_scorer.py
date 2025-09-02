# stephanie/agents/agent_scorer_agent.py
"""
AgentScorerAgent

Meta-agent that scores the outputs of other agents.

- Activates only if `scorable_details` exist in context
- Relates output back to the goal
- Runs multi-scorer evaluation (SICQL, MRQ, EBT, etc.)
- Performs MARS analysis for consensus
- Ranks performance vs similar past cases
- Suggests alternative agent options for improvement
"""

import traceback
from typing import Dict, List
from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.data.score_bundle import ScoreBundle
from stephanie.scoring.scorable import Scorable
from stephanie.data.score_corpus import ScoreCorpus
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.scoring_manager import ScoringManager
from stephanie.scoring.scorer.scorable_ranker import ScorableRanker
from stephanie.scoring.calculations.mars_calculator import MARSCalculator
from stephanie.constants import SCORABLE_DETAILS
from stephanie.scoring.scorable_factory import TargetType
import time
from uuid import uuid4

import logging

logger = logging.getLogger(__name__)


class AgentScorerAgent(BaseAgent):
    def __init__(self, cfg, memory, log):
        super().__init__(
            cfg.get("agents", {}).get("agent_scorer", {}), memory, log
        )
        self.dimensions = self.cfg.get(
            "dimensions",
            [
                "novelty",
                "clarity",
                "relevance",
                "implementability",
                "alignment",
            ],
        )
        self.include_mars = self.cfg.get("include_mars", True)
        self.enabled_scorers = self.cfg.get(
            "enabled_scorers", ["sicql", "mrq", "ebt"]
        )

        self.include_ranking = self.cfg.get("include_ranking", True)
        self.rank_top_k = self.cfg.get("rank_top_k", 5)

        dimension_config = self.cfg.get("dimension_config", {})

        # Components
        self.mars_calculator = MARSCalculator(
            dimension_config, self.memory, self.logger
        )
        self.ranker = ScorableRanker(cfg, memory, log)

        logger.debug(
            f"AgentScorerInitialized: "
            f"enabled_scorers={self.enabled_scorers}, "
            f"include_mars={self.include_mars}, "
            f"include_ranking={self.include_ranking}, "
            f"rank_top_k={self.rank_top_k}"
        )

    async def run(self, context: Dict) -> Dict:
        start_time = time.time()

        try:
            goal = context.get("goal", {})
            scorable_details = context.get(SCORABLE_DETAILS)
            output_text = scorable_details.get("output_text", "")
            self.report(
                {
                    "event": "start",
                    "step": "AgentScoring",
                    "details": f"{output_text}",
                }
            )

            if not output_text:
                self.logger.log(
                    "AgentScorerSkipped",
                    {"reason": "No output text available"},
                )
                return context

            scorable_id = str(uuid4())
            scorable = Scorable(
                id=scorable_id,
                text=scorable_details.get("output_text"),
                target_type=TargetType.AGENT_OUTPUT,
                metadata={
                    "agent_name": scorable_details.get("agent_name"),
                    "stage_name": scorable_details.get("stage_name"),
                    "pipeline_run_id": context.get("pipeline_run_id"),
                    "goal_id": goal.get("id"),
                }
            )

            # === 1. Run configured scorers === You need to do a better one you need to do this one
            scores, bundle = self._score(context=context, scorable=scorable)
            corpus = ScoreCorpus(bundles={scorable_id: bundle})
            # === 2. MARS Analysis ===
            mars_result = {}
            if self.include_mars:
                mars_result = self.mars_calculator.calculate(
                    corpus, context=context
                )

            # === 3. Ranking vs similar cases ===
            ranking = []
            if self.include_ranking:
                candidates = self.memory.embedding.search_related_scorables(
                    scorable.text, TargetType.AGENT_OUTPUT
                )
                ranking = self.ranker.rank(
                    query=scorable, candidates=candidates, context=context
                )

            # === 4. Suggest Alternatives ===
            alternatives = self._suggest_alternatives(
                ranking, scorable_details
            )

            # Store everything back
            context["agent_scoring"] = {
                "scorable_id": scorable.id,
                "scores": scores,
                "mars": mars_result,
                "ranking": ranking,
                "alternatives": alternatives,
            }

            # Persist via ScoringManager
            ScoringManager.save_score_to_memory(
                bundle=bundle,
                scorable=scorable,
                context=context,
                cfg=self.cfg,
                memory=self.memory,
                logger=self.logger,
                source="agent_scorer",
                model_name="ensemble",
                evaluator_name=str(self.enabled_scorers),
            )

            self.logger.log(
                "AgentScored",
                {
                    "agent": scorable_details.get("agent_name"),
                    "goal": goal.get("goal_text", "")[:80],
                    "scores": scores,
                    "mars": mars_result,
                    "top_alt": alternatives[:2],
                },
            )

            logger.info(f"Time taken for scoring: {time.time() - start_time:.2f} seconds")
            return context

        except Exception as e:
            self.logger.error(
                "AgentScorerError",
                {"error": str(e), "trace": traceback.format_exc()},
            )
            return context

    def _suggest_alternatives(
        self, ranking: List[Dict], scorable: Scorable
    ) -> List[Dict]:
        """Suggest alternative agents from ranking results"""
        alternatives = []
        for r in ranking:
            alt_agent = r.get("agent_name")
            if alt_agent and alt_agent != scorable.meta.get("agent_name"):
                alternatives.append(r)
        return alternatives

    def _score(self, context: dict, scorable: Scorable) -> tuple:
        """Score one paper with all scorers"""
        goal = context.get("goal", {"goal_text": ""})
        score_results = {}

        for scorer_name in self.enabled_scorers:
            try:
                bundle = self.scoring.score(
                    scorer_name,
                    context=context,
                    scorable=scorable,
                    dimensions=self.dimensions,
                )
                for dim, result in bundle.results.items():
                    # ensure the result carries its dimension and source
                    if not getattr(result, "dimension", None):
                        result.dimension = dim
                    if not getattr(result, "source", None):
                        result.source = (
                            scorer_name  # fallback if scorer didn't set it
                        )

                    # use a composite key to avoid overwriting, but keep result.dimension == dim
                    key = f"{dim}::{result.source}"
                    score_results[key] = result
            except Exception as e:
                self.logger.log(
                    "ScorerError", {"scorer": scorer_name, "error": str(e)}
                )
                continue

        bundle = ScoreBundle(results=dict(score_results))

        # Save to memory
        ScoringManager.save_score_to_memory(
            bundle,
            scorable,
            context,
            self.cfg,
            self.memory,
            self.logger,
            source="paper_score",
            model_name="ensemble",
            evaluator_name=str(self.enabled_scorers),
        )

        report_scores = {
            dim: {
                "score": result.score,
                "rationale": result.rationale,
                "source": result.source,
            }
            for dim, result in score_results.items()
        }

        return {
            "scores": report_scores,
            "goal_text": goal.get("goal_text", ""),
        }, bundle
