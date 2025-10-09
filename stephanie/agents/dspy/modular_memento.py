# stephanie/agents/cbr/modular_memento.py
from __future__ import annotations

from stephanie.agents.dspy.mcts_reasoning import MCTSReasoningAgent
from stephanie.components.cbr.ab_validator import DefaultABValidator
from stephanie.components.cbr.case_selector import DefaultCaseSelector
from stephanie.components.cbr.casebook_scope_manager import \
    DefaultCasebookScopeManager
from stephanie.components.cbr.champion_promoter import DefaultChampionPromoter
from stephanie.components.cbr.context_namespacer import \
    DefaultContextNamespacer
from stephanie.components.cbr.goal_state_tracker import DefaultGoalStateTracker
from stephanie.components.cbr.micro_learner import DefaultMicroLearner
from stephanie.components.cbr.middleware import CBRMiddleware
from stephanie.components.cbr.quality_assessor import DefaultQualityAssessor
from stephanie.components.cbr.rank_and_analyze import DefaultRankAndAnalyze
from stephanie.components.cbr.retention_policy import DefaultRetentionPolicy
from stephanie.constants import AGENT_NAME, INCLUDE_MARS
from stephanie.scoring.calculations.mars_calculator import MARSCalculator
from stephanie.scoring.scorer.scorable_ranker import ScorableRanker


class ModularMementoAgent(MCTSReasoningAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        ns = DefaultContextNamespacer()
        scope = DefaultCasebookScopeManager(cfg, memory, container, logger)
        selector = DefaultCaseSelector(cfg, memory, container, logger)
        ranker = DefaultRankAndAnalyze(cfg, memory, container, logger, ranker=ScorableRanker(cfg, memory, container, logger),
                                       mars=MARSCalculator(cfg, memory, container, logger) if cfg.get(INCLUDE_MARS, True) else None)
        retention = DefaultRetentionPolicy(cfg, memory, container, logger, casebook_scope_mgr=scope)
        assessor = DefaultQualityAssessor(cfg, memory, container, logger)
        promoter = DefaultChampionPromoter(cfg, memory, container, logger)
        tracker = DefaultGoalStateTracker(cfg, memory, container, logger)
        ab = DefaultABValidator(cfg, memory, container, logger, ns=ns, assessor=assessor)
        micro = DefaultMicroLearner(cfg, memory, container, logger)

        self._cbr = CBRMiddleware(cfg, memory, container, logger, ns, scope, selector, ranker, retention, assessor, promoter, tracker, ab, micro)

    async def run(self, context: dict) -> dict:
        self._cbr.container = self.container
        self._cbr.ranker.scoring = self.container.get("scoring")
        # This delegates “CBR extras” to the middleware, using this agent’s base run as the core.
        parent_run = super(ModularMementoAgent, self).run  # <-- bound coroutine fn
        context[AGENT_NAME] = self.name
        async def base_run(ctx):  # what CBR wraps: your monolithic base behavior
            return await parent_run(ctx)
        result_ctx = await self._cbr.run(context, base_run, self.output_key)
        return result_ctx
