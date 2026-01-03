# stephanie/agents/universal_scorer.py
from __future__ import annotations

import logging

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.data.score_bundle import ScoreBundle
from stephanie.orm.cartridge_triple import CartridgeTripleORM
from stephanie.orm.casebook import CaseScorableORM
from stephanie.orm.chat import ChatTurnORM
from stephanie.orm.document import DocumentORM
from stephanie.orm.hypothesis import HypothesisORM
from stephanie.orm.prompt import PromptORM
from stephanie.orm.theorem import CartridgeORM, TheoremORM
from stephanie.scoring.calculations.mars_calculator import MARSCalculator
from stephanie.scoring.metrics.scorable_processor import ScorableProcessor
from stephanie.scoring.scorable import ScorableFactory, ScorableType
from stephanie.scoring.score_display import ScoreDisplay
from stephanie.scoring.scorer.scorable_ranker import ScorableRanker
from stephanie.utils.db_scope import session_scope

log = logging.getLogger(__name__)

class UniversalScorerAgent(BaseAgent):

    ORM_MAP = {
        "document": DocumentORM,
        "prompt": PromptORM,
        "hypothesis": HypothesisORM,
        "cartridge": CartridgeORM,
        "theorem": TheoremORM,
        "triple": CartridgeTripleORM,
        "case_scorable": CaseScorableORM,
        "conversation_turn": ChatTurnORM,
    }

    """
    Scores any scorable object (documents, cartridges, theorems, triples, etc.)
    if not already scored. Uses ensemble of configured scorers.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.enabled_scorers = cfg.get("enabled_scorers", ["tiny", "hrm"])
        self.progress = cfg.get("progress", True)
        self.force_rescore = cfg.get("force_rescore", False)
        self.save_results = cfg.get("save_results", False)
        self.target_types = cfg.get(
            "target_types",
            [
                ScorableType.CONVERSATION_TURN
            ],
        )
        self.dimensions = self.cfg.get(
            "dimensions",
            [
                "coverage",
                "clarity",
                "faithfulness",
                "knowledge",
                "reasoning",
            ],
        )
        self.include_mars = self.cfg.get("include_mars", True)
        self.include_ranking = self.cfg.get("include_ranking", True)
        self.rank_top_k = self.cfg.get("rank_top_k", 5)
        self.max_candidates = self.cfg.get("max_candidates", 100)
        dimension_config = self.cfg.get("dimension_config", {})
        # Components
        self.mars_calculator = MARSCalculator(
            dimension_config, memory=self.memory, container=self.container, logger=self.logger
        )
        self.ranker = ScorableRanker(cfg, memory=self.memory, container=self.container, logger=self.logger)

        self.scorable_processor = ScorableProcessor(
            cfg=cfg.get("processor", {}),
            memory=self.memory,
            container=self.container,
            logger=self.logger,
        )

        log.debug(
            f"AgentScorerInitialized: "
            f"enabled_scorers={self.enabled_scorers}, "
            f"include_mars={self.include_mars}, "
            f"force_rescore={self.force_rescore}, "
            f"target_types={self.target_types}, "
            f"dimensions={self.dimensions}, "
            f"include_mars={self.include_mars}, "
            f"include_ranking={self.include_ranking}, "
            f"rank_top_k={self.rank_top_k}"
        )

    async def run(self, context: dict) -> dict:
        scorables = [] 

        for ttype in self.target_types:
            if ttype == ScorableType.CONVERSATION_TURN:
                objs = self.memory.chats.list_turns(limit=self.max_candidates)

                objs = [o.to_dict() for o in objs]
                for obj in objs:
                    obj.setdefault("text", obj.get("assistant_message", {}).get("conversation", ""))
            else:
                objs = context.get(ttype.lower() + "s", [])

                if not objs and ttype in self.ORM_MAP:
                    orm_cls = self.ORM_MAP[ttype]
                    sessionmaker = self.memory.session  # now a sessionmaker, not a live session
                    with session_scope(sessionmaker) as session:
                        objs = session.query(orm_cls).all()
                    objs = [o.to_dict() for o in objs]

            for obj in objs:
                scorables.append(ScorableFactory.from_dict(obj, ttype))

        scorables = scorables[:self.max_candidates]
        total_candidates = len(scorables)

        self.report({"event": "scoring_start", "total": total_candidates})

        scored_items = await self.scorable_processor.process_many(scorables, context=context)

        self.report({"event": "scoring_completed", "total": total_candidates})


        self.report({"event": "scoring_completed", "total_scored": len(scored_items)})
        context[self.output_key] = scored_items
        return context

    def _score_item(self, context, scorable, ttype):
        score_results = {}

        for scorer_name in self.enabled_scorers:
            try:
                bundle = self.container.get("scoring").score(
                    scorer_name,
                    context=context,
                    scorable=scorable,
                    dimensions=self.dimensions
                )
                for dim, result in bundle.results.items():
                    result.dimension = dim
                    result.source = scorer_name
                    key = f"{dim}::{result.source}"
                    score_results[key] = result
            except Exception as e:
                self.logger.log("ScorerError", {"scorer": scorer_name, "scorable_id": scorable.id, "error": str(e)})
                continue

        bundle = ScoreBundle(results=dict(score_results))
        weighted_score = bundle.aggregate()
        ScoreDisplay.show(scorable, bundle.to_dict(), weighted_score)

        # Save to 
        if self.save_results:
            for key, result in score_results.items():
                eval_id = self.memory.evaluations.save_bundle(
                    bundle=bundle,
                    scorable=scorable,
                    context=context,
                    cfg=self.cfg,
                    agent_name=self.name,
                    source=result.source, 
                    model_name=result.source,
                    evaluator_name=self.name,
                )
                self.logger.log("EvaluationSaved", {"id": eval_id})

        report_scores = {
            key: {"score": result.score,
                "rationale": result.rationale,
                "source": result.source}
            for key, result in score_results.items()
        }

        return {
            "scorable_id": scorable.id,
            "text": scorable.text,
            "scores": report_scores,
        }, bundle
