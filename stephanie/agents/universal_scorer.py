# stephanie/agents/universal_scorer.py
from __future__ import annotations


import logging

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.data.score_bundle import ScoreBundle
from stephanie.models.cartridge_triple import CartridgeTripleORM
from stephanie.models.casebook import CaseScorableORM
from stephanie.models.chat import ChatTurnORM
from stephanie.models.document import DocumentORM
from stephanie.models.hypothesis import HypothesisORM
from stephanie.models.prompt import PromptORM
from stephanie.models.theorem import CartridgeORM, TheoremORM
from stephanie.scoring.calculations.mars_calculator import MARSCalculator
from stephanie.scoring.scorable import ScorableFactory, ScorableType
from stephanie.scoring.score_display import ScoreDisplay
from stephanie.scoring.scorer.scorable_ranker import ScorableRanker
from stephanie.utils.db_scope import session_scope

_logger = logging.getLogger(__name__)

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
        self.enabled_scorers = cfg.get("enabled_scorers", ["tiny"])
        self.progress = cfg.get("progress", True)
        self.force_rescore = cfg.get("force_rescore", False)
        self.save_results = cfg.get("save_results", False)
        self.target_types = cfg.get(
            "target_types",
            [
                ScorableType.DOCUMENT,
                ScorableType.CARTRIDGE,
                ScorableType.THEOREM,
                ScorableType.TRIPLE,
                ScorableType.CASE_SCORABLE,
                ScorableType.CONVERSATION_TURN
            ],
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
        self.include_ranking = self.cfg.get("include_ranking", True)
        self.rank_top_k = self.cfg.get("rank_top_k", 5)
        dimension_config = self.cfg.get("dimension_config", {})
        # Components
        self.mars_calculator = MARSCalculator(
            dimension_config, memory=self.memory, container=self.container, logger=self.logger
        )
        self.ranker = ScorableRanker(cfg, memory=self.memory, container=self.container, logger=self.logger)

        _logger.debug(
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
        scored_items = []
        candidates = []

        for ttype in self.target_types:
            if ttype == ScorableType.CONVERSATION_TURN:
                objs = self.memory.chats.list_turns()
                objs = [o.to_dict() for o in objs]
                for obj in objs:
                    candidates.append((obj, ttype))
            else:
                objs = context.get(ttype.lower() + "s", [])

                if not objs and ttype in self.ORM_MAP:
                    orm_cls = self.ORM_MAP[ttype]
                    sessionmaker = self.memory.session  # now a sessionmaker, not a live session
                    with session_scope(sessionmaker) as session:
                        objs = session.query(orm_cls).all()
                    objs = [o.to_dict() for o in objs]

                for obj in objs:
                    candidates.append((obj, ttype))

        total_candidates = len(candidates)

        if not candidates:
            self.report({"event": "no_scorables", "target_types": self.target_types})
            return context

        self.report({"event": "scoring_start", "total": total_candidates})

        pbar = tqdm(candidates, desc="Scoring Scorables", disable=not self.progress)

        for idx, (obj, ttype) in enumerate(pbar):
            try:
                scorable = ScorableFactory.from_dict(obj, ttype)

                existing = self.memory.scores.get_scores_for_target(
                    target_id=scorable.id,
                    target_type=ttype,
                    dimensions=self.dimensions,
                )
                if existing and not self.force_rescore:
                    self.report({
                        "event": "skipped",
                        "id": scorable.id,
                        "type": ttype,
                        "reason": "already_scored"
                    })
                    continue

                result, _ = self._score_item(context, scorable, ttype)
                scored_items.append(result)

                self.report({
                    "event": "scored",
                    "id": scorable.id,
                    "type": ttype,
                    "scores": result["scores"]
                })

                pbar.set_postfix({"done": f"{idx + 1}/{total_candidates}"})

            except Exception as e:
                self.report({
                    "event": "error",
                    "id": obj.get("id"),
                    "type": ttype,
                    "error": str(e)
                })

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
