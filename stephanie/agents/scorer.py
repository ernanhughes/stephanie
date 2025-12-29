# stephanie/agents/scorer.py
from __future__ import annotations

import logging
from typing import List, Tuple

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.data.score_corpus import ScoreCorpus
from stephanie.orm.cartridge_triple import CartridgeTripleORM
from stephanie.orm.casebook import CaseScorableORM
from stephanie.orm.chat import ChatTurnORM
from stephanie.orm.document import DocumentORM
from stephanie.orm.hypothesis import HypothesisORM
from stephanie.orm.prompt import PromptORM
from stephanie.orm.theorem import CartridgeORM, TheoremORM
from stephanie.scoring.calculations.mars_calculator import MARSCalculator
from stephanie.scoring.scorable import ScorableFactory, ScorableType
from stephanie.utils.db_scope import session_scope

log = logging.getLogger(__name__)


class ScorerAgent(BaseAgent):
    """
    Service-driven scorer agent.

    - Pulls candidates (documents/turns/etc.) from context or DB.
    - Scores with an ensemble of scorers via ScoringService.
    - Optionally persists via ScoringService.save_bundle / score_and_persist.
    - Never imports scorer classes directly (all via service).

    Config keys:
      enabled_scorers: [ "tiny", "hrm", ... ]          # registered in ScoringService
      dimensions:     [ "reasoning", "knowledge", ... ]
      target_types:   [ "document", "conversation_turn", ... ]
      force_rescore:  bool
      save_results:   bool
      progress:       bool
    """

    ORM_MAP = {
        ScorableType.DOCUMENT: DocumentORM,
        ScorableType.PROMPT: PromptORM,
        ScorableType.HYPOTHESIS: HypothesisORM,
        ScorableType.CARTRIDGE: CartridgeORM,
        ScorableType.THEOREM: TheoremORM,
        ScorableType.TRIPLE: CartridgeTripleORM,
        ScorableType.CASE_SCORABLE: CaseScorableORM,
        ScorableType.CONVERSATION_TURN: ChatTurnORM,
    }

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # knobs
        self.enabled_scorers: List[str] = cfg.get("enabled_scorers", ["tiny", "hf_tiny"])
        self.dimensions: List[str] = cfg.get(
            "dimensions",
            ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"],
        )
        self.target_types: List[str] = cfg.get(
            "target_types",
            [
                ScorableType.DOCUMENT,
                ScorableType.CONVERSATION_TURN,
            ],
        )
        dimension_config = self.cfg.get("dimension_config", {})
        self.mars_calculator = MARSCalculator(dimension_config, self.memory, self.logger)

        self.force_rescore: bool = bool(cfg.get("force_rescore", False))
        self.save_results: bool = bool(cfg.get("save_results", True))
        self.progress: bool = bool(cfg.get("progress", True))

        # service handle
        self.scoring_service = self.container.get("scoring")

        log.debug(
            "ScorerAgentInitialized: enabled=%s dims=%s targets=%s force_rescore=%s save_results=%s",
            self.enabled_scorers, self.dimensions, self.target_types,
            self.force_rescore, self.save_results
        )

    async def run(self, context: dict) -> dict:
        candidates: List[Tuple[dict, str]] = self._collect_candidates(context)
        total = len(candidates)
        if total == 0:
            self.logger.log("ScorerAgentNoCandidates", {"target_types": self.target_types})
            return context

        self.logger.log("ScorerAgentStart", {
            "total": total,
            "scorers": self.enabled_scorers,
            "dimensions": self.dimensions,
        })

        pbar = tqdm(candidates, desc="Scoring (service)", disable=not self.progress)
        results_out = []
        all_bundles = {}  # scorable_id -> ScoreBundle
        
        for idx, (obj, ttype) in enumerate(pbar):
            try:
                scorable = ScorableFactory.from_dict(obj, ttype)
                if not scorable or not scorable.text:
                    self.logger.log("ScorerAgentSkipEmpty", {"type": ttype, "obj_id": obj.get("id")})
                    continue

                # skip if scored & not forcing
                if not self.force_rescore:
                    has_any = self.memory.scores.has_any_score_for_target(
                        target_id=scorable.id,
                        target_type=ttype,
                        dimensions=self.dimensions,
                    )
                    if has_any:
                        self.logger.log("ScorerAgentAlreadyScored", {"type": ttype, "id": scorable.id})
                        continue

                # score through the service (and optionally persist)
                per_scorer = {}
                for scorer_name in self.enabled_scorers:
                    try:
                        if self.save_results:
                            bundle = self.scoring_service.score_and_persist(
                                scorer_name=scorer_name,
                                scorable=scorable,
                                context=context,
                                dimensions=self.dimensions,
                                source=scorer_name,
                                evaluator=self.name,
                                model_name=scorer_name,
                            )
                        else:
                            bundle = self.scoring_service.score(
                                scorer_name=scorer_name,
                                scorable=scorable,
                                context=context,
                                dimensions=self.dimensions,
                            )

                        # flatten for report
                        per_scorer[scorer_name] = {
                            d: {
                                "score": float(sr.score),
                                "rationale": sr.rationale,
                                "source": sr.source,
                                "attributes": getattr(sr, "attributes", {}) or {},
                            }
                            for d, sr in bundle.results.items()
                        }

                    except Exception as e:
                        self.logger.log("ScorerAgentServiceError", {
                            "scorer": scorer_name, "id": scorable.id, "type": ttype, "error": str(e)
                        })

                results_out.append({
                    "id": scorable.id,
                    "type": ttype,
                    "text": scorable.text,
                    "scores": per_scorer,
                })
                pbar.set_postfix({"done": f"{idx+1}/{total}"})

            except Exception as e:
                self.logger.log("ScorerAgentItemError", {
                    "type": ttype, "obj_id": obj.get("id"), "error": str(e)
                })

        self.logger.log("ScorerAgentDone", {"count": len(results_out)})
# Create ScoreCorpus for MARS analysis
        corpus = ScoreCorpus(bundles=all_bundles)

        self.logger.log("ScoreCorpusSummary", {
            "dims": corpus.dimensions,
            "scorers": corpus.scorers,
            "shape_example": corpus.get_dimension_matrix(self.dimensions[0]).shape
        })

        # Save corpus to context for potential future analysis
        context["score_corpus"] = corpus.to_dict()

        # Run MARS analysis if requested
        mars_results = {}
        if self.include_mars and all_bundles:
            mars_results = self.mars_calculator.calculate(corpus, context=context)
            context["mars_analysis"] = {
                "summary": mars_results,
                "recommendations": self.mars_calculator.generate_recommendations(
                    mars_results
                ),
            }
            self.logger.log(
                "MARSAnalysisCompleted",
                {
                    "document_count": len(all_bundles),
                    "dimensions": self.dimensions,
                },
            )

        context[self.output_key] = results_out
        return context

    # ---------------- helpers ----------------

    def _collect_candidates(self, context: dict) -> List[Tuple[dict, str]]:
        items: List[Tuple[dict, str]] = []

        for ttype in self.target_types:
            # 1) prefer objects in context
            ctx_key = ttype.lower() + "s"
            from_ctx = context.get(ctx_key, [])
            if from_ctx:
                for o in from_ctx:
                    items.append((o, ttype))
                continue

            # 2) fallback to DB fetch via ORM map
            orm_cls = self.ORM_MAP.get(ttype)
            if orm_cls is None:
                continue
            try:
                sessionmaker = self.memory.session
                with session_scope(sessionmaker) as session:
                    rows = session.query(orm_cls).all()
                for r in rows:
                    items.append((r.to_dict(), ttype))
            except Exception as e:
                self.logger.log("ScorerAgentDBError", {"type": ttype, "error": str(e)})

        return items
