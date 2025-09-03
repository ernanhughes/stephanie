from typing import Any, Dict

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.data.score_bundle import ScoreBundle
from stephanie.models.cartridge_triple import CartridgeTripleORM
from stephanie.models.document import DocumentORM
from stephanie.models.hypothesis import HypothesisORM
from stephanie.models.prompt import PromptORM
from stephanie.models.theorem import CartridgeORM, TheoremORM
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.scorer.contrastive_ranker_scorer import \
    ContrastiveRankerScorer
from stephanie.scoring.scorer.ebt_scorer import EBTScorer
from stephanie.scoring.scorer.hrm_scorer import HRMScorer
from stephanie.scoring.scorer.mrq_scorer import MRQScorer
from stephanie.scoring.scorer.sicql_scorer import SICQLScorer
from stephanie.scoring.scorer.svm_scorer import SVMScorer
from stephanie.scoring.scoring_manager import ScoringManager


class UniversalScorerAgent(BaseAgent):

    ORM_MAP = {
        "document": DocumentORM,
        "prompt": PromptORM,
        "hypothesis": HypothesisORM,
        "cartridge": CartridgeORM,
        "theorem": TheoremORM,
        "triple": CartridgeTripleORM,
    }

    """
    Scores any scorable object (documents, cartridges, theorems, triples, etc.)
    if not already scored. Uses ensemble of configured scorers.
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get("dimensions", ["alignment", "clarity", "relevance"])
        self.enabled_scorers = cfg.get("enabled_scorers", ["sicql"])
        self.progress = cfg.get("progress", True)
        self.force_rescore = cfg.get("force_rescore", False)
        self.target_types = cfg.get(
            "target_types",
            [
                # TargetType.DOCUMENT,
                # TargetType.CARTRIDGE,
                # TargetType.THEOREM,
                TargetType.TRIPLE,
            ],
        )
        self.scorers = self._initialize_scorers()

    def _initialize_scorers(self) -> Dict[str, Any]:
        """Initialize all configured scorers"""
        scorers = {}
        if "svm" in self.enabled_scorers:
            scorers["svm"] = SVMScorer(self.cfg.get("svm"), memory=self.memory, logger=self.logger)
        if "mrq" in self.enabled_scorers:
            scorers["mrq"] = MRQScorer(self.cfg.get("mrq"), memory=self.memory, logger=self.logger)
        if "sicql" in self.enabled_scorers:
            scorers["sicql"] = SICQLScorer(self.cfg.get("sicql"), memory=self.memory, logger=self.logger)
        if "ebt" in self.enabled_scorers:
            scorers["ebt"] = EBTScorer(self.cfg.get("ebt"), memory=self.memory, logger=self.logger)
        if "hrm" in self.enabled_scorers:
            scorers["hrm"] = HRMScorer(self.cfg.get("hrm"), memory=self.memory, logger=self.logger)
        if "contrastive_ranker" in self.enabled_scorers:
            scorers["contrastive_ranker"] = ContrastiveRankerScorer(
                self.cfg.get("contrastive_ranker"), memory=self.memory, logger=self.logger
            )
        return scorers

    async def run(self, context: dict) -> dict:
        scored_items = []
        candidates = []

        # --- Collect candidates ---
        for ttype in self.target_types:
            objs = context.get(ttype.lower() + "s", [])

            if not objs and ttype in self.ORM_MAP:
                orm_cls = self.ORM_MAP[ttype]
                objs = self.memory.session.query(orm_cls).all()
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

                result, bundle = self._score_item(context, scorable, ttype)
                scored_items.append(result)

                ScoringManager.save_score_to_memory(
                    bundle,
                    scorable,
                    context,
                    self.cfg,
                    self.memory,
                    self.logger,
                    source="universal_scorer",
                    model_name="ensemble",
                    evaluator_name=str(self.scorers.keys()
                )

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
        for _, scorer in self.scorers.items():
            bundle = scorer.score(context, scorable=scorable, dimensions=self.dimensions)
            for dim, result in bundle.results.items():
                if dim not in score_results:
                    score_results[dim] = result
        return {
            "id": scorable.id,
            "type": ttype,
            "scores": {dim: {"score": r.score, "source": r.source} for dim, r in score_results.items()},
        }, ScoreBundle(results=score_results)
