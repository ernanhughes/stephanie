# stephanie/scoring/scoring_engine.py

from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.scoring_manager import ScoringManager


class ScoringEngine:
    def __init__(self, cfg, memory, prompt_loader, logger, call_llm):
        self.cfg = cfg
        self.memory = memory
        self.prompt_loader = prompt_loader
        self.logger = logger
        self.call_llm = call_llm
        self.scoring_managers = {}

    def get_manager(self, scoring_profile: str) -> ScoringManager:
        if scoring_profile not in self.scoring_managers:
            config_path = self.cfg.get(
                f"{scoring_profile}_score_config",
                f"config/scoring/{scoring_profile}.yaml",
            )
            self.scoring_managers[scoring_profile] = ScoringManager.from_file(
                filepath=config_path,
                prompt_loader=self.prompt_loader,
                cfg=self.cfg,
                logger=self.logger,
                memory=self.memory,
                llm_fn=self.call_llm,
            )
        return self.scoring_managers[scoring_profile]

    def score(
        self,
        target_id,
        target_type: TargetType,
        text: str,
        context: dict,
        scoring_profile: str,
    ) -> ScoreBundle:
        try:
            scorable = Scorable(id=target_id, text=text, target_type=target_type)
            scoring_manager = self.get_manager(scoring_profile)

            merged_context = {
                "target_type": target_type.value,
                "target": scorable.to_dict(),
                **context,
            }

            scorer = scoring_manager.scorer
            if not scorer:
                score_result = scoring_manager.evaluate(
                    scorable=scorable, context=merged_context, llm_fn=self.call_llm
                )
            else:
                score_result = scorer.score(
                    context, scorable, scoring_manager.dimensions
                )

            self.logger.log("ItemScored", score_result.to_dict())
            return score_result

        except Exception as e:
            self.logger.log(
                "ScoringFailed",
                {"target_id": target_id, "target_type": target_type, "error": str(e)},
            )
            return {}

    def score_item(
        self, scorable: Scorable, context: dict, scoring_profile: str
    ) -> ScoreBundle:
        try:
            scoring_manager = self.get_manager(scoring_profile)

            merged_context = {
                "target_type": scorable.target_type.value,
                "target": scorable.to_dict(),
                **context,
            }

            scorer = scoring_manager.scorer
            if not scorer:
                score_result = scoring_manager.evaluate(
                    scorable=scorable, context=merged_context, llm_fn=self.call_llm
                )
            else:
                score_result = scorer.score(
                    context, scorable, scoring_manager.dimensions
                )

            self.logger.log("ItemScored", score_result.to_dict())
            return score_result

        except Exception as e:
            self.logger.log(
                "ScoreItemFailed",
                {
                    "scrable": scorable,
                    "error": str(e),
                },
            )
            return {}
