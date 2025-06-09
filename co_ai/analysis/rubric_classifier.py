from datetime import datetime, timezone

from co_ai.constants import GOAL
from co_ai.models import PatternStatORM


class RubricClassifierMixin:
    def _load_enabled_rubrics(self, cfg):
        enabled_rubrics = []
        rubrics_cfg = cfg.get("rubrics", [])
        for entry in rubrics_cfg:
            if entry.get("enabled", False):
                enabled_rubrics.append(
                    {
                        "dimension": entry["dimension"],
                        "rubric": entry["rubric"],
                        "options": entry["options"],
                    }
                )
        return enabled_rubrics

    def classify_with_rubrics(self, hypothesis, context, prompt_loader, cfg, logger):
        results = {}
        pattern_file = cfg.get("pattern_prompt_file", "cot_pattern.txt")
        rubrics = self._load_enabled_rubrics(cfg)

        for rubric in rubrics:
            rubric["hypotheses"] = hypothesis.get("text")
            merged = {**context, **rubric}
            prompt_text = prompt_loader.from_file(pattern_file, cfg, merged)
            custom_llm = cfg.get("analysis_model", None)
            result = self.call_llm(prompt_text, merged, custom_llm)
            results[rubric["dimension"]] = result
            logger.log(
                "RubricClassified",
                {
                    "dimension": rubric["dimension"],
                    "rubric": rubric["rubric"],
                    "classification": result,
                },
            )

        return results

    def classify_and_store_patterns(
        self,
        hypothesis,
        context,
        prompt_loader,
        cfg,
        memory,
        logger,
        agent_name,
        score=None,  # Optional numeric score or win count
    ):
        """Classifies rubrics and stores pattern stats for the given hypothesis."""
        pattern = self.classify_with_rubrics(
            hypothesis=hypothesis,
            context=context,
            prompt_loader=prompt_loader,
            cfg=cfg,
            logger=logger,
        )

        goal = context.get(GOAL)
        summarized = self._summarize_pattern(pattern)

        pattern_stats = self.generate_pattern_stats(
            goal, hypothesis, summarized, cfg, agent_name, score
        )
        memory.pattern_stats.insert(pattern_stats)
        logger.log(
            "RubricPatternsStored",
            {"summary": summarized, "goal": goal, "hypothesis": hypothesis},
        )

        context["pattern_stats"] = summarized
        return summarized

    def generate_pattern_stats(
        self,
        goal,
        hypothesis,
        pattern_dict,
        cfg,
        agent_name,
        confidence_score=None,
    ):
        """
        Create PatternStatORM entries for a classified CoT using DB lookup for IDs.
        """
        try:
            # Get or create goal
            goal_id = self.get_goal_id(goal)

            # Get hypothesis by text
            hypothesis_id = self.get_hypothesis_id(hypothesis)
            model_name = cfg.get("model", {}).get("name", "unknown")

            stats = []
            for dimension, label in pattern_dict.items():
                stat = PatternStatORM(
                    goal_id=goal_id,
                    hypothesis_id=hypothesis_id,
                    model_name=model_name,
                    agent_name=agent_name,
                    dimension=dimension,
                    label=label,
                    confidence_score=confidence_score,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                stats.append(stat)

            return stats
        except Exception as e:
            print(f"❌ Failed to generate pattern stats: {e}")
            raise
