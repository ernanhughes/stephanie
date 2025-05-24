from co_ai.constants import GOAL
from co_ai.models.pattern_stat import generate_pattern_stats


class RubricClassifierMixin:
    def _load_enabled_rubrics(self, cfg):
        enabled_rubrics = []
        rubrics_cfg = cfg.get("rubrics", [])
        for entry in rubrics_cfg:
            if entry.get("enabled", False):
                enabled_rubrics.append({
                    "dimension": entry["dimension"],
                    "rubric": entry["rubric"],
                    "options": entry["options"]
                })
        return enabled_rubrics

    def classify_with_rubrics(self, hypothesis, context, prompt_loader, cfg, logger):
        results = {}
        pattern_file = cfg.get("pattern_prompt_file", "cot_pattern.txt")
        rubrics = self._load_enabled_rubrics(cfg)

        for rubric in rubrics:
            rubric["goal"] = context["goal"]["goal_text"]
            rubric["hypotheses"] = hypothesis.text
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

        goal = self.extract_goal_text(context.get(GOAL))
        summarized = self._summarize_pattern(pattern)

        goal_id, hypothesis_id, pattern_stats = generate_pattern_stats(
            goal, hypothesis.text, summarized, memory, cfg, agent_name, score
        )
        memory.hypotheses.store_pattern_stats(goal_id, hypothesis_id, pattern_stats)
        logger.log(
            "RubricPatternsStored",
            {"goal_id": goal_id, "hypothesis_id": hypothesis_id, "summary": summarized},
        )

        context["pattern_stats"] = summarized
        return summarized