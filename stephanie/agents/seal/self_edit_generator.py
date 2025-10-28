# stephanie/agents/seal/self_edit_generator.py
from __future__ import annotations


from dataclasses import dataclass

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.scoring.scorer.mrq_scorer import MRQScorer


@dataclass
class SelfEditGeneratorConfig:
    prompt_file: str = "implication.txt"
    num_edits: int = 5
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 512


class SelfEditGeneratorAgent(BaseAgent):
    def __init__(
        self,
        cfg,
        memory,
        container,
        logger=None,
        config: SelfEditGeneratorConfig = SelfEditGeneratorConfig(),
    ):
        super().__init__(cfg, memory, container, logger)
        self.config = config
        self.prompt_files = self.cfg.get("prompt_files", [])

    async def run(self, context: dict) -> dict:
        all_edits = []

        mrq_scorer = MRQScorer(self.cfg, memory=self.memory, container=self.container, logger=self.logger)
        mrq_scorer.train_from_database(cfg=self.cfg)

        for prompt_file in self.prompt_files:
            prompt_text = self.prompt_loader.from_file(
                prompt_file, config=self.cfg, context=context
            )
            self.logger.log("PromptFileLoaded", {"file": prompt_file})
            response = self.call_llm(prompt_text, context)
            strategy = prompt_file.replace(".txt", "")
            self.memory.prompt_programs.insert(
                {
                    "pipeline_run_id": context.get("pipeline_run_id"),
                    "goal": context.get("goal", {}).get("goal_text", ""),
                    "strategy": strategy,
                    "prompt_text": prompt_text,
                    "hypothesis": response,
                    "template": prompt_file,
                    "mutation_type": strategy,
                }
            )
            print(f"Generated response: {response}...")
            hypothesis = self.save_hypothesis(
                {
                    "text": response,
                    "features": {"prompt_file": prompt_file, "strategy": strategy},
                },
                context=context,
            )
            scorable = hypothesis.to_dict()
            context.setdefault("hypotheses", []).append(scorable)
            context["prompt"] = prompt_text  # we use thie to score in the evaluator

            scorable = Scorable(
                text=scorable.text,
                id=hypothesis.id,
                target_type=ScorableType.HYPOTHESIS,
            )
            score = self.score_item(
                scorable, context, metrics="compiler", scorer=self.scorer
            )
            all_edits.append(
                {"edit": response, "strategy": strategy, "score": score.to_dict()}
            )
            self.logger.log(
                "EditGenerated",
                {"edit": response[:100], "strategy": strategy, "score": score},
            )

        context["self_edits"] = all_edits
        return context
