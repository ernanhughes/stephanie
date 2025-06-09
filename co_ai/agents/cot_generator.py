from co_ai.agents import BaseAgent
from co_ai.analysis.rubric_classifier import RubricClassifierMixin
from co_ai.constants import GOAL, PIPELINE, PIPELINE_RUN_ID
from co_ai.evaluator.llm_judge_evaluator import LLMJudgeEvaluator
from co_ai.evaluator.mrq_self_evaluator import MRQSelfEvaluator
from co_ai.models import HypothesisORM


class ChainOfThoughtGeneratorAgent(BaseAgent, RubricClassifierMixin): 
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.logger.log("AgentInit", {"agent": "ChainOfThoughtGeneratorAgent"})
        self.evaluator = self._init_evaluator()
        self.num_candidates = cfg.get("num_candidates", 2)

    async def run(self, context: dict):
        goal = context.get(GOAL)
        self.logger.log("AgentRunStarted", {"goal": goal})

        if isinstance(self.evaluator, MRQSelfEvaluator):
            self.logger.log("MRQTraining", {"type": "MRQ"})
            self.evaluator.train_from_database(goal=goal.goal_text, cfg=self.cfg)

        prompt = self.prompt_loader.load_prompt(self.cfg, context)
        self.logger.log("PromptGenerated", {"prompt": prompt[:200]})

        # Step 1: Generate candidates
        self.logger.log("GenerationStarted", {"num_candidates": self.num_candidates})
        candidates = [
            self.call_llm(prompt, context) for _ in range(self.num_candidates)
        ]
        self.logger.log(
            "GenerationCompleted", {"candidates": [c[:100] for c in candidates]}
        )

        # Step 2: Evaluate pairwise
        best = candidates[0]
        scores = {}
        for candidate in candidates[1:]:
            best, scores = self.evaluator.judge(
                prompt=prompt,
                output_a=best,
                output_b=candidate, 
                context=context
            )
        self.logger.log("EvaluationCompleted", {"best_output": best[:100], **scores})

        # Step 3: Store hypothesis and patterns
        value_a = scores.get("value_a", 0)
        value_b = scores.get("value_b", 0)
        score = max(value_a, value_b)
        features = {
            "prompt": prompt,
            "best_output": best,
            "candidates": candidates,
        }

        best_orm = HypothesisORM(
            goal_id=self.get_goal_id(goal),
            text=best,
            confidence=score,
            features=features,
            prompt_id=self.get_prompt_id(prompt),
            pipeline_signature=context.get(PIPELINE),
            pipeline_run_id=context.get(PIPELINE_RUN_ID),
        )
        self.memory.hypotheses.insert(best_orm)
        self.logger.log("HypothesisStored", {"text": best[:100], "confidence": score})

        self.classify_and_store_patterns(
            hypothesis=best_orm.to_dict(),
            context=context,
            prompt_loader=self.prompt_loader,
            cfg=self.cfg,
            memory=self.memory,
            logger=self.logger,
            agent_name=self.name,
            score=score,
        )

        context[self.output_key] = [best_orm.to_dict()]
        self.logger.log("AgentRunCompleted", {"output_key": self.output_key})
        return context

    def _init_evaluator(self):
        if self.cfg.get("evaluator", "mrq") == "llm":
            llm = self.cfg.get("evaluator_model", self.cfg.get("model"))
            prompt_file = self.cfg.get("evaluator_prompt_file", "evaluation.txt")
            self.logger.log(
                "EvaluatorInit", {"strategy": "LLM", "prompt_file": prompt_file}
            )
            return LLMJudgeEvaluator(
                self.cfg, llm, prompt_file, self.call_llm, self.logger
            )
        else:
            self.logger.log("EvaluatorInit", {"strategy": "MRQ"})
            return MRQSelfEvaluator(self.memory, self.logger)

    def _summarize_pattern(self, pattern: dict):
        stats = {}
        for dimension, label in pattern.items():
            if label not in stats:
                stats[label] = 0
            stats[label] += 1
        return stats
