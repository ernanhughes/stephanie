from co_ai.agents import BaseAgent
from co_ai.evaluator.llm_judge_evaluator import LLMJudgeEvaluator
from co_ai.evaluator.mrq_self_evaluator import MRQSelfEvaluator
from co_ai.constants import GOAL
from co_ai.models import Hypothesis
from co_ai.analysis import RubricClusterer
from co_ai.models.pattern_stat import generate_pattern_stats


class ChainOfThoughtGeneratorAgent(BaseAgent):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.logger.log("AgentInit", {"agent": "ChainOfThoughtGeneratorAgent"})
        self.evaluator = self._init_evaluator()
        self.num_candidates = cfg.get("num_candidates", 2)
        self.rubrics = self._load_enabled_rubrics()
        self._cluster_rubrics()

    async def run(self, context: dict):
        goal = context.get(GOAL)
        self.logger.log("AgentRunStarted", {"goal": goal})

        if isinstance(self.evaluator, MRQSelfEvaluator):
            self.logger.log("MRQTraining", {"type": "MRQ"})
            self.evaluator.train_from_database(goal=goal, cfg=self.cfg)

        prompt = self.prompt_loader.load_prompt(self.cfg, context)
        self.logger.log("PromptGenerated", {"prompt": prompt[:200]})

        # Step 1: Generate candidates
        self.logger.log("GenerationStarted", {"num_candidates": self.num_candidates})
        candidates = [self.call_llm(prompt, context) for _ in range(self.num_candidates)]
        self.logger.log("GenerationCompleted", {"candidates": [c[:100] for c in candidates]})

        # Step 2: Evaluate pairwise
        best = candidates[0]
        scores = {}
        for candidate in candidates[1:]:
            best, scores = self.evaluator.evaluate(
                prompt=prompt,
                goal=goal,
                output_a=best,
                output_b=candidate,
            )
        self.logger.log("EvaluationCompleted", {"best_output": best[:100], **scores})

        # Step 3: Classify the winning hypothesis
        self.logger.log("ClassificationStarted", {"rubric_count": len(self.rubrics)})
        pattern = self.classify(prompt, best, context)
        self.logger.log("ClassificationCompleted", {"pattern": pattern})

        # Step 4: Store results
        value_a = scores.get("value_a", 0)
        value_b = scores.get("value_b", 0)
        score = max(value_a, value_b)
        features = {
            "prompt": prompt,
            "best_output": best,
            "candidates": candidates,
            "pattern": pattern,
        }
        hyp = Hypothesis(goal=goal, text=best, confidence=score, features=features, prompt=prompt)
        self.memory.hypotheses.store(hyp)
        self.logger.log("HypothesisStored", {"text": best[:100], "confidence": score})

        context[self.output_key] = [best]

        patterns = self._summarize_pattern(pattern)
        self.logger.log(
            "CoTPatternStats",
            {
                "pattern_summary": patterns,
                "goal": goal,
                "model": self.cfg.get("model", {}).get("name", "unknown"),
            },
        )
        context["pattern_stats"] = patterns
        goal_id, hypothesis_id, pattern_stats = generate_pattern_stats(goal, best, patterns, self.memory, self.cfg, self.name, score)
        self.memory.hypotheses.store_pattern_stats(goal_id, hypothesis_id, pattern_stats)

        self.logger.log("AgentRunCompleted", {"output_key": self.output_key})
        return context

    def classify(self, prompt, cot_response, context: dict):
        results = {}
        pattern_file = self.cfg.get("pattern_prompt_file", "cot_pattern.txt")

        for rubric in self.rubrics:
            rubric["goal"] = prompt
            rubric["hypotheses"] = cot_response
            merged = {**context, **rubric}
            prompt_text = self.prompt_loader.from_file(pattern_file, self.cfg, merged)
            custom_llm = self.cfg.get("analysis_model", None)
            result = self.call_llm(prompt_text, merged, custom_llm)
            results[rubric["dimension"]] = result
            self.logger.log("RubricClassified", {
                "dimension": rubric["dimension"],
                "rubric": rubric["rubric"],
                "classification": result
            })

        return results

    def _load_enabled_rubrics(self):
        enabled_rubrics = []
        rubrics_cfg = self.cfg.get("rubrics", [])
        for entry in rubrics_cfg:
            if entry.get("enabled", False):
                enabled_rubrics.append({
                    "dimension": entry["dimension"],
                    "rubric": entry["rubric"],
                    "options": entry["options"]
                })
        self.logger.log("RubricsLoaded", {"count": len(enabled_rubrics)})
        return enabled_rubrics

    def _init_evaluator(self):
        if self.cfg.get("evaluator", "mrq") == "llm":
            llm = self.cfg.get("evaluator_model", self.cfg.get("model"))
            prompt_file = self.cfg.get("evaluator_prompt_file", "evaluation.txt")
            self.logger.log("EvaluatorInit", {"strategy": "LLM", "prompt_file": prompt_file})
            return LLMJudgeEvaluator(self.cfg, llm, prompt_file, self.call_llm, self.logger)
        else:
            self.logger.log("EvaluatorInit", {"strategy": "MRQ"})
            return MRQSelfEvaluator(self.memory, self.logger)

    def _cluster_rubrics(self):
        if not self.rubrics:
            self.logger.log("RubricClusteringSkipped", {"reason": "No rubrics loaded"})
            return

        self.logger.log("RubricClusteringStarted", {"rubric_count": len(self.rubrics)})

        def embed_fn(text):
            return self.memory.embedding.get_or_create(text)

        clusterer = RubricClusterer(self.memory)
        embedded = clusterer.embed_rubrics(self.rubrics)
        clustered = clusterer.cluster_rubrics(
            embedded, num_clusters=self.cfg.get("num_rubric_clusters", 6)
        )
        summary = clusterer.summarize_clusters(clustered)

        self.logger.log("RubricClusteringCompleted", {"summary": summary})


    def _summarize_pattern(self, pattern: dict):
        """Counts rubric label frequencies (e.g., 'Top-Down': 1) across the classified dimensions."""
        stats = {}
        for dimension, label in pattern.items():
            if label not in stats:
                stats[label] = 0
            stats[label] += 1
        return stats
