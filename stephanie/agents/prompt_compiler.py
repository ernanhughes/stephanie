import dspy

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.prompt_evolver_mixin import PromptEvolverMixin
from stephanie.compiler.llm_compiler import LLMCompiler
from stephanie.compiler.passes.strategy_mutation_pass import StrategyMutationPass
from stephanie.constants import GOAL
from stephanie.evaluator.evaluator_loader import get_evaluator


class PromptCompilerAgent(BaseAgent, PromptEvolverMixin):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.prompt_key = cfg.get("prompt_key", "default")
        self.sample_size = cfg.get("sample_size", 20)
        self.generate_count = cfg.get("generate_count", 10)
        self.version = cfg.get("version", 1)
        self.use_strategy_mutation = cfg.get("use_strategy_mutation", False)

        # Initialize the LLM compiler through DSPy
        llm = dspy.LM("ollama_chat/qwen3", api_base="http://localhost:11434")
        self.init_evolver(llm, logger=logger)
        self.compiler = LLMCompiler(llm=self.llm, logger=self.logger)
        self.evaluator = get_evaluator(cfg, memory, self.call_llm, logger)
        if self.use_strategy_mutation:
            self.strategy_pass = StrategyMutationPass(self.evaluator,
                compiler=self.compiler, logger=self.logger
            )


    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        goal_text = self.extract_goal_text(goal)
        total_count = self.sample_size + self.generate_count

        examples = self.memory.prompt.get_prompt_training_set(goal_text, total_count)
        if not examples:
            self.logger.log("PromptCompilerSkipped", {"reason": "no_examples", "goal": goal})
            return context

        refined_prompts = self.evolve_prompts(
            examples, context=context, sample_size=self.sample_size
        )

        scored = []
        hypotheses = []
        for prompt in refined_prompts:
            # Generate hypothesis from the compiled prompt
            hypothesis_text = self.call_llm(prompt, context).strip()
            prompt_id  = self.get_or_save_prompt(prompt).id
            self.logger.log("CompiledPromptHypothesisGenerated", {
                "prompt": prompt[:100],
                "hypothesis": hypothesis_text[:100]
            })
            hyp = self.save_hypothesis(
                {
                    "prompt_id": prompt_id,
                    "text": hypothesis_text,
                    "features": {"source": "compiled_prompt"},
                },
                context=context
            )
            hypotheses.append(hyp.to_dict())
            score = self.score_prompt(
                prompt=prompt,
                reference_output=examples[0].get("hypothesis_text", ""),
                context=context
            )
            scored.append((prompt, score))
            self.add_to_prompt_history(context, prompt, {"source": "dspy_compiler"})

        self.logger.log(
            "PromptCompilerCompleted",
            {"goal": goal, "generated_count": len(refined_prompts)},
        )


        scored_sorted = sorted(scored, key=lambda x: get_winner_score(x[1]), reverse=True)

        context["refined_prompts"] = scored_sorted
        context["hypotheses"] = hypotheses

        # Log top results
        if self.logger:
            for i, (text, score) in enumerate(scored_sorted[:5]):
                self.logger.log(
                    "CompiledPromptScore",
                    {"rank": i + 1, "score": score, "prompt": text[:200]},
                )
        sorted_prompt = scored_sorted[0][0] if scored_sorted else None
        best_score_dict = scored_sorted[0][1] if scored_sorted else {}

        # Assume you already have: sorted_prompt, best_score_dict
        context["compiled_prompt"] = {
            "text": sorted_prompt,  # best-performing prompt
            "score_summary": best_score_dict,
            "compiler_agent": self.name,
        }
        self.logger.log("CompiledPromptSelected", context["compiled_prompt"])

        return context

    def score_prompt(self, prompt: str, reference_output, context:dict) -> float:
        if not self.evaluator:
            return 0.0
        try:
            score = self.evaluator.score_single(prompt, reference_output, context)
            if self.logger:
                self.logger.log("ScoringPrompt", {"score": score, "prompt": prompt[:100], "reference_output": reference_output[:100]})
            return score 
        except Exception as e:
            if self.logger:
                self.logger.log("PromptScoreError", {"prompt": prompt[:100], "error": str(e)})
            return 0.0

def get_winner_score(score_dict):
    if isinstance(score_dict, float):
        return score_dict
    if score_dict["winner"] == "A":
        return score_dict.get("score_a", 0)
    elif score_dict["winner"] == "B":
        return score_dict.get("score_b", 0)
    return 0  # fallback