# stephanie/compiler/prompt_evolver.py
import dspy
from dspy import BootstrapFewShot, Example, Predict

from stephanie.compiler.llm_compiler import LLMCompiler
from stephanie.compiler.passes.strategy_mutation_pass import \
    StrategyMutationPass
from stephanie.compiler.prompt_tuning_signature import PromptTuningSignature
from stephanie.evaluator.evaluator_loader import get_evaluator


class PromptEvolver:
    def __init__(
        self,
        llm,
        logger=None,
        use_strategy_mutation=False,
        evaluator_cfg=None,
        memory=None,
    ):
        self.llm = llm
        self.logger = logger
        self.use_strategy_mutation = use_strategy_mutation
        dspy.configure(lm=self.llm)

        self.compiler = LLMCompiler(llm=self.llm, logger=self.logger)
        if self.use_strategy_mutation:
            self.strategy_pass = StrategyMutationPass(
                compiler=self.compiler, logger=self.logger
            )

        self.evaluator = None
        if evaluator_cfg:
            self.evaluator = get_evaluator(
                evaluator_cfg, memory=memory, llm=llm, logger=logger
            )

    def evolve(
        self, examples: list[dict], context: dict = {}, sample_size: int = 10
    ) -> list[str]:
        """
        Use DSPy to tune prompts based on performance signals.
        Optionally use symbolic strategy mutation.
        Returns a list of refined prompt strings.
        """
        if not examples:
            return []

        training_set = [
            Example(
                goal=ex["goal"],
                input_prompt=ex["prompt_text"],
                hypotheses=ex["hypothesis_text"],
                review=ex.get("review", ""),
                score=ex.get("elo_rating", 1000),
            ).with_inputs("goal", "input_prompt", "hypotheses", "review", "score")
            for ex in examples[:sample_size]
        ]

        def fallback_metric(example, pred, trace=None):
            return 1.0  # fallback metric for training

        tuner = BootstrapFewShot(metric=fallback_metric)
        student = Predict(PromptTuningSignature)
        tuned_program = tuner.compile(student=student, trainset=training_set)

        refined_prompts = []

        # Use DSPy tuned program
        for ex in examples[sample_size:]:
            try:
                result = tuned_program(
                    goal=ex["goal"],
                    input_prompt=ex["prompt_text"],
                    hypotheses=ex["hypothesis_text"],
                    review=ex.get("review", ""),
                    score=ex.get("elo_rating", 1000),
                )
                refined = result.refined_prompt.strip()
                refined_prompts.append(refined)
            except Exception as e:
                if self.logger:
                    self.logger.log("DSPyPromptEvolutionFailed", {"error": str(e)})

        # Optionally add symbolic strategy mutations
        if self.use_strategy_mutation:
            for ex in examples:
                base_prompt = ex["prompt_text"]
                metadata = {
                    "goal": ex["goal"],
                    "hypotheses": ex.get("hypothesis_text", ""),
                    "review": ex.get("review", ""),
                    "score": ex.get("elo_rating", 1000),
                }
                try:
                    mutations = self.strategy_pass.apply(base_prompt, metadata)
                    for mut in mutations:
                        prompt_text = mut["prompt"]
                        score = self.score_prompt(
                            prompt_text,
                            reference_output=metadata["hypotheses"],
                            context=context,
                        )
                        if score >= 0:  # optionally apply a score threshold
                            refined_prompts.append(prompt_text)
                except Exception as e:
                    if self.logger:
                        self.logger.log("StrategyMutationFailed", {"error": str(e)})

        return refined_prompts

    def score_prompt(
        self, prompt: str, reference_output: str = "", context: dict = {}
    ) -> float:
        return self.evaluator.score_single(prompt, reference_output, context)

    def score_prompt_only(
        self, prompt: str, goal: str = "", context: dict = {}
    ) -> float:
        """
        Score a prompt independently of any output. Returns a numeric score.
        Requires the evaluator to support prompt-only scoring.
        """
        if not self.evaluator:
            if self.logger:
                self.logger.log(
                    "PromptScoreFailed", {"error": "Evaluator not initialized"}
                )
            return -1.0

        try:
            return self.evaluator.score_single(prompt, goal=goal, context=context)
        except Exception as e:
            if self.logger:
                self.logger.log(
                    "PromptScoreFailed", {"error": str(e), "prompt": prompt}
                )
            return -1.0
