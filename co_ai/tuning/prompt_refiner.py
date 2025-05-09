# co_ai/tuning/prompt_refiner.py

from dspy import LM, InputField, OutputField, Predict, Signature, configure
from dspy.teleprompt import BootstrapFewShot


class GenerateSignature(Signature):
    goal = InputField()
    hypotheses = OutputField()


def refine_prompt(seed_prompts, few_shot_data, model_config):

    # Setup LM based on config
    lm = LM(
        model_config["name"],
        api_base=model_config["api_base"],
        api_key=model_config.get("api_key")
    )
    configure(lm=lm)

    tuner = BootstrapFewShot(metric="exact_match")  # Simple metric placeholder

    program = Predict(GenerateSignature)
    tuned_program = tuner.compile(
        signature=GenerateSignature,
        trainset=few_shot_data,
        program=program,
        init_prompts=seed_prompts,
    )

    return tuned_program.prompt
