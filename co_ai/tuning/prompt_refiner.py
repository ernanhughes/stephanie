# co_ai/tuning/prompt_refiner.py

from dspy import InputField, OutputField, Predict, Signature, configure, LM, BootstrapFewShot
import requests
from typing import List, Dict, Any


class HypothesisRefinementSignature(Signature):
    """Signature for refining scientific hypothesis generation prompts"""
    goal = InputField(desc="Scientific research objective")
    hypothesis = InputField(desc="Current hypothesis under evaluation")
    review = InputField(desc="Expert review of hypothesis")
    score = InputField(desc="Elo rating or ranking score")

    refined_hypothesis = OutputField(desc="Improved version of hypothesis")


def refine_prompt(seed_prompts: List[str], few_shot_data: List[Dict[str, Any]], model_config: dict) -> str:
    """
    Refine a prompt using few-shot examples.
    
    Args:
        seed_prompts: list of original prompts
        few_shot_data: list of {"goal", "hypothesis", "review", "score"}
        model_config: dict with model settings
    
    Returns:
        refined_prompt: improved prompt string
    """
    # Set up local LLM via Ollama
    ollama_model = model_config.get("name", "ollama/qwen3")
    ollama_api_base = model_config.get("api_base", "http://localhost:11434/api/generate")

    class LocalLLM(LM):
        def __init__(self):
            super().__init__(model=ollama_model, api_base=ollama_api_base)

        def _generate(self, prompt: str, max_tokens: int = 2048):
            payload = {
                "model": self.model.split("/")[-1],
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False
            }

            try:
                response = requests.post(self.api_base, json=payload)
                return [response.json().get("response", "").strip()]
            except Exception as e:
                print(f"[LLM] Call failed: {str(e)}")
                return [""]

    # Mock API key since Ollama doesn't need one
    model_config["api_key"] = "local"
    lm = LocalLLM()
    configure(lm=lm)

    # Build trainset from few-shot data
    trainset = []
    for item in few_shot_data:
        trainset.append({
            "goal": item["goal"],
            "hypothesis": item["hypothesis"],
            "review": item.get("review", ""),
            "score": item.get("elo_rating", 1000)
        })

    # Define metric – we'll use a simple match for now
    def exact_match(example, pred, trace=None):
        return example.refined_hypothesis.lower() == pred.refined_hypothesis.lower()

    # Run prompt optimizer
    tuner = BootstrapFewShot(metric=exact_match)
    program = Predict(HypothesisRefinementSignature)
    tuned_program = tuner.compile(
        program=program,
        trainset=trainset,
        valset=trainset[:2]
    )

    # Return optimized prompt
    return tuned_program.prompt