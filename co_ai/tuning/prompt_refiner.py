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


class PromptRefiner:
    def __init__(self, agent_config: dict, logger=None, memory=None):
        self.model_config = agent_config
        self.memory = memory
        self.logger = logger

        # Set up local LLM
        ollama_model = agent_config.get("name", "ollama/qwen3")
        ollama_api_base = agent_config.get("api_base", "http://localhost:11434")

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

        self.lm = LocalLLM()
        configure(lm=self.lm)

    def refine_prompt(
        self,
        seed_prompts: List[str],
        few_shot_data: List[Dict[str, Any]],
        agent_name: str,
        prompt_key: str,
        current_version: int = 1
    ) -> str:
        """
        Attempt to refine a prompt using few-shot examples.
        
        Only stores new prompt if it improves over previous versions.
        
        Args:
            seed_prompts: list of original prompts
            few_shot_data: list of {"goal", "hypotheses", "review", "score"}
            agent_name: name of agent using this prompt (e.g., generation, reflection)
            prompt_key: identifier like 'generation_goal_aligned.txt'
            current_version: current version number
            
        Returns:
            refined_prompt: improved version or fallback to original
        """
        training_set = []
        few_shot_data = self._get_few_shot_data(seed_prompts[0])
        for item in few_shot_data:
            training_set.append({
                "goal": item["goal"],
                "hypotheses": item["hypotheses"],
                "review": item.get("review", ""),
                "score": item.get("elo_rating", 1000)
            })

        # Run DSPy-based prompt optimization
        tuner = BootstrapFewShot(metric=self._exact_match_metric)
        program = Predict(HypothesisRefinementSignature)
        tuned_program = tuner.compile(
            program=program,
            trainset=training_set,
            valset=training_set[:2]
        )

        refined_prompt = tuned_program.prompt

        # Compare against baseline performance
        old_prompt = seed_prompts[0]
        old_score = self._evaluate_prompt(old_prompt, few_shot_data)
        new_score = self._evaluate_prompt(refined_prompt, few_shot_data)

        # Only store if improvement detected
        if new_score > old_score:
            self.logger.log("PromptRefinedAndImproved", {
                "agent": agent_name,
                "prompt_key": prompt_key,
                "old_score": old_score,
                "new_score": new_score
            })

            self.memory.store_prompt_version(
                agent_name=agent_name,
                prompt_key=prompt_key,
                prompt_text=refined_prompt,
                input_keys=["goal", "literature", "preferences"],
                output_key="hypotheses",
                extraction_regex=r"Hypothesis 1:\n(.+?)\n\nHypothesis 2:",
                source="dsp_refinement",
                version=current_version + 1,
                is_current=True,
                metadata={
                    "few_shot_count": len(few_shot_data),
                    "improvement_score": new_score - old_score
                }
            )

            return refined_prompt
        else:
            self.logger.log("PromptRefiningNoImprovement", {
                "agent": agent_name,
                "prompt_key": prompt_key,
                "old_score": old_score,
                "new_score": new_score
            })
            return old_prompt

    def _exact_match_metric(self, example, pred, trace=None):
        """Simple metric to evaluate exact match between hypothesis and prediction"""
        return example.refined_hypothesis.lower() == pred.refined_hypothesis.lower()

    def _evaluate_prompt(self, prompt: str, test_data: List[Dict[str, Any]]) -> float:
        """Evaluate prompt quality by running it on test data and scoring results"""
        total_score = 0
        for item in test_data:
            try:
                # Simulate prompt use
                response = self.lm(prompt.format(**item))
                refined = response[0].strip()

                # Score based on whether hypothesis matches expected one
                if refined:
                    total_score += item.get("score", 1000) * 0.1  # Weighted by prior ranking
            except Exception as e:
                self.logger.log("PromptEvaluationFailed", {"error": str(e)})
                continue

        return total_score / len(test_data) if test_data else 0


    def _get_few_shot_data(self, goal: str) -> List[Dict[str, Any]]:
        """Pull ranked hypotheses and reviews from memory"""
        return self.memory.get_ranked_hypotheses(goal)