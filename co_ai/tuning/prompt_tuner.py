import re

from co_ai.memory import MemoryTool
from co_ai.tuning import PromptRefiner
from co_ai.utils import get_text_from_file, write_text_to_file


class PromptTuner:
    def __init__(self, agent_name, signature_class=None):
        self.agent_name = agent_name
        self.signature = signature_class
        self.memory = MemoryTool()
        self.base_prompt_path = f"prompts/{agent_name}/base_prompt.txt"
        self.current_prompt_path = f"prompts/{agent_name}/current_prompt.txt"

        # Load base prompt as fallback
        self.base_prompt = get_text_from_file(self.base_prompt_path)
        self.current_prompt = self._load_current_prompt()
        self.prompt_refiner = PromptRefiner(None, None, None)


    def _load_current_prompt(self):
        """Load latest prompt — base if no tuned version exists"""
        try:
            return get_text_from_file(self.current_prompt_path)
        except FileNotFoundError:
            return self.base_prompt

    def get_current_prompt(self):
        return self.current_prompt

    def tune_prompt(self, few_shot_data, model_config):
        """
        Attempt to improve the prompt using real-world data.
        
        Args:
            few_shot_data: list of {"goal", "hypotheses", "review", "score"}
            model_config: dict with LLM settings
        
        Returns:
            bool: whether the prompt was updated
        """
        print(f"[PromptTuner] Tuning prompt for {self.agent_name}")
        try:
            refined_prompt = refine_prompt(
                seed_prompts=[self.current_prompt],
                few_shot_data=few_shot_data,
                signature=self.signature,
                metric="exact_match",
                model_config=model_config
            )

            print("[PromptTuner] Evaluating refined prompt...")
            improvement_score = self._evaluate_improvement(refined_prompt, few_shot_data)

            if improvement_score > 0.8:
                print("[PromptTuner] New prompt validated. Updating.")
                write_text_to_file(self.current_prompt_path, refined_prompt)
                self.current_prompt = refined_prompt

                # Store in memory for traceability
                self.memory.log_prompt_version(
                    agent=self.agent_name,
                    prompt=refined_prompt,
                    score=improvement_score
                )
                return True
            else:
                print("[PromptTuner] No significant improvement. Keeping current prompt.")
                return False

        except Exception as e:
            print(f"[PromptTuner] Error during tuning: {e}. Reverting to base prompt.")
            self.revert_to_base()
            return False

    def revert_to_base(self):
        """Revert to original base prompt"""
        self.current_prompt = self.base_prompt
        write_text_to_file(self.current_prompt_path, self.base_prompt)

    def evaluate_improvement(self, new_prompt, few_shot_data):
        """
        Test refined prompt against few-shot examples.
        Returns a score between 0–1 indicating improvement.
        """
        correct_count = 0
        total = len(few_shot_data)

        for example in few_shot_data:
            response = self._run_prompt(new_prompt, example)
            match = re.search(r"hypothesis:(.*)", response, re.IGNORECASE)
            if match:
                hyp = match.group(1).strip()
                if self._is_high_quality(hyp, example):
                    correct_count += 1

        return correct_count / total if total else 0

    def _run_prompt(self, prompt_template, example):
        """Run prompt using local LLM and return raw response"""
        # Use Ollama/Qwen3 here
        pass

    def _is_high_quality(self, generated_hypothesis, example):
        """Determine if hypothesis meets criteria like novelty, feasibility"""
        # Use ranking scores, reflection reviews, etc.
        pass