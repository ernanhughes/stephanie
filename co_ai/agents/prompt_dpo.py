from co_ai.agents import BaseAgent
from co_ai.constants import GOAL

from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re


class PromptDPOAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_name = cfg.get("model", "Qwen3")
        self.local_model_path = cfg.get("local_model_path", "./models/qwen3-sharp")
        self.beta = cfg.get("beta", 0.1)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_name)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, "")
        pairs = self.memory.get_prompt_hypothesis_pairs(goal, limit=200)
        
        # Convert to DPO-ready format
        dpo_ready = []
        for pair in pairs:
            dpo_ready.append({
                "prompt": pair["prompt"],
                "chosen": pair["hypothesis"],
                "rejected": self._generate_opposite_hypothesis(pair),
            })

        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            beta=self.beta,
            train_dataset=dpo_ready,
            tokenizer=self.tokenizer
        )

        # Run training
        trainer.train()

        # Save updated model
        self.model.save_pretrained(self.local_model_path)
        self.tokenizer.save_pretrained(self.local_model_path)

        context["refined_prompter"] = self.local_model_path
        return context

    def _generate_opposite_hypothesis(self, pair):
        """Generate a weaker version for contrastive learning"""
        critique_prompt = f"""
You are an expert reviewer identifying weaknesses in hypothesis generation.

Review this hypothesis:
{pair['hypothesis']}

Goal:
{pair['goal']}

Instructions:
1. Identify what makes this hypothesis strong
2. Generate a weaker version with reduced clarity or novelty
3. Keep mechanism intact but reduce quality slightly
4. Output only:
weaker hypothesis:<your response>
"""

        response = self.call_llm(critique_prompt).strip()
        match = re.search(r"weaker hypothesis:<(.+)", response, re.DOTALL)
        return match.group(1).strip() if match else pair["hypothesis"][::-1]