# co_ai/evaluator/llm_judge_evaluator.py

import re

from co_ai.evaluator.base import BaseEvaluator
from co_ai.prompts import PromptLoader


class LLMJudgeEvaluator(BaseEvaluator):
    def __init__(self, cfg, llm_cfg, prompt_file, llm, logger):
        self.cfg = cfg
        self.llm_cfg = llm_cfg
        self.prompt_file = prompt_file
        self.llm = llm  # callable: prompt, context, llm_cfg -> response
        self.logger = logger

    def judge(self, prompt, goal, output_a, output_b):
        context = {
            "goal": goal,
            "hypothesis_a": output_a,
            "hypothesis_b": output_b,
            "comparison_notes": self.cfg.get("comparison_notes", "")
        }

        # Load prompt template
        prompt_loader = PromptLoader(None, self.logger)
        prompt_text = prompt_loader.from_file(self.prompt_file, self.cfg, context)

        # Call LLM with optional llm_cfg
        response = self.llm(prompt_text, context, llm_cfg=self.llm_cfg)
        response = remove_think_blocks(response)

        parsed = parse_response(response)

        preferred = output_a if parsed["winner"] == "A" else output_b
        scores = {
            "winner": parsed["winner"],
            "reason": parsed["reason"],
            "score_a": parsed["score_a"],
            "score_b": parsed["score_b"],
        }

        self.logger.log("LLMJudgeResult", {
            "prompt": prompt,
            "output_a": output_a[:100],
            "output_b": output_b[:100],
            "winner": parsed["winner"],
            "score_a": parsed["score_a"],
            "score_b": parsed["score_b"],
            "reason": parsed["reason"],
            "raw_response": response[:300]
        })

        return preferred, scores


def parse_response(response: str):
    # Normalize spacing
    lines = response.strip().splitlines()
    text = "\n".join([line.strip() for line in lines if line.strip()])  # remove extra spaces

    # Flexible matchers
    winner_match = re.search(r"better hypothesis[:：]\s*<?([AB])>?", text, re.IGNORECASE)
    reason_match = re.search(r"reason[:：]\s*(.+?)(?=\n(?:score_a|score_b)[:：])", text, re.IGNORECASE | re.DOTALL)
    score_a_match = re.search(r"score_a[:：]\s*<?(\d{1,3})>?", text, re.IGNORECASE)
    score_b_match = re.search(r"score_b[:：]\s*<?(\d{1,3})>?", text, re.IGNORECASE)

    return {
        "winner": (winner_match.group(1).upper() if winner_match else "A"),
        "reason": (reason_match.group(1).strip() if reason_match else "No reason provided."),
        "score_a": int(score_a_match.group(1)) if score_a_match else 0,
        "score_b": int(score_b_match.group(1)) if score_b_match else 0,
    }


def remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
