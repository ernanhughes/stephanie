import json
import random
from collections import Counter
from typing import Dict, List, Optional

from datasets import load_dataset

REASONING_FORMATS = {
    "direct": "<Direct>",
    "short_cot": "<Short_CoT>",
    "code": "<Code>",
    "long_cot": "<Long_CoT>"
}

FORMAT_END_TAGS = {
    "direct": "</Direct>",
    "short_cot": "</Short_CoT>",
    "code": "</Code>",
    "long_cot": "</Long_CoT>"
}


class ARMDataLoader:
    def __init__(
        self,
        dataset_name: str = "aqua_rat",
        subset: Optional[str] = None,
        split: str = "train",
        max_samples: int = 500,
        memory=None,
        logger=None,
    ):
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.max_samples = max_samples
        self.memory = memory
        self.logger = logger

        # Format tokens
        self.format_tokens = {
            "direct": "<Direct>",
            "short_cot": "<Short_CoT>",
            "code": "<Code>",
            "long_cot": "<Long_CoT>",
        }
        self.format_end_tokens = {
            "direct": "</Direct>",
            "short_cot": "</Short_CoT>",
            "code": "</Code>",
            "long_cot": "</Long_CoT>",
        }

        self._debug_count = 0
        self.dataset = None

    def log(self, event_name: str, payload: dict):
        if self.logger:
            self.logger.log(event_name, payload)
        else:
            print(f"[{event_name}] {json.dumps(payload)}")

    def adapt(self, context: dict):
        """Main method: Load → Convert → Save to Memory"""
        self.log("DatasetLoading", {"name": self.dataset_name, "split": self.split})
        self.load_dataset()
        self.summarize_difficulties()
        self.print_samples_by_difficulty()

        total_samples = len(self.dataset)
        indices = random.sample(
            range(total_samples), min(self.max_samples, total_samples)
        )

        count = 0
        goal_text = context.get("goal").get("goal_text")
        run_id = context.get("run_id")
        for idx in indices:
            sample = self.dataset[idx]
            pairs = self.build_preference_pairs(sample)
            for pair in pairs:
                prompt = pair["prompt"]
                chosen = pair["chosen"]
                rejected = pair["rejected"]
                preferred = pair["preferred_format"]
                fmt_a = self.detect_format(chosen)
                fmt_b = self.detect_format(rejected)
                difficulty = self.detect_difficulty(prompt)
                # Embed everything once
                self._get_or_cache_embedding(prompt)
                self._get_or_cache_embedding(chosen)
                self._get_or_cache_embedding(rejected)

                # Save to database
                try:
                    self.memory.mrq.add_preference_pair(
                        goal=goal_text,
                        prompt=prompt,
                        output_a=chosen,
                        output_b=rejected,
                        preferred=preferred,
                        fmt_a=fmt_a,
                        fmt_b=fmt_b,
                        difficulty=difficulty,
                        run_id=run_id,
                    )
                    count += 1
                except Exception as e:
                    self.log(
                        "PreferencePairSaveError",
                        {
                            "error": str(e),
                            "prompt": prompt[:80],
                            "chosen": chosen[:80],
                            "rejected": rejected[:80],
                        },
                    )

        self.log("PreferencePairsSaved", {"count": count, "goal": "arm_dpo"})
        context["dpo_samples"] = count
        return context

    def _get_or_cache_embedding(self, text: str) -> List[float]:
        """
        Get embedding from cache or compute and store.
        Uses your existing memory.embedding.get_or_create() method.
        """
        emb = self.memory.embedding.get_or_create(text)
        return emb

    def load_dataset(self):
        """Load dataset from Hugging Face."""
        try:
            self.dataset = load_dataset(
                self.dataset_name, self.subset, split=self.split
            )
            self.log("DatasetLoaded", {"count": len(self.dataset)})
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset '{self.dataset_name}': {str(e)}"
            )

    def _detect_difficulty(self, question: str) -> str:
        words = question.split()
        if len(words) < 20:
            return "easy"
        elif len(words) < 50:
            return "medium"
        else:
            return "hard"

    def build_preference_pairs(self, sample: Dict) -> List[Dict]:
        """
        Build DPO-style preference pairs by comparing formats.
        Returns list of dicts like:
        {
          'prompt': ...,
          'chosen': ...,
          'rejected': ...,
          'preferred_format': ...,
          'difficulty': ...
        }
        """
        question = sample.get("question", "").strip()
        ground_truth = sample.get("correct", "").strip()
        difficulty = self._detect_difficulty(question)

        # Generate all four reasoning formats
        direct = self.generate_direct(ground_truth)
        short_cot = self.generate_short_cot(question, ground_truth)
        code = self.generate_code(question, ground_truth)
        long_cot = self.generate_long_cot(question, ground_truth)

        format_to_response = {
            "direct": direct,
            "short_cot": short_cot,
            "code": code,
            "long_cot": long_cot,
        }

        # Filter out empty responses
        valid_formats = [
            fmt for fmt, resp in format_to_response.items() if resp.strip()
        ]
        format_to_response = {
            k: v for k, v in format_to_response.items() if k in valid_formats
        }

        # Define which formats are preferred based on difficulty
        if difficulty == "easy":
            preferred_formats = ["direct", "short_cot", "code"]
            non_preferred_formats = ["long_cot"]
        elif difficulty == "hard":
            preferred_formats = ["long_cot", "code"]
            non_preferred_formats = ["direct", "short_cot"]
        else:  # medium or default case
            preferred_formats = ["short_cot", "code"]
            non_preferred_formats = ["direct", "long_cot"]

        # Build all possible pairs
        pairs = []
        for pref in preferred_formats:
            p_resp = format_to_response.get(pref)
            if not p_resp:
                continue
            for non_pref in non_preferred_formats:
                np_resp = format_to_response.get(non_pref)
                if not np_resp:
                    continue
                pairs.append(
                    {
                        "prompt": question,
                        "chosen": p_resp,
                        "rejected": np_resp,
                        "preferred_format": pref,
                        "rejected_format": non_pref,
                        "difficulty": difficulty,
                    }
                )

        return pairs

    def summarize_difficulties(self):
        counts = Counter()
        for sample in self.dataset:
            question = sample.get("question", "")
            detected = self._detect_difficulty(question)
            counts[detected] += 1
        self.log("DifficultySummary", dict(counts))
        return counts

    def print_samples_by_difficulty(self, count_per_level=3):
        buckets = {"easy": [], "medium": [], "hard": []}
        for sample in self.dataset:
            question = sample.get("question", "")
            difficulty = self._detect_difficulty(question)
            if len(buckets[difficulty]) < count_per_level:
                buckets[difficulty].append(question)

        for diff, questions in buckets.items():
            self.log("SampleByDifficulty", {"difficulty": diff, "examples": questions})

    def _detect_difficulty(self, question: str) -> str:
        """Basic heuristic to infer difficulty based on question length."""
        words = question.split()
        if len(words) < 20:
            return "easy"
        elif len(words) < 50:
            return "medium"
        else:
            return "hard"

    def generate_direct(self, answer: str) -> str:
        return f"{self.format_tokens['direct']}The answer is {answer}.{self.format_end_tokens['direct']}"

    def generate_short_cot(self, question: str, answer: str) -> str:
        return (
            f"{self.format_tokens['short_cot']}"
            "Let me think briefly:\n"
            "Step 1: Understand the question.\n"
            "Step 2: Apply basic logic.\n"
            f"Final Answer: {answer}"
            f"{self.format_end_tokens['short_cot']}"
        )

    def generate_code(self, question: str, answer: str) -> str:
        return (
            f"{self.format_tokens['code']}"
            "def solve():\n"
            "    # Placeholder code generated by GPT-4o\n"
            f"    return '{answer}'\n"
            "solve()\n"
            f"# Output: {answer}"
            f"{self.format_end_tokens['code']}"
        )

    def generate_long_cot(self, question: str, answer: str) -> str:
        return (
            f"{self.format_tokens['long_cot']}"
            "Let's analyze this step-by-step:\n\n"
            "1. Read the question carefully.\n"
            "2. Identify key information.\n"
            "3. Consider multiple approaches.\n"
            "4. Evaluate thoroughly.\n"
            "...\n"
            "Reflection: This approach ensures correctness by exploring multiple paths.\n"
            f"Final Answer: {answer}"
            f"{self.format_end_tokens['long_cot']}"
        )

    def detect_difficulty(self, text: str) -> str:
        words = text.split()
        if len(words) < 20:
            return "easy"
        elif len(words) < 50:
            return "medium"
        else:
            return "hard"

    @staticmethod
    def detect_format(text: str) -> str:
        text = text.strip().lower()
        if not text:
            return "unknown"
        if "<direct>" in text:
            return "direct"
        elif "<short_cot>" in text:
            return "short_cot"
        elif "<code>" in text:
            return "code"
        elif "<long_cot>" in text:
            return "long_cot"
        
        # Direct Answer
        if text.startswith("the answer is") or text.startswith("answer:"):
            return "direct"

        # Short CoT
        elif text.startswith("let me think briefly"):
            return "short_cot"

        # Long CoT
        elif text.startswith("let's analyze this step-by-step"):
            return "long_cot"

        # Code
        elif any(kw in text for kw in ["def", "return", "solve()", "print(", "for ", "if "]):
            return "code"

        else:
            print(f"[WARNING] Unknown format:\n{text[:100]}...")
            return "unknown"