# stephanie/agents/icl/icl_cartridge.py
from datetime import datetime

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.icl.prompt_builder import PromptBuilder
from stephanie.core.knowledge_cartridge import KnowledgeCartridge
from stephanie.scoring.svm_scorer import SVMScorer


class ICLExample:
    def __init__(self, prompt, response, task_type, embedding=None, score=0.5):
        self.prompt = prompt
        self.response = response
        self.task_type = task_type
        self.timestamp = datetime.utcnow().isoformat()
        self.embedding = embedding
        self.score = score  # Score for relevance or quality, default to 0.5

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "response": self.response,
            "task_type": self.task_type,
            "timestamp": self.timestamp,
            "embedding": self.embedding,
            "score": self.score,
        }


class ICLHelper:
    def __init__(self, embedding_fn=None):
        self.examples = []
        self.embedding_fn = embedding_fn

    def add_example(self, prompt, response, task_type):
        self.examples.append(ICLExample(prompt, response, task_type))
        if self.embedding_fn:
            embedding = self.embedding_fn([prompt])[0]
        self.examples.append(ICLExample(prompt, response, task_type, embedding))

    def get_top_k_similar(self, query, task_type, k=5):
        filtered = [
            ex
            for ex in self.examples
            if ex.task_type == task_type and ex.embedding is not None
        ]
        if not filtered or not self.embedding_fn:
            return [ex.to_dict() for ex in filtered[:k]]

        query_emb = self.embedding_fn([query])[0]
        scores = [
            cosine_similarity([query_emb], [ex.embedding])[0][0] for ex in filtered
        ]
        top_indices = np.argsort(scores)[-k:][::-1]
        return [filtered[i].to_dict() for i in top_indices]

    def get_examples_by_type(self, task_type):
        return [ex.to_dict() for ex in self.examples if ex.task_type == task_type]

    def to_prompt_context(self, task_type, query=None, top_k=5):
        if query and self.embedding_fn:
            relevant = self.get_top_k_similar(query, task_type, k=top_k)
        else:
            relevant = self.get_examples_by_type(task_type)[:top_k]
        return "\n\n".join(
            [f"Q: {ex['prompt']}\nA: {ex['response']}" for ex in relevant]
        )


class ICLAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.helper = ICLHelper()

    async def run(self, context: dict):
        input_text = context.get(self.input_key, "")
        task_type = self.cfg.get("task_type", "default")
        context = self.helper.to_prompt_context(task_type, query=input_text)
        full_prompt = f"{context}\n\nNow, {input_text}"

        # Simulated LLM call (replace with actual model invocation)
        response = self.simulate_llm(full_prompt)

        self.helper.add_example(input_text, response, task_type)

        # 1. Run original prompt
        original_response = self.simulate_llm(input_text)

        # 2. Generate ICL-augmented prompt
        icl_context = self.helper.to_prompt_context(task_type)
        if icl_context:
            alt_prompt = f"{icl_context}\n\nNow, {input_text}"
            icl_response = self.simulate_llm(alt_prompt)
        else:
            icl_response = None

        # 3. Score both
        if (
            self.cfg.get("use_scorer")
            and self.memory
            and hasattr(self.memory, "svm_scorer")
        ):
            scorer = self.memory.svm_scorer
            goal = {"goal_text": input_text}
            original_score = scorer.score(
                goal, {"text": original_response}, ["alignment"]
            ).aggregate()
            icl_score = (
                scorer.score(goal, {"text": icl_response}, ["alignment"]).aggregate()
                if icl_response
                else 0
            )
        else:
            original_score, icl_score = 0.5, 0.5  # fallback

        # 4. Choose best
        if icl_response and icl_score > original_score:
            self.helper.add_example(input_text, icl_response, task_type)

        context["original_score"] = original_score
        context["icl_score"] = icl_score

        context["icl_response"] = icl_response
        context["original"] = original_response

        return context

    def simulate_llm(self, prompt):
        return f"[Simulated Response to: {prompt[:60]}...]"

    def export_to_cartridge(self, cartridge: KnowledgeCartridge):
        for example in self.helper.examples:
            cartridge.schema.setdefault("icl_examples", []).append(example.to_dict())

        return cartridge


class ICLPromptEngineeringAgent:
    """Agent for engineering effective prompts for In-Context Learning"""

    def __init__(self, prompt_builder: PromptBuilder):
        self.prompt_builder = prompt_builder

    def construct_prompt(self, task_context: dict, examples: list[dict]) -> str:
        """Construct a complete prompt with instructions, examples, and constraints"""
        self.prompt_builder.reset()

        # Add task instructions
        self._add_task_instructions(task_context)

        # Add relevant examples
        self._add_examples(examples)

        # Add constraints and format requirements
        self._add_constraints_and_format(task_context)

        return self.prompt_builder.build()

    def _add_task_instructions(self, task_context: dict):
        """Add task-specific instructions"""
        instruction = task_context.get(
            "instruction", "Follow the examples to complete the task."
        )
        self.prompt_builder.add_instruction(instruction)

    def _add_examples(self, examples: list[dict]):
        """Add formatted examples to the prompt"""
        for example in examples:
            if "cot" in example.get("metadata", {}).get("example_type", ""):
                self.prompt_builder.add_cot_example(
                    input_text=example["input"],
                    thought_process=example["metadata"].get("thought_process", ""),
                    output_text=example["output"],
                )
            else:
                self.prompt_builder.add_example(
                    input_text=example["input"], output_text=example["output"]
                )

    def _add_constraints_and_format(self, task_context: dict):
        """Add constraints and format requirements"""
        constraints = task_context.get("constraints", [])
        if constraints:
            self.prompt_builder.add_constraints(constraints)

        output_format = task_context.get("output_format")
        if output_format:
            self.prompt_builder.add_section("Output Format", output_format)

        persona = task_context.get("persona")
        if persona:
            self.prompt_builder.add_section("Persona", persona)


class ICLSelfEditorAgent:
    def __init__(self, llm_call_fn, logger=None):
        self.llm_call_fn = llm_call_fn
        self.logger = logger

    def improve_prompt(
        self, original_prompt: str, initial_response: str, goal: str
    ) -> str:
        critique_prompt = f"""
        The goal is: {goal}

        Original Prompt:
        {original_prompt}

        Initial Response:
        {initial_response}

        Critique the prompt's effectiveness and suggest an improved version.
        Return ONLY the improved prompt, no explanation.
        """
        improved = self.llm_call_fn(critique_prompt)
        if self.logger:
            self.logger.log(
                "PromptEdited", {"before": original_prompt, "after": improved}
            )
        return improved


class PromptEditorAgent:
    """Agent that explores prompt variants and selects the best via scoring"""

    def __init__(self, cfg, scorer: SVMScorer, llm_call_fn, logger=None):
        self.cfg = cfg
        self.scorer = scorer
        self.llm_call_fn = llm_call_fn
        self.logger = logger

    def edit_and_select(
        self, prompt: str, variations: list[str], goal: dict, task_type: str
    ) -> str:
        """Try multiple prompt variations and return the highest scoring result"""
        best_score = -1.0
        best_output = None
        for variant in variations:
            full_prompt = f"{variant}\n\nNow, {prompt}"
            output = self.llm_call_fn(full_prompt)
            score_bundle = self.scorer.score(
                goal, {"text": output}, dimensions=["alignment"]
            )
            score = score_bundle.aggregate()

            self.logger.log(
                "PromptVariantScored",
                {"variant": variant, "score": score, "output": output},
            )

            if score > best_score:
                best_score = score
                best_output = output

        return best_output
