# co_ai/compiler/final_prompt_builder.py
from co_ai.agents.compiler.reasoning_trace import ReasoningNode


class FinalPromptBuilder:
    def build_prompt(self, path: list[ReasoningNode]) -> str:
        prompt = f"Goal: {path[0].goal}\n\n"
        for i, node in enumerate(path):
            prompt += f"{i+1}. {node.thought}\n"
            prompt += f"Action: {node.action}\n"
            prompt += f"Response: {node.response}\n\n"
        prompt += "Now solve this problem using the above reasoning pattern."
        return prompt