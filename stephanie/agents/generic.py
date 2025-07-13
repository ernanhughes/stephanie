# stephanie/agents/generic.py
import re

from stephanie.agents.base_agent import BaseAgent


class GenericAgent(BaseAgent):
    def __init__(self, cfg: dict, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.name = cfg.get("name")
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        # Input/output mapping
        self.strategy = cfg.get("strategy", "default")

        # Regex pattern to extract result
        self.extraction_regex = cfg.get("extraction_regex", r"response:(.*)")

        # Optional refinement
        self.refine_prompts = cfg.get("refine_prompts", False)

    async def run(self, context: dict) -> dict:
        """Run agent based on config-defined behavior"""
        try:
            # Build prompt from template and context
            prompt = self.prompt_loader.load_prompt(self.cfg, context)

            # Call LLM
            response = self.call_llm(prompt, context)

            # Extract result using regex
            match = re.search(self.extraction_regex, response, re.DOTALL)
            result = match.group(1).strip() if match else response

            # Store in context
            context[self.output_key] = {
                "title": self.name,
                "content": result,
                "prompt_used": prompt[:300],
                "strategy": self.strategy,
            }

            self.logger.log(
                "AgentRanSuccessfully",
                {
                    "agent": self.name,
                    "input_key": self.input_key,
                    "output_key": self.output_key,
                    "prompt_snippet": prompt[:200],
                    "response_snippet": result[:300],
                },
            )

            return context

        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            self.logger.log(
                "AgentFailed",
                {
                    "agent": self.name,
                    "error": str(e),
                    "context_snapshot": {k: len(str(v)) for k, v in context.items()},
                },
            )
            return context
