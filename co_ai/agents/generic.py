# co_ai/agents/generic_agent.py
import re

from co_ai.agents.base import BaseAgent


class GenericAgent(BaseAgent):
    def __init__(self, cfg: dict, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.name = cfg.name
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        # Input/output mapping
        self.input_keys = cfg.get("input_keys", [])
        self.output_key = cfg.get("output_key", "result")
        self.strategy = cfg.get("strategy", "default")

        # Regex pattern to extract result
        self.extraction_regex = cfg.get("extraction_regex", r"response:(.*)")

        # Optional refinement
        self.refine_prompts = cfg.get("refine_prompts", False)

    async def run(self, context: dict) -> dict:
        """Run agent based on config-defined behavior"""
        try:
            # Validate required inputs exist
            missing = [key for key in self.input_keys if key not in context]
            if missing:
                self.logger.log("MissingInputKeys", {
                    "agent": self.name,
                    "missing": missing
                })
                return context

            # Build prompt from template and context
            prompt = self.prompt_loader.load_prompt(self.cfg, context)

            # Call LLM
            response = self.call_llm(prompt).strip()

            # Extract result using regex
            match = re.search(self.extraction_regex, response, re.DOTALL)
            result = match.group(1).strip() if match else response

            # Store in context
            context[self.output_key] = {
                "title": self.name,
                "content": result,
                "prompt_used": prompt[:300],
                "strategy": self.strategy
            }

            self.logger.log("AgentRanSuccessfully", {
                "agent": self.name,
                "inputs_used": self.input_keys,
                "output_key": self.output_key,
                "prompt_snippet": prompt[:200],
                "response_snippet": result[:300]
            })

            if self.cfg.get("save_context", False):
                self._save_context(context)
            return context

        except Exception as e:
            self.logger.log("AgentFailed", {
                "agent": self.name,
                "error": str(e),
                "context_snapshot": {k: len(str(v)) for k, v in context.items()}
            })
            return context
