# stephanie/agents/qwen3_prompt_agent.py
from __future__ import annotations

import re
from typing import Any, Dict, List

from stephanie.agents.base_agent import BaseAgent


class Qwen3PromptAgent(BaseAgent):
    """
    Simple prompt→Qwen3→result agent.

    - Loads a prompt template via the BaseAgent prompt_loader
    - Sends it to the configured LLM (intended: Qwen3)
    - Optionally extracts a region via regex
    - Optionally generates multiple samples
    - Stores the result(s) under context[self.output_key]

    Configure Qwen3 via your Hydra / LLM config (e.g. model name, endpoint).
    This agent doesn't hard-code the transport; it just calls `self.call_llm`.
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        self.name: str = cfg.get("name", "qwen3_prompt")
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        # Strategy label (purely informational / logging)
        self.strategy: str = cfg.get("strategy", "qwen3-default")

        # Regex pattern to extract main payload from the raw response
        # Example: r"response:(.*)$" or r"<result>(.*)</result>"
        self.extraction_regex: str = cfg.get(
            "extraction_regex",
            r"response:(.*)",  # default compatible with your GenericAgent
        )

        # Number of completions to generate per prompt
        self.num_samples: int = int(cfg.get("num_samples", 1))

        # Optional: whether to also store the raw, un-extracted responses
        self.store_raw: bool = bool(cfg.get("store_raw", True))

        # Optional: label for the model we're expecting (for logging only);
        # actual model routing should be done in your LLM service config.
        self.model_name: str = cfg.get("model_name", "qwen3")

    def _extract(self, text: str) -> str:
        """
        Apply the configured regex extraction to the raw LLM text.
        Falls back to the full text if no match is found.
        """
        if not self.extraction_regex:
            return text.strip()

        m = re.search(self.extraction_regex, text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects:
          - Any keys required by your prompt template (via prompt_loader)

        Produces:
          context[self.output_key] = {
              "title": <agent name>,
              "strategy": <strategy label>,
              "model_name": <string>,
              "prompt_used": <first 300 chars>,
              "samples": [
                  {
                      "index": i,
                      "raw": <optional raw text>,
                      "result": <extracted text>,
                  },
                  ...
              ],
          }
        """
        try:
            # 1) Build prompt from template + context
            prompt: str = self.prompt_loader.load_prompt(self.cfg, context)

            samples: List[Dict[str, Any]] = []

            # 2) Call LLM N times (Qwen3 via your LLM config)
            for i in range(self.num_samples):
                # NOTE:
                # - We keep the signature identical to GenericAgent:
                #       self.call_llm(prompt, context)
                # - If your BaseAgent.call_llm supports model overrides,
                #   you can extend this to pass model_name / temperature, e.g.:
                #       self.call_llm(prompt, context, model_name=self.model_name)
                raw_text: str = self.call_llm(prompt, context)

                extracted: str = self._extract(raw_text)

                sample_rec: Dict[str, Any] = {
                    "index": i,
                    "result": extracted,
                }
                if self.store_raw:
                    sample_rec["raw"] = raw_text

                samples.append(sample_rec)

            # 3) Store structured result in context
            context[self.output_key] = {
                "title": self.name,
                "strategy": self.strategy,
                "model_name": self.model_name,
                "prompt_used": prompt[:300],
                "num_samples": self.num_samples,
                "samples": samples,
            }

            # 4) Log success for observability
            first_result = samples[0]["result"] if samples else ""
            self.logger.log(
                "AgentRanSuccessfully",
                {
                    "agent": self.name,
                    "input_key": self.input_key,
                    "output_key": self.output_key,
                    "prompt_snippet": prompt[:200],
                    "response_snippet": first_result[:300],
                    "num_samples": self.num_samples,
                    "model_name": self.model_name,
                },
            )

            return context

        except Exception as e:
            # Make sure failures are visible but don't crash the pipeline
            err_msg = f"{type(e).__name__}: {e}"
            print(f"❌ Qwen3PromptAgent exception: {err_msg}")
            self.logger.log(
                "AgentFailed",
                {
                    "agent": self.name,
                    "error": err_msg,
                    "input_key": self.input_key,
                    "output_key": self.output_key,
                    "context_snapshot": {k: len(str(v)) for k, v in context.items()},
                },
            )
            return context
