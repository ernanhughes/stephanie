# stephanie/agents/dspy/cbr_dspy.py
"""
CBR Hypothesis DSPy Generator Agent

This agent uses DSPy to review one or more documents in light of a goal,
and extract strong, testable hypotheses aligned with Case-Based Reasoning (CBR).

Key features:
- Takes documents + goal as input
- Uses DSPy signature tailored to hypothesis extraction
- Embeds CBR definition + example into the prompt automatically
- Returns hypotheses as Scorables for downstream scoring/ranking
"""
from __future__ import annotations

import re
from typing import List

import dspy
from dspy import InputField, OutputField, Signature

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL, GOAL_TEXT
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType


class CBRSignature(Signature):
    """DSPy signature for CBR hypothesis generation."""

    documents: str = InputField(
        desc="Relevant documents or text passages to review"
    )
    goal: str = InputField(desc="The research or reasoning goal")
    task_instructions: str = InputField(
        desc="Task framing and example for CBR"
    )
    hypotheses: list[str] = OutputField(
        desc="List of clear, testable, actionable hypotheses"
    )


class CBRModule(dspy.Module):
    """DSPy module for generating CBR hypotheses."""

    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(CBRSignature)

    def forward(self, documents, goal, task_instructions=""):
        return self.generator(
            documents=documents,
            goal=goal,
            task_instructions=task_instructions,
        )


class LoggingLM(dspy.LM):
    def __call__(self, *args, **kwargs):
        print("📡 Sending request to LLM...")
        result = super().__call__(*args, **kwargs)
        print("✅ Received response.")
        return result


class CBRDSPyAgent(BaseAgent):
    """
    Agent that generates hypotheses using Case-Based Reasoning (CBR) with DSPy.

    CBR involves solving new problems by adapting solutions that were used
    to solve similar problems in the past.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Get model configuration from agent config with fallbacks
        model_config = cfg.get("model", {})
        model_name = model_config.get("name", "ollama_chat/qwen3")
        api_base = model_config.get("api_base", "http://localhost:11434")
        api_key = model_config.get("api_key", "")

        # Configure DSPy with the specified model
        lm = LoggingLM(
            model_name,
            api_base=api_base,
            api_key=api_key,
        )
        dspy.configure(lm=lm)

        self.module = CBRModule()
        self.max_hypotheses = cfg.get("max_hypotheses", 3)

        # CBR instructions template
        self.cbr_instructions_template = """
        Case-Based Reasoning (CBR) means learning from past cases:
        retrieve the most similar one, adapt to the current goal, revise as needed.

        Example of a good hypothesis:
        "If each execution step in the pipeline is stored as a case and scored,
        then similar future steps can be retrieved and reused, improving decision quality over time."

        Please extract {max_hypotheses} strong hypotheses from the documents that directly support
        applying CBR to improve pipeline execution, agent scoring, or knowledge reuse.
        
        Format your response as a numbered list without any additional commentary.
        """

    def _parse_hypotheses(self, raw) -> List[str]:
        """
        Normalize hypotheses whether returned as list[str] or str.
        """
        hypotheses = []

        if isinstance(raw, list):
            # Clean each string in the list
            for h in raw:
                if isinstance(h, str):
                    cleaned = h.strip()
                    if cleaned and len(cleaned) > 10:
                        hypotheses.append(cleaned)
        elif isinstance(raw, str):
            # Fallback: parse numbered list from a big text block
            lines = raw.strip().split("\n")
            for line in lines:
                line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
                if line and len(line) > 10:
                    hypotheses.append(line)
        else:
            self.logger.log(
                "ParseHypothesesError",
                {"type": str(type(raw)), "value": str(raw)[:200]},
            )

        return hypotheses[: self.max_hypotheses]

    async def run(self, context: dict):
        """
        Generate CBR hypotheses from documents in the context.

        Args:
            context: Execution context containing goal and documents

        Returns:
            Updated context with generated hypotheses
        """
        goal = context.get(GOAL, {})
        documents = context.get("documents", [])

        if not documents:
            self.logger.log(
                "NoDocumentsForCBR",
                {
                    "goal_id": goal.get("id", "unknown"),
                    "goal_text": goal.get(GOAL_TEXT, "")[:100],
                },
            )
            context[self.output_key] = []
            return context

        # Combine documents into a single string
        docs_text = "\n\n".join(
            d.get("text", "") if isinstance(d, dict) else str(d)
            for d in documents
        )

        # Truncate if too long to avoid context window issues
        max_length = self.cfg.get("max_document_length", 4000)
        if len(docs_text) > max_length:
            docs_text = docs_text[:max_length] + "... [truncated]"
            self.logger.log(
                "DocumentsTruncated",
                {
                    "original_length": len(docs_text),
                    "truncated_length": max_length,
                },
            )

        # Build instructions with configured max hypotheses
        task_instructions = self.cbr_instructions_template.format(
            max_hypotheses=self.max_hypotheses
        )

        try:
            # Run DSPy generation
            result = self.module(
                documents=docs_text,
                goal=goal.get(GOAL_TEXT, ""),
                task_instructions=task_instructions,
            )

            # Parse the hypotheses from the result
            if result and hasattr(result, "hypotheses"):
                raw_hypotheses_text = result.hypotheses
                parsed_hypotheses = self._parse_hypotheses(raw_hypotheses_text)
            else:
                parsed_hypotheses = []
                self.logger.log(
                    "NoHypothesesGenerated",
                    {"goal_id": goal.get("id", "unknown")},
                )

        except Exception as e:
            self.logger.log(
                "CBRGenerationError",
                {"error": str(e), "goal_id": goal.get("id", "unknown")},
            )
            parsed_hypotheses = []

        # Create Scorable objects for each hypothesis
        hypotheses = []
        for i, hyp_text in enumerate(parsed_hypotheses):
            sc = Scorable(
                text=hyp_text.strip(),
                target_type=TargetType.HYPOTHESIS,
            )
            result = self.save_hypothesis({"text": sc.text}, context=context)
            hypotheses.append(result.to_dict())

            self.logger.log(
                "HypothesisExtracted",
                {
                    "goal": goal.get(GOAL_TEXT, "")[:80],
                    "hypothesis": hyp_text.strip()[:100] + "..."
                    if len(hyp_text) > 100
                    else hyp_text.strip(),
                },
            )

        return context
