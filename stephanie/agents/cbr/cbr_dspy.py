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

import dspy
from dspy import InputField, OutputField, Signature

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable
from stephanie.constants import GOAL, GOAL_TEXT

# --- Signature must be inside a class ---
class CBRHypothesisSignature(Signature):
    documents = InputField(desc="Relevant documents or text passages to review")
    goal = InputField(desc="The research or reasoning goal")
    instructions = InputField(desc="Task framing and example for CBR")
    hypotheses = OutputField(desc="List of clear, testable, actionable hypotheses")


class CBRHypothesisModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(CBRHypothesisSignature)

    def forward(self, documents, goal, instructions=""):
        return self.generator(
            documents=documents,
            goal=goal,
            instructions=instructions,
        )


class CBRHypothesisDSPyAgent(BaseAgent):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)

        lm = dspy.LM(
            "ollama_chat/qwen3",
            api_base="http://localhost:11434",
            api_key="",
        )
        dspy.configure(lm=lm)

        self.module = CBRHypothesisModule()

    async def run(self, context: dict):
        goal = context.get(GOAL, {})
        documents = context.get("documents", [])

        # Combine documents into a single string (simple version)
        docs_text = "\n\n".join(
            d.get("text", "") if isinstance(d, dict) else str(d) for d in documents
        )

        # Build instructions (with embedded CBR example)
        instructions = """
        Case-Based Reasoning (CBR) means learning from past cases:
        retrieve the most similar one, adapt to the current goal, revise as needed.

        Example of a good hypothesis:
        "If each execution step in the pipeline is stored as a case and scored,
        then similar future steps can be retrieved and reused, improving decision quality over time."

        Please extract 1–3 strong hypotheses from the documents that directly support
        applying CBR to improve pipeline execution, agent scoring, or knowledge reuse.
        """

        # Run DSPy generation
        result = self.module(
            documents=docs_text,
            goal=goal.get(GOAL_TEXT, ""),
            instructions=instructions
        )

        raw_hypotheses = result.hypotheses if result else []
        hypotheses = []

        for i, hyp_text in enumerate(raw_hypotheses):
            sc = Scorable(
                id=f"{goal.get('id', 'goal')}_hyp_{i}",
                text=hyp_text.strip(),
                target_type="hypothesis",
                metadata={"source": "cbr_dspy", "goal": goal.get("goal_text")}
            )
            self.memory.hypotheses.add(sc)
            hypotheses.append(sc.to_dict())

            self.logger.log("HypothesisExtracted", {
                "goal": goal.get("goal_text", "")[:80],
                "hypothesis": hyp_text.strip()
            })

        context[self.output_key] = hypotheses

        self.report({
            "event": "cbr_hypotheses_extracted",
            "count": len(hypotheses),
            "examples": [h["text"][:80] for h in hypotheses[:2]]
        })




        return context
