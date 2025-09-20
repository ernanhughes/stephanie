# stephanie/agents/knowledge_summarizer_agent.py
import re
import uuid
from datetime import datetime
from stephanie.models.plan_trace import ExecutionStep, PlanTrace
from stephanie.agents.base_agent import BaseAgent

class KnowledgeSummarizerAgent(BaseAgent):
    """
    4th Agent: Summarization using knowledge retrieval + CARE/UR² logic.
    """

    def __init__(self, llm, knowledge_store, scorer=None, logger=None):
        super().__init__(logger=logger)
        self.llm = llm
        self.knowledge_store = knowledge_store  # e.g., CaseBooks/Knowledge Graph
        self.scorer = scorer

    async def run(self, goal: str, paper_section: str, meta=None) -> PlanTrace:
        trace_id = str(uuid.uuid4())
        steps = []

        # Step 1: Difficulty-aware trigger
        if self._needs_retrieval(paper_section):
            retrievals = self.knowledge_store.search(paper_section, top_k=5)
        else:
            retrievals = []

        # Step 2: Construct prompt (CARE: enforce retrieval tags)
        retrieval_context = "\n".join(r["text"] for r in retrievals)
        system_prompt = (
            "Summarize the paper section with reasoning inside <think> tags. "
            "If using prior knowledge, wrap retrieved spans inside <retrieval> tags. "
            "Use evidence from the context below when necessary."
        )
        prompt = f"Goal: {goal}\n\nSection:\n{paper_section}\n\nKnowledge:\n{retrieval_context}\n\n"

        response = await self.llm.ainvoke(system_prompt + prompt)

        reasoning = self._extract_tagged(response, "think")
        answer = self._extract_answer(response)
        used_retrievals = self._extract_all_tags(reasoning, "retrieval")

        step = ExecutionStep(
            id=str(uuid.uuid4()),
            trace_id=trace_id,
            goal=goal,
            reasoning=reasoning,
            answer=answer,
            retrievals=used_retrievals,
            created_at=datetime.now(),
            meta={"raw": response, "knowledge_refs": retrievals},
        )
        steps.append(step)

        trace = PlanTrace(id=trace_id, goal=goal, steps=steps, created_at=datetime.now(), meta=meta or {})

        if self.scorer:
            trace = await self.scorer.score_trace(trace, context=paper_section)

        return trace

    def _needs_retrieval(self, section: str) -> bool:
        return len(section.split()) > 150  # toy heuristic: long sections = hard → retrieve

    def _extract_tagged(self, text: str, tag: str) -> str:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    def _extract_all_tags(self, text: str, tag: str):
        return re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)

    def _extract_answer(self, text: str):
        m = re.search(r"Answer:\s*(.*)", text, re.DOTALL)
        return m.group(1).strip() if m else ""
