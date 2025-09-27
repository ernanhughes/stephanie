import logging
import re

from stephanie.agents.base_agent import BaseAgent

_logger = logging.getLogger(__name__)


class ChatAnalyzeAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.limit = cfg.get("limit", 10000)

    async def run(self, context: dict) -> dict:
        if context.get("chats"):
            batch = context["chats"]
        else:
            batch = self.memory.chats.list_turns_with_texts(
                min_assistant_len=50,  # skip trivial replies
                limit=self.limit,
                order_desc=False
           )
        out = []
        for row in batch:
            turn_id = row.get("id")
            if row.get("ai_score") is not None:
                _logger.info(f"ChatAnalyzeAgent: Already analyzed turn_id: {turn_id}")
                continue

            merged_context = {**row, **context}
            prompt = self.prompt_loader.load_prompt(self.cfg, merged_context)

            response = self.call_llm(prompt, context)
            try:
                parsed = parse_knowledge_judge_text(response)
            except ParseError as e:
                _logger.error(f"ChatAnalyzeAgent: Failed to parse LLM response for turn_id: {turn_id}, error: {e}")
                continue

            self.memory.chats.set_turn_ai_eval(turn_id=turn_id, score=parsed["score"], rationale=parsed["rationale"])

            _logger.info(f"ChatAnalyzeAgent: Upserted turn analysis for turn_id: {turn_id} with score: {parsed['score']}")
            out.append(turn_id)
        context["analyzed_turn_ids"] = out
        return context


class ParseError(ValueError):
    pass

def parse_knowledge_judge_text(raw: str) -> dict:
    """
    Parse a plain-text knowledge-judge response into:
      {'rationale': str, 'score': int}

    Expected format (case-insensitive keys, rationale may be multi-line):
      rationale: <text...>
      score: <0-100>

    Tolerates leading/trailing whitespace and code fences.
    Raises ParseError on failure or out-of-range score.
    """
    if not raw or not raw.strip():
        raise ParseError("Empty response")

    # Strip common code fences (```...```)
    text = raw.strip()
    if text.startswith("```"):
        # remove first fence line and any trailing ```
        text = re.sub(r"^```[^\n]*\n", "", text, flags=re.DOTALL)
        text = re.sub(r"\n```$", "", text, flags=re.DOTALL)

    # Normalize line endings
    text = text.strip().replace("\r\n", "\n").replace("\r", "\n")

    # Primary pattern: rationale (greedy, up to the 'score:' line), then score
    m = re.search(
        r"(?is)^\s*rationale\s*:\s*(.*?)\n\s*score\s*:\s*([0-9]{1,3})\s*$",
        text,
        flags=re.MULTILINE,
    )
    if not m:
        # Fallback: find score line anywhere; take everything before as rationale
        ms = re.search(r"(?im)^\s*score\s*:\s*([0-9]{1,3})\s*$", text)
        if not ms:
            raise ParseError("Could not find 'score:' line")
        score = int(ms.group(1))
        if not (0 <= score <= 100):
            raise ParseError("Score out of range 0..100")

        # Prefer an explicit 'rationale:' label before score; else take preceding text
        mr = re.search(r"(?is)^\s*rationale\s*:\s*(.*)$", text[:ms.start()], flags=re.MULTILINE)
        rationale = (mr.group(1).strip() if mr else text[:ms.start()].strip())
        if not rationale:
            raise ParseError("Could not find 'rationale:' text")
        return {"rationale": rationale, "score": score}

    rationale = m.group(1).strip()
    score = int(m.group(2))
    if not (0 <= score <= 100):
        raise ParseError("Score out of range 0..100")

    return {"rationale": rationale, "score": score}

