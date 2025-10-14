import logging
import re
from stephanie.agents.base_agent import BaseAgent
from stephanie.data.score_result import ScoreResult
from stephanie.data.score_bundle import ScoreBundle
from stephanie.scoring.scorable import Scorable, ScorableType

_logger = logging.getLogger(__name__)


class ChatAnalyzeAgent(BaseAgent):
    """
    ChatAnalyzeAgent
    ----------------
    Scores chat turns (user → assistant) using an LLM prompt, parses rationale + score,
    and persists results via the unified ScoringService / EvaluationStore pipeline.
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.limit = cfg.get("limit", 10000)
        self.dimensions = cfg.get("dimensions", ["reasoning", "knowledge", "clarity", "faithfulness", "coverage"])
        self.force_rescore = cfg.get("force_rescore", False) # Test initially with force rescore

    async def run(self, context: dict) -> dict:
        """
        Analyze chat turns (user→assistant pairs), assign AI knowledge scores,
        and persist both chat_turn updates and EvaluationORM records.
        """
        if context.get("chats"):
            batch = context["chats"]
        else:
            batch = self.memory.chats.list_turns_with_texts(
                min_assistant_len=50,
                limit=self.limit,
                order_desc=False,
            )

        scoring = self.container.get("scoring")  # ScoringService
        prompt_service = self.container.get("prompt")  # PromptService
        pipeline_run_id = context.get("pipeline_run_id")
        
        out = []
        for row in batch:
            turn_id = row.get("id")
            assistant_message_id = row.get("assistant_message_id")
            if row.get("ai_score") is not None:
                _logger.debug(
                    f"[ChatAnalyzeAgent] Skipping already analyzed turn {turn_id}"
                )
                continue

            user_text = row.get("user_text", "").strip()
            assistant_text = row.get("assistant_text", "").strip()
            if not user_text or not assistant_text:
                _logger.info(
                    f"[ChatAnalyzeAgent] Skipping incomplete turn {turn_id}"
                )
                continue

            # 1️⃣ Create teh goal from the user text if not exists
            goal = self.memory.goals.get_or_create(
                {
                    "goal_text": user_text,
                    "description": "Goal Created from Chat Analyze Agent",
                    "pipeline_run_id": pipeline_run_id,
                    "meta": {"source": "chat_analyze_agent"},
                }
            )

            # 2️⃣ Call LLM to get knowledge score + rationale
            merged_context = {**row, **context}


            results = {}
            for dim in self.dimensions:
                prompt = self.prompt_loader.from_file(f"{dim}.txt", self.cfg, merged_context)
                response = await prompt_service.run_prompt(prompt, merged_context)
                # 3️⃣ Parse the response
                score = 0.0
                rationale = ""
                try:
                    parsed = parse_knowledge_judge_text(response)
                    score = parsed["score"]
                    rationale = parsed["rationale"]
                except ParseError as e:
                    _logger.error(
                        f"[ChatAnalyzeAgent] Parse error for turn {assistant_message_id}: {e}"
                    )
                    continue

                # 3️⃣ Save the scores on the chat turn object ... we do this for the gui  
                if dim == "knowledge":
                    self.memory.chats.set_turn_ai_eval(
                        turn_id=turn_id,
                        score=score,
                        rationale=rationale,
                    )


                # 2️⃣ Create the EvaluationORM object
                score_result = ScoreResult(
                    dimension=dim,
                    score=score,
                    source="knowledge_llm",
                    rationale=rationale,
                    attributes={"raw_response": response},
                )
                results[dim] = score_result

            bundle = ScoreBundle(results=results)

            #3️⃣ Wrap as Scorable and persist via standard save_bundle
            scorable = Scorable(
                id=assistant_message_id,
                text=assistant_text,
                target_type=ScorableType.CONVERSATION_TURN,
            )

            # 2️⃣ Create and save the bundle
            scored_context = {**context, "goal": goal.to_dict()}
            scoring.save_bundle(
                bundle=bundle,
                scorable=scorable,
                context=scored_context,
                cfg=self.cfg,
                agent_name=self.name,
                scorer_name="knowledge_llm",
                source="knowledge_llm",
                model_name="llm"
            )

            out.append(
                {
                    "turn_id": assistant_message_id,
                    "score": parsed["score"],
                    "rationale": parsed["rationale"],
                }
            )

        context["analyzed_turns"] = out
        return context


class ParseError(ValueError):
    pass


def parse_knowledge_judge_text(raw: str) -> dict:
    """
    Parse a plain-text knowledge-judge response into:
      {'rationale': str, 'score': int}
    """
    if not raw or not raw.strip():
        raise ParseError("Empty response")

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text, flags=re.DOTALL)
        text = re.sub(r"\n```$", "", text, flags=re.DOTALL)

    text = text.strip().replace("\r\n", "\n").replace("\r", "\n")

    m = re.search(
        r"(?is)rationale\s*:\s*(.*?)\n\s*score\s*:\s*([0-9]{1,3})\s*$", text
    )
    if m:
        rationale, score = m.group(1).strip(), int(m.group(2))
    else:
        ms = re.search(r"(?im)score\s*:\s*([0-9]{1,3})", text)
        if not ms:
            raise ParseError("Could not find score")
        score = int(ms.group(1))
        rationale = text[: ms.start()].strip()

    if not (0 <= score <= 100):
        raise ParseError(f"Score out of range: {score}")

    return {"rationale": rationale, "score": score}
