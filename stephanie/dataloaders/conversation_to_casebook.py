# dataloaders/conversation_to_casebook.py
import logging
import re
from datetime import datetime

from stephanie.memory.memory_tool import MemoryTool
from stephanie.models.casebook import CaseBookORM, CaseORM, CaseScorableORM
from stephanie.models.goal import GoalORM

logger = logging.getLogger(__name__)

def summarize_goal(bundle: list) -> str:
    """Infers a high-level goal from the conversation history."""
    # A simple, heuristic-based summarization.
    # Can be replaced with a more advanced summarization agent later.
    if not bundle:
        return "Untitled Conversation"
    first_user_turn = next((t for t in bundle if t["role"] == "user"), None)
    if first_user_turn:
        return first_user_turn["text"].split("\n")[0][:100] + "..."
    return "Untitled Conversation"

def make_name(bundle: list) -> str:
    """Generates a unique name for the CaseBook."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    goal_slug = re.sub(r'[^a-zA-Z0-9]+', '_', summarize_goal(bundle).strip()).lower()
    return f"conv_{timestamp}_{goal_slug}"

def user_assistant_pairs(bundle: list):
    """Yields (user, assistant) pairs from the conversation bundle."""
    user_turn = None
    for turn in bundle:
        if turn["role"] == "user":
            user_turn = turn
        elif turn["role"] == "assistant" and user_turn:
            yield (user_turn, turn)
            user_turn = None

def conversation_to_casebook(memory, bundle: dict, context: dict, tags: list[str]=None) -> tuple[CaseBookORM, dict]:
    """
    Convert a parsed conversation bundle into a CaseBook + Cases + Scorables.
    Returns (CaseBookORM, counts).
    """
    counts = {"cases_created": 0, "scorables_created": 0}

    title = bundle.get("title", "Untitled Conversation")
    conversation_id = bundle.get("id") or bundle.get("conversation_id")

    cb = memory.casebooks.ensure_casebook(
        name=f"chat_{conversation_id or title[:200]}",
        pipeline_run_id=context.get("pipeline_run_id"),
        description=f"Imported chat conversation: {title}",
        tags=tags or [],
    )
    logger.info(f"[CaseBook] Using: {cb.name} (id={cb.id})")

    # --- 1. Extract turns ---
    mapping = bundle.get("mapping", {})
    if mapping:
        turns = _extract_turns_from_mapping(mapping)
    else:
        turns = bundle.get("turns", [])

    if not turns:
        logger.info(f"⚠️ Skipping {title}, no turns found")
        return cb, counts

    # --- 2. Goal setup ---
    goal = memory.goals.get_or_create(
        {"goal_text": f"{cb.name}", "description": f"{cb.description}"}
    )
    goal = goal.to_dict()

    # --- 3. Uniqueness guard: load existing scorable hashes ---
    existing_hashes = {
        sc.meta.get("turn_hash")
        for sc in memory.session.query(CaseScorableORM)
        .join(CaseORM, CaseScorableORM.case_id == CaseORM.id)
        .filter(CaseORM.casebook_id == cb.id)
        .all()
        if sc.meta and sc.meta.get("turn_hash")
    }

    # --- 4. Add new turns as cases/scorables ---
    for i in range(len(turns) - 1):
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            user_text = turns[i]["text"].strip()
            assistant_text = turns[i + 1]["text"].strip()
            thash = _turn_hash(user_text, assistant_text)

            if thash in existing_hashes:
                logger.debug(f"[Skip] Duplicate scorable hash {thash[:10]}…")
                continue

            case = memory.casebooks.add_case(
                casebook_id=cb.id,
                goal_id=goal.get("id"),
                goal_text=goal.get("goal_text"),
                agent_name="chat_import",
                prompt_text=user_text,
                scorables=[{
                    "text": assistant_text,
                    "role": "assistant",
                    "source": "chat",
                    "meta": {"turn_hash": thash},
                }],
                response_texts=assistant_text
            )

            counts["cases_created"] += 1
            counts["scorables_created"] += 1  # one scorable per case here
            existing_hashes.add(thash)

            logger.info(f"[Case] Created case {case.id}, +1 case, +1 scorable")

    return cb, counts
