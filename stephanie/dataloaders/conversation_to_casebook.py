# dataloaders/conversation_to_casebook.py
from stephanie.memory.memory_tool import MemoryTool
from stephanie.models.casebook import CaseBookORM, CaseORM, CaseScorableORM
from stephanie.models.goal import GoalORM
from datetime import datetime
import re

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

def conversation_to_casebook(memory: MemoryTool, bundle: list, goal_id: str = None, tags: list = None):
    """
    Creates CaseBook, Case, and Scorable ORM instances from a conversation bundle.
    """
    # 1) Ensure goal
    goal = memory.goals.get_by_id(goal_id) if goal_id else None
    if not goal:
        goal = memory.goals.create({"text": summarize_goal(bundle)})
    
    cb = memory.casebooks.create({
        "name": make_name(bundle),
        "goal_id": goal.id,
        "tags": tags or ["chat-import", "zero-model", "pacs"]
    })
    
    # 2) Create cases and scorables
    for u, a in user_assistant_pairs(bundle):
        # Semantic dedupe check (a simple version)
        if memory.embedding.search_related_scorables(query=a["text"], top_k=1, min_score=0.95):
            continue # Skip if a very similar scorable already exists

        case = memory.cases.create({
            "prompt_text": u["text"],
            "casebook_id": cb.id
        })
        
        memory.scorables.create({
            "text": a["text"],
            "target_type": "message",
            "source": "chat",
            "case_id": case.id
        })
    
    # Placeholder for outcome linking
    # attach_outcomes_if_any(memory, cb, bundle)
    
    return cb