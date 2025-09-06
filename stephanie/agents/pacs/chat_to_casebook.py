# stephanie/agents/pacs/chat_to_casebook.py
from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL
from stephanie.models.chat import ChatConversationORM
from stephanie.models.casebook import CaseBookORM
from datetime import datetime

class ChatToCaseBookAgent(BaseAgent):
    """
    Converts ChatConversationORMs into CaseBooks + Cases + Scorables.
    Hybrid mapping: one conversation = one case with full-text scorable.
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.limit = cfg.get("limit", 30)  # process top-N convos
        self.metric = cfg.get("metric", "messages")  # "turns" or "messages"

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, {})
        self.logger.log("ChatToCaseBookStart", {
            "metric": self.metric,
            "limit": self.limit,
            "goal": goal.get("goal_text") if goal else None
        })

        # --- 1. Select top conversations
        top_convs = self.memory.chats.get_top_conversations(
            limit=self.limit, by=self.metric
        )

        casebooks_created = []
        for conv, count in top_convs:
            try:
                cb = self._convert_conversation(conv)
                casebooks_created.append(cb)
                self.logger.log("ChatToCaseBookConverted", {
                    "conversation_id": conv.id,
                    "title": conv.title,
                    "metric_count": count,
                    "casebook_id": cb.id
                })
            except Exception as e:
                self.logger.log("ChatToCaseBookError", {
                    "conversation_id": conv.id,
                    "title": conv.title,
                    "error": str(e)
                })

        context["casebooks_created"] = [cb.id for cb in casebooks_created]
        return context

    def _convert_conversation(self, conv: ChatConversationORM) -> CaseBookORM:
        """
        Internal: convert one conversation into a CaseBook + Case + Scorable.
        """
        # --- 1. CaseBook
        cb = self.memory.casebooks.ensure_casebook(
            name=f"{conv.title}",
            description=f"Imported chat conversation: {conv.id} - {conv.title}"
        )

        # --- 2. Goal (tie to conversation topic)
        goal = self.memory.goals.get_or_create({
            "goal_text": conv.title,
            "description": f"Conversation imported on {conv.created_at or datetime.now()}"
        }).to_dict()

        # --- 3. Build case
        full_text = self._conversation_to_text(conv)
        case = self.memory.casebooks.add_case(
            casebook_id=cb.id,
            goal_id=goal["id"],
            goal_text=goal["goal_text"],
            agent_name="chat_to_casebook",
            prompt_text=conv.title,
            response_texts=full_text,
            scorables=[{
                "text": full_text,
                "role": "conversation",
                "source": "chat",
                "meta": {"conversation_id": conv.id}
            }]
        )

        return cb

    def _conversation_to_text(self, conv: ChatConversationORM) -> str:
        """
        Serialize conversation into plain text (user/assistant turns).
        """
        lines = []
        for msg in conv.messages:
            ts = msg.created_at.isoformat() if msg.created_at else "?"
            lines.append(f"[{ts}] {msg.role.upper()}: {msg.text.strip()}")
        return "\n".join(lines)
