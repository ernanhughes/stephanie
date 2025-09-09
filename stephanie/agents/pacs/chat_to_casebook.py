"""
Chat to CaseBook Conversion Agent

This module implements an agent that converts chat conversations into structured CaseBooks
with associated cases and scorables. It transforms raw conversational data from multiple
AI agents into learnable trajectories for Stephanie's continuous improvement system.

Key Features:
- Multiple granularity levels: conversation, turns, or messages
- Integration with chat memory and casebook systems
- Automated goal creation based on conversation topics
- Comprehensive reporting and error handling

The agent supports three granularity levels:
1. Conversation: One case per entire conversation (single scorable)
2. Turns: One case per user→assistant turn
3. Messages: One case per individual message

This transformation is crucial for creating the training data that enables
Stephanie to learn from human-AI collaboration patterns.
"""

from datetime import datetime

# stephanie/agents/pacs/chat_to_casebook.py
from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL
from stephanie.models.casebook import CaseBookORM
from stephanie.models.chat import ChatConversationORM


class ChatToCaseBookAgent(BaseAgent):
    """
    Converts ChatConversationORMs into structured CaseBooks with Cases and Scorables.
    
    This agent transforms raw conversational data from multiple AI agents (OpenAI, Qwen,
    DeepSeek, Gemini) into structured learning materials for Stephanie's continuous
    improvement system. It creates CaseBooks that capture the trajectory of conversations,
    enabling Stephanie to learn from patterns of human-AI collaboration.
    
    Parameters:
        cfg (dict): Configuration dictionary with:
            - limit: Number of top conversations to process (default: 10)
            - metric: Ranking metric for conversations ("messages" or "turns")
            - granularity: Processing level ("conversation", "turns", or "messages")
        memory: Stephanie's memory system instance
        logger: Logger instance for reporting
        
    The agent supports three granularity levels:
    1. "conversation": One case per entire conversation (single scorable)
    2. "turns": One case per user→assistant turn (preserves dialogue structure)
    3. "messages": One case per individual message (maximum granularity)
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.limit = cfg.get("limit", 10)      # Number of top conversations to process
        self.metric = cfg.get("metric", "messages")  # Ranking metric: "messages" or "turns"
        self.granularity = cfg.get("granularity", "conversation")  
        # Processing granularity: "conversation", "turns", or "messages"

    async def run(self, context: dict) -> dict:
        """
        Main execution method that converts conversations to CaseBooks.
        
        Args:
            context: Execution context containing goal information and other state
            
        Returns:
            Updated context with list of created CaseBook IDs
            
        Process:
            1. Selects top conversations based on configured metric
            2. Converts each conversation to a CaseBook with appropriate granularity
            3. Creates associated goals based on conversation topics
            4. Generates scorables at the specified granularity level
            5. Reports progress and errors throughout the process
        """
        goal = context.get(GOAL, {})
        self.report({
            "event": "start",
            "metric": self.metric,
            "limit": self.limit,
            "granularity": self.granularity,
            "goal": goal.get("goal_text") if goal else None
        })

        # --- 1. Select top conversations based on configured metric
        top_convs = self.memory.chats.get_top_conversations(
            limit=self.limit, by=self.metric
        )
        self.report({
            "event": "selected_conversations",
            "count": len(top_convs),
            "metric": self.metric
        })

        casebooks_created = []
        for idx, (conv, count) in enumerate(top_convs, 1):
            try:
                cb = self._convert_conversation(conv)
                casebooks_created.append(cb)
                self.report({
                    "event": "converted",
                    "conversation_id": conv.id,
                    "title": conv.title,
                    "metric_count": count,
                    "casebook_id": cb.id,
                    "index": idx,
                    "total": len(top_convs)
                })
            except Exception as e:
                self.report({
                    "event": "error",
                    "conversation_id": conv.id,
                    "title": conv.title,
                    "error": str(e),
                    "index": idx,
                    "total": len(top_convs)
                })

        self.report({
            "event": "completed",
            "casebooks_created": len(casebooks_created)
        })

        context["casebooks_created"] = [cb.id for cb in casebooks_created]
        return context

    def _convert_conversation(self, conv: ChatConversationORM) -> CaseBookORM:
        """
        Convert a single conversation into a CaseBook with Cases and Scorables.
        
        Args:
            conv: The ChatConversationORM to convert
            
        Returns:
            Created CaseBookORM instance
            
        Process:
            1. Creates or retrieves a CaseBook for the conversation
            2. Creates or links a goal based on the conversation topic
            3. Generates scorables based on configured granularity
            4. Creates cases with associated scorables
        """
        # --- 1. Create or retrieve CaseBook for this conversation
        cb = self.memory.casebooks.ensure_casebook(
            name=f"{conv.title}",
            description=f"Imported chat conversation: {conv.id} - {conv.title}"
        )
        self.report({
            "event": "casebook_created",
            "conversation_id": conv.id,
            "casebook_id": cb.id,
            "title": conv.title
        })

        # --- 2. Create or link goal based on conversation topic
        goal = self.memory.goals.get_or_create({
            "goal_text": conv.title,
            "description": f"Conversation imported on {conv.created_at or datetime.now()}"
        }).to_dict()
        self.report({
            "event": "goal_linked",
            "conversation_id": conv.id,
            "goal_id": goal["id"],
            "goal_text": goal["goal_text"]
        })

        # --- 3. Generate scorables based on configured granularity
        if self.granularity == "conversation":
            # Single scorable for the entire conversation
            scorables = [self.memory.chats.scorable_from_conversation(conv)]
        elif self.granularity == "turns":
            # One scorable per user→assistant turn
            turns = self.memory.chats.get_turns_for_conversation(conv.id)
            scorables = [self.memory.chats.scorable_from_turn(t) for t in turns]
        elif self.granularity == "messages":
            # One scorable per individual message
            msgs = self.memory.chats.get_messages(conv.id)
            scorables = [self.memory.chats.scorable_from_message(m) for m in msgs]
        else:
            raise ValueError(f"Unsupported granularity: {self.granularity}")

        self.report({
            "event": "scorables_generated",
            "conversation_id": conv.id,
            "granularity": self.granularity,
            "count": len(scorables)
        })

        # --- 4. Create cases with associated scorables
        for sc in scorables:
            case = self.memory.casebooks.add_case(
                prompt_text=conv.title, 
                casebook_id=cb.id,
                goal_id=goal["id"],
                agent_name="chat_to_casebook",
                scorables=[{
                    "scorable_id": sc.id,              
                    "scorable_type": sc.target_type,   
                    "text": sc.text,
                    "source": self.name,
                    "meta": {
                        "conversation_id": conv.id,
                        **(sc.meta or {})
                    },
                }]
            )
            self.report({
                "event": "case_created",
                "case_id": case.id,
                "casebook_id": cb.id,
                "scorable_id": sc.id,
                "conversation_id": conv.id
            })

        return cb