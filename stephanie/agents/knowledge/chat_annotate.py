# stephanie/agents/knowledge/chat_annotate.py
"""
Chat Annotation Agent

This agent enriches chat conversations with domain classification and named entity recognition (NER)
to support the knowledge extraction and learning process. It processes conversations in bulk,
adding semantic annotations that enable better organization, retrieval, and analysis of chat content.

Key Features:
- Domain classification using both seed-based and goal-aware classifiers
- Named Entity Recognition with optional Knowledge Graph integration
- Progress tracking with tqdm progress bars
- Idempotent operation (only processes missing annotations by default)
- Comprehensive logging and reporting
"""

from __future__ import annotations

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.tools.domain_tool import annotate_conversation_domains
from stephanie.tools.scorable_classifier import ScorableClassifier
from stephanie.tools.turn_ner_tool import annotate_conversation_ner


class ChatAnnotateAgent(BaseAgent):
    """
    Agent that enriches chat conversations with domain and NER annotations.
    
    This agent processes one or multiple conversations, adding:
    1. Domain classifications (what the conversation is about)
    2. Named Entity Recognition (people, places, concepts mentioned)
    
    The annotations are stored directly in the database and can be used for
    improved retrieval, filtering, and knowledge extraction.
    """

    def __init__(self, cfg, memory, container, logger):
        # Initialize parent class with configuration, memory, container and logger
        super().__init__(cfg, memory, container, logger)
        
        # Initialize domain classifiers with configuration paths
        self.seed_classifier = ScorableClassifier(
            memory, logger, config_path=cfg.get("seed_config", "config/domain/seeds.yaml")
        )
        self.goal_classifier = ScorableClassifier(
            memory, logger, config_path=cfg.get("goal_config", "config/domain/goal_prompt.yaml")
        )

        # Domain classification settings
        self.max_k = int(cfg.get("max_domains_per_source", 3))  # Max domains per turn
        self.min_conf = float(cfg.get("min_confidence", 0.10))  # Minimum confidence threshold

        # Processing controls
        self.limit = int(cfg.get("limit", 1000))  # Maximum conversations to process
        self.only_missing = bool(cfg.get("only_missing", True))  # Skip already annotated turns
        self.force = bool(cfg.get("force", False))  # Force re-annotation of all turns
        self.progress_enabled = bool(cfg.get("progress", True))  # Enable/disable progress bars

    async def run(self, context: dict) -> dict:
        """
        Execute the annotation process on available chat conversations.
        
        Args:
            context: Execution context dictionary
            
        Returns:
            Updated context with annotation summary
        """
        # Retrieve conversations to process
        chats = self.memory.chats.get_all(limit=self.limit)

        kg = self.container.get("knowledge_graph") 

        # Pre-count turns that need processing (respects only_missing/force settings)
        def count_turns_for(chat_id: int, missing: str | None) -> int:
            rows = self.memory.chats.get_turn_texts_for_conversation(
                chat_id, only_missing=missing if (self.only_missing and not self.force) else None
            )
            return len(rows)

        # Calculate totals for progress tracking
        total_domains = sum(count_turns_for(c.id, "domains") for c in chats)
        total_ner = sum(count_turns_for(c.id, "ner") for c in chats)
        total_turns = total_domains + total_ner

        # Log start of annotation process
        self.logger.log("ChatAnnotateStart", {
            "conversations": len(chats),
            "turns_domains": total_domains,
            "turns_ner": total_ner,
            "turns_total": total_turns
        })

        # Create global progress bar for both domains and NER
        pbar = tqdm(
            total=total_turns or 1,
            desc="Annotating chats (domains+NER)",
            disable=not self.progress_enabled,
        )

        # Track progress for each annotation type
        dom_done = ner_done = 0

        def _bump_domains(n=1):
            """Callback to update progress for domain annotations"""
            nonlocal dom_done
            dom_done += n
            pbar.update(n)
            pbar.set_postfix({"domains": dom_done, "ner": ner_done})

        def _bump_ner(n=1):
            """Callback to update progress for NER annotations"""
            nonlocal ner_done
            ner_done += n
            pbar.update(n)
            pbar.set_postfix({"domains": dom_done, "ner": ner_done})

        # Initialize statistics counters
        totals = {
            "conversations": 0,
            "dom_seen": 0, "dom_updated": 0,
            "ner_seen": 0, "ner_updated": 0,
        }

        # Process each conversation
        for chat in chats:
            # Link conversation to a goal (using title as goal text)
            goal = self.memory.goals.get_or_create({
                "goal_text": chat.title,
                "description": f"Conversation imported on {chat.created_at}",
            }).to_dict()
            
            # Report goal linking
            self.report({
                "event": "goal_linked", 
                "conversation_id": chat.id,
                "goal_id": goal["id"], 
                "goal_text": goal["goal_text"]
            })

            # Annotate domains for this conversation
            dom_stats = annotate_conversation_domains(
                self.memory, 
                chat.id,
                seed_classifier=self.seed_classifier,
                goal_classifier=self.goal_classifier,
                goal=goal,
                max_k=self.max_k,
                min_conf=self.min_conf,
                only_missing=(self.only_missing and not self.force),
                progress_cb=_bump_domains,  # Progress callback
            )
            
            # Report domain annotation results
            self.report({
                "event": "domains_annotated", 
                "conversation_id": chat.id,
                "seen": dom_stats["seen"], 
                "updated": dom_stats["updated"]
            })

            # Annotate NER for this conversation
            ner_stats = annotate_conversation_ner(
                self.memory, 
                chat.id,
                kg=kg,  # Knowledge Graph for entity detection
                only_missing=(self.only_missing and not self.force),
                publish_to_kg=True,  # Publish entities to Knowledge Graph
                progress_cb=_bump_ner,  # Progress callback
            )
            
            # Report NER annotation results
            self.report({
                "event": "ner_annotated", 
                "conversation_id": chat.id,
                "seen": ner_stats["seen"], 
                "updated": ner_stats["updated"]
            })

            # Update totals
            totals["conversations"] += 1
            totals["dom_seen"] += dom_stats["seen"]
            totals["dom_updated"] += dom_stats["updated"]
            totals["ner_seen"] += ner_stats["seen"]
            totals["ner_updated"] += ner_stats["updated"]

            # Log progress for this conversation
            self.logger.log("ChatAnnotateProgress", {
                "conversation_id": chat.id, 
                **dom_stats, 
                **(ner_stats or {})
            })

        # Close progress bar and log completion
        pbar.close()
        self.logger.log("ChatAnnotateDone", {
            **totals, 
            "turns_domains": total_domains,
            "turns_ner": total_ner, 
            "turns_total": total_turns
        })
        
        # Add summary to context for downstream processing
        context["chat_annotation_summary"] = {
            **totals, 
            "turns_domains": total_domains,
            "turns_ner": total_ner, 
            "turns_total": total_turns
        }
        
        return context