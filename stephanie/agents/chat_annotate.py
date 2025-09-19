# stephanie/agents/chat_annotate_agent.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.tools.turn_domains_tool import annotate_conversation_domains
from stephanie.tools.turn_ner_tool import annotate_conversation_ner


class ChatAnnotateAgent(BaseAgent):
    """
    Runs NER + Domains enrichment on chat conversations (one or many).
    Mirrors ChatImportAgent style.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        # Classifiers for domains
        self.seed_classifier = ScorableClassifier(
            memory,
            logger,
            config_path=cfg.get("seed_config", "config/domain/seeds.yaml"),
        )
        self.goal_classifier = ScorableClassifier(
            memory,
            logger,
            config_path=cfg.get(
                "goal_config", "config/domain/goal_prompt.yaml"
            ),
        )

        # NER settings
        self.ner_backend = cfg.get("ner_backend", "kg")
        self.spacy_model = cfg.get("spacy_model", "en_core_web_sm")
        self.hf_model = cfg.get("hf_model", "dslim/bert-base-NER")
        self.only_missing = bool(cfg.get("only_missing", True))
        self.force = bool(cfg.get("force", False))
        self.publish_to_kg = bool(cfg.get("publish_to_kg", True))

        # Domain settings
        self.max_k = int(cfg.get("max_domains_per_source", 3))
        self.min_conf = float(cfg.get("min_confidence", 0.10))
        
        # Sweep limit for batch mode
        self.limit = int(cfg.get("limit", 1000))

    async def run(self, context: dict) -> dict:
        chats = self.memory.chats.get_all(limit=self.limit)

        self.logger.log(
            "ChatAnnotateStart",
            {"conversations": len(chats), "limit": self.limit},
        )

        total = {
            "conversations": 0,
            "ner_seen": 0,
            "ner_updated": 0,
            "dom_seen": 0,
            "dom_updated": 0,
        }

        kg = self.container.get("knowledge_graph")
        kg.initialize()  # ensure ready
        
        for chat in chats:
            # Create/link goal for this conversation
            goal = self.memory.goals.get_or_create(
                {
                    "goal_text": chat.title,
                    "description": f"Conversation imported on {chat.created_at}",
                }
            ).to_dict()
            self.report(
                {
                    "event": "goal_linked",
                    "conversation_id": chat.id,
                    "goal_id": goal["id"],
                    "goal_text": goal["goal_text"],
                }
            )

            # Domains
            dom_stats = annotate_conversation_domains(
                self.memory,
                chat.id,
                seed_classifier=self.seed_classifier,
                goal_classifier=self.goal_classifier,
                goal=goal,
                max_k=self.max_k,
                min_conf=self.min_conf,
            )

            self.report(
                {
                    "event": "domains_annotated",
                    "conversation_id": chat.id,
                    "seen": dom_stats["seen"],
                    "updated": dom_stats["updated"],
                }
            )

            # NER
            ner_stats = annotate_conversation_ner(
                self.memory,
                chat.id,
                kg=kg,
                publish_to_kg=True,)
            self.report(
                {
                    "event": "ner_annotated",
                    "conversation_id": chat.id,
                    "seen": ner_stats["seen"],
                    "updated": ner_stats["updated"],
                }
            )

            total["conversations"] += 1
            total["dom_seen"] += dom_stats["seen"]
            total["dom_updated"] += dom_stats["updated"]
            total["ner_seen"] += ner_stats["seen"]
            total["ner_updated"] += ner_stats["updated"]

            self.logger.log(
                "ChatAnnotateProgress",
                {"conversation_id": chat.id, **dom_stats, **ner_stats},
            )

        self.logger.log("ChatAnnotateDone", total)
        context["chat_annotation_summary"] = total
        return context
