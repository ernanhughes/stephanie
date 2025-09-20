# stephanie/agents/chat_annotate_agent.py
from __future__ import annotations

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.tools.turn_domains_tool import annotate_conversation_domains
from stephanie.tools.turn_ner_tool import annotate_conversation_ner


class ChatAnnotateAgent(BaseAgent):
    """
    Runs NER + Domains enrichment on chat conversations (one or many).
    Mirrors ChatImportAgent style, now with tqdm progress.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        # Domain classifiers
        self.seed_classifier = ScorableClassifier(
            memory, logger, config_path=cfg.get("seed_config", "config/domain/seeds.yaml")
        )
        self.goal_classifier = ScorableClassifier(
            memory, logger, config_path=cfg.get("goal_config", "config/domain/goal_prompt.yaml")
        )


        # Domain settings
        self.max_k = int(cfg.get("max_domains_per_source", 3))
        self.min_conf = float(cfg.get("min_confidence", 0.10))

        # Controls
        self.limit = int(cfg.get("limit", 1000))
        self.only_missing = bool(cfg.get("only_missing", True))  # idempotent by default
        self.force = bool(cfg.get("force", False))
        self.progress_enabled = bool(cfg.get("progress", True))   # enable/disable tqdm

    async def run(self, context: dict) -> dict:
        chats = self.memory.chats.get_all(limit=self.limit)

        # Pre-count total turns we intend to touch (respects only_missing/force)
        def count_turns_for(chat_id: int, missing: str | None) -> int:
            rows = self.memory.chats.get_turn_texts_for_conversation(
                chat_id, only_missing=missing if (self.only_missing and not self.force) else None
            )
            return len(rows)

        total_domains = sum(count_turns_for(c.id, "domains") for c in chats)
        total_ner     = sum(count_turns_for(c.id, "ner")     for c in chats)
        total_turns   = total_domains + total_ner

        self.logger.log("ChatAnnotateStart", {
            "conversations": len(chats),
            "turns_domains": total_domains,
            "turns_ner": total_ner,
            "turns_total": total_turns
        })

        # Initialize KG once if using kg backend
        kg = self.container.get("knowledge_graph") 
        if kg:
            kg.initialize()

        # Single global progress bar (covers domains + ner)
        pbar = tqdm(
            total=total_turns or 1,
            desc="Annotating chats (domains+NER)",
            disable=not self.progress_enabled,
        )

        # Per-phase counters (for set_postfix)
        dom_done = ner_done = 0

        def _bump_domains(n=1):
            nonlocal dom_done
            dom_done += n
            pbar.update(n)
            pbar.set_postfix({"domains": dom_done, "ner": ner_done})

        def _bump_ner(n=1):
            nonlocal ner_done
            ner_done += n
            pbar.update(n)
            pbar.set_postfix({"domains": dom_done, "ner": ner_done})

        totals = {
            "conversations": 0,
            "dom_seen": 0, "dom_updated": 0,
            "ner_seen": 0, "ner_updated": 0,
        }

        for chat in chats:
            # Link a goal to the conversation (title as goal text)
            goal = self.memory.goals.get_or_create({
                "goal_text": chat.title,
                "description": f"Conversation imported on {chat.created_at}",
            }).to_dict()
            self.report({"event": "goal_linked", "conversation_id": chat.id,
                         "goal_id": goal["id"], "goal_text": goal["goal_text"]})

            # ---- Domains (with progress callback + only_missing) ----
            dom_stats = annotate_conversation_domains(
                self.memory, chat.id,
                seed_classifier=self.seed_classifier,
                goal_classifier=self.goal_classifier,
                goal=goal,
                max_k=self.max_k,
                min_conf=self.min_conf,
                only_missing=(self.only_missing and not self.force),
                progress_cb=_bump_domains,   # ✅ tqdm callback
            )
            self.report({"event": "domains_annotated", "conversation_id": chat.id,
                         "seen": dom_stats["seen"], "updated": dom_stats["updated"]})

            # ---- NER (with progress callback + only_missing) ----
            ner_stats = annotate_conversation_ner(
                self.memory, chat.id,
                kg=kg,                       # pass KG so we share detector + publish
                only_missing=(self.only_missing and not self.force),
                publish_to_kg=True,
                progress_cb=_bump_ner,       # ✅ tqdm callback
            )
            self.report({"event": "ner_annotated", "conversation_id": chat.id,
                         "seen": ner_stats["seen"], "updated": ner_stats["updated"]})

            totals["conversations"] += 1
            totals["dom_seen"] += dom_stats["seen"]; totals["dom_updated"] += dom_stats["updated"]
            totals["ner_seen"] += ner_stats["seen"]; totals["ner_updated"] += ner_stats["updated"]

            # periodic log
            self.logger.log("ChatAnnotateProgress", {
                "conversation_id": chat.id, **dom_stats, **(ner_stats or {})
            })

        pbar.close()
        self.logger.log("ChatAnnotateDone", {**totals, "turns_domains": total_domains,
                                             "turns_ner": total_ner, "turns_total": total_turns})
        context["chat_annotation_summary"] = {**totals, "turns_domains": total_domains,
                                              "turns_ner": total_ner, "turns_total": total_turns}
        return context
