# stephanie/agents/learning/corpus_service.py
from __future__ import annotations
from typing import List, Dict, Any
from stephanie.agents.knowledge.chat_analyze import ChatAnalyzeAgent
from stephanie.agents.knowledge.scorable_annotate import ScorableAnnotateAgent
from stephanie.tools.chat_corpus_tool import build_chat_corpus_tool

import logging

_logger = logging.getLogger(__name__)


class CorpusService:
    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self.chat_corpus = build_chat_corpus_tool(
            memory=memory, container=container, cfg=cfg.get("chat_corpus", {})
        )
        # Sub-agents / utilities
        self.annotate = ScorableAnnotateAgent(
            cfg.get("annotate", {}), memory, container, logger
        )
        self.analyze = ChatAnalyzeAgent(
            cfg.get("analyze", {}), memory, container, logger
        )

    async def fetch(self, section_text: str) -> List[Dict[str, Any]]:
        try:
            res = self.chat_corpus(
                section_text,
                k=self.cfg.get("chat_corpus_k", 60),
                weights={"semantic": 0.6, "entity": 0.25, "domain": 0.15},
                include_text=True,
            )
            items = res.get("items", []) or []
            try:
                if self.annotate:
                    await self.annotate.run(context={"scorables": items})
                if self.analyze:
                    await self.analyze.run(context={"chats": items})
            except Exception as e:
                _logger.warning(f"Corpus annotate/analyze skipped: {e}")
            return items
        except Exception as e:
            _logger.warning(f"Chat corpus retrieval failed: {e}")
            return []
