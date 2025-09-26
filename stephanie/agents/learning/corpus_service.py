# stephanie/agents/learning/corpus_service.py
from __future__ import annotations
from typing import List, Dict, Any, Set, Optional
from stephanie.agents.learning.attribution import AttributionTracker
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

    @staticmethod
    def _corpus_key(it: Dict[str, Any]) -> str:
        return f"corpus:{str(it.get('id'))}"

    async def fetch(
        self,
        section_text: str,
        *,
        mask_keys: Optional[Set[str]] = None,
        allow_keys: Optional[Set[str]] = None,
        attribution_tracker: Optional[AttributionTracker] = None,  # NEW
    ) -> List[Dict[str, Any]]:
        try:
            res = self.chat_corpus(
                section_text,
                k=self.cfg.get("chat_corpus_k", 60),
                weights={"semantic": 0.6, "entity": 0.25, "domain": 0.15},
                include_text=True,
            )
            items = res.get("items", []) or []

            # NEW: allowlist/mask
            if allow_keys is not None:
                items = [it for it in items if self._corpus_key(it) in allow_keys]
            if mask_keys:
                mk = set(mask_keys)
                items = [it for it in items if self._corpus_key(it) not in mk]

            if attribution_tracker:
                for it in items:
                    k = self._corpus_key(it)
                    it["attribution_id"] = k
                    attribution_tracker.record_contribution(k, {
                        "source": "corpus",
                        "id": it.get("id"),
                        "score": float((it.get("score") or 0.0)),
                        "section_text": section_text[:240],
                        "retrieval_context": "section processing",
                    })

            # (annotate/analyze) unchanged
            try:
                if self.annotate: await self.annotate.run(context={"scorables": items})
                if self.analyze:  await self.analyze.run(context={"chats": items})
            except Exception as e:
                _logger.warning(f"Corpus annotate/analyze skipped: {e}")
            return items
        except Exception as e:
            _logger.warning(f"Chat corpus retrieval failed: {e}")
            return []
