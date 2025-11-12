# stephanie/agents/learning/corpus_retriever.py
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set

from stephanie.agents.knowledge.chat_analyze import ChatAnalyzeAgent
from stephanie.agents.learning.attribution import AttributionTracker
from stephanie.agents.maintenance.scorable_annotate import \
    ScorableAnnotateAgent
from stephanie.tools.chat_corpus_tool import build_chat_corpus_tool

log = logging.getLogger(__name__)


class CorpusRetriever:
    """
    Retrieval with optional tag-aware filtering/boosting.
    Works even if underlying chat_corpus tool doesn't support tags natively
    (falls back to local filter/boost using item.meta/conversation tags).
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        self.chat_corpus = build_chat_corpus_tool(
            memory=memory, container=container, cfg=cfg.get("chat_corpus", {})
        )

        # Sub-agents / utilities
        self.annotate = ScorableAnnotateAgent(cfg.get("annotate", {}), memory, container, logger)
        self.analyze = ChatAnalyzeAgent(cfg.get("analyze", {}), memory, container, logger)

        # --- Tag controls (config defaults, can be overridden per fetch) ---
        tf = cfg.get("tag_filters", {}) or {}
        self.default_tag_any:   List[str] = list(tf.get("any", []) or [])
        self.default_tag_all:   List[str] = list(tf.get("all", []) or [])
        self.default_tag_none:  List[str] = list(tf.get("none", []) or [])
        self.default_tag_mode:  str       = str(cfg.get("tag_mode", "require")).lower()  # "require" | "prefer"
        self.default_tag_boost: float     = float(cfg.get("tag_boost", 0.25))            # used when mode="prefer"

        # Optional: restrict to a dedicated sub-index / corpus
        self.default_corpus_id: Optional[str] = cfg.get("corpus_id")

    @staticmethod
    def _corpus_key(it: Dict[str, Any]) -> str:
        return f"corpus:{str(it.get('id'))}"

    @staticmethod
    def _tags_from_item(it: Dict[str, Any]) -> List[str]:
        """
        Try common places where conversation/message tags might be stored.
        Adjust to your actual schema if needed.
        """
        meta = (it.get("meta") or {})
        # Prefer explicit conversation tags if present
        conv = meta.get("conversation") or {}
        if isinstance(conv, dict) and isinstance(conv.get("tags"), list):
            return list(conv.get("tags") or [])
        # Fallbacks
        if isinstance(meta.get("tags"), list):
            return list(meta.get("tags") or [])
        if isinstance(it.get("conversation_tags"), list):
            return list(it.get("conversation_tags") or [])
        return []

    def _ensure_tags(self, it: Dict[str, Any]) -> List[str]:
        return self._tags_from_item(it)

    @staticmethod
    def _match_tags(
        tags: Iterable[str],
        any_of: Iterable[str],
        all_of: Iterable[str],
        none_of: Iterable[str],
    ) -> bool:
        """Return True if tags satisfy (any | all) and do not include excluded."""
        tags = set(t.lower() for t in (tags or []))
        any_of = set(t.lower() for t in (any_of or []))
        all_of = set(t.lower() for t in (all_of or []))
        none_of = set(t.lower() for t in (none_of or []))

        if none_of and tags & none_of:
            return False
        if all_of and not all_of.issubset(tags):
            return False
        if any_of and not (tags & any_of):
            return False
        # If no constraints, accept
        return True if (any_of or all_of or none_of) else True

    async def fetch(
        self,
        section_text: str,
        *,
        mask_keys: Optional[Set[str]] = None,
        allow_keys: Optional[Set[str]] = None,
        attribution_tracker: Optional[AttributionTracker] = None,
        # --- per-call tag overrides ---
        tags_any: Optional[List[str]] = None,
        tags_all: Optional[List[str]] = None,
        tags_none: Optional[List[str]] = None,
        tag_mode: Optional[str] = None,     # "require" (hard filter) or "prefer" (soft boost)
        tag_boost: Optional[float] = None,  # only used if tag_mode="prefer"
        corpus_id: Optional[str] = None,    # restrict to a specific corpus/index if supported
    ) -> List[Dict[str, Any]]:
        tag_mode = (tag_mode or self.default_tag_mode).lower()
        tag_boost = float(tag_boost if tag_boost is not None else self.default_tag_boost)

        tags_any  = list(tags_any  if tags_any  is not None else self.default_tag_any)
        tags_all  = list(tags_all  if tags_all  is not None else self.default_tag_all)
        tags_none = list(tags_none if tags_none is not None else self.default_tag_none)

        # Try to pass tag/corpus hints through to the tool if it supports them
        # We'll catch TypeError and fall back to local filtering/boosting.
        tool_kwargs = dict(
            k=self.cfg.get("chat_corpus_k", 60),
            weights={"semantic": 0.6, "entity": 0.25, "domain": 0.15},
            include_text=True,
        )
        if corpus_id or self.default_corpus_id:
            tool_kwargs["corpus_id"] = corpus_id or self.default_corpus_id
        # Hypothetical API; safe to ignore by try/except
        if tags_any or tags_all or tags_none:
            tool_kwargs["filters"] = {
                "tags_any": tags_any,
                "tags_all": tags_all,
                "tags_none": tags_none,
                "mode": tag_mode,  # in case tool supports it
            }

        try:
            res = self.chat_corpus(section_text, **tool_kwargs)
        except TypeError:
            # Old tool signature, retry without unsupported kwargs
            log.debug("chat_corpus tool does not support filters/corpus_id; falling back to local filtering.")
            tool_kwargs.pop("filters", None)
            tool_kwargs.pop("corpus_id", None)
            res = self.chat_corpus(section_text, **tool_kwargs)

        items = res.get("items", []) or []

        # Allowlist/mask
        if allow_keys is not None:
            ak = set(allow_keys)
            items = [it for it in items if self._corpus_key(it) in ak]
        if mask_keys:
            mk = set(mask_keys)
            items = [it for it in items if self._corpus_key(it) not in mk]

        # If tool didn’t natively filter by tags (or we want boost), do it here
        if tags_any or tags_all or tags_none:
            if tag_mode == "require":
                kept = []
                for it in items:
                    tgs = self._ensure_tags(it)
                    if self._match_tags(tgs, tags_any, tags_all, tags_none):
                        kept.append(it)
                items = kept
            elif tag_mode == "prefer":
                # Soft boost items that match; keep others
                for it in items:
                    tgs = self._ensure_tags(it)
                    if self._match_tags(tgs, tags_any, tags_all, tags_none):
                        try:
                            it["score"] = float(it.get("score", 0.0)) + float(tag_boost)
                        except Exception:
                            # leave score untouched on failure
                            pass
                # re-sort by the adjusted score (desc)
                items.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        # Attribution tracking
        if attribution_tracker:
            for it in items:
                k = self._corpus_key(it)
                it["attribution_id"] = k
                try:
                    attribution_tracker.record_contribution(
                        k,
                        {
                            "source": "corpus",
                            "id": it.get("id"),
                            "score": float((it.get("score") or 0.0)),
                            "section_text": section_text[:240],
                            "retrieval_context": "section processing",
                            "tags": self._ensure_tags(it),
                            "corpus_id": corpus_id or self.default_corpus_id,
                        },
                    )
                except Exception:
                    # never break retrieval on attribution logging
                    pass

        # (annotate/analyze) — best-effort
        try:
            if self.annotate:
                await self.annotate.run(context={"scorables": items})
            if self.analyze:
                await self.analyze.run(context={"chats": items})
        except Exception as e:
            log.warning("Corpus annotate/analyze skipped: %s", e)

        return items
