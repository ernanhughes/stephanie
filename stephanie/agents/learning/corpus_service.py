# stephanie/agents/learning/corpus_service.py
from typing import List, Dict, Any, Optional, Set
from .attribution import AttributionTracker  # NEW
import logging
_logger = logging.getLogger(__name__)

class CorpusService:
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

            # NEW: attribution breadcrumbs
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
