# stephanie/tools/ner_tool.py
from __future__ import annotations

import logging

from stephanie.models.gliner2_entity_detector import Gliner2EntityDetector
from stephanie.tools.base_tool import BaseTool

log = logging.getLogger(__name__)

class NerTool(BaseTool):
    """
    Tool for NER extraction + persistence.
    Wraps DB hydration + EntityDetector inference.
    """

    name = "ner"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        self.use_model = bool(cfg.get("use_model", True))
        self.persist = bool(cfg.get("persist", True))
        self.hydrate = bool(cfg.get("hydrate", True))
        self.force = bool(cfg.get("force", False))  # recompute even if DB exists
        self.min_conf = float(cfg.get("min_conf", 0.05))

        # The low-level model
        model_name = cfg.get("model_name", "fastino/gliner2-large-v1")
        labels = cfg.get("labels")        # could be list or dict
        threshold = float(cfg.get("threshold", 0.35))

        self.detector = Gliner2EntityDetector(
            model_name=model_name,
            labels=labels,
            threshold=threshold,
        )


    # ------------------------------------------------------------------
    async def apply(self, scorable, context):
        """
        Main entry point: load from DB, or compute, then store.
        """
        text = scorable.text or ""
        if not text.strip():
            return scorable

        # 1. Try DB hydration
        ner_from_db = None
        if self.hydrate and not self.force:
            ner_from_db = self._load_from_db(scorable)

        if ner_from_db:
            scorable.meta["ner"] = ner_from_db
            return scorable

        # 2. Run NER model if allowed
        if self.use_model and self.detector:
            ner = self._run_detector(text)
        else:
            ner = []

        # 3. Persist results
        if self.persist and ner:
            self._persist_entities(scorable, ner)

        # Attach to scorable.meta
        scorable.meta["ner"] = ner
        return scorable

    # ------------------------------------------------------------------
    def _load_from_db(self, scorable):
        """
        Hydrate entities from scorable_entities table.
        """
        try:
            rows = self.memory.scorable_entities.find(
                scorable_id=str(scorable.id),
                scorable_type=scorable.target_type,
            )
            if not rows:
                return None

            ner = [{
                "text": r["entity_text"],
                "type": r.get("entity_type"),
                "span": [r.get("start"), r.get("end")],
                "score": r.get("score", 1.0),
                "source": "db",
            } for r in rows]

            return ner

        except Exception as e:
            log.error(f"[NER] DB hydration failed for {scorable.id}: {e}")
            return None

    def detect(self, text: str):
        """
        Public method to detect entities in text.
        """
        if not self.use_model or not self.detector:
            return []

        return self._run_detector(text)    


    # ------------------------------------------------------------------
    def _run_detector(self, text: str):
        """
        Execute NER model and return standardized entity records.
        """
        try:
            raw = self.detector.detect_entities(text) or []
        except Exception as e:
            log.warning(f"[NER] detector failed: {e}")
            return []

        out = []
        for ent in raw:
            score = float(ent.get("score", 1.0))
            if score < self.min_conf:
                continue

            out.append({
                "text": ent.get("text"),
                "type": ent.get("type"),
                "span": ent.get("span") or [None, None],
                "score": score,
                "source": "model",
            })

        return out

    # ------------------------------------------------------------------
    def _persist_entities(self, scorable, entities):
        """
        Insert entities into scorable_entities table.
        """
        for ent in entities:
            try:
                self.memory.scorable_entities.insert({
                    "scorable_id": scorable.id,
                    "scorable_type": scorable.target_type,
                    "entity_text": ent["text"],
                    "entity_type": ent["type"],
                    "start": ent["span"][0],
                    "end": ent["span"][1],
                    "score": ent.get("score", 1.0),
                })
            except Exception as e:
                log.error(f"[NER] persist failed for {scorable.id}: {e}")
