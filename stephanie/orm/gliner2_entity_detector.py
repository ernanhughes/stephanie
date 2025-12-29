# stephanie/orm/gliner2_entity_detector.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from gliner2 import GLiNER2  # pip install gliner2

log = logging.getLogger(__name__)

# You can tune this for papers later
DEFAULT_ENTITY_LABELS: Dict[str, str] = {
    "person": "Names of individual people (authors, researchers, etc.)",
    "organization": "Research labs, universities, companies, conferences, workshops",
    "method": "Named methods, algorithms, models, or techniques",
    "dataset": "Named datasets or benchmarks",
    "task": "Named tasks or problem settings (e.g. autonomous driving, RL, NER)",
    "metric": "Named evaluation metrics (e.g. F1, BLEU, accuracy)",
    "location": "Cities, countries, institutions with a location",
    "paper": "Paper titles or clearly marked work names",
}

class Gliner2EntityDetector:
    """
    Thin wrapper around GLiNER2 for Stephanie-style NER.

    Exposes:
        detect_entities(text: str) -> List[Dict[str, Any]]

    Each entity dict has:
        {
            "text": <entity text>,
            "type": <entity type/label>,
            "span": (start, end),
            "score": float
        }
    so it plugs into NerTool + NERRetriever without changes.
    """

    def __init__(
        self,
        model_name: str = "fastino/gliner2-large-v1",
        labels: Optional[Union[Dict[str, str], List[str]]] = None,
        threshold: float = 0.35,
    ) -> None:
        """
        Args:
            model_name: HF model id for GLiNER2.
            labels: either
                - dict[label] = description, or
                - list[str] of label names.
            threshold: confidence threshold passed to extract_entities.
        """
        self.model_name = model_name
        self.threshold = float(threshold)

        # Use your paper-friendly default schema unless caller overrides
        self.labels: Union[Dict[str, str], List[str]] = labels or DEFAULT_ENTITY_LABELS

        log.info(f"[GLiNER2] Loading model {model_name} …")
        # device is handled internally by gliner2; you can later add a device arg if needed
        self.model = GLiNER2.from_pretrained(model_name)
        log.info("[GLiNER2] Model loaded")

    # ------------------------------------------------------------------
    def detect_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Main API: GLiNER2 → Stephanie-style entity list.
        """
        text = text or ""
        if not text.strip():
            return []

        try:
            # GLiNER2 API:
            # results = model.extract_entities(text, labels, threshold=...)
            results = self.model.extract_entities(
                text,
                self.labels,                # dict or list both supported
                threshold=self.threshold,   # supported by GLiNER2 lib
            )
        except Exception as e:
            log.exception(f"[GLiNER2] extract_entities failed: {e}")
            return []

        raw = results.get("entities", {})
        entities: List[Dict[str, Any]] = []

        # GLiNER2 returns mapping: {label: [entity_text, ...]}
        for label, texts in raw.items():
            for ent_text in texts:
                ent_text = ent_text.strip()
                if not ent_text:
                    continue

                # Map the returned string back to spans in the original text.
                # For now: greedy left-to-right search, allowing multiple occurrences.
                start_search = 0
                found_at_least_once = False
                while True:
                    idx = text.find(ent_text, start_search)
                    if idx == -1:
                        break
                    found_at_least_once = True
                    start = idx
                    end = idx + len(ent_text)
                    entities.append(
                        {
                            "text": ent_text,
                            "type": label,
                            "span": (start, end),
                            # GLiNER2 Python API doesn’t expose scores by default;
                            # you can revise this once they’re available.
                            "score": 1.0,
                        }
                    )
                    start_search = end

                # If we somehow didn’t find a span (e.g. casing issues), still keep one
                if not found_at_least_once:
                    entities.append(
                        {
                            "text": ent_text,
                            "type": label,
                            "span": (0, 0),
                            "score": 1.0,
                        }
                    )

        return entities
