# stephanie/tools/domain_tool.py
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

from stephanie.tools.base_tool import BaseTool
from stephanie.tools.scorable_classifier import ScorableClassifier

log = logging.getLogger(__name__)

class DomainTool(BaseTool):
    """
    Tool that classifies a Scorable into domains and persists them.
    """

    name = "domain"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Initialize domain classifiers with configuration paths
        self.seed_classifier = ScorableClassifier(
            memory, logger, config_path=cfg.get("seed_config", "config/domain/seeds.yaml")
        )
        self.goal_classifier = ScorableClassifier(
            memory, logger, config_path=cfg.get("goal_config", "config/domain/goal_prompt.yaml")
        )

        # Domain classification settings
        self.min_conf = float(cfg.get("min_confidence", 0.10))  # Minimum confidence threshold
        self.max_k = int(cfg.get("max_k", 3))
        self.min_conf = float(cfg.get("min_conf", 0.10))
        self.store_to_memory = bool(cfg.get("store_to_memory", True))

    async def apply(self, scorable, context: dict):
        """
        Extract domains from the scorable text and store them.
        """
        text = scorable.text or ""
        if not text.strip():
            return scorable

        goal = context.get("goal", {})

        try:
            domains = self.memory.scorable_domains.get_by_scorable(str(scorable.id), scorable.target_type)            
            if (domains and not self.cfg.get("force", False)):
                scorable.meta["domains"] = domains
                log.debug(f"[DomainTool] {scorable.id}, domains retrieved from db.")
                return scorable


            domains = classify_text_domains(
                text,
                seed_classifier=self.seed_classifier,
                goal_classifier=self.goal_classifier,
                goal=goal,
                max_k=self.max_k,
                min_conf=self.min_conf,
            )

            # Persist to DB
            if self.store_to_memory:
                for d in domains:
                    self.memory.scorable_domains.insert({
                        "scorable_id": scorable.id,
                        "scorable_type": scorable.target_type,
                        "domain": d["domain"],
                        "score": d["score"],
                        "source": d["source"],
                    })

            scorable.meta["domains"] = domains
            log.debug(f"[DomainTool] {scorable.id} → {domains}")

        except Exception as e:
            log.error(f"[DomainTool] failed for scorable {scorable.id}: {e}")

        return scorable



def classify_text_domains(
    text: str,
    *,
    seed_classifier,
    goal_classifier=None,
    goal: Optional[dict] = None,
    max_k: int = 3,
    min_conf: float = 0.10,
) -> List[Dict]:
    out: List[Dict] = []
    try:
        seed_hits = seed_classifier.classify(text, top_k=max_k)
        for d, s in (seed_hits or []):
            if s >= min_conf:
                out.append({"domain": d, "score": float(s), "source": "seed"})
    except Exception:
        pass

    if goal and goal_classifier:
        try:
            goal_hits = goal_classifier.classify(text, context=goal, top_k=max_k)
            for d, s in (goal_hits or []):
                if s >= min_conf:
                    out.append({"domain": d, "score": float(s), "source": "goal"})
        except Exception:
            pass

    # Dedup by (domain,source)
    dedup: Dict[Tuple[str, str], Dict] = {}
    for item in out:
        k = (item["domain"], item["source"])
        if k not in dedup or item["score"] > dedup[k]["score"]:
            dedup[k] = item
    return list(dedup.values())


def annotate_conversation_domains(
    memory,
    conversation_id: int,
    *,
    seed_classifier,
    goal_classifier=None,
    goal: Optional[dict] = None,
    max_k: int = 3,
    min_conf: float = 0.10,
    only_missing: bool = True,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Dict[str, int]:

    turns = memory.chats.get_turn_texts_for_conversation(
        conversation_id,
        only_missing=("domains" if only_missing else None)
    )

    seen = updated = 0
    for t in turns:
        seen += 1
        txt = f"USER: {t['user_text']}\nASSISTANT: {t['assistant_text']}".strip()
        if not txt:
            progress_cb and progress_cb(1)
            continue

        payload = classify_text_domains(
            txt,
            seed_classifier=seed_classifier,
            goal_classifier=goal_classifier,
            goal=goal,
            max_k=max_k,
            min_conf=min_conf,
        )
        memory.chats.set_turn_domains(t["id"], payload)
        updated += 1
        progress_cb and progress_cb(1)

    return {"seen": seen, "updated": updated}


def classify_and_store_domains(self, text: str, scorable_id: int, scorable_type: str) -> None:
    """Classify the paper and assign domains."""
    results = self.domain_classifier.classify(text, self.top_k_domains, self.min_classification_score)
    for domain, score in results:
        self.memory.scorable_domains.insert(
            {
                "scorable_id": scorable_id,
                "scorable_type": scorable_type,
                "domain": domain,
                "score": score,
            }
        )
        log.debug(f"DomainAssigned: domain={domain}, score={score}")