# stephanie/tools/turn_domains_tool.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional

def classify_text_domains(
    text: str,
    *,
    seed_classifier,
    goal_classifier=None,
    goal: Optional[str] = None,
    max_k: int = 3,
    min_conf: float = 0.10,
) -> List[Dict]:
    """
    Returns a list of {domain, score, source}.
    - seed: controlled ontology
    - goal: prompt-driven, if provided
    """
    out: List[Dict] = []

    # Seed
    try:
        seed_hits = seed_classifier.classify(text, top_k=max_k)
        for d, s in (seed_hits or []):
            if s >= min_conf:
                out.append({"domain": d, "score": float(s), "source": "seed"})
    except Exception:
        pass

    # Goal
    if goal and goal_classifier:
        try:
            goal_hits = goal_classifier.classify(text, context=goal, top_k=max_k)
            for d, s in (goal_hits or []):
                if s >= min_conf:
                    out.append({"domain": d, "score": float(s), "source": "goal"})
        except Exception:
            pass

    # Dedup by (domain, source) keeping highest score
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
    goal: Optional[str] = None,
    max_k: int = 3,
    min_conf: float = 0.10,
) -> Dict[str, int]:
    turns = memory.chats.get_turn_texts_for_conversation(conversation_id)
    seen = updated = 0
    for t in turns:
        seen += 1
        txt = f"USER: {t['user_text']}\nASSISTANT: {t['assistant_text']}".strip()
        if not txt:
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
    return {"seen": seen, "updated": updated}
