from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from stephanie.components.gap.risk.explore.runner import ExplorationRunner
from stephanie.components.gap.risk.orchestrator import GapRiskOrchestrator
from stephanie.jobs.turn_cursor import TurnCursor

OUT_DIR = Path("./data/runs/overnight").resolve()

def _cfg(app) -> dict:
    return getattr(app.state, "config", {}) or {}

async def select_seeds_from_db(container: Any, *, k: int, min_len: int,
                               require_nonempty_ner: bool, cursor_path: str):
    memory = container.memory                     # ✅ single gateway
    chats = memory.chats                          # stephanie.memory.chat_store.ChatStore
    # newest conversations first
    convs = chats.get_all(limit=500)
    conv_ids = [c.id for c in convs]

    cur = TurnCursor(cursor_path)
    if not cur.state.get("conv_ids"):
        cur.set_conversations(conv_ids)

    seeds = []
    attempts, max_attempts = 0, max(5 * k, 50)
    while len(seeds) < k and attempts < max_attempts:
        conv_id, offset = cur.next_batch_hint()
        if conv_id is None:
            break
        rows = chats.get_turn_texts_for_conversation(
            conv_id,
            limit=20,
            offset=offset,
            order_by_id=True,
            only_missing=None,
        )
        kept = 0
        for r in rows:
            if len((r.get("assistant_text") or "")) < int(min_len):
                continue
            # (optional) if you truly need NER here, switch to list_turns_* variants
            seeds.append({
                "turn_id": r["id"],
                "text": r["user_text"],
                "assistant_text": r["assistant_text"],
                "conversation_id": conv_id,
            })
            kept += 1
            if len(seeds) >= k:
                break
        cur.advance(consumed=len(rows))
        attempts += 1
    return seeds

async def run_once(container: Any, *, seeds_k=None, triggers_k=None, divergence=None, top_n=None):
    app = container.app
    ocfg = (_cfg(app).get("overnight") or {})
    # ... (unchanged defaults)
    seeds = await select_seeds_from_db(
        container,
        k=seeds_k or int(ocfg.get("seeds_per_night", 6)),
        min_len=int(ocfg.get("min_assistant_len", 80)),
        require_nonempty_ner=bool(ocfg.get("require_nonempty_ner", False)),
        cursor_path=ocfg.get("cursor_state_file", "./data/cursors/overnight_turn_cursor.json"),
    )
    # ... remainder of run_once unchanged


def _ts() -> str: 
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%SZ")



def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)  # atomic move on same filesystem

# ... inside run_once(...)
async def select_seeds(container: Any, *, k: int = 10) -> List[Dict[str, Any]]:
    """
    Pull k seed turns from SIS (recent + diverse). If you have embeddings in DB,
    prefer MMR sampling across recent days; fallback: last k user turns.
    """
    db = getattr(container, "db", None)
    seeds: List[Dict[str, Any]] = []
    if db and hasattr(db, "recent_user_turns"):
        rows = await db.recent_user_turns(limit=200)  # you likely have something similar
        # TODO: MMR/diversity by embed if available; for now take first k
        for r in rows[:k]:
            seeds.append({"turn_id": r["id"], "text": r["text"], "when": r["created_at"]})
    else:
        # absolute fallback
        seeds = [{"turn_id": i, "text": s, "when": None} for i, s in enumerate([
            "How to detect OOD drift in chat agents?",
            "Visual badge for uncertainty and faithfulness?",
            "Using disagreement as a routing signal?",
            "Closing Δ-gaps with tiny monitors?",
            "Efficient RAG faithfulness checks?"
        ][:k])]
    return seeds

async def verify_candidate(container: Any, prompt: str, reply: str) -> Dict[str, Any]:
    """
    Cascaded verifier: risk (badge) + retrieval checks (web/papers) + entailment.
    Wire your real adapters here (container.search_web/search_papers/tiny_nli).
    """
    orch = GapRiskOrchestrator(container, policy_profile="chat.standard")
    rec = await orch.evaluate(goal=prompt, reply=reply, model_alias="chat-hrm", monitor_alias="selfcheck")

    sources, support_scores = [], []
    search_web = getattr(container, "search_web", None)
    search_papers = getattr(container, "search_papers", None)
    entail = getattr(container, "tiny_nli", None)

    queries = [prompt[:128], reply[:128]]
    for fn in (search_web, search_papers):
        if callable(fn):
            hits = await fn(queries[0])  # simple: use prompt
            for h in (hits or [])[:3]:
                ent = 0.5
                if callable(entail):
                    res = await entail([h.get("snippet","")], [reply])
                    ent = float(res[0].get("entail", 0.5))
                sources.append({"type": h.get("type","web"), **{k:v for k,v in h.items() if k!='type'}, "entailment01": ent})
                support_scores.append(ent)

    support = float(sum(support_scores)/len(support_scores)) if support_scores else 0.5
    return {**rec, "support": support, "sources": sources}

def classify_type(prompt: str, reply: str) -> str:
    # very light heuristic; upgrade later
    if "code" in reply.lower() or "```" in reply: return "code"
    if "we propose" in reply.lower() or "method" in reply.lower(): return "paper"
    return "post"

def package_card(rec: Dict[str, Any], *, seed: Dict[str, Any], prompt: str, reply: str) -> Dict[str, Any]:
    novelty = float(rec.get("novelty", 0.0))
    support = float(rec.get("support", 0.5))
    reasons = rec.get("reasons", {})
    interest = 0.4*novelty + 0.25*(1.0-reasons.get("risk_faith",0.5)) + 0.15*support + 0.10*(1.0-reasons.get("risk_delta",0.5)) + 0.10*(1.0-reasons.get("risk_ood",0.5))

    title = reply.strip().split("\n",1)[0][:100]
    abstract = reply.strip()[:280]
    outline = []
    for line in reply.split("\n"):
        if line.strip().startswith(("-", "*")) and len(outline) < 6:
            outline.append(line.strip("-* ").strip())

    return {
        "sid": f"sug_{_ts()}",
        "type": classify_type(prompt, reply),
        "title": title or "Speculative Idea",
        "abstract": abstract,
        "outline": outline,
        "seed_ref": {"turn_id": seed.get("turn_id"), "when": seed.get("when")},
        "prompt_used": prompt,
        "reply_text": reply,
        "metrics": {
            "novelty": novelty,
            "support": support,
            "risk_faith": rec["reasons"]["risk_faith"],
            "risk_delta": rec["reasons"]["risk_delta"],
            "risk_ood": rec["reasons"]["risk_ood"]
        },
        "sources": rec.get("sources", []),
        "badge_svg": rec.get("badge_svg"),
        "run_id": rec.get("run_id"),
        "interest": interest
    }

async def run_once(container: Any, *, seeds_k=6, triggers_k=4, divergence=0.8, top_n=10) -> List[Dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = await select_seeds(container, k=seeds_k)
    explorer = ExplorationRunner(container, profile="chat.standard")

    cards: List[Dict[str, Any]] = []
    for seed in seeds:
        rows = await explorer.explore_goal(seed["text"], k_triggers=triggers_k, divergence=divergence)
        for r in rows:
            v = await verify_candidate(container, r["prompt_used"], r["reply"])
            packaged = package_card({**r, **v}, seed=seed, prompt=r["prompt_used"], reply=r["reply"])
            cards.append(packaged)

    # Rank and persist
    cards.sort(key=lambda x: x["interest"], reverse=True)
    cards = cards[:top_n]
    os.makedirs(OUT_DIR, exist_ok=True)
    safe_name = f"{_ts()}_suggestions.json"
    out_path = OUT_DIR / safe_name
    _write_json_atomic(out_path, cards)
    return cards

if __name__ == "__main__":
    # Expect your app to provide a Container; for CLI, import it from SIS if available
    from sis.main import app
    container = getattr(app.state, "container", object())
    asyncio.run(run_once(container))
