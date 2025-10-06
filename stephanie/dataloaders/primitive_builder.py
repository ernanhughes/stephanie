"""
Stephanie AKTN Transfer Module – Enhanced with ORM Integration
=============================================================

This builds on the AKTN+CARE implementation, but now wires `PrimitiveBuilder`
methods directly to Stephanie’s ORM classes:
- ChatTurnORM
- CaseBookORM

These methods output both a feature vector tensor and a `PrimitiveMetadata`
object for MemCube/CartridgeORM persistence.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from stephanie.models.casebook import CaseBookORM

# ORM imports – adjust paths to your repo
from stephanie.models.chat import ChatTurnORM


# ======= Primitive Metadata =======

@dataclass
class PrimitiveMetadata:
    primitive_id: str
    source_artifact_id: str
    source_type: str  # "chat" or "casebook"
    created_at: float
    embedding_model: str
    tags: List[str]
    transfer_confidence: float = 0.0
    last_used: float = 0.0
    version: int = 1


# ======= PrimitiveBuilder with ORM integration =======


class PrimitiveBuilder:
    """ORM-aware primitive feature builder for Stephanie.

    Works with either:
      • SQLAlchemy ORM objects (ChatTurnORM, CaseBookCaseORM/CaseBookORM)
      • Plain dicts that mirror the ORM payloads

    Design:
      - Strong defaults based on common fields we saw in your codebase
        (chat: user_text, assistant_text, goal, entities, tags; case: summary, citations, tags)
      - Optional field maps let you override names without changing code
      - Returns (tensor, PrimitiveMetadata) for persistence
    """

    def __init__(
        self,
        embedder,
        embedding_model_name: str,
        chat_field_map: Optional[Dict[str, str]] = None,
        case_field_map: Optional[Dict[str, str]] = None,
    ):
        self.embedder = (
            embedder  # e.g., HNet/Ollama/HF embedding fn -> np.ndarray/list
        )
        self.embedding_model_name = embedding_model_name
        # Defaults (override via maps)
        self.chat_map = {
            "id": "id",
            "user_text": "user_text",  # alt: "user_message", "question"
            "assistant_text": "assistant_text",  # alt: "assistant_message", "answer"
            "goal": "goal",  # JSON/dict or str
            "entities": "entities",  # list/JSON
            "tags": "tags",  # list/JSON
        }
        self.case_map = {
            "id": "id",
            "summary": "summary",
            "citations": "citations",  # list[str]
            "tags": "tags",
            # Optional extras (if present):
            "policy": "policy",  # dict/JSON of structured params
            "constraints": "constraints",  # list/JSON
        }
        if chat_field_map:
            self.chat_map.update(chat_field_map)
        if case_field_map:
            self.case_map.update(case_field_map)

    # ---------- Public API ----------
    def from_chat_turn(
        self, turn_obj: Any
    ) -> Tuple[torch.Tensor, PrimitiveMetadata]:
        data = self._as_dict(turn_obj, which="chat")
        text = self._fmt_chat_text(data)
        vec = self._embed(text)
        pm = PrimitiveMetadata(
            primitive_id=f"chat_{data.get('id', 'unknown')}",
            source_artifact_id=str(data.get("id", "unknown")),
            source_type="chat",
            created_at=time.time(),
            embedding_model=self.embedding_model_name,
            tags=self._to_list(data.get("tags")),
        )
        return torch.tensor(vec, dtype=torch.float32), pm

    def from_casebook_case(
        self, case_obj: Any
    ) -> Tuple[torch.Tensor, PrimitiveMetadata]:
        data = self._as_dict(case_obj, which="case")
        text = self._fmt_case_text(data)
        vec = self._embed(text)
        pm = PrimitiveMetadata(
            primitive_id=f"case_{data.get('id', 'unknown')}",
            source_artifact_id=str(data.get("id", "unknown")),
            source_type="casebook",
            created_at=time.time(),
            embedding_model=self.embedding_model_name,
            tags=self._to_list(data.get("tags")),
        )
        return torch.tensor(vec, dtype=torch.float32), pm

    # ---------- Batch helpers (DB session optional) ----------
    def build_from_chat_turns(
        self, turns: List[Any]
    ) -> Tuple[torch.Tensor, List[PrimitiveMetadata]]:
        vecs, metas = [], []
        for t in turns:
            v, m = self.from_chat_turn(t)
            vecs.append(v)
            metas.append(m)
        return torch.stack(vecs, dim=0), metas

    def build_from_cases(
        self, cases: List[Any]
    ) -> Tuple[torch.Tensor, List[PrimitiveMetadata]]:
        vecs, metas = [], []
        for c in cases:
            v, m = self.from_casebook_case(c)
            vecs.append(v)
            metas.append(m)
        return torch.stack(vecs, dim=0), metas

    # ---------- Internals ----------
    def _embed(self, text: str):
        return self.embedder(text)

    def _fmt_chat_text(self, d: Dict[str, Any]) -> str:
        user_text = str(d.get("user_text") or d.get("question") or "").strip()
        assistant_text = str(
            d.get("assistant_text") or d.get("answer") or ""
        ).strip()
        goal = d.get("goal")
        entities = self._to_list(d.get("entities"))
        tags = self._to_list(d.get("tags"))
        goal_str = goal if isinstance(goal, str) else json.dumps(goal or {})
        meta = json.dumps(
            {"entities": entities, "tags": tags}, ensure_ascii=False
        )
        return f"""<CHAT_USER>
{user_text}
<CHAT_ASSISTANT>
{assistant_text}
<GOAL>
{goal_str}
<META>
{meta}"""

    def _fmt_case_text(self, d: Dict[str, Any]) -> str:
        summary = str(d.get("summary") or "").strip()
        citations = self._to_list(d.get("citations"))
        policy = d.get("policy")
        constraints = self._to_list(d.get("constraints"))
        policy_str = json.dumps(policy or {}, ensure_ascii=False)
        meta = json.dumps(
            {"citations": citations, "constraints": constraints},
            ensure_ascii=False,
        )
        cite_blob = "".join(citations)
        return f"""<CASE_SUMMARY>
{summary}
<POLICY>
{policy_str}
<CITATIONS>
{cite_blob}
<META>
{meta}"""

    def _as_dict(self, obj: Any, which: str) -> Dict[str, Any]:
        """Extract a dict from ORM or dict using field maps and getattr fallbacks."""
        fmap = self.chat_map if which == "chat" else self.case_map
        if isinstance(obj, dict):
            return {k: obj.get(v) for k, v in fmap.items()}
        # ORM path — tolerate missing attrs
        out = {}
        for k, v in fmap.items():
            out[k] = getattr(obj, v, None)
        # Common nested containers
        if which == "chat":
            # Try nested .meta JSON if available
            meta = getattr(obj, "meta", None)
            if isinstance(meta, dict):
                out["goal"] = out.get("goal") or meta.get("goal")
                out["entities"] = out.get("entities") or meta.get("entities")
                out["tags"] = out.get("tags") or meta.get("tags")
        else:  # case
            meta = getattr(obj, "meta", None)
            if isinstance(meta, dict):
                out["citations"] = out.get("citations") or meta.get(
                    "citations"
                )
                out["constraints"] = out.get("constraints") or meta.get(
                    "constraints"
                )
                out["tags"] = out.get("tags") or meta.get("tags")
        return out

    @staticmethod
    def _to_list(x: Any) -> List[Any]:
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        # JSON string
        if isinstance(x, str):
            try:
                val = json.loads(x)
                return list(val) if isinstance(val, (list, tuple)) else [x]
            except Exception:
                return [x]
        return [x]


# ======= Edge Builder (unchanged, for PlanTraces) =======


class EdgeBuilder:
    @staticmethod
    def from_plan_traces(nodes: List[Dict[str, Any]]) -> torch.Tensor:
        src, dst = [], []
        for i, n in enumerate(nodes):
            for p in n.get("parents", []) or []:
                src.append(p)
                dst.append(i)
            for s in n.get("siblings", []) or []:
                src.append(s)
                dst.append(i)
            for c in n.get("contradicts", []) or []:
                src.append(c)
                dst.append(i)
        if not src:
            src = list(range(len(nodes) - 1))
            dst = list(range(1, len(nodes)))
        return torch.tensor([src, dst], dtype=torch.long)


# ======= Example usage with ORM =======
if __name__ == "__main__":
    pb = PrimitiveBuilder("hnet-v4")

    # Fake ORM objects (replace with session query)
    chat_turn = ChatTurnORM(
        id=1,
        user_text="How do we handle refunds?",
        assistant_text="By escalation policy.",
    )
    case = CaseBookORM(
        id=10,
        summary="Refunds must be approved if >30 days.",
        citations=["Policy123"],
    )

    x_chat, meta_chat = pb.from_chat_turn(chat_turn)
    x_case, meta_case = pb.from_casebook_case(case)

    print("chat vector shape:", x_chat.shape, "metadata:", meta_chat)
    print("case vector shape:", x_case.shape, "metadata:", meta_case)
