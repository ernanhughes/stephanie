# stephanie/agents/thought/sot_v01_dataset_builder_pg.py
from __future__ import annotations

import json
from sqlalchemy.orm import Session
from stephanie.memory.chat_store import ChatStore
from stephanie.trajectory.pgvector_ann import PgVectorANN  # from earlier


class SoTV01DatasetBuilderPg:
    """
    Build SoT v0.1 dataset using pgvector ANN and span expansion.
    Each example:
      - query (user text)
      - retrieved_turns (few examples from spans)
      - response (assistant text)
      - predicted_move (from meta, if present)
    """

    def __init__(self, session: Session, memory, logger=None, window: int = 2):
        self.session = session
        self.store = ChatStore(session)
        self.memory = memory
        self.logger = logger
        self.window = window
        self.ann = PgVectorANN(
            memory=memory
        )  # table/cols default to your earlier names

        # batch embed via your memory (ad-hoc namespace)
        import numpy as np

        def embed_batch(texts):
            vecs = []
            for t in texts:
                v = self.memory.embedding.get_or_create(t)
                import numpy as _np

                v = _np.array(v, dtype=float).reshape(-1)
                v /= float(_np.linalg.norm(v)) + 1e-12
                vecs.append(v)
            return np.vstack(vecs) if vecs else np.zeros((0, 1))

        self.embed_batch = embed_batch

    def _reasoning_move(self, turn) -> str:
        meta = (
            (turn.assistant_message.meta or {})
            if turn.assistant_message
            else {}
        )
        return (meta.get("reasoning_move") or "VOICE").upper()

    def _user_text(self, turn) -> str:
        return (
            (turn.user_message.text or "").strip() if turn.user_message else ""
        )

    def _assistant_text(self, turn) -> str:
        return (
            (turn.assistant_message.text or "").strip()
            if turn.assistant_message
            else ""
        )

    def _retrieve_spans_fast(
        self,
        section_text: str,
        exclude_conv_id: int,
        exclude_turn_id: int,
        k: int = 20,
        top_spans: int = 3,
    ):
        # ANN → expand spans → score mean sim → exclude same conversation vicinity
        import numpy as np

        q = self.embed_batch([section_text])[0]
        hits = self.ann.search(q, k=k)  # [(turn_id, sim)]

        out, seen_convs = [], set()
        for turn_id, _ in hits:
            t = self.store.get_turn_by_id(turn_id)
            if not t:
                continue
            # exclude same conversation within +/- window to avoid leakage
            if t.conversation_id == exclude_conv_id:
                # find index to compare distance
                turns = self.store.get_turns_for_conversation(exclude_conv_id)
                idx = next(
                    (
                        i
                        for i, x in enumerate(turns)
                        if x.id == exclude_turn_id
                    ),
                    None,
                )
                jdx = next(
                    (j for j, x in enumerate(turns) if x.id == turn_id), None
                )
                if (
                    idx is not None
                    and jdx is not None
                    and abs(idx - jdx) <= self.window
                ):
                    continue

            # expand span around the hit
            turns = self.store.get_turns_for_conversation(t.conversation_id)
            center = next(
                (i for i, x in enumerate(turns) if x.id == turn_id), 0
            )
            lo, hi = (
                max(0, center - self.window),
                min(len(turns) - 1, center + self.window),
            )
            span = turns[lo : hi + 1]

            # mean sim across span
            texts = []
            for s in span:
                u = (s.user_message.text or "") if s.user_message else ""
                a = (
                    (s.assistant_message.text or "")
                    if s.assistant_message
                    else ""
                )
                texts.append(f"USER: {u}\nYOU: {a}")
            V = self.embed_batch(texts) if texts else None
            span_sim = (
                float(np.mean(V @ q))
                if (V is not None and len(V) > 0)
                else 0.0
            )

            if t.conversation_id in seen_convs:
                continue
            seen_convs.add(t.conversation_id)

            out.append(
                {
                    "score": span_sim,
                    "conversation_id": t.conversation_id,
                    "span": span,
                }
            )
            if len(out) >= top_spans:
                break

        out.sort(key=lambda d: d["score"], reverse=True)
        return out

    def build_dataset(
        self,
        output_path: str,
        max_conversations: int = 1000,
        top_spans: int = 3,
    ):
        top = self.store.get_top_conversations(
            limit=max_conversations, by="turns"
        )
        total = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for conv, _ in top:
                turns = self.store.get_turns_for_conversation(conv.id)
                for t in turns:
                    q = self._user_text(t)
                    a = self._assistant_text(t)
                    if not q or not a:
                        continue
                    spans = self._retrieve_spans_fast(
                        q,
                        exclude_conv_id=conv.id,
                        exclude_turn_id=t.id,
                        top_spans=top_spans,
                    )
                    ex = {
                        "query": q,
                        "retrieved_turns": [
                            {
                                "user": (s.user_message.text or "")
                                if s.user_message
                                else "",
                                "assistant": (s.assistant_message.text or "")
                                if s.assistant_message
                                else "",
                                "conversation_id": s.conversation_id,
                            }
                            for block in spans
                            for s in block["span"]
                        ],
                        "response": a,
                        "predicted_move": self._reasoning_move(t),
                        "conversation_id": conv.id,
                        "turn_id": t.id,
                    }
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    total += 1
        if self.logger:
            self.logger.info(
                f"[SoTV01DatasetBuilderPg] wrote {total} examples to {output_path}"
            )
