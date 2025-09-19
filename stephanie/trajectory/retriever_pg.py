from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sqlalchemy.orm import Session

from stephanie.memory.chat_store import ChatStore
from stephanie.trajectory.pgvector_ann import PgVectorANN


def _turn_text(t) -> str:
    u = (t.user_message.text or "") if t.user_message else ""
    a = (t.assistant_message.text or "") if t.assistant_message else ""
    return f"USER: {u}\nYOU: {a}"


class TrajectoryRetrieverPg:
    def __init__(
        self, session: Session, ann: PgVectorANN, embed_batch, window: int = 2
    ):
        self.session, self.ann, self.embed_batch, self.window = (
            session,
            ann,
            embed_batch,
            window,
        )
        self.store = ChatStore(session)

    def _expand_span(self, turn_id: int):
        t = self.store.get_turn_by_id(turn_id)
        turns = self.store.get_turns_for_conversation(t.conversation_id)
        i = next((i for i, x in enumerate(turns) if x.id == turn_id), 0)
        lo, hi = max(0, i - self.window), min(len(turns) - 1, i + self.window)
        return turns[lo : hi + 1]

    def _coherence_penalty(self, span) -> float:
        ids = [s.id for s in span]
        if len(ids) < 2:
            return 0.0

        gaps = np.diff(sorted(ids))
        return 0.1 * float((gaps > 1).sum())

    def retrieve_spans(
        self,
        section_text: str,
        k_turns: int = 20,
        top_spans: int = 3,
        move_filter: str = None,
        require_image: bool = False,
    ) -> List[Dict[str, Any]]:
        q = self.embed_batch([section_text])[0]
        hits = self.ann.search(q, k=k_turns)

        out, seen = [], set()
        for turn_id, _ in hits:
            span = self._expand_span(turn_id)

            # 👇 Filter by move (if specified)
            if move_filter:
                move_found = False
                for s in span:
                    if s.assistant_message and s.assistant_message.meta:
                        if (
                            s.assistant_message.meta.get("reasoning_move")
                            == move_filter
                        ):
                            move_found = True
                            break
                if not move_found:
                    continue

            # 👇 Filter by image (if required)
            if require_image:
                image_found = False
                for s in span:
                    if s.user_message and s.user_message.meta:
                        if s.user_message.meta.get("image_latent") is not None:
                            image_found = True
                            break
                if not image_found:
                    continue

            texts = [_turn_text(s) for s in span]
            V = self.embed_batch(texts) if texts else np.zeros((0, q.shape[0]))
            span_sim = float(np.mean(V @ q)) if len(V) else 0.0
            score = span_sim - self._coherence_penalty(span)
            conv_id = span[0].conversation_id if span else None
            if conv_id in seen:
                continue
            seen.add(conv_id)

            out.append(
                {
                    "score": score,
                    "conversation_id": conv_id,
                    "span_turns": span,
                    "moves": [
                        s.assistant_message.meta.get("reasoning_move", "VOICE")
                        for s in span
                        if s.assistant_message
                    ], 
                    "has_image": any(
                        s.user_message.meta.get("image_latent") is not None
                        for s in span
                        if s.user_message
                    ), 
                }
            )
            if len(out) >= top_spans:
                break

        out.sort(key=lambda d: d["score"], reverse=True)
        return out
