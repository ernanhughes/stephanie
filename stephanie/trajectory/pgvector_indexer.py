from __future__ import annotations
from typing import Iterable, Callable
from stephanie.memory.chat_store import ChatStore
from sqlalchemy.orm import Session
import re

# (text, cfg) -> vector; must persist to your embeddings table when given namespace/key
EmbedOneFn = Callable[[str, dict], list]

def _turn_text(t) -> str:
    u = (t.user_message.text or "") if t.user_message else ""
    a = (t.assistant_message.text or "") if t.assistant_message else ""
    return f"USER: {u.strip()}\nYOU: {a.strip()}"

class PgVectorTurnIndexer:
    """
    Ensures every ChatTurn has a persisted vector in your existing embeddings table,
    via memory.get_or_create(text, {"namespace":"chat_turn","key":"chat_turn:{id}", ...}).
    """
    def __init__(self, session: Session, embed_one: EmbedOneFn, cfg: dict | None = None):
        self.session = session
        self.embed_one = embed_one
        self.cfg = cfg or {}
        self.store = ChatStore(session)
        self.move_predictor = _predict_reasoning_move  # 👈 Add this


    def build(self, limit: int | None = None, batch: int = 400):
        convs = self.store.get_all(limit=limit or 10**9)
        buf: list = []
        for conv in convs:
            for t in self.store.get_turns_for_conversation(conv.id):
                buf.append(t)
                if len(buf) >= batch:
                    self._process(buf); buf = []
        if buf:
            self._process(buf)

    def _process(self, turns: Iterable):
        for t in turns:
            text = _turn_text(t)
            _ = self.embed_one(text)  # e.g., memory.embeddings.get_or_create(text)
            key = f"chat_turn:{t.id}"
            
            move = "VOICE"  # default 
            image_latent = None  

            if t.assistant_message and t.assistant_message.meta:
                move = t.assistant_message.meta.get("reasoning_move", "VOICE")
            else:
                # Fallback to heuristic
                if t.assistant_message and t.assistant_message.text:
                    move = self.move_predictor(t.assistant_message.text)
            
            if t.user_message and t.user_message.meta:
                image_url = t.user_message.meta.get("image_url")
                if image_url and self.image_encoder:
                    try:
                        # Load image, encode to latent
                        image_latent = self.image_encoder.encode_image(image_url)
                        # Store as JSON string or separate column (we'll use meta)
                    except Exception:
                        pass



            # 👇 Embed with move context
            # Option A: Embed "MOVE: {move}\n{text}"
            # Option B: Store move separately (we'll do this)
            _ = self.embed_one(text, {
                "namespace": "chat_turn", 
                "key": key, 
                "reasoning_move": move,
                "image_latent": image_latent.tolist() if image_latent is not None else None,
                **self.cfg
            })


def _predict_reasoning_move(text: str) -> str:
    text_lower = text.lower()
    if any(phrase in text_lower for phrase in ["define", "what is", "meaning of"]):
        return "DEFINE"
    if any(phrase in text_lower for phrase in ["step", "first", "second", "process"]):
        return "STEPS"
    if any(phrase in text_lower for phrase in ["therefore", "in conclusion", "summary"]):
        return "TIEBACK"
    if any(phrase in text_lower for phrase in ["because", "since", "due to", "framework"]):
        return "FRAME"
    if re.search(r'\d+\.?\d*\s*(%|percent|accuracy|score)', text_lower):
        return "CITE"
    return "VOICE"
