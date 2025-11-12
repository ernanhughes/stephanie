# stephanie/agents/thought/krm_v1_trainer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

ACCEPT_LEX = [
    "ok ok","okay okay","yes exactly","lock it","do this","that’s it","perfect",
    "this is the one","bingo","aha","eureka","yes — do this","lock this in"
]

@dataclass
class KRMSample:
    turn_id: int
    conv_id: int
    turn_text: str
    next_user_text: str
    windows: List[str]

class KRMv1:
    def __init__(self, memory, logger, vpm=None):
        self.memory = memory
        self.logger = logger
        self.vpm = vpm  # optional ZeroModel/VPM signature service
        self.model = LogisticRegression(max_iter=1000)

    # ---------- dataset mining ----------
    def mine_samples(self, limit_convs: int = 1000, pos_limit: int = 2000, neg_limit: int = 2000) -> Tuple[List[KRMSample], List[KRMSample]]:
        positives: List[KRMSample] = []
        negatives: List[KRMSample] = []

        top = self.memory.chats.get_top_conversations(limit=limit_convs, by="turns")
        for conv, _ in top:
            turns = self.memory.chats.get_turns_for_conversation(conv.id)
            msgs = self.memory.chats.get_messages(conv.id)
            # prebuild windows ANN by using user text as query → pull 2 nearest [WIN:*]
            def nearest_windows(txt: str) -> List[str]:
                q = self.memory.embeddings.get_or_create(txt or "")
                hits = self.memory.embeddings.find_neighbors(q, k=10)  # your default search
                wins = [h.get("text","") for h in hits if "[WIN:" in (h.get("text","") or "")]
                return wins[:2]

            # map message id → its index to look up next user
            msg_by_id = {m.id: m for m in msgs}
            for i, t in enumerate(turns):
                if not t.user_message or not t.assistant_message:
                    continue
                u = t.user_message.text or ""; a = t.assistant_message.text or ""
                if not u.strip() or not a.strip():
                    continue

                # find NEXT user message after assistant in this conversation
                next_user_text = ""
                # crude linear lookahead in messages order_index
                a_id = t.assistant_message_id
                a_msg = msg_by_id.get(a_id)
                next_user = None
                if a_msg:
                    for m in msgs:
                        if m.order_index > a_msg.order_index and m.role == "user":
                            next_user = m; break
                if next_user:
                    next_user_text = next_user.text or ""

                sample = KRMSample(
                    turn_id=t.id, conv_id=conv.id, turn_text=f"USER: {u}\nASSISTANT: {a}", 
                    next_user_text=next_user_text, windows=nearest_windows(u)
                )

                # positive if acceptance lexicon appears in next user text
                if any(p in (next_user_text or "").lower() for p in ACCEPT_LEX):
                    positives.append(sample)
                else:
                    negatives.append(sample)

                if len(positives) >= pos_limit and len(negatives) >= neg_limit:
                    break
            if len(positives) >= pos_limit and len(negatives) >= neg_limit:
                break

        self.logger and self.logger.info(f"[KRMv1] mined positives={len(positives)} negatives={len(negatives)}")
        return positives, negatives

    # ---------- features ----------
    def _cos(self, a: str, b: str) -> float:
        va = self.memory.embeddings.get_or_create(a or "")
        vb = self.memory.embeddings.get_or_create(b or "")
        va, vb = np.array(va), np.array(vb)
        den = (np.linalg.norm(va)*np.linalg.norm(vb) + 1e-8)
        return float(np.dot(va, vb)/den)

    def _coverage(self, answer: str, windows: List[str]) -> float:
        if not answer or not windows: return 0.0
        aw = set((answer or "").lower().split())
        if not aw: return 0.0
        ctx = set((" ".join(windows)).lower().split())
        return len(aw & ctx)/max(1,len(aw))

    def _unsupported(self, answer: str, windows: List[str]) -> float:
        # crude: sentences not semantically similar to any window
        sents = [s.strip() for s in (answer or "").split(".") if s.strip()]
        ctx = windows or []
        bad = 0
        for s in sents:
            if max([self._cos(s, w) for w in ctx] or [0.0]) < 0.55:
                bad += 1
        return float(bad)

    def _vpm_dist(self, text: str) -> float:
        if not self.vpm: return 0.0
        try:
            sig = np.array(self.vpm.get_signature(text))  # expects list[float]
            centroid = np.array(self.vpm.good_centroid()) # implement however you like
            return float(np.linalg.norm(sig - centroid))
        except Exception:
            return 0.0

    def _featurize(self, s: KRMSample) -> np.ndarray:
        # split turn
        try:
            user_part = s.turn_text.split("ASSISTANT:")[0]
            assist_part = s.turn_text.split("ASSISTANT:",1)[1]
        except Exception:
            user_part, assist_part = s.turn_text, ""

        feats = [
            1.0 if any(p in (s.next_user_text or "").lower() for p in ACCEPT_LEX) else 0.0,
            self._cos(s.next_user_text, "yes exactly ok ok lock this perfect"),
            self._coverage(assist_part, s.windows),
            -self._unsupported(assist_part, s.windows),
            1.0 if any(k in assist_part for k in ["steps:", "1.", "•", "checklist", "rule:","recipe:"]) else 0.0,
            1.0 if (" if " in assist_part.lower() and " then " in assist_part.lower()) else 0.0,
            1.0 if ("```" in assist_part or "\\(" in assist_part or "$" in assist_part) else 0.0,
            1.0 - self._cos(assist_part, user_part),  # novelty proxy
            self._vpm_dist(assist_part),
        ]
        return np.array(feats, dtype=float)

    # ---------- train / predict ----------
    def fit(self, positives: List[KRMSample], negatives: List[KRMSample]):
        X, y = [], []
        for s in positives:  X.append(self._featurize(s)); y.append(1)
        for s in negatives:  X.append(self._featurize(s)); y.append(0)
        if not X:
            raise RuntimeError("No KRM training data found.")
        self.model.fit(np.vstack(X), np.array(y))
        self.logger and self.logger.info("[KRMv1] trained")

    def score_turn(self, turn_text: str, next_user_text: str, windows: List[str]) -> float:
        s = KRMSample(turn_id=0, conv_id=0, turn_text=turn_text, next_user_text=next_user_text, windows=windows)
        f = self._featurize(s).reshape(1, -1)
        return float(self.model.predict_proba(f)[0,1])
