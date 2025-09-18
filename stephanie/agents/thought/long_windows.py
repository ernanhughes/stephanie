from __future__ import annotations
from typing import Iterable
from stephanie.models.chat import ChatConversationORM

# naive tokenizer is fine for now; replace with your real splitter if you like
def _tok(s: str) -> list: 
    return (s or "").split()

def iter_long_windows_texts(messages, max_tokens=4000, stride=2000) -> Iterable[str]:
    buf, toks = [], 0
    for m in messages:
        t = m.text or ""
        n = len(_tok(t))
        if toks + n > max_tokens and buf:
            yield " ".join(x.text or "" for x in buf)
            # stride tail
            tail_tokens = _tok(" ".join(x.text or "" for x in buf))[-stride:]
            buf, toks = [], 0
            if tail_tokens:
                tail = " ".join(tail_tokens)
                class Tmp: pass
                tmp = Tmp(); tmp.text = tail
                buf.append(tmp); toks = len(tail_tokens)
        buf.append(m); toks += n
    if buf:
        yield " ".join(x.text or "" for x in buf)

def index_conversation_windows(memory, conv: ChatConversationORM, logger=None, max_tokens=4000, stride=2000):
    msgs = memory.chats.get_messages(conv.id)
    for i, win_text in enumerate(iter_long_windows_texts(msgs, max_tokens, stride)):
        memory.embeddings.get_or_create(f"[WIN:{conv.id}:{i}] {win_text}")
    logger and logger.info(f"[LongWindows] indexed conv={conv.id} ({conv.title})")
