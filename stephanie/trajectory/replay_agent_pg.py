# stephanie/trajectory/replay_agent_pg.py
from __future__ import annotations

from stephanie.trajectory.composer import compose_from_spans
from stephanie.trajectory.pgvector_ann import PgVectorANN
from stephanie.trajectory.pgvector_indexer import PgVectorTurnIndexer
from stephanie.trajectory.retriever_pg import TrajectoryRetrieverPg


class TrajectoryReplayAgentPg:
    """
    Uses your memory.get_or_create for embeddings, pgvector for ANN, and your LLM for composing.
    """
    def __init__(self, session, memory, llm, window=2,
                 emb_table="embeddings", key_col="key", ns_col="namespace", vec_col="vector"):
        self.session, self.memory, self.llm, self.window = session, memory, llm, window
        self._ANN = PgVectorANN(memory=memory)  # table/cols default to your earlier names
        self._Indexer = PgVectorTurnIndexer
        self._Retriever = TrajectoryRetrieverPg
        self._compose = compose_from_spans
        self.ann = self._ANN(session, table=emb_table, key_col=key_col, ns_col=ns_col, vec_col=vec_col)

        # Batch wrapper on your memory embedder
        import numpy as np
        def embed_batch(texts):
            vecs = []
            for t in texts:
                v = self.memory.get_or_create(t)  # one-off vector; does not pollute chat_turn ns
                v = np.array(v, dtype=float).reshape(-1)
                v /= (np.linalg.norm(v) + 1e-12)
                vecs.append(v)
            return np.vstack(vecs) if vecs else np.zeros((0, 1))
        self.embed_batch = embed_batch

    def build_index(self, limit=None):
        # Persist vectors under namespace='chat_turn', key='chat_turn:{turn_id}'
        indexer = self._Indexer(
            self.session,
            embed_one=lambda text, cfg: self.memory.get_or_create(text, cfg),
            cfg={"normalize": True}  # forwarded into your memory layer if it supports it
        )
        indexer.build(limit=limit)
        return self

    def write_section(self, section_text: str, k_turns=20, top_spans=3,
                     target_move: str = "VOICE", require_image_context: bool = False) -> str:
        retr = self._Retriever(self.session, self.ann, self.embed_batch, window=self.window)
        spans = retr.retrieve_spans(
            section_text, 
            k_turns=k_turns, 
            top_spans=top_spans,
            move_filter=target_move if target_move != "VOICE" else None,  # Only filter if not default
            require_image=require_image_context
        )
        self.logger.log("TrajectoryReplay", {
            "query": section_text[:100],
            "retrieved_turns": [t.id for s in spans for t in s['span_turns']],
            "moves": list(set([m for s in spans for m in s.get('moves', [])])),
            "has_image": any(s.get('has_image') for s in spans),
        })
        return self._compose(
            self.llm, 
            section_text, 
            spans, 
            target_move=target_move,
            require_image_context=require_image_context
        )
