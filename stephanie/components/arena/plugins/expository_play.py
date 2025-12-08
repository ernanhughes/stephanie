# stephanie/arena/plays/expository_play.py
from __future__ import annotations

from stephanie.components.arena.plugins.interfaces import JobCtx, PlayResult
from stephanie.components.arena.plugins.registry import register_play
from stephanie.memory.expository_store import (
    BlogDraftStore,
    ExpositoryBufferStore,
)
from stephanie.components.arena.plugins.stictcher import BlogStitcher


@register_play
class ExpositoryPlay:
    name = "expository_v1"

    def __init__(
        self,
        session_maker,
        logger,
        k=12,
        target_words=700,
        min_exp=0.45,
        min_blog=0.50,
    ):
        self.sm = session_maker
        self.log = logger
        self.k = k
        self.target_words = target_words
        self.min_exp = min_exp
        self.min_blog = min_blog

    def run(self, ctx: JobCtx) -> PlayResult:
        # rank & buffer
        buf = ExpositoryBufferStore(self.sm, self.log).rank_and_create_buffer(
            topic=ctx.topic,
            k=self.k,
            min_exp=self.min_exp,
            min_blog=self.min_blog,
        )
        # stitch → draft

        draft = BlogStitcher(None, self.sm, None, self.log).run(
            buffer_id=buf.id, topic=ctx.topic, target_words=self.target_words
        )
        # if stitcher is async in your codebase, await it and adapt here
        draft = (
            draft if isinstance(draft, object) else draft
        )  # noop for illustration

        # store-level internal eval
        r = BlogDraftStore(self.sm, self.log).evaluate_and_mark(
            draft.id,
            min_readability=45,
            min_adjacent_coherence=0.62,
            keep_threshold=0.6,
            repetition_penalty_weight=0.1,
        )

        metrics = {
            "readability": draft.readability or 0.0,
            "coherence": draft.local_coherence or 0.0,
            "r_solve": float(r or 0.0),
            "kept": 1.0 if draft.kept else 0.0,
        }
        return PlayResult(
            artefact_id=draft.id,
            artefact_type="blog_draft",
            metrics=metrics,
            payload={"topic": ctx.topic},
            ok=True,
        )
