# stephanie/agents/expository/stitcher.py
from typing import List, Optional

from sqlalchemy.orm import sessionmaker
from stephanie.memory.expository_store import (
    ExpositoryBuffer,
    ExpositorySnippet,
    BlogDraft,
)


class BlogStitcher:
    def __init__(
        self,
        agent: Optional[object],
        session_maker: sessionmaker,
        unused_param: Optional[object],
        logger: object
    ):
        self.agent = agent
        self.sm = session_maker
        self.unused = unused_param
        self.log = logger

    def run(
        self,
        buffer_id: int,
        topic: str,
        target_words: int
    ) -> BlogDraft:
        """
        Stitches ranked snippets into a blog draft meeting target word count.
        
        Args:
            buffer_id: ID of the expository buffer containing ranked snippets
            topic: Topic for the blog draft
            target_words: Target word count for the draft
            
        Returns:
            BlogDraft object with stitched content
        """
        session = self.sm()
        try:
            # Fetch buffer and validate
            buffer = session.query(ExpositoryBuffer).get(buffer_id)
            if not buffer:
                raise ValueError(f"Buffer {buffer_id} not found")

            # Get snippets ordered by rank (highest first)
            snippets = (
                session.query(ExpositorySnippet)
                .filter(ExpositorySnippet.buffer_id == buffer_id)
                .order_by(ExpositorySnippet.rank.desc())
                .all()
            )
            
            if not snippets:
                self.log.warning(f"No snippets found in buffer {buffer_id}")
                return self._create_empty_draft(session, topic, buffer_id)

            # Stitch snippets to meet target word count
            draft_text, total_words = self._stitch_snippets(
                snippets, target_words
            )
            
            # Create and save draft
            draft = BlogDraft(
                topic=topic,
                text=draft_text,
                word_count=total_words,
                buffer_id=buffer_id,
                readability=None,  # Will be evaluated later
                local_coherence=None,  # Will be evaluated later
                kept=None  # Will be determined by evaluation
            )
            session.add(draft)
            session.commit()
            session.refresh(draft)
            return draft

        finally:
            session.close()

    def _stitch_snippets(
        self,
        snippets: List[ExpositorySnippet],
        target_words: int
    ) -> tuple[str, int]:
        """Stitches snippets while respecting target word count."""
        current_words = 0
        selected_texts = []
        
        for snippet in snippets:
            # Calculate available space
            words_needed = max(0, target_words - current_words)
            if words_needed <= 0:
                break
                
            # Handle snippet truncation if needed
            snippet_words = snippet.word_count
            if snippet_words <= words_needed:
                # Take full snippet
                selected_texts.append(snippet.text)
                current_words += snippet_words
            else:
                # Truncate to available space
                words = snippet.text.split()[:words_needed]
                truncated = " ".join(words)
                selected_texts.append(truncated)
                current_words = target_words
                break
        
        # Join with paragraph breaks
        return "\n\n".join(selected_texts), current_words

    def _create_empty_draft(
        self,
        session: sessionmaker,
        topic: str,
        buffer_id: int
    ) -> BlogDraft:
        """Creates an empty draft when no snippets are available."""
        draft = BlogDraft(
            topic=topic,
            text="",
            word_count=0,
            buffer_id=buffer_id,
            readability=0.0,
            local_coherence=0.0,
            kept=False
        )
        session.add(draft)
        session.commit()
        session.refresh(draft)
        self.log.error(f"Created empty draft for buffer {buffer_id} - no valid snippets")
        return draft