# stephanie/models/search_result.py
# models/search_result.py
from datetime import datetime, timezone

from sqlalchemy import (JSON, Column, DateTime, ForeignKey, Integer, String,
                        Text)

from stephanie.models.base import Base


class SearchResultORM(Base):
    __tablename__ = "search_results"

    id = Column(Integer, primary_key=True)
    query = Column(Text, nullable=False)
    source = Column(String, nullable=False)
    result_type = Column(String)
    title = Column(Text)
    summary = Column(Text)
    url = Column(Text)
    author = Column(String)
    published_at = Column(DateTime)
    tags = Column(JSON)
    goal_id = Column(Integer, ForeignKey("goals.id"))
    parent_goal = Column(Text)
    strategy = Column(String)
    focus_area = Column(String)
    extra_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    # 🔍 New Fields Below — For Knowledge Refinement & Idea Linking
    key_concepts = Column(JSON)
    technical_insights = Column(JSON)
    relevance_score = Column(Integer)  # 1–10 score for how relevant this is to the goal
    novelty_score = Column(Integer)  # Estimated novelty vs. prior knowledge
    related_ideas = Column(JSON)  # List of idea IDs or descriptions
    refined_summary = Column(Text)  # A concise, processed summary for downstream agents
    extracted_methods = Column(JSON)  # Techniques or methods described in the result
    domain_knowledge_tags = Column(JSON)  # e.g., "self-modifying", "graph transformer"
    critique_notes = Column(Text)  # Feedback from evaluator agent (Mr Q), if any

    def to_dict(self, include_relationships: bool = False) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "source": self.source,
            "result_type": self.result_type,
            "title": self.title,
            "summary": self.summary,
            "url": self.url,
            "author": self.author,
            "published_at": self.published_at.isoformat()
            if self.published_at
            else None,
            "tags": self.tags,
            "goal_id": self.goal_id,
            "parent_goal": self.parent_goal,
            "strategy": self.strategy,
            "focus_area": self.focus_area,
            "key_concepts": self.key_concepts,
            "technical_insights": self.technical_insights,
            "relevance_score": self.relevance_score,
            "novelty_score": self.novelty_score,
            "related_ideas": self.related_ideas,
            "extracted_methods": self.extracted_methods,
            "critique_notes": self.critique_notes,
            "extra_data": self.extra_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)
