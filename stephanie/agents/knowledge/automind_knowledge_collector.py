# stephanie/agents/knowledge/automind_knowledge_collector.py
"""
AutoMind Knowledge Collector Module

This module provides the AutoMindKnowledgeCollector class for intelligent knowledge acquisition
and processing in the co-ai framework. It specializes in collecting, labeling, and ranking
research papers and Kaggle solutions based on task descriptions.

Key Features:
    - Automated research paper collection from various sources
    - Kaggle competition solution gathering
    - Intelligent document labeling using hierarchical classification
    - Similarity-based filtering and ranking
    - Label-priority based re-ranking for relevance optimization

Classes:
    AutoMindKnowledgeCollector: Main class for knowledge collection and processing

Constants:
    LABEL_HIERARCHY: Hierarchical mapping of ML/AI domains to specific tasks

Dependencies:
    - BaseAgent: Core agent functionality
    - WebSearchTool, WikipediaTool: Web-based information retrieval
    - ArxivTool: Academic paper search
    - CosineSimilarityTool: Document similarity computation
    - HuggingFaceTool: Dataset search capabilities
"""

from typing import Dict, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.tools.cos_sim_tool import get_top_k_similar

LABEL_HIERARCHY = {
    "Computer Vision": ["Image Classification", "Object Detection", "Segmentation"],
    "NLP": ["Text Classification", "NER", "Summarization"],
    "Tabular Data": ["Classification", "Regression", "Anomaly Detection"],
    "Graph Learning": ["Node Classification", "Link Prediction"]
}


class AutoMindKnowledgeCollector:
    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.memory = agent.memory
        self.logger = agent.logger
        self.cfg = agent.cfg

    async def collect_papers(self, query: str) -> List[Dict]:
        context = {
            "goal": {
                "id": "paper_search",
                "goal_text": query,
                "goal_type": "model_review"
            },
            "search_queries": [{"goal_text": query}]
        }
        result = await self.agent.run(context)
        return result.get("search_results", [])

    async def collect_kaggle_solutions(self, task_description: str) -> List[Dict]:
        query = f"top {task_description} kaggle solution"
        context = {
            "goal": {
                "id": "kaggle_search",
                "goal_text": query,
                "goal_type": "data_search"
            },
            "search_queries": [{"goal_text": query}]
        }
        result = await self.agent.run(context)
        return result.get("search_results", [])

    def assign_labels_to_document(self, doc_title: str, doc_summary: str) -> List[str]:
        combined_text = f"{doc_title} {doc_summary}".lower()
        matched_labels = []

        for category, subcategories in LABEL_HIERARCHY.items():
            if any(kw in combined_text for kw in category.lower().split()):
                matched_labels.append(category)
                for subcat in subcategories:
                    if any(kw in combined_text for kw in subcat.lower().split()):
                        matched_labels.append(subcat)

        # Fallback using similarity
        if not matched_labels:
            all_labels = [label for cat in LABEL_HIERARCHY.values() for label in cat]
            top = get_top_k_similar(combined_text, all_labels, self.memory, top_k=2)
            matched_labels = [label for label, _ in top]

        return list(set(matched_labels))

    async def retrieve_knowledge(self, task_description: str) -> List[Dict]:
        papers = await self.collect_papers(task_description)
        kaggle_tricks = await self.collect_kaggle_solutions(task_description)
        all_docs = papers + kaggle_tricks

        labeled_docs = [
            {**doc, "labels": self.assign_labels_to_document(doc["title"], doc["summary"])}
            for doc in all_docs
        ]

        # Filter by relevance to task description
        relevant_docs = self.filter_by_similarity(task_description, labeled_docs)

        # Re-rank by label priority
        reranked_docs = self.rerank_by_label_priority(relevant_docs)

        return reranked_docs

    def filter_by_similarity(self, query: str, docs: List[Dict]) -> List[Dict]:
        titles_and_summaries = [f"{doc['title']} {doc['summary']}" for doc in docs]
        scores = get_top_k_similar(query, titles_and_summaries, self.memory, top_k=len(docs))
        ranked_indices = [i for i, _ in scores]
        return [docs[i] for i in ranked_indices]

    def rerank_by_label_priority(self, docs: List[Dict]) -> List[Dict]:
        label_priority = {
            "Computer Vision": 5,
            "NLP": 5,
            "Tabular Data": 4,
            "Image Classification": 3,
            "Text Classification": 3,
            "Classification": 2
        }

        def score_doc(doc):
            return sum(label_priority.get(label, 0) for label in doc.get("labels", []))

        return sorted(docs, key=lambda x: score_doc(x), reverse=True)