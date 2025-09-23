<!-- Merged Python Code Files -->


## File: __init__.py

`python
# stephanie/agents/knowledge/__init__.py
``n

## File: adaptive_reasoner.py

`python
# stephanie/agents/knowledge/adaptive_reasoner.py
from __future__ import annotations

from typing import Union

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL
from stephanie.dataloaders import ARMDataLoader
from stephanie.evaluator import ARMReasoningSelfEvaluator, LLMJudgeEvaluator


class AdaptiveReasonerAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        self.modes = ["adaptive", "instruction_guided", "consensus_guided"]
        self.mode = self.cfg.get("mode", "adaptive")
        self.format_list = self.cfg.get(
            "format_list", ["direct", "short_cot", "code", "long_cot"]
        )
        self.judge = self._init_judge()

    async def run(self, context: dict):
        goal = context.get(GOAL)

        self.judge.train_from_database(goal.get("goal_text"), self.cfg)

        prompt = goal.get("goal_text")

        response = ""
        if self.mode == "instruction_guided":
            format_name = self.cfg.get("format", "long_cot")
            response = self._generate_with_format(format_name, context)
        elif self.mode == "consensus_guided":
            response = self._run_consensus_mode(context)
        else:  # default to adaptive
            response = self._run_adaptive_mode(prompt, context)

        self.logger.log("AdaptiveReasoningResponse", response)

        context[self.output_key] = response
        return context

    def _generate_with_format(self, fmt, context):
        prompt = self.prompt_loader.from_file(fmt, self.cfg, context)
        response = self.call_llm(prompt, context)
        return {
            "prompt": prompt,
            "response": response,
            "format_used": ARMDataLoader.detect_format(response) or fmt,
        }

    def _run_consensus_mode(self, context: dict):
        outputs = {}
        for fmt in ["direct", "short_cot", "code"]:
            outputs[fmt] = self._generate_with_format(fmt, context)["response"]

        responses = list(outputs.values())
        unique_responses = set(responses)

        if len(unique_responses) == 1:
            return {
                "response": responses[0],
                "format": "consensus-simple",
                "source_formats": list(outputs.keys()),
            }
        else:
            long_cot_response = self._generate_with_format("long_cot", context)
            return {
                "response": long_cot_response["response"],
                "format": "long_cot",
                "source_formats": list(outputs.keys()),
                "fallback_reason": "no_consensus",
            }

    def _run_adaptive_mode(
        self, prompt: str, context: dict
    ) -> dict[str, Union[str, float]]:
        prioritized_formats = ["direct", "short_cot", "code", "long_cot"]

        scores = {}
        for fmt in prioritized_formats:
            dict_response = self._generate_with_format(fmt, context)
            response = dict_response["response"]
            base_score = self.judge.score(prompt, response)

            token_len = len(response.split())
            rarity_bonus = 1.0 / (1 + self.judge.format_freq.get(fmt, 0))

            final_score = base_score - 0.01 * token_len + rarity_bonus
            scores[fmt] = final_score
            self.judge._update_format_stats(fmt, final_score)

        best_format = max(scores, key=scores.get)
        chosen_response = self._generate_with_format(best_format, context)
        # Log decision
        self.logger.log(
            "AdaptiveModeDecision",
            {"goal": prompt, "scores": scores, "chosen": best_format},
        )

        return {
            "response": chosen_response,
            "format_used": best_format,
            "scores": scores,
        }

    def get_format_for_goal(self, goal: dict):
        if "preferred_format" in goal:
            return goal["preferred_format"]
        goal_type = goal.get("goal_type", "default")
        if goal_type == "math":
            return "code"
        elif goal_type == "commonsense":
            return "short_cot"
        else:
            return "long_cot"

    def _get_prioritized_formats(self, context: dict):
        if "preferred_format" in context:
            return [context["preferred_format"]]

        priority_map = self.cfg.get("format_priority_by_difficulty", {})
        difficulty = context.get("difficulty", "default").lower()
        return priority_map.get(difficulty, priority_map.get("default", ["long_cot"]))

    def _init_judge(self):
        judge_strategy = self.cfg.get("judge", "mrq")
        if judge_strategy == "llm":
            llm = self.cfg.get("judge_model", self.cfg.get("model"))
            prompt_file = self.cfg.get(
                "judge_prompt_file", "judge_pairwise_comparison.txt"
            )
            self.logger.log(
                "EvaluatorInit", {"strategy": "LLM", "prompt_file": prompt_file}
            )
            return LLMJudgeEvaluator(
                self.cfg, llm, prompt_file, self.call_llm, self.logger
            )
        else:
            self.logger.log("EvaluatorInit", {"strategy": "ARM"})
            return ARMReasoningSelfEvaluator(self.cfg, self.memory, self.logger)
``n

## File: arxiv_search.py

`python
# stephanie/agents/knowledge/arxiv_search.py
"""
arXiv Search Agent Module

Provides functionality to search and retrieve academic papers from arXiv.org
based on research goals and extracted keywords. Includes robust error handling,
query construction, and result processing capabilities.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

import arxiv

from stephanie.agents.base_agent import BaseAgent


class ArxivSearchAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        # Configuration with defaults
        self.year_start = cfg.get("year_start", 2021)
        self.year_end = cfg.get("year_end", 2025)
        self.category = cfg.get("category", "cs.AI")  # Default to AI category
        self.max_results = cfg.get("max_results", 50)
        self.return_top_n = cfg.get("top_n", 10)
        self.date_filter = cfg.get("date_filter", "")

    async def run(self, context: dict) -> dict:
        """Main execution method for arXiv search agent"""
        goal = context.get("goal", {}).get("goal_text", "")

        # --- Performance reporting ---
        self.report({
            "event": "start",
            "step": "ArxivSearch",
            "details": f"Searching arXiv for goal: {goal}",
        })

        # Step 1: Keyword extraction
        keywords = self.extract_keywords(context)
        context["search_keywords"] = keywords

        self.report({
            "event": "keywords_extracted",
            "step": "ArxivSearch",
            "details": f"Extracted {len(keywords)} keywords",
            "keywords": keywords,
        })

        # Step 2: Query construction
        query = self.build_arxiv_query_from_goal(
            context=context,
            year_start=self.year_start,
            year_end=self.year_end,
            category=self.category,
            keywords=keywords,
        )

        self.report({
            "event": "query_built",
            "step": "ArxivSearch",
            "details": f"Built query with {len(keywords)} keywords",
            "query": query,
        })

        # Step 3: Fetch results
        results = []
        try:
            results = self.fetch_arxiv_results(
                context, query, max_results=self.max_results
            )
            context["raw_arxiv_results"] = results

            len_results = len(results) if results else 0

            self.report({
                "event": "search_complete",
                "step": "ArxivSearch",
                "details": f"Fetched {len_results} papers",
                "sample_titles": [r['title'] for r in results[:3]],  # First 3 titles
            })
        except Exception as e:
            self.report({
                "event": "error",
                "step": "ArxivSearch",
                "details": f"Error fetching arXiv results: {str(e)}",
            })

        context[self.output_key] = results

        # --- Completion report ---
        self.report({
            "event": "end",
            "step": "ArxivSearch",
            "details": f"Completed with {len(results)} results",
        })

        return context

    def extract_keywords(self, merged_context: dict) -> list:
        """Extract keywords using prompt-based approach with regex parsing"""
        response = self.execute_prompt(merged_context)
        # Match lines separated by newlines
        pattern = r"(?:\n|\r|\r\n)([^\n\r]+?)(?=(?:\n|\r|\r\n|$))"
        lines = re.findall(pattern, response.strip())
        # Clean numbering/bullets from lines
        keywords = [re.sub(r"^[-•\d\.\s]+", "", line).strip() for line in lines]

        # Debug logging
        self.logger.log(
            "KeywordsExtracted", {"raw_keywords": lines, "cleaned_keywords": keywords}
        )
        return [kw for kw in keywords if kw]  # Return non-empty keywords

    def build_arxiv_query_from_goal(
        self,
        context: dict,
        keywords: list[str],
        year_start: int = None,
        year_end: int = None,
        category: str = None,
    ) -> str:
        """Construct arXiv-compatible search query with filters"""
        keyword_filter = " OR ".join(f'"{kw.strip()}"' for kw in keywords if kw.strip())
        filters = [f"({keyword_filter})"]

        # Date filtering logic
        date_filter_mode = self.cfg.get("date_filter", "").lower()
        now = datetime.now()

        if date_filter_mode == "today":
            day = now.strftime("%Y%m%d")
            filters.append(f"submittedDate:[{day} TO {day}]")
        elif date_filter_mode == "week":
            start = (now - timedelta(days=7)).strftime("%Y%m%d")
            end = now.strftime("%Y%m%d")
            filters.append(f"submittedDate:[{start} TO {end}]")
        elif date_filter_mode == "month":
            start = (now - timedelta(days=30)).strftime("%Y%m%d")
            end = now.strftime("%Y%m%d")
            filters.append(f"submittedDate:[{start} TO {end}]")
        elif date_filter_mode == "year":
            start = (now - timedelta(days=365)).strftime("%Y%m%d")
            end = now.strftime("%Y%m%d")
            filters.append(f"submittedDate:[{start} TO {end}]")
        elif year_start or year_end:
            # Use year range if specified
            start = f"{year_start}0101" if year_start else "00000101"
            end = f"{year_end}1231" if year_end else "99991231"
            filters.append(f"submittedDate:[{start} TO {end}]")

        # Category filter
        if category:
            filters.append(f"cat:{category}")

        return " AND ".join(filters)

    def fetch_arxiv_results(
        self, context: dict, query: str, max_results: int = 50
    ) -> list[dict]:
        """
        Fetch papers from arXiv API with error handling and retry logic.
        Returns list of paper dictionaries with metadata.
        """
        results: list[dict] = []
        goal = context.get("goal", {})
        goal_id = goal.get("id", "")
        parent_goal = goal.get("goal_text")
        strategy = goal.get("strategy")
        focus_area = goal.get("focus_area")

        # Safety clamp on results
        max_results = min(max_results or self.max_results, 100)

        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending,
            )

            # Retry mechanism for transient errors
            for attempt in range(3):
                try:
                    for result in search.results():
                        try:
                            arxiv_url = getattr(result, "entry_id", "") or ""
                            pid = arxiv_url.split("/")[-1] if "/" in arxiv_url else None
                            if not pid:
                                continue  # Skip entries without ID

                            # Extract and clean fields
                            title = (getattr(result, "title", "") or "Unknown").strip()
                            summary = (getattr(result, "summary", "") or "").strip()
                            published = getattr(result, "published", None)
                            published_str = (
                                published.isoformat() if published else ""
                            )
                            authors = [
                                a.name for a in getattr(result, "authors", []) or []
                            ]
                            primary_category = getattr(
                                result, "primary_category", "unknown"
                            )

                            # Build result dictionary
                            results.append(
                                {
                                    "query": query,
                                    "source": self.name,
                                    "result_type": "paper",
                                    "title": title,
                                    "summary": summary,
                                    "url": f"https://arxiv.org/pdf/{pid}.pdf",
                                    "goal_id": goal_id,
                                    "parent_goal": parent_goal,
                                    "strategy": strategy,
                                    "focus_area": focus_area,
                                    "authors": authors,
                                    "published": published_str,
                                    "pid": pid,
                                    "primary_category": primary_category,
                                }
                            )
                        except Exception as parse_err:
                            self.logger.log(
                                "ArxivResultParseError",
                                {
                                    "error": str(parse_err),
                                    "entry_id": getattr(result, "entry_id", "unknown"),
                                },
                            )
                            continue

                    break  # Success - exit retry loop

                except arxiv.UnexpectedEmptyPageError as e:
                    self.logger.log(
                        "ArxivEmptyPageRetry",
                        {"query": query, "attempt": attempt + 1, "error": str(e)},
                    )
                    if attempt < 2:
                        import time
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        return []  # Failed after retries

        except Exception as e:
            self.logger.log(
                "ArxivSearchFailed",
                {"query": query, "error": str(e)},
            )
            return []

        # Log results
        if not results:
            self.logger.log(
                "ArxivNoResults",
                {"query": query, "goal_id": goal_id, "parent_goal": parent_goal},
            )
        else:
            self.logger.log(
                "ArxivResultsFetched",
                {
                    "query": query, 
                    "count": len(results), 
                    "first_title": results[0]["title"]
                },
            )

        return results
``n

## File: automind_knowledge_collector.py

`python
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
    "Graph Learning": ["Node Classification", "Link Prediction"],
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
                "goal_type": "model_review",
            },
            "search_queries": [{"goal_text": query}],
        }
        result = await self.agent.run(context)
        return result.get("search_results", [])

    async def collect_kaggle_solutions(self, task_description: str) -> List[Dict]:
        query = f"top {task_description} kaggle solution"
        context = {
            "goal": {
                "id": "kaggle_search",
                "goal_text": query,
                "goal_type": "data_search",
            },
            "search_queries": [{"goal_text": query}],
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
            {
                **doc,
                "labels": self.assign_labels_to_document(doc["title"], doc["summary"]),
            }
            for doc in all_docs
        ]

        # Filter by relevance to task description
        relevant_docs = self.filter_by_similarity(task_description, labeled_docs)

        # Re-rank by label priority
        reranked_docs = self.rerank_by_label_priority(relevant_docs)

        return reranked_docs

    def filter_by_similarity(self, query: str, docs: List[Dict]) -> List[Dict]:
        titles_and_summaries = [f"{doc['title']} {doc['summary']}" for doc in docs]
        scores = get_top_k_similar(
            query, titles_and_summaries, self.memory, top_k=len(docs)
        )
        ranked_indices = [i for i, _ in scores]
        return [docs[i] for i in ranked_indices]

    def rerank_by_label_priority(self, docs: List[Dict]) -> List[Dict]:
        label_priority = {
            "Computer Vision": 5,
            "NLP": 5,
            "Tabular Data": 4,
            "Image Classification": 3,
            "Text Classification": 3,
            "Classification": 2,
        }

        def score_doc(doc):
            return sum(label_priority.get(label, 0) for label in doc.get("labels", []))

        return sorted(docs, key=lambda x: score_doc(x), reverse=True)
``n

## File: cartridge.py

`python
# stephanie/agents/knowledge/cartridge.py

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.builders.cartridge_builder import CartridgeBuilder
from stephanie.builders.theorem_extractor import TheoremExtractor
from stephanie.builders.triplet_extractor import TripletExtractor
from stephanie.models.theorem import CartridgeORM
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.scorer.ebt_scorer import EBTScorer
from stephanie.scoring.scorer.mrq_scorer import MRQScorer
from stephanie.scoring.scorer.sicql_scorer import SICQLScorer
from stephanie.scoring.scorer.svm_scorer import SVMScorer


class CartridgeAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger, full_cfg=None):
        super().__init__(cfg, memory, container, logger)

        self.input_key = cfg.get("input_key", "documents")
        self.score_cartridges = cfg.get("score_cartridges", True)
        self.score_triplets = cfg.get("score_triplets", True)
        self.scorer_type = cfg.get("scorer_type", "sicql")  # default
        self.scorer = self._init_scorer(full_cfg, self.scorer_type)
        self.dimensions = cfg.get(
            "dimensions", ["relevance", "acy", "completeness"]
        )
        self.force_rescore = cfg.get("force_rescore", False)
        self.top_k_domains = cfg.get("top_k_domains", 3)
        self.min_classification_score = cfg.get(
            "min_classification_score", 0.6
        )
        self.force_rebuild_cartridges = cfg.get(
            "force_rebuild_cartridges", False
        )

        self.domain_classifier = ScorableClassifier(
            memory=self.memory,
            logger=self.logger,
            config_path=cfg.get(
                "domain_seed_config_path", "config/domain/cartridges.yaml"
            ),
        )

        self.builder = CartridgeBuilder(
            cfg,
            memory=self.memory,
            prompt_loader=self.prompt_loader,
            logger=self.logger,
            call_llm=self.call_llm,
        )
        self.triplet_extractor = TripletExtractor(
            cfg=cfg,
            prompt_loader=self.prompt_loader,
            memory=self.memory,
            logger=self.logger,
            call_llm=self.call_llm,
        )
        self.theorem_extractor = TheoremExtractor(
            cfg=cfg,
            prompt_loader=self.prompt_loader,
            memory=self.memory,
            logger=self.logger,
            call_llm=self.call_llm,
        )

    def _init_scorer(self, full_cfg, scorer_type):
        if scorer_type == "sicql":
            return SICQLScorer(
                full_cfg.scorer.sicql, memory=self.memory, logger=self.logger
            )
        if scorer_type == "mrq":
            return MRQScorer(
                full_cfg.scorer.mrq, memory=self.memory, logger=self.logger
            )
        if scorer_type == "svm":
            return SVMScorer(
                full_cfg.scorer.svm, memory=self.memory, logger=self.logger
            )
        if scorer_type == "ebt":
            return EBTScorer(
                full_cfg.scorer.ebt, memory=self.memory, logger=self.logger
            )
        raise ValueError(f"Unsupported scorer_type: {scorer_type}")

    async def run(self, context: dict) -> dict:
        documents = context.get(self.input_key, [])
        cartridges = []
        total_docs = len(documents)

        self.report(
            {
                "event": "start",
                "step": "CartridgeAgent",
                "total_documents": total_docs,
            }
        )

        for idx, doc in enumerate(
            tqdm(documents, desc="🔨 Building Cartridges", unit="doc"), start=1
        ):
            try:
                self.report(
                    {
                        "event": "document_start",
                        "step": "CartridgeAgent",
                        "doc_number": idx,
                        "doc_id": doc.get("id"),
                        "title": doc.get("title", "")[:100],
                    }
                )

                goal = context.get("goal")

                existing = self.memory.cartridges.get_by_source_uri(
                    uri=str(doc.get("id")), source_type="document"
                )
                if existing:
                    if context.get("pipeline_run_id"):
                        existing.pipeline_run_id = context["pipeline_run_id"]
                        self.memory.session.commit()

                    self.report(
                        {
                            "event": "cartridge_skipped_existing",
                            "step": "CartridgeAgent",
                            "doc_id": doc.get("id"),
                            "cartridge_id": existing.id,
                        }
                    )
                    cartridges.append(existing.to_dict())
                    continue

                # 1. Build CartridgeORM
                cartridge = self.builder.build(doc, goal=goal)
                if not cartridge:
                    self.report(
                        {
                            "event": "cartridge_skipped",
                            "step": "CartridgeAgent",
                            "doc_id": doc.get("id"),
                            "reason": "Builder returned None",
                        }
                    )
                    continue

                # 🔑 Ensure embedding_id is properly registered in document_embeddings
                scorable_text = cartridge.markdown_content or (
                    doc.get("summary") or doc.get("content")
                )
                scorable = Scorable(
                    id=cartridge.id,
                    text=scorable_text,
                    target_type=TargetType.CARTRIDGE,
                )
                if scorable_text:
                    # Guarantee row exists in document_embeddings
                    doc_embedding_id = (
                        self.memory.scorable_embeddings.get_or_create(scorable)
                    )
                    cartridge.embedding_id = doc_embedding_id

                self.report(
                    {
                        "event": "cartridge_built",
                        "step": "CartridgeAgent",
                        "cartridge_id": cartridge.id,
                        "title": cartridge.title[:80]
                        if cartridge.title
                        else "Untitled",
                    }
                )

                # 2. Extract Triplets
                if not self.memory.cartridge_triples.has_triples(cartridge.id):
                    triplets = self.triplet_extractor.extract(
                        cartridge.sections, context
                    )
                    self.report(
                        {
                            "event": "triplets_extracted",
                            "step": "CartridgeAgent",
                            "cartridge_id": cartridge.id,
                            "triplet_count": len(triplets),
                            "examples": triplets[:3],
                        }
                    )

                    for subj, pred, obj in triplets:
                        triple_orm = self.memory.cartridge_triples.insert(
                            {
                                "cartridge_id": cartridge.id,
                                "subject": subj,
                                "predicate": pred,
                                "object": obj,
                            }
                        )
                        if self.score_triplets:
                            scorable = ScorableFactory.from_text(
                                f"({subj}, {pred}, {obj})", TargetType.TRIPLE
                            )
                            score = self.scorer.score(
                                context=context,
                                scorable=scorable,
                                dimensions=self.dimensions,
                            )
                            context.setdefault("triplet_scores", []).append(
                                score
                            )

                # 3. Extract Theorems
                theorems = self.theorem_extractor.extract(
                    cartridge.sections, context
                )
                self.report(
                    {
                        "event": "theorems_extracted",
                        "step": "CartridgeAgent",
                        "cartridge_id": cartridge.id,
                        "theorem_count": len(theorems),
                        "examples": [t.statement[:80] for t in theorems[:2]],
                    }
                )

                for theorem in theorems:
                    # Save theorem first
                    self.memory.session.add(theorem)
                    self.memory.session.commit()

                    # Create embedding in document_embeddings (doc_id = theorem.id, type = "theorem")
                    scorable = Scorable(
                        id=theorem.id,
                        text=theorem.statement,
                        target_type=TargetType.THEOREM,
                    )
                    doc_embedding_id = (
                        self.memory.scorable_embeddings.get_or_create(scorable)
                    )

                    # Update theorem with FK to document_embeddings
                    theorem.embedding_id = doc_embedding_id
                    self.memory.session.commit()

                    # Link to cartridge
                    theorem.cartridges.append(cartridge)

                    # Score theorem
                    scorable = ScorableFactory.from_text(
                        theorem.statement, TargetType.THEOREM
                    )
                    theorem_score = self.scorer.score(
                        context={**context, "theorem": theorem.to_dict()},
                        scorable=scorable,
                        dimensions=self.dimensions,
                    )
                    context.setdefault("theorem_scores", []).append(
                        theorem_score
                    )

                self.memory.session.commit()

                # 4. Score Cartridge
                if self.score_cartridges:
                    scorable = ScorableFactory.from_text(
                        cartridge.markdown_content, TargetType.CARTRIDGE
                    )
                    score = self.scorer.score(
                        context=context,
                        scorable=scorable,
                        dimensions=self.dimensions,
                    )
                    context.setdefault("cartridge_scores", []).append(score)
                    self.report(
                        {
                            "event": "cartridge_scored",
                            "step": "CartridgeAgent",
                            "cartridge_id": cartridge.id,
                            "score": getattr(score, "value", None),
                        }
                    )

                # 5. Assign Domains
                self.assign_domains(cartridge)

                self.report(
                    {
                        "event": "document_done",
                        "step": "CartridgeAgent",
                        "cartridge_id": cartridge.id,
                        "doc_number": idx,
                        "total_documents": total_docs,
                    }
                )

                cartridges.append(cartridge.to_dict())

            except Exception as e:
                self.report(
                    {
                        "event": "error",
                        "step": "CartridgeAgent",
                        "doc_number": idx,
                        "error": str(e),
                    }
                )

        self.report(
            {
                "event": "end",
                "step": "CartridgeAgent",
                "processed_cartridges": len(cartridges),
                "total_documents": total_docs,
            }
        )

        context[self.output_key] = cartridges
        context["cartridge_ids"] = [c.get("id") for c in cartridges]
        return context

    def assign_domains(self, cartridge: CartridgeORM):
        """Classify and log domains for the cartridge."""
        if not cartridge.markdown_content:
            return

        existing = self.memory.cartridge_domains.get_domains(cartridge.id)
        if existing:
            self.logger.log(
                "DomainAssignmentSkipped",
                {
                    "cartridge_id": cartridge.id,
                    "existing_domains": [e.domain for e in existing],
                },
            )
            return

        results = self.domain_classifier.classify(
            cartridge.markdown_content,
            top_k=self.top_k_domains,
            min_value=self.min_classification_score,
        )
        for domain, score in results:
            self.memory.cartridge_domains.insert(
                {
                    "cartridge_id": cartridge.id,
                    "domain": domain,
                    "score": score,
                }
            )
            self.logger.log(
                "DomainAssigned",
                {
                    "title": cartridge.title[:60],
                    "domain": domain,
                    "score": score,
                },
            )
``n

## File: chat_analyze.py

`python
from stephanie.agents.base_agent import BaseAgent
import re
import logging

_logger = logging.getLogger(__name__)


class ChatAnalyzeAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.limit = cfg.get("limit", 10000)

    async def run(self, context: dict) -> dict:
        batch = self.memory.chats.list_turns_with_texts(
            min_assistant_len=50,  # skip trivial replies 
            limit=self.limit,
            order_desc=False
        )
        out = []
        for row in batch:
            turn_id = row.get("id")
            if row.get("ai_score") is not None:
                _logger.info(f"ChatAnalyzeAgent: Already analyzed turn_id: {turn_id}")
                continue

            merged_context = {**row, **context}
            prompt = self.prompt_loader.load_prompt(self.cfg, merged_context)

            response = self.call_llm(prompt, context)
            try:
                parsed = parse_knowledge_judge_text(response)
            except ParseError as e:
                _logger.error(f"ChatAnalyzeAgent: Failed to parse LLM response for turn_id: {turn_id}, error: {e}")
                continue

            self.memory.chats.set_turn_ai_eval(turn_id=turn_id, score=parsed["score"], rationale=parsed["rationale"])

            _logger.info(f"ChatAnalyzeAgent: Upserted turn analysis for turn_id: {turn_id} with score: {parsed['score']}")
            out.append(turn_id)
        context["analyzed_turn_ids"] = out
        return context


class ParseError(ValueError):
    pass

def parse_knowledge_judge_text(raw: str) -> dict:
    """
    Parse a plain-text knowledge-judge response into:
      {'rationale': str, 'score': int}

    Expected format (case-insensitive keys, rationale may be multi-line):
      rationale: <text...>
      score: <0-100>

    Tolerates leading/trailing whitespace and code fences.
    Raises ParseError on failure or out-of-range score.
    """
    if not raw or not raw.strip():
        raise ParseError("Empty response")

    # Strip common code fences (```...```)
    text = raw.strip()
    if text.startswith("```"):
        # remove first fence line and any trailing ```
        text = re.sub(r"^```[^\n]*\n", "", text, flags=re.DOTALL)
        text = re.sub(r"\n```$", "", text, flags=re.DOTALL)

    # Normalize line endings
    text = text.strip().replace("\r\n", "\n").replace("\r", "\n")

    # Primary pattern: rationale (greedy, up to the 'score:' line), then score
    m = re.search(
        r"(?is)^\s*rationale\s*:\s*(.*?)\n\s*score\s*:\s*([0-9]{1,3})\s*$",
        text,
        flags=re.MULTILINE,
    )
    if not m:
        # Fallback: find score line anywhere; take everything before as rationale
        ms = re.search(r"(?im)^\s*score\s*:\s*([0-9]{1,3})\s*$", text)
        if not ms:
            raise ParseError("Could not find 'score:' line")
        score = int(ms.group(1))
        if not (0 <= score <= 100):
            raise ParseError("Score out of range 0..100")

        # Prefer an explicit 'rationale:' label before score; else take preceding text
        mr = re.search(r"(?is)^\s*rationale\s*:\s*(.*)$", text[:ms.start()], flags=re.MULTILINE)
        rationale = (mr.group(1).strip() if mr else text[:ms.start()].strip())
        if not rationale:
            raise ParseError("Could not find 'rationale:' text")
        return {"rationale": rationale, "score": score}

    rationale = m.group(1).strip()
    score = int(m.group(2))
    if not (0 <= score <= 100):
        raise ParseError("Score out of range 0..100")

    return {"rationale": rationale, "score": score}

``n

## File: chat_annotate.py

`python
"""
Chat Annotation Agent

This agent enriches chat conversations with domain classification and named entity recognition (NER)
to support the knowledge extraction and learning process. It processes conversations in bulk,
adding semantic annotations that enable better organization, retrieval, and analysis of chat content.

Key Features:
- Domain classification using both seed-based and goal-aware classifiers
- Named Entity Recognition with optional Knowledge Graph integration
- Progress tracking with tqdm progress bars
- Idempotent operation (only processes missing annotations by default)
- Comprehensive logging and reporting
"""

from __future__ import annotations

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.tools.turn_domains_tool import annotate_conversation_domains
from stephanie.tools.turn_ner_tool import annotate_conversation_ner


class ChatAnnotateAgent(BaseAgent):
    """
    Agent that enriches chat conversations with domain and NER annotations.
    
    This agent processes one or multiple conversations, adding:
    1. Domain classifications (what the conversation is about)
    2. Named Entity Recognition (people, places, concepts mentioned)
    
    The annotations are stored directly in the database and can be used for
    improved retrieval, filtering, and knowledge extraction.
    """

    def __init__(self, cfg, memory, container, logger):
        # Initialize parent class with configuration, memory, container and logger
        super().__init__(cfg, memory, container, logger)
        
        # Initialize domain classifiers with configuration paths
        self.seed_classifier = ScorableClassifier(
            memory, logger, config_path=cfg.get("seed_config", "config/domain/seeds.yaml")
        )
        self.goal_classifier = ScorableClassifier(
            memory, logger, config_path=cfg.get("goal_config", "config/domain/goal_prompt.yaml")
        )

        # Domain classification settings
        self.max_k = int(cfg.get("max_domains_per_source", 3))  # Max domains per turn
        self.min_conf = float(cfg.get("min_confidence", 0.10))  # Minimum confidence threshold

        # Processing controls
        self.limit = int(cfg.get("limit", 1000))  # Maximum conversations to process
        self.only_missing = bool(cfg.get("only_missing", True))  # Skip already annotated turns
        self.force = bool(cfg.get("force", False))  # Force re-annotation of all turns
        self.progress_enabled = bool(cfg.get("progress", True))  # Enable/disable progress bars

    async def run(self, context: dict) -> dict:
        """
        Execute the annotation process on available chat conversations.
        
        Args:
            context: Execution context dictionary
            
        Returns:
            Updated context with annotation summary
        """
        # Retrieve conversations to process
        chats = self.memory.chats.get_all(limit=self.limit)

        # Pre-count turns that need processing (respects only_missing/force settings)
        def count_turns_for(chat_id: int, missing: str | None) -> int:
            rows = self.memory.chats.get_turn_texts_for_conversation(
                chat_id, only_missing=missing if (self.only_missing and not self.force) else None
            )
            return len(rows)

        # Calculate totals for progress tracking
        total_domains = sum(count_turns_for(c.id, "domains") for c in chats)
        total_ner = sum(count_turns_for(c.id, "ner") for c in chats)
        total_turns = total_domains + total_ner

        # Log start of annotation process
        self.logger.log("ChatAnnotateStart", {
            "conversations": len(chats),
            "turns_domains": total_domains,
            "turns_ner": total_ner,
            "turns_total": total_turns
        })

        # Initialize Knowledge Graph if available
        kg = self.container.get("knowledge_graph") 
        if kg:
            kg.initialize()

        # Create global progress bar for both domains and NER
        pbar = tqdm(
            total=total_turns or 1,
            desc="Annotating chats (domains+NER)",
            disable=not self.progress_enabled,
        )

        # Track progress for each annotation type
        dom_done = ner_done = 0

        def _bump_domains(n=1):
            """Callback to update progress for domain annotations"""
            nonlocal dom_done
            dom_done += n
            pbar.update(n)
            pbar.set_postfix({"domains": dom_done, "ner": ner_done})

        def _bump_ner(n=1):
            """Callback to update progress for NER annotations"""
            nonlocal ner_done
            ner_done += n
            pbar.update(n)
            pbar.set_postfix({"domains": dom_done, "ner": ner_done})

        # Initialize statistics counters
        totals = {
            "conversations": 0,
            "dom_seen": 0, "dom_updated": 0,
            "ner_seen": 0, "ner_updated": 0,
        }

        # Process each conversation
        for chat in chats:
            # Link conversation to a goal (using title as goal text)
            goal = self.memory.goals.get_or_create({
                "goal_text": chat.title,
                "description": f"Conversation imported on {chat.created_at}",
            }).to_dict()
            
            # Report goal linking
            self.report({
                "event": "goal_linked", 
                "conversation_id": chat.id,
                "goal_id": goal["id"], 
                "goal_text": goal["goal_text"]
            })

            # Annotate domains for this conversation
            dom_stats = annotate_conversation_domains(
                self.memory, 
                chat.id,
                seed_classifier=self.seed_classifier,
                goal_classifier=self.goal_classifier,
                goal=goal,
                max_k=self.max_k,
                min_conf=self.min_conf,
                only_missing=(self.only_missing and not self.force),
                progress_cb=_bump_domains,  # Progress callback
            )
            
            # Report domain annotation results
            self.report({
                "event": "domains_annotated", 
                "conversation_id": chat.id,
                "seen": dom_stats["seen"], 
                "updated": dom_stats["updated"]
            })

            # Annotate NER for this conversation
            ner_stats = annotate_conversation_ner(
                self.memory, 
                chat.id,
                kg=kg,  # Knowledge Graph for entity detection
                only_missing=(self.only_missing and not self.force),
                publish_to_kg=True,  # Publish entities to Knowledge Graph
                progress_cb=_bump_ner,  # Progress callback
            )
            
            # Report NER annotation results
            self.report({
                "event": "ner_annotated", 
                "conversation_id": chat.id,
                "seen": ner_stats["seen"], 
                "updated": ner_stats["updated"]
            })

            # Update totals
            totals["conversations"] += 1
            totals["dom_seen"] += dom_stats["seen"]
            totals["dom_updated"] += dom_stats["updated"]
            totals["ner_seen"] += ner_stats["seen"]
            totals["ner_updated"] += ner_stats["updated"]

            # Log progress for this conversation
            self.logger.log("ChatAnnotateProgress", {
                "conversation_id": chat.id, 
                **dom_stats, 
                **(ner_stats or {})
            })

        # Close progress bar and log completion
        pbar.close()
        self.logger.log("ChatAnnotateDone", {
            **totals, 
            "turns_domains": total_domains,
            "turns_ner": total_ner, 
            "turns_total": total_turns
        })
        
        # Add summary to context for downstream processing
        context["chat_annotation_summary"] = {
            **totals, 
            "turns_domains": total_domains,
            "turns_ner": total_ner, 
            "turns_total": total_turns
        }
        
        return context
``n

## File: chat_import.py

`python
"""
Chat Import Agent

This agent handles the import of chat conversations from various formats (JSON/HTML)
into Stephanie's memory system. It serves as the operational wrapper that coordinates
the import process within the broader pipeline workflow.

Key Responsibilities:
- Initializes import configuration and optional database purging
- Executes the chat import process via the chat_importer tool
- Provides comprehensive logging of import operations
- Updates execution context to signal successful import completion

The agent ensures chat data is properly ingested and available for downstream
processing including casebook creation and knowledge extraction.
"""

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL
from stephanie.tools.chat_importer import import_conversations


class ChatImportAgent(BaseAgent):
    """
    Agent that imports chat conversations from a directory
    and stores them as CaseBooks in Stephanie's memory.
    """

    def __init__(self, cfg, memory, container, logger):
        # Initialize parent class with configuration, memory, container and logger
        super().__init__(cfg, memory, container, logger)
        
        # Set import path from config or use default
        self.import_path = cfg.get("import_path", "data/chats")
        
        # Optionally purge existing conversations if configured
        if cfg.get("purge_existing", False):
            self.memory.chats.purge_all(True)

    async def run(self, context: dict) -> dict:
        # Extract goal from context for logging purposes
        goal = context.get(GOAL, {})
        
        # Log the start of import operation
        self.logger.log("ChatImportStart", {
            "import_path": self.import_path,
            "goal": goal.get("goal_text") if goal else None
        })

        # Execute the import process
        try:
            # Call the import_conversations tool to handle actual import
            summary = import_conversations(self.memory, self.import_path, context=context)

            # Log detailed summary of import results
            self.logger.log("ChatImportSuccess", {
                "import_path": self.import_path,
                "files_processed": summary.get("files_processed", 0),
                "files_skipped": summary.get("files_skipped", 0),
                "conversations_imported": summary.get("conversations_imported", 0),
                # Optional: if you add case/scorable counts to summary, log them too
                "cases_created": summary.get("cases_created", 0),
                "scorables_created": summary.get("scorables_created", 0),
            })

        except Exception as e:
            # Log any errors during import process
            self.logger.log("ChatImportError", {
                "error": str(e),
                "import_path": self.import_path
            })
            raise

        # Update context to indicate successful import
        context["chat_imported"] = True
        return context
``n

## File: chat_knowledge_builder.py

`python
# stephanie/agents/knowledge/chat_knowledge_builder.py
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch

from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.data.knowledge_unit import KnowledgeUnit
from stephanie.knowledge.casebook_store import Scorable
from stephanie.memory.chat_store import ChatStore
from stephanie.models.ner_retriever import EntityDetector
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType

_logger = logging.getLogger(__name__)

class ChatKnowledgeBuilder:
    """
    AI-powered builder of structured knowledge units from conversations and documents.

    Integrates:
      - Domain classification (ScorableClassifier)
      - Named Entity Recognition (EntityDetector)
      - Knowledge Graph linking (KnowledgeGraphService)
      - Conversation history (ChatStore)

    Follows Stephanie agent pattern: initialized with cfg, memory, container, logger.
    Gracefully degrades when subsystems fail.
    """

    def __init__(self, cfg: Dict[str, Any], memory: Any, container, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        # Lazily initialize components only if needed
        try:
            self.classifier = ScorableClassifier(
                memory=memory,
                logger=self.logger,
                config_path=cfg.get("domain_config", "config/domain/seeds.yaml"),
                metric=cfg.get("domain_metric", "cosine")
            )
            self.logger.info("Domain classifier loaded.")
        except Exception as e:
            self.classifier = None
            _logger.error(f"Failed to initialize ScorableClassifier: {e}")

        try:
            self.entity_detector = EntityDetector(
                device=cfg.get("device", "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu")
            )
            self.logger.info("NER detector loaded.")
        except Exception as e:
            self.entity_detector = None
            _logger.error(f"Failed to initialize EntityDetector: {e}")

        try:
            self.kg_service = self.container.get("knowledge_graph")
            self.logger.info("KnowledgeGraphService connected.")
        except Exception as e:
            self.kg_service = None
            _logger.error(f"Failed to initialize KnowledgeGraphService: {e}")

        try:
            self.chat_store = ChatStore(memory.session, logger=self.logger)
            self.logger.info("ChatStore connected.")
        except Exception as e:
            self.chat_store = None
            _logger.error(f"Failed to initialize ChatStore: {e}")

    # ---------------------------
    # Public API
    # ---------------------------
    def build(
        self,
        chat_messages: List[Dict[str, Any]],
        paper_text: str,
        conversation_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, KnowledgeUnit]:
        """
        Build aligned knowledge units from user chat and paper text.
        Optionally enrich with prior conversation context.
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Extract plain text from messages
            chat_text = " ".join(
                m["text"] for m in chat_messages
                if isinstance(m.get("text"), str) and m["text"].strip()
            ).strip()

            # Generate stable IDs
            chat_hash = hashlib.sha256(chat_text.encode()).hexdigest()[:16]
            paper_hash = hashlib.sha256(paper_text.encode()).hexdigest()[:16]


            chat_scorable = self.memory.dynamic_scorables.add(
                pipeline_run_id=context.get("pipeline_run_id"),
                scorable_type=TargetType.DYNAMIC,
                source="chat_knowledge_builder",
                text=chat_text,
                meta={"hash": chat_hash, "kind": "chat"}
            ) 

            paper_scorable = self.memory.dynamic_scorables.add(
                pipeline_run_id=context.get("pipeline_run_id"),
                scorable_type=TargetType.DYNAMIC,
                source="chat_knowledge_builder",
                text=paper_text,
                meta={"hash": paper_hash, "kind": "paper"}
            ) 
            chat_ku = self._process_with_ai(ScorableFactory.from_orm(chat_scorable), source="chat")
            paper_ku = self._process_with_ai(ScorableFactory.from_orm(paper_scorable), source="paper")


            # Enrich with historical context
            if conversation_id and self.chat_store:
                ctx_ku = self._build_contextual_knowledge(conversation_id)
                if ctx_ku.stats.get("error"):
                    _logger.debug(f"Context enrichment skipped: {ctx_ku.stats['error']}")
                else:
                    chat_ku.provenance["context_from_conversation"] = ctx_ku.to_dict()

            # Final stats
            duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.logger.info(f"Built knowledge pair in {duration:.0f}ms", extra={
                "chat_entities": sum(len(v) for v in chat_ku.entities.values()),
                "paper_entities": sum(len(v) for v in paper_ku.entities.values()),
                "domains_matched": len(set(chat_ku.domains) & set(paper_ku.domains))
            })

            return {"chat": chat_ku, "paper": paper_ku}

        except Exception:
            self.logger.error("ChatKnowledgeBuilder.build failed", exc_info=True)
            raise

    # ---------------------------
    # Core Processing Pipeline
    # ---------------------------
    def _process_with_ai(self, scorable: Scorable, source: str) -> KnowledgeUnit:
        text = scorable.text
        scorable_id = scorable.id
        if not text.strip():
            return KnowledgeUnit(text="", stats={"empty": True, "source": source})

        start_time = datetime.now(timezone.utc)

        domains, raw_entities, entities_by_type, kg_nodes = {}, [], {}, []
        phrases, anchors = [], []

        # 1. Domain Classification
        if self.classifier:
            try:
                domain_matches = self.classifier.classify(text=text, top_k=5)
                domain_scores = [{"domain": d, "score": float(s)} for d, s in domain_matches]

                domains = {d["domain"]: float(d["score"]) for d in domain_scores if d.get("score", 0) > 0.01}
            except Exception as e:
                _logger.error(f"[{source}] Domain classification failed: {e}")
        else:
            domains = {}

        # 2. Entity Detection
        if self.entity_detector:
            try:
                raw_entities = self.entity_detector.detect_entities(text)
                for ent in raw_entities:
                    entities_by_type.setdefault(ent["type"], []).append(ent)
            except Exception as e:
                _logger.error(f"[{source}] NER extraction failed: {e}")
        else:
            entities_by_type = {}

        # 3. Phrase Extraction
        phrases = self._extract_salient_phrases(text, domains, entities_by_type)

        # 4. Link to Knowledge Graph
        if self.kg_service and self.kg_service._initialized:
            try:
                for ent in raw_entities:
                    # Unique local provenance for this entity
                    node_id = f"{scorable_id}:{ent['type']}:{ent['start']}-{ent['end']}"
                    matched_nodes = []

                    # Fallback: search by embedding of entity text
                    query_vec = self.kg_service._retriever.embed_type_query(ent["text"])
                    results = self.kg_service._graph.search(query_vec, k=3)

                    for _, score, meta in results:
                        if meta.get("text", "").lower() == ent["text"].lower():
                            matched_nodes.append({
                                **meta,                       # KG-provided metadata
                                "score": float(score),        # similarity score
                                "node_id": node_id,           # our stable local ID
                                "scorable_id": scorable_id,   # provenance back to source scorable
                                "entity_text": ent["text"],   # original surface form
                                "entity_type": ent["type"],   # PERSON, ORG, etc
                                "span": f"{ent['start']}-{ent['end']}",
                                "source": source              # chat | paper | context
                            })

                    kg_nodes.extend(matched_nodes)
            except Exception as e:
                _logger.error(f"[{source}] KG linking failed: {e}")
        else:
            kg_nodes = []

        # 5. Select Anchors
        anchors = self._select_anchors(phrases, domains, entities_by_type)

        return KnowledgeUnit(
            text=text,
            domains=domains,
            phrases=phrases,
            anchors=anchors,
            entities=entities_by_type,
            linked_kg_nodes=kg_nodes,
            provenance={
                "source": source,
                "scorable_id": scorable_id,
                "used_classifier": bool(self.classifier and domains),
                "used_ner": bool(self.entity_detector and raw_entities),
                "used_kg": bool(self.kg_service and kg_nodes),
            },
            stats={
                "char_length": len(text),
                "word_count": len(text.split()),
                "phrase_count": len(phrases),
                "entity_count": sum(len(v) for v in entities_by_type.values()),
                "kg_link_count": len(kg_nodes),
                "processing_duration_ms": int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000),
                "timestamp": start_time.isoformat(),
            },
        )

    # ---------------------------
    # Helpers
    # ---------------------------
    def _extract_salient_phrases(self, text: str, domains: Dict[str, float], entities: Dict[str, List[Dict]]) -> List[str]:
        import re
        sentences = [s.strip() for s in re.split(r"[.;!?]", text) if len(s.strip().split()) >= 2]
        domain_terms = {d for d in domains if domains[d] > 0.2}
        entity_texts = {e["text"].lower() for ents in entities.values() for e in ents}

        scored = []
        for sent in sentences:
            words = sent.lower().split()
            score = len(words)
            if any(d.lower() in words for d in domain_terms):
                score += 2.0
            if any(e in words for e in entity_texts):
                score += 1.5
            scored.append((sent, score))

        scored.sort(key=lambda x: -x[1])
        return [s for s, _ in scored[:50]]

    def _select_anchors(self, phrases: List[str], domains: Dict[str, float], entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        domain_terms = {d.lower() for d in domains if domains[d] > 0.2}
        entity_texts = {e["text"].lower() for ents in entities.values() for e in ents}

        anchors = []
        for p in phrases[:20]:
            words = p.lower().split()
            score = len(words)
            if any(d in words for d in domain_terms):
                score += 1.5
            if any(e in words for e in entity_texts):
                score += 1.0
            anchors.append({
                "span": p,
                "score": float(score),
                "length": len(words),
                "contains_entity": any(e in words for e in entity_texts),
                "overlaps_domain": any(d in words for d in domain_terms)
            })

        anchors.sort(key=lambda x: -x["score"])
        return anchors[:10]

    def _build_contextual_knowledge(self, conversation_id: int) -> KnowledgeUnit:
        if not self.chat_store:
            return KnowledgeUnit(text="", stats={"error": "chat_store_unavailable"})

        try:
            conv = self.chat_store.get_conversation(conversation_id)
            if not conv:
                return KnowledgeUnit(text="", stats={"error": "conversation_not_found"})

            turns = self.chat_store.get_turns_for_conversation(conversation_id)[-5:]
            snippets = []
            for turn in turns:
                u = (turn.user_message.text or "").strip()
                a = (turn.assistant_message.text or "").strip()
                if u or a:
                    snippets.append(f"USER: {u}\nASSISTANT: {a}")

            combined = "\n\n".join(snippets).strip()
            if not combined:
                return KnowledgeUnit(text="", stats={"error": "no_content_in_context"})

            return self._process_with_ai(
                text=combined,
                source=f"context:{conversation_id}",
                scorable_id=f"context:{conversation_id}:{hash(combined)}"
            )
        except Exception as e:
            _logger.error(f"Context enrichment failed for conv={conversation_id}: {e}")
            return KnowledgeUnit(text="", stats={"error": str(e)}) 
``n

## File: conversation_filter.py

`python
"""
ConversationFilterAgent (Enhanced)
----------------------------------
Scores chat messages for relevance to a paper section and keeps only the
high-signal ones for downstream KnowledgeFusion/Drafting.

Key improvements:
- Windowed processing for long conversations (memory-safe)
- Hybrid scoring: embeddings + VPM + evidence/technical cues
- Dynamic thresholds (IQR-based) per section quality
- Optional domain-specific adjustments
- Critical path identification (causal learning trajectory)
- Robust embedding fallbacks (won’t crash on failures)

Input context:
  paper_section: { section_name, section_text, paper_id, domain? }
  chat_corpus:   [ { id?, role, text, timestamp? }, ... ]
  goal_template?: str (default "academic_summary")

Output context additions:
  scored_messages:   [ { ...message, score, similarity, vpm_score, reason, ... } ]
  critical_messages: high-signal subset (score ≥ dynamic threshold)
  critical_path:     ordered “learning path” through critical messages
  filter_threshold:  the dynamic threshold used
"""

from __future__ import annotations

import logging
import re
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.paper_improver.goals import GoalScorer

# ---- cues / patterns ----
EVIDENCE_HINTS = (
    "see figure", "fig.", "figure", "table", "tbl.", "[#]", "as shown",
    "we report", "results in", "demonstrated in", "shown in", "according to"
)
PROGRESS_HINTS = (
    "solution", "solve", "fixed", "works", "implemented", "approach",
    "method", "technique", "here's what we tried", "next we", "then we",
    "now we can", "this led to", "therefore", "consequently", "thus", "hence"
)
FACTUAL_HINTS = (
    "result", "show", "prove", "achiev", "increase", "decrease",
    "outperform", "error", "accuracy", "loss", "significant", "statistically"
)
TECHNICAL_TERMS = (
    "transformer", "adapter", "loss", "optimization", "pipeline",
    "retrieval", "graph", "policy", "reward", "ablation", "evaluation"
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _sentences(t: str) -> List[str]:
    if not t:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(t) if len(s.strip()) > 2]

def _lex_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    aw = set(re.findall(r"\b\w+\b", a.lower()))
    bw = set(re.findall(r"\b\w+\b", b.lower()))
    return (len(aw & bw) / max(1, len(aw))) if aw else 0.0

def _cos(u: np.ndarray, v: np.ndarray) -> float:
    if u is None or v is None:
        return 0.0
    num = float(np.dot(u, v))
    den = (float(np.dot(u, u)) ** 0.5) * (float(np.dot(v, v)) ** 0.5) + 1e-8
    return num / den


@dataclass
class CFConfig:
    # thresholds
    min_keep: float = 0.55
    z_sigma: float = 0.5  # legacy (kept; dynamic threshold takes precedence)
    # weights
    w_embed: float = 0.55
    w_vpm: float   = 0.30
    w_extra: float = 0.15
    # limits
    max_msgs: int = 1200
    window_size: int = 60
    window_stride: int = 30
    # toggles
    use_embeddings: bool = True
    use_temporal_boost: bool = True
    use_domain_rules: bool = True
    use_critical_path: bool = True
    # evidence / penalties
    evidence_bonus: float = 0.10
    chatter_penalty: float = 0.08
    length_cutoff: int = 18
    # logging
    log_top_k: int = 5


class ConversationFilterAgent(BaseAgent):
    def __init__(self, cfg: Dict[str, Any], memory: Any, container: Any, logger):
        super().__init__(cfg, memory, container, logger)
        self.kfg = CFConfig(**cfg.get("conversation_filter", {}))
        self.goal_scorer = GoalScorer(logger=logger)
        self.logger.info("ConversationFilterAgent initialized", {"config": asdict(self.kfg)})

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()

        section = context.get("paper_section") or {}
        chat    = (context.get("chat_corpus") or [])[: self.kfg.max_msgs]
        goal_template = context.get("goal_template") or "academic_summary"

        section_text = (section.get("section_text") or "").strip()
        section_name = section.get("section_name", "section")
        section_domain = (section.get("domain") or "general").lower()

        if not section_text or not chat:
            self.logger.log("ConversationFilterSkipped", {
                "has_section_text": bool(section_text), "chat_count": len(chat)
            })
            return context

        # drop empties/very short
        chat = [m for m in chat if m.get("text") and len(m["text"].strip()) > 5]

        # section embedding (robust)
        sec_emb = None
        if self.kfg.use_embeddings:
            try:
                sec_emb = self._section_embedding(section_text)
            except Exception as e:
                self.logger.log("SectionEmbeddingFailed", {"error": str(e), "section": section_name})

        try:
            scored = self._score_messages(chat, section_text, sec_emb, goal_template)

            if self.kfg.use_domain_rules:
                scored = self._apply_domain_rules(scored, section_domain)

            thr = self._dynamic_threshold([m["score"] for m in scored], self.kfg.min_keep)
            critical = [m for m in scored if m["score"] >= thr]
            critical.sort(key=lambda x: x["score"], reverse=True)

            critical_path = []
            if self.kfg.use_critical_path and critical:
                critical_path = self._critical_path(critical)

            context.update({
                "scored_messages": scored,
                "critical_messages": critical,
                "critical_path": critical_path,
                "filter_threshold": thr,
                "section_domain": section_domain,
            })

            self.logger.log("ConversationFilterComplete", {
                "section": section_name,
                "domain": section_domain,
                "chat_in": len(chat),
                "kept": len(critical),
                "threshold": round(thr, 3),
                "top_samples": [
                    {"score": round(m["score"],3), "sim": round(m.get("similarity",0.0),3), "text": m["text"][:120]}
                    for m in critical[: self.kfg.log_top_k]
                ],
                "elapsed_s": round(time.time() - start, 2),
            })
            return context

        except Exception as e:
            self.logger.log("ConversationFilterError", {
                "error": str(e), "traceback": traceback.format_exc(), "section": section_name
            })
            context["filter_error"] = str(e)
            return context

    # ---------------- internal: scoring ----------------

    def _score_messages(self, chat: List[Dict[str, Any]], section_text: str,
                        sec_emb: Optional[np.ndarray], goal_template: str) -> List[Dict[str, Any]]:
        """Score all messages with overlapping windows to limit memory and add locality bias."""
        if len(chat) <= self.kfg.window_size:
            return self._score_window(chat, section_text, sec_emb, goal_template)

        all_scored: List[Dict[str, Any]] = []
        for start in range(0, len(chat), self.kfg.window_stride):
            end = min(start + self.kfg.window_size, len(chat))
            window = chat[start:end]
            window_scored = self._score_window(window, section_text, sec_emb, goal_template)

            # slight position bias toward window center (more coherent clusters)
            n = len(window_scored) or 1
            mid = (n - 1) / 2.0
            for i, m in enumerate(window_scored):
                m["score"] *= (1.0 - (abs(i - mid) / max(1.0, mid)) * 0.1)
            all_scored.extend(window_scored)

        return self._dedup_by_id(all_scored)

    def _score_window(self, chat: List[Dict[str, Any]], section_text: str,
                      sec_emb: Optional[np.ndarray], goal_template: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for msg in chat:
            text = (msg.get("text") or "").strip()
            if not text:
                continue

            # embedding similarity (with robust fallback)
            sim = 0.0
            if self.kfg.use_embeddings and sec_emb is not None:
                sim = self._embedding_similarity(text, sec_emb, section_text)

            # VPM dimensions + score
            dims = self._vpm_dims(text, section_text)
            vpm_score = self.goal_scorer.score(kind="text", goal=goal_template, vpm_row=dims).get("score", 0.0)

            # extras: evidence/technical/chatter/factual cues
            extra = self._extra_score(text, dims)

            score = (
                self.kfg.w_embed * sim +
                self.kfg.w_vpm   * vpm_score +
                self.kfg.w_extra * extra
            )

            # light temporal smoothing: boost msg after a strong one
            if self.kfg.use_temporal_boost and out and out[-1]["score"] > 0.6:
                score = min(1.0, score * 1.15)

            out.append({
                **msg,
                "score": float(score),
                "similarity": float(sim),
                "vpm_score": float(vpm_score),
                "extra_score": float(extra),
                "vpm_dims": dims,
                "reason": self._reason_from_dims(dims, extra),
            })
        return out

    # ---------------- internal: features ----------------

    def _section_embedding(self, section_text: str) -> Optional[np.ndarray]:
        if not section_text or len(section_text) < 10:
            return None
        try:
            if len(section_text) <= 2000:
                return self._embed(section_text)
            sents = _sentences(section_text)
            if len(sents) >= 3:
                first = self._embed(sents[0][:2000])
                mid   = self._embed(sents[len(sents)//2][:2000])
                last  = self._embed(sents[-1][:2000])
                return (first + mid + last) / 3.0
            return self._embed(section_text[:2000])
        except Exception:
            return None

    def _embedding_similarity(self, text: str, sec_emb: np.ndarray, section_text: str) -> float:
        try:
            if len(text) > 50:
                return _cos(self._embed(text[:2000]), sec_emb)
            # very short → lexical overlap proxy
            if len(text.split()) < 5:
                return _lex_overlap(text, section_text)
            sents = _sentences(text)
            if sents:
                return _cos(self._embed(sents[0][:200]), sec_emb)
        except Exception:
            self.logger.log("EmbeddingFallback", {"len": len(text)})
            return _lex_overlap(text, section_text)
        return 0.0

    def _vpm_dims(self, text: str, section_text: str) -> Dict[str, float]:
        t = text.lower()
        # correctness: overlap boosted when factual terms appear
        correctness = 0.35 + 0.65 * _lex_overlap(text, section_text)
        if any(k in t for k in FACTUAL_HINTS):
            correctness = min(1.0, correctness * 1.2)
        # progress: pushes toward solutions; penalize raw “error” unless analyzed
        if any(k in t for k in PROGRESS_HINTS):
            progress = 0.75
        elif "error" in t or "not working" in t:
            progress = 0.25
        else:
            progress = 0.5
        # evidence: hard references; otherwise overlap proxy
        evidence = 0.9 if any(h in t for h in EVIDENCE_HINTS) else min(1.0, 1.2 * _lex_overlap(text, section_text))
        # novelty: “we found/new/alternative…”
        novelty = 0.8 if any(k in t for k in ("new approach", "alternative", "novel", "innovative", "we found", "surprising", "counterintuitive")) else 0.35
        return {
            "correctness": float(correctness),
            "progress": float(progress),
            "evidence": float(evidence),
            "novelty": float(novelty),
        }

    def _extra_score(self, text: str, dims: Dict[str, float]) -> float:
        t = text.lower()
        extra = 0.0
        if any(h in t for h in EVIDENCE_HINTS):
            extra += self.kfg.evidence_bonus
        if len(text) < self.kfg.length_cutoff or t in {"ok", "thanks", "got it", "yup", "cool"}:
            extra -= self.kfg.chatter_penalty
        if any(term in t for term in TECHNICAL_TERMS):
            extra += 0.05
        if any(kw in t for kw in FACTUAL_HINTS) and dims["correctness"] > 0.7:
            extra += 0.07
        return max(-0.2, min(0.2, extra))

    def _reason_from_dims(self, dims: Dict[str, float], extra: float) -> str:
        bits = []
        if dims["correctness"] > 0.7: bits.append("factually accurate")
        if dims["progress"]   > 0.7: bits.append("advances understanding")
        if dims["evidence"]   > 0.7: bits.append("well-supported")
        if dims["novelty"]    > 0.7: bits.append("provides new insight")
        if extra > 0.05: bits.append("strong evidence refs")
        if extra < -0.05: bits.append("likely chatter")
        return " and ".join(bits) if bits else "moderate relevance"

    # ---------------- internal: post-processing ----------------

    def _dynamic_threshold(self, scores: List[float], floor: float) -> float:
        if not scores:
            return floor
        s = sorted(scores)
        q1 = s[int(0.25 * (len(s)-1))]
        q3 = s[int(0.75 * (len(s)-1))]
        iqr = q3 - q1
        thr = q3 + 0.5 * iqr
        return max(floor, min(0.9, thr))

    def _apply_domain_rules(self, scored: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
        changed = 0
        for m in scored:
            before = m["score"]
            tl = m["text"].lower()
            if "machine learning" in domain:
                if "correlates with" in tl and "caus" not in tl:
                    m["score"] = max(0.0, m["score"] - 0.15)
                if any(t in tl for t in ["p-value", "confidence interval", "statistically significant"]):
                    m["score"] = min(1.0, m["score"] + 0.10)
            elif "biology" in domain:
                if any(t in tl for t in ["gene expression", "protein folding", "cellular mechanism"]):
                    m["score"] = min(1.0, m["score"] + 0.15)
            elif "math" in domain or "theory" in domain:
                if any(t in tl for t in ["proof", "theorem", "lemma", "corollary"]):
                    m["score"] = min(1.0, m["score"] + 0.12)
                if "basically" in tl or "kind of" in tl:
                    m["score"] = max(0.0, m["score"] - 0.10)
            if m["score"] != before:
                changed += 1
                m["domain_adjustment"] = round(m["score"] - before, 4)
        if changed:
            self.logger.log("DomainRulesApplied", {"domain": domain, "adjusted": changed})
        return scored

    def _critical_path(self, msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Greedy chain: high score + forward in time + causal/lexical link."""
        if not msgs:
            return []
        # sort by score desc, then by timestamp asc
        ranked = sorted(msgs, key=lambda x: (-x["score"], x.get("timestamp", 0)))
        path = [ranked[0]]
        last_ts = ranked[0].get("timestamp", 0)
        for m in ranked[1:]:
            ts = m.get("timestamp", 0)
            if ts < last_ts:
                continue
            if self._linked(path[-1]["text"], m["text"]):
                path.append(m)
                last_ts = ts
        # final path sorted by timestamp
        return sorted(path, key=lambda x: x.get("timestamp", 0))

    def _linked(self, prev: str, curr: str) -> bool:
        prev_l, curr_l = prev.lower(), curr.lower()
        causal_patterns = [
            r"so\s+we", r"therefore", r"as\s+a\s+result", r"because\s+of\s+this",
            r"consequently", r"thus", r"hence", r"this\s+led\s+to", r"follows\s+that"
        ]
        if any(re.search(p, prev_l + " " + curr_l) for p in causal_patterns):
            return True
        prev_words = set(prev_l.split()[:10])
        curr_words = set(curr_l.split())
        return len(prev_words & curr_words) > 2

    def _dedup_by_id(self, scored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        uniq: List[Dict[str, Any]] = []
        # keep best-scoring duplicate
        for m in sorted(scored, key=lambda x: -x["score"]):
            mid = m.get("id") or str(hash(m["text"][:64]))
            if mid in seen:
                continue
            seen.add(mid)
            uniq.append(m)
        return uniq

    def _embed(self, text: str) -> np.ndarray:
        vec = self.memory.embedding.get_or_create(text)
        return vec if isinstance(vec, np.ndarray) else np.asarray(vec, dtype=np.float32)
``n

## File: conversation_trajectory_mapper.py

`python
# stephanie/agents/knowledge/conversation_trajectory_mapper.py
"""
ConversationTrajectoryMapper (Upgraded)
---------------------------------------
Maps conversation trajectories to paper sections with causal relevance scoring.
Embedding alignment + HRM/MRQ (if available) + sentence-level evidence linking.
Emits graph-friendly mappings and can publish edges to the knowledge bus.

Input context:
  - paper_section: { id?, section_name, section_text, paper_id, goal? }
  - chat_corpus: [ { role, text, timestamp, id? }, ... ]  # not strictly required
  - conversation_trajectories: [
      {
        start_idx, end_idx,
        messages: [ { text, role?, id?, ts?, score?, is_critical? }, ... ],
        score, goal_achieved
      }, ...
    ]

Output added to context:
  - trajectory_mappings: [ { ... see return schema below ... } ]
  - critical_trajectory_mappings: [subset]
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stephanie.agents.base_agent import BaseAgent

_logger = logging.getLogger(__name__)


# ----------------------------
# Utilities
# ----------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\b\w+\b", re.UNICODE)

def _sentences(text: str, max_sents: int = 80) -> List[str]:
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 2]
    if max_sents and len(sents) > max_sents:
        return sents[:max_sents]
    return sents

def _words(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())

def _lexical_overlap(a: str, b: str) -> float:
    A, B = set(_words(a)), set(_words(b))
    if not A:
        return 0.0
    return len(A & B) / max(1, len(A))

def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    if u is None or v is None:
        return 0.0
    num = float(np.dot(u, v))
    den = (float(np.dot(u, u)) ** 0.5) * (float(np.dot(v, v)) ** 0.5) + 1e-8
    return num / den


@dataclass
class CTMConfig:
    min_causal_strength: float = 0.60         # floor for causal thresholding
    critical_sigma: float = 0.5               # z-threshold: mean + sigma * std
    evidence_top_k: int = 3                   # top-K span pairs to keep
    max_traj_sents: int = 80                  # cap per-trajectory sentence calc
    max_section_sents: int = 120              # cap section sentence calc
    use_embeddings: bool = True               # embedding similarity if available
    use_hrm: bool = True                      # try HRM scorer if available
    use_mrq: bool = True                      # try MRQ scorer if available
    publish_edges: bool = True                # publish graph edges to bus
    bus_subject: str = "knowledge.trajectory.mapping"


class ConversationTrajectoryMapper(BaseAgent):
    """
    Embedding + scorer hybrid mapper from conversation trajectories to paper sections.
    """

    def __init__(self, cfg: Dict[str, Any], memory: Any, logger: logging.Logger):
        super().__init__(cfg, memory, container, logger)
        self.kfg = CTMConfig(
            min_causal_strength=cfg.get("min_causal_strength", 0.60),
            critical_sigma=cfg.get("critical_sigma", 0.5),
            evidence_top_k=cfg.get("evidence_top_k", 3),
            max_traj_sents=cfg.get("max_traj_sents", 80),
            max_section_sents=cfg.get("max_section_sents", 120),
            use_embeddings=cfg.get("use_embeddings", True),
            use_hrm=cfg.get("use_hrm", True),
            use_mrq=cfg.get("use_mrq", True),
            publish_edges=cfg.get("publish_edges", True),
            bus_subject=cfg.get("bus_subject", "knowledge.trajectory.mapping"),
        )

        # Optional scorers (best-effort discovery)
        self.hrm_scorer = cfg.get("hrm_scorer")
        self.mrq_scorer = cfg.get("mrq_scorer")
        if self.kfg.use_hrm and self.hrm_scorer is None:
            self.hrm_scorer = getattr(self.memory, "hrm_scorer", None)
        if self.kfg.use_mrq and self.mrq_scorer is None:
            self.mrq_scorer = getattr(self.memory, "mrq_scorer", None)

        # Bus is optional; HybridKnowledgeBus recommended (NATS/InProcess)
        self.bus = getattr(self.memory, "bus", None)

        self.logger.info("ConversationTrajectoryMapper initialized", {
            "config": self.kfg.__dict__,
            "hrm_available": bool(self.hrm_scorer),
            "mrq_available": bool(self.mrq_scorer),
            "bus_available": bool(self.bus),
        })

    # ----------------------------
    # Agent entry point
    # ----------------------------
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        paper_section = context.get("paper_section") or {}
        trajectories = context.get("conversation_trajectories", [])
        section_text = (paper_section.get("section_text") or "").strip()

        if not section_text or not trajectories:
            self.logger.log("TrajectoryMappingSkipped", {
                "reason": "missing_inputs",
                "has_section_text": bool(section_text),
                "num_trajectories": len(trajectories)
            })
            return context

        # Pre-embed section if embeddings are enabled
        section_emb = None
        section_sents = _sentences(section_text, self.kfg.max_section_sents)
        section_sent_embs: List[np.ndarray] = []

        if self.kfg.use_embeddings:
            try:
                section_emb = self._embed(section_text)
                for s in section_sents:
                    section_sent_embs.append(self._embed(s))
            except Exception as e:
                self.logger.log("SectionEmbeddingFailed", {"error": str(e)})
                section_emb = None
                section_sent_embs = []

        # Score each trajectory
        mappings: List[Dict[str, Any]] = []
        for i, traj in enumerate(trajectories):
            try:
                mapping = self._map_trajectory_to_section(
                    idx=i,
                    trajectory=traj,
                    section_text=section_text,
                    section_emb=section_emb,
                    section_sents=section_sents,
                    section_sent_embs=section_sent_embs
                )
                mappings.append(mapping)
            except Exception as e:
                self.logger.log("TrajectoryMappingError", {
                    "trajectory_index": i, "error": str(e)
                })

        # Adaptive critical thresholding on causal_strength
        strengths = [m["causal_strength"] for m in mappings if "causal_strength" in m]
        if strengths:
            mu = float(np.mean(strengths))
            sigma = float(np.std(strengths))
            dynamic_thr = max(self.kfg.min_causal_strength, mu + self.kfg.critical_sigma * sigma)
        else:
            mu, sigma, dynamic_thr = 0.0, 0.0, self.kfg.min_causal_strength

        for m in mappings:
            m["is_critical"] = bool(m["causal_strength"] >= dynamic_thr)

        critical = [m for m in mappings if m["is_critical"]]

        # Optionally publish graph edges for each mapping
        if self.kfg.publish_edges and self.bus:
            await self._publish_edges(paper_section, mappings)

        # Update context
        self.logger.log("TrajectoryMappingComplete", {
            "section": paper_section.get("section_name"),
            "trajectories_in": len(trajectories),
            "mappings_out": len(mappings),
            "critical": len(critical),
            "mu": mu, "sigma": sigma, "dynamic_thr": dynamic_thr
        })
        context["trajectory_mappings"] = mappings
        context["critical_trajectory_mappings"] = critical
        return context

    # ----------------------------
    # Core mapping logic
    # ----------------------------
    def _map_trajectory_to_section(
        self,
        idx: int,
        trajectory: Dict[str, Any],
        section_text: str,
        section_emb: Optional[np.ndarray],
        section_sents: List[str],
        section_sent_embs: List[np.ndarray],
    ) -> Dict[str, Any]:
        msgs = trajectory.get("messages", []) or []
        traj_text = "\n".join(m.get("text", "") for m in msgs if m.get("text"))
        traj_sents = _sentences(traj_text, self.kfg.max_traj_sents)

        # --- Relevance (embedding + lexical)
        section_relevance = self._section_relevance(
            msgs=msgs,
            traj_sents=traj_sents,
            section_text=section_text,
            section_emb=section_emb
        )

        # --- Causal strength (hybrid)
        causal_strength = self._causal_strength(
            trajectory_text=traj_text,
            section_text=section_text,
            hrm_score=self._score_hrm(trajectory),
            mrq_score=self._score_mrq(trajectory, section_text),
        )

        # --- Evidence linking (top-K sentence pairs)
        evidence = self._evidence_links(
            traj_sents=traj_sents,
            section_sents=section_sents,
            section_sent_embs=section_sent_embs
        )

        return {
            "trajectory_id": f"traj_{idx}",
            "section_relevance": float(section_relevance),
            "causal_strength": float(causal_strength),
            "supporting_evidence": evidence,  # [{trajectory_span, section_span, strength}]
            "trajectory": trajectory,
        }

    # ----------------------------
    # Scoring subroutines
    # ----------------------------
    def _section_relevance(
        self,
        msgs: List[Dict[str, Any]],
        traj_sents: List[str],
        section_text: str,
        section_emb: Optional[np.ndarray]
    ) -> float:
        # Message-level embedding max/avg sim (if available)
        emb_sims: List[float] = []
        if self.kfg.use_embeddings and section_emb is not None:
            for m in msgs:
                t = (m.get("text") or "").strip()
                if not t:
                    continue
                try:
                    e = self._embed(t[:2000])
                    emb_sims.append(_cosine(e, section_emb))
                except Exception:
                    pass

        # Lexical overlap fallback
        lex = _lexical_overlap("\n".join(traj_sents), section_text)

        if emb_sims:
            return 0.75 * float(np.mean(emb_sims)) + 0.25 * lex
        return lex

    def _causal_strength(
        self,
        trajectory_text: str,
        section_text: str,
        hrm_score: float,
        mrq_score: float,
    ) -> float:
        # Causal cues (regex)
        patterns = [
            r"\bso\s+we\s+(decided|concluded|implemented)",
            r"\btherefore\b",
            r"\bthis\s+led\s+to\b",
            r"\bas\s+a\s+result\b",
            r"\bbecause\s+of\s+this\b",
            r"\bconsequently\b",
            r"\bthus\b",
            r"\bhence\b",
        ]
        cue_hits = sum(1 for p in patterns if re.search(p, trajectory_text, re.IGNORECASE))
        cue_score = min(1.0, cue_hits / max(1, len(patterns) / 2.0))

        # Content overlap
        overlap = _lexical_overlap(trajectory_text, section_text)

        # Hybrid weighting:
        #   cues + overlap as base, HRM and MRQ (if present) as boosters
        base = 0.55 * cue_score + 0.45 * overlap
        booster = 0.0
        if hrm_score > 0:
            booster += 0.20 * hrm_score   # reasoning progress signal
        if mrq_score > 0:
            booster += 0.15 * mrq_score   # question/goal satisfaction signal

        return max(0.0, min(1.0, base + booster))

    def _evidence_links(
        self,
        traj_sents: List[str],
        section_sents: List[str],
        section_sent_embs: List[np.ndarray],
    ) -> List[Dict[str, Any]]:
        if not traj_sents or not section_sents:
            return []

        pairs: List[Tuple[str, str, float]] = []

        # If embeddings available, match by cosine; else lexical overlap
        if self.kfg.use_embeddings and section_sent_embs:
            # Pre-embed trajectory sentences
            traj_embs: List[np.ndarray] = []
            for s in traj_sents:
                try:
                    traj_embs.append(self._embed(s))
                except Exception:
                    traj_embs.append(None)

            for ti, (ts, te) in enumerate(zip(traj_sents, traj_embs)):
                # best match over section sentences
                best = 0.0
                best_ss = ""
                if te is not None:
                    for ss, se in zip(section_sents, section_sent_embs):
                        if se is None:
                            continue
                        sim = _cosine(te, se)
                        if sim > best:
                            best, best_ss = sim, ss
                else:
                    # fallback to lexical for this sentence
                    for ss in section_sents:
                        sim = _lexical_overlap(ts, ss)
                        if sim > best:
                            best, best_ss = sim, ss

                if best > 0:
                    pairs.append((ts, best_ss, float(best)))
        else:
            # lexical only
            for ts in traj_sents:
                best = 0.0
                best_ss = ""
                for ss in section_sents:
                    sim = _lexical_overlap(ts, ss)
                    if sim > best:
                        best, best_ss = sim, ss
                if best > 0:
                    pairs.append((ts, best_ss, float(best)))

        # Sort and take top-K
        pairs.sort(key=lambda x: x[2], reverse=True)
        top = pairs[: self.kfg.evidence_top_k]
        return [
            {"trajectory_span": t, "section_span": s, "strength": round(float(sc), 4)}
            for t, s, sc in top
        ]

    # ----------------------------
    # Optional scorer hooks
    # ----------------------------
    def _score_hrm(self, trajectory: Dict[str, Any]) -> float:
        """Human-Reasoning Model (progress) score if available; else heuristic."""
        try:
            if self.hrm_scorer and hasattr(self.hrm_scorer, "score"):
                # Expect interface: score(messages=[...]) -> {"score": float}
                out = self.hrm_scorer.score(messages=trajectory.get("messages", []))
                return float(out.get("score", 0.0))
        except Exception as e:
            self.logger.log("HRMScoreError", {"error": str(e)})

        # Heuristic fallback: look for resolution signals
        txt = "\n".join((m.get("text") or "").lower() for m in trajectory.get("messages", []))
        pos = int(bool(re.search(r"\b(solved|fixed|works|resolved|implemented)\b", txt)))
        neg = int(bool(re.search(r"\b(error|not working|issue|failed)\b", txt)))
        return max(0.0, min(1.0, 0.6 * pos + 0.2 * (1 - neg)))

    def _score_mrq(self, trajectory: Dict[str, Any], section_text: str) -> float:
        """Goal/Q alignment score if available; else heuristic overlap."""
        try:
            if self.mrq_scorer and hasattr(self.mrq_scorer, "score"):
                out = self.mrq_scorer.score(
                    messages=trajectory.get("messages", []),
                    section_text=section_text
                )
                return float(out.get("score", 0.0))
        except Exception as e:
            self.logger.log("MRQScoreError", {"error": str(e)})
        traj_text = "\n".join(m.get("text", "") for m in trajectory.get("messages", []))
        return _lexical_overlap(traj_text, section_text)

    # ----------------------------
    # Embedding
    # ----------------------------
    def _embed(self, text: str) -> np.ndarray:
        """Use memory.embedding to get/create embeddings (NumPy vector)."""
        if not text or not self.kfg.use_embeddings:
            return None
        vec = self.memory.embedding.get_or_create(text)
        # Ensure numpy array
        if isinstance(vec, np.ndarray):
            return vec
        try:
            return np.asarray(vec, dtype=np.float32)
        except Exception:
            return None

    # ----------------------------
    # Bus publishing
    # ----------------------------
    async def _publish_edges(self, paper_section: Dict[str, Any], mappings: List[Dict[str, Any]]) -> None:
        if not self.bus or not hasattr(self.bus, "publish"):
            return
        section_id = paper_section.get("id") or paper_section.get("paper_id")
        tasks = []
        for m in mappings:
            payload = {
                "event_type": "conversation.trajectory.mapping",
                "payload": {
                    "section_id": section_id,
                    "section_name": paper_section.get("section_name"),
                    "trajectory_id": m.get("trajectory_id"),
                    "causal_strength": m.get("causal_strength"),
                    "section_relevance": m.get("section_relevance"),
                    "is_critical": m.get("is_critical", False),
                    "evidence": m.get("supporting_evidence", []),
                    "timestamp": None,
                    "source_agent": "ConversationTrajectoryMapper",
                }
            }
            try:
                # HybridKnowledgeBus.publish(subject, payload) is async
                tasks.append(self.bus.publish(self.kfg.bus_subject, payload))
            except Exception as e:
                self.logger.log("TrajectoryMappingPublishError", {"error": str(e)})
        # Best-effort: publish concurrently
        if tasks:
            try:
                import asyncio
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                self.logger.log("TrajectoryMappingPublishGatherError", {"error": str(e)})
``n

## File: debate.py

`python
# stephanie/agents/knowledge/debate.py
from stephanie.agents.base_agent import BaseAgent


class OptimistDebater(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = context.get("hypotheses", [])
        reviews = []

        for h in hypotheses:
            prompt = (
                f"As an optimistic analyst, critique the following hypothesis:\n\n"
                f"{h}\n\n"
                f"Focus on strengths, positive implications, and reasons it might be valid."
            )
            review = self.call_llm(prompt, context)
            reviews.append({"hypotheses": h, "review": review, "persona": "Optimist"})

        return {"reviews": reviews}


class SkepticDebater(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = context.get("hypotheses", [])
        reviews = []

        for h in hypotheses:
            prompt = (
                f"As a skeptical analyst, critique the following hypothesis:\n\n"
                f"{h}\n\n"
                f"Focus on weaknesses, uncertainties, or reasons it might be flawed."
            )
            review = self.call_llm(prompt, context)
            reviews.append({"hypotheses": h, "review": review, "persona": "Skeptic"})

        return {"reviews": reviews}


class BalancedDebater(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = context.get("hypotheses", [])
        reviews = []

        for h in hypotheses:
            prompt = (
                f"As a balanced analyst, critique the following hypothesis:\n\n"
                f"{h}\n\n"
                f"Provide both positive and negative aspects."
            )
            review = self.call_llm(prompt, context)
            reviews.append({"hypotheses": h, "review": review, "persona": "Balanced"})

        return {"reviews": reviews}


class DebateAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.optimist = OptimistDebater(cfg, memory, container, logger)
        self.skeptic = SkepticDebater(cfg, memory, container, logger)
        self.balanced = BalancedDebater(cfg, memory, container, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = context.get("hypotheses", [])
        optimist_reviews = await self.optimist.run({"hypotheses": hypotheses})
        skeptic_reviews = await self.skeptic.run({"hypotheses": hypotheses})
        balanced_reviews = await self.balanced.run({"hypotheses": hypotheses})

        return {
            "optimist_reviews": optimist_reviews,
            "skeptic_reviews": skeptic_reviews,
            "balanced_reviews": balanced_reviews,
        }
``n

## File: dimension_generator.py

`python
# stephanie/agents/knowledge/dimension_generator.py

import re

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL


class DimensionGeneratorAgent(BaseAgent):
    """
    Generates goal-informed evaluation dimensions dynamically.

    Instead of relying on fixed domains or pre-defined scoring axes,
    this agent reads the current goal and invents relevant dimensions
    like 'stability', 'generalization', 'adaptivity', etc.

    These dimensions are then used by the IdeaParser and SVMScorer
    to evaluate ideas in context.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Settings
        self.default_dimensions = cfg.get(
            "default_dimensions",
            ["usefulness", "novelty", "alignment", "epistemic_gain"]
        )
        self.max_dimensions = cfg.get("max_dimensions", 6)
        self.min_similarity_score = cfg.get("min_similarity_score", 0.65)
        self.use_memory = cfg.get("use_memory", True)

        # Injected tools
        self.prompt_loader = None
        self.call_llm = None

    async def run(self, context: dict) -> dict:
        """
        Main pipeline:
        1. Get goal from context
        2. Generate dimensions based on goal + memory
        3. Store them in context for downstream scorers
        """
        try:
            goal = context.get(GOAL, {})
            goal_text = goal.get("goal_text", "").strip()
            goal_id = goal.get("id")

            if not goal_text:
                self.logger.log("NoGoalText", {"stage": "dimension_generation"})
                context["dimensions"] = self.default_dimensions
                return context

            # Step 1: Use memory to find similar past goals
            similar_goals = []
            if self.use_memory:
                similar_goals = self._find_similar_goals(goal_text, top_k=3)

            # Step 2: Prompt LLM to generate dimensions
            prompt_context = {
                "goal_text": goal_text,
                "similar_goals": [g["goal_text"] for g in similar_goals],
                "existing_dimensions": self.default_dimensions,
                "max_dimensions": self.max_dimensions,
            }

            prompt = self.prompt_loader.from_file(
                self.cfg.get("dimension_prompt_file", "generate_dimensions.txt"),
                self.cfg, prompt_context
            )

            raw_output = self.call_llm(prompt, prompt_context)
            dimensions = self._parse_dimension_response(raw_output)

            # Step 3: Optionally filter duplicates or irrelevant ones
            filtered = self._filter_irrelevant_dimensions(dimensions)

            # Step 4: Store in memory for future reuse
            if goal_id and filtered:
                self._save_dimensions_to_goal(goal_id, filtered)

            # Step 5: Return in context
            context["dimensions"] = filtered
            context["dynamic_dimensions"] = filtered
            context["dimension_source"] = "DimensionGeneratorAgent"

            self.logger.log(
                "DimensionsGenerated",
                {
                    "count": len(filtered),
                    "dimensions": filtered,
                    "source": "LLM"
                }
            )

            return context

        except Exception as e:
            self.logger.log("DimensionGenerationFailed", {"error": str(e)})
            context["dimensions"] = self.default_dimensions
            return context

    def _find_similar_goals(self, goal_text: str, top_k: int = 3) -> list:
        """
        Find previous goals with similar intent using embeddings
        """
        self.memory.embedding.get_or_create(goal_text)
        embedding_id = self.memory.embedding.get_id_for_text(goal_text)

        results = self.memory.goal_embeddings.find_similar(embedding_id, top_k=top_k)

        similar_goals = []
        for result in results:
            goal_id = result[0]
            score = result[1]
            if score > self.min_similarity_score:
                goal_data = self.memory.goals.get(goal_id)
                similar_goals.append({
                    "goal_id": goal_id,
                    "goal_text": goal_data.text,
                    "score": score
                })

        return similar_goals

    def _parse_dimension_response(self, response: str) -> list:
        """
        Parses LLM output into clean dimension list.
        Handles both numbered lists and free-form text.
        """
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        dimensions = []

        for line in lines:
            # Match bullet points or numbers
            match = re.match(r"(?:\d+\.|\-)\s*([a-zA-Z][a-zA-Z_\-\s]+)", line)
            if match:
                dim = match.group(1).strip().lower()
                dim = re.sub(r"[^\w\s]", "", dim)
                dim = re.sub(r"\s+", "_", dim)
                if dim:
                    dimensions.append(dim)

        # Remove duplicates while preserving order
        seen = set()
        unique_dims = []
        for d in dimensions:
            if d not in seen:
                seen.add(d)
                unique_dims.append(d)

        return unique_dims[:self.max_dimensions]

    def _filter_irrelevant_dimensions(self, dimensions: list) -> list:
        """
        Optional filtering step to remove low-value dimensions.
        Could be replaced with an SVM or heuristic-based filter.
        """
        common_stopwords = [
            "relevance", "importance", "impact", "quality", "value", "meaning"
        ]
        return [d for d in dimensions if d not in common_stopwords]

    def _save_dimensions_to_goal(self, goal_id: int, dimensions: list):
        """
        Stores generated dimensions with the goal for future reference
        """
        for i, dim in enumerate(dimensions):
            self.memory.goal_dimensions.insert({
                "goal_id": goal_id,
                "dimension": dim,
                "rank": i + 1
            })

        self.logger.log("GoalDimensionsSaved", {"goal_id": goal_id, "dimensions": dimensions})
``n

## File: document_loader.py

`python
# stephanie/agents/knowledge/document_loader.py
"""
Document Loader Agent Module

This module provides the DocumentLoaderAgent class for automated retrieval, processing, and storage
of research documents in the co-ai framework. It handles the complete document ingestion pipeline
from URL-based retrieval to structured database storage with domain classification.

Key Features:
    - Automated PDF document downloading from URLs
    - Text extraction from PDF files using PDFConverter
    - Optional document summarization using LLMs
    - ArXiv metadata integration for enhanced document information
    - Domain classification and scoring using DomainClassifier
    - Embedding generation and storage for similarity search
    - Persistent storage in document database with relationship tracking
    - Duplicate document detection and handling
    - Error handling and comprehensive logging

Classes:
    DocumentLoaderAgent: Main agent class for document loading and processing

Functions:
    guess_title_from_text: Utility function to extract document title from text content

Configuration Options:
    - max_chars_for_summary: Maximum characters for document summarization
    - summarize_documents: Enable/disable automatic document summarization
    - force_domain_update: Force re-classification of existing documents
    - top_k_domains: Number of top domains to assign per document
    - download_directory: Temporary directory for PDF downloads
    - min_classification_score: Minimum confidence score for domain classification
    - domain_seed_config_path: Path to domain classification configuration

Dependencies:
    - BaseAgent: Core agent functionality and LLM integration
    - DomainClassifier: Document domain classification and scoring
    - PDFConverter: PDF text extraction utilities
    - ArxivTool: ArXiv metadata retrieval
    - Memory system: Document and embedding storage

Usage:
    Typically used as part of a document processing pipeline after search orchestrator
    agents to prepare documents for further analysis, scoring, or hypothesis generation.

"""

import os
import re

import requests
from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.constants import GOAL
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.tools.arxiv_tool import fetch_arxiv_metadata
from stephanie.tools.pdf_tools import PDFConverter


def guess_title_from_text(text: str) -> str:
    """Extract a likely document title from text content by analyzing the first few lines"""
    lines = text.strip().split("\n")
    lines = [line.strip() for line in lines if line.strip()]
    # Look for lines with at least 4 words in the first 15 lines
    candidates = [line for line in lines[:15] if len(line.split()) >= 4]
    return candidates[0] if candidates else None


class DocumentLoaderAgent(BaseAgent):
    """Agent responsible for downloading, processing, and storing research documents"""
    
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        # Configuration parameters with defaults
        self.max_chars_for_summary = cfg.get("max_chars_for_summary", 8000)
        self.summarize_documents = cfg.get("summarize_documents", False)
        self.force_domain_update = cfg.get("force_domain_update", False)
        self.top_k_domains = cfg.get("top_k_domains", 3)
        self.download_directory = cfg.get("download_directory", "/tmp")
        self.min_classification_score = cfg.get(
            "min_classification_score", 0.6
        )
        self.embed_full_document = cfg.get("embed_full_document", True)
        self.scorable_type = cfg.get("scorable_type", "document")
        # Initialize domain classifier for categorizing documents
        self.domain_classifier = ScorableClassifier(
            memory=self.memory,
            logger=self.logger,
            config_path=cfg.get(
                "domain_seed_config_path", "config/domain/seeds.yaml"
            ),
        )

    async def run(self, context: dict) -> dict:
        """Main execution method for document loading pipeline"""
        search_results = context.get(self.input_key, [])
        goal = context.get(GOAL, {})
        goal_id = goal.get("id")

        stored_documents = []
        document_domains = []

        # --- Report: start ---
        self.report(
            {
                "event": "start",
                "step": "DocumentLoader",
                "details": f"Processing {len(search_results)} search results",
            }
        )
        
        # Process each search result with progress tracking
        for result in tqdm(search_results, desc="📄 Loading documents", unit="doc"):
            try:
                url = result.get("url")
                title = result.get("title")
                summary = result.get("summary")

                # Skip existing documents to avoid duplicates
                existing = self.memory.documents.get_by_url(url)
                if existing:
                    self.report(
                        {
                            "event": "skipped_existing",
                            "step": "DocumentLoader",
                            "details": f"Document already exists: {title}",
                            "url": url,
                        }
                    )
                    doc_dict = existing.to_dict()
                    stored_documents.append(doc_dict)
                    self.ensure_scorable(doc_dict, context)
                    # Assign domains if needed (new or forced update)
                    if (
                        not self.memory.scorable_domains.has_domains(str(existing.id), self.scorable_type)
                        or self.force_domain_update
                    ):
                        self.assign_domains_to_document(existing)

                    continue

                # Download PDF document
                response = requests.get(url, stream=True)
                if response.status_code != 200:
                    self.report(
                        {
                            "event": "download_failed",
                            "step": "DocumentLoader",
                            "details": f"HTTP {response.status_code} for {title}",
                            "url": url,
                        }
                    )
                    continue

                # Create safe filename for temporary storage
                file_name = (
                    result.get("pid")
                    or result.get("arxiv_id")
                    or self.sanitize_filename(title)
                    or "document"
                )
                pdf_path = f"{self.download_directory}/{file_name}"

                # Save PDF to temporary location
                with open(pdf_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                # Validate PDF integrity
                if not PDFConverter.validate_pdf(pdf_path):
                    self.report(
                        {
                            "event": "invalid_pdf",
                            "step": "DocumentLoader",
                            "details": f"Invalid PDF format for {title}",
                            "url": url,
                        }
                    )
                    os.remove(pdf_path)
                    continue

                # Extract text from PDF
                text = PDFConverter.pdf_to_text(pdf_path)
                os.remove(pdf_path)  # Clean up temporary file

                # Summarize document content if enabled
                if self.summarize_documents:
                    pid = result.get("pid") or result.get("arxiv_id")
                    meta_data = fetch_arxiv_metadata(pid)
                    if meta_data:
                        # Use arXiv metadata if available
                        title = meta_data["title"]
                        summary = meta_data["summary"]
                    else:
                        # Generate summary using LLM
                        merged = {"document_text": text, **context}
                        prompt_text = self.prompt_loader.load_prompt(
                            self.cfg, merged
                        )
                        summary = self.call_llm(prompt_text, context)
                        guessed_title = guess_title_from_text(text)
                        if guessed_title:
                            title = guessed_title

                # Store document in database
                doc = {
                    "goal_id": goal_id,
                    "title": title,
                    "external_id": result.get("title"),
                    "summary": summary,
                    "source": self.name,
                    "text": text,
                    "url": url,
                }
                stored = self.memory.documents.add_document(doc)
                doc_id = stored.id

                doc_dict = stored.to_dict()
                stored_documents.append(doc_dict)
                self.ensure_scorable(doc_dict, context)

                self.report(
                    {
                        "event": "stored",
                        "step": "DocumentLoader",
                        "details": f"Stored document: {title}",
                        "doc_id": doc_id,
                        "url": url,
                    }
                )

                # Assign domain classifications to document
                self.assign_domains_to_document(stored)
                self.report(
                    {
                        "event": "domains_assigned",
                        "step": "DocumentLoader",
                        "details": f"Domains assigned for {title}",
                    }
                )

            except Exception as e:
                self.report(
                    {
                        "event": "error",
                        "step": "DocumentLoader",
                        "details": f"Error loading {result.get('url')}: {str(e)}",
                    }
                )

        # Update context with results
        context[self.output_key] = stored_documents
        context["document_ids"] = [doc.get("id") for doc in stored_documents]
        context["document_domains"] = document_domains

        # --- Report: end ---
        self.report(
            {
                "event": "end",
                "step": "DocumentLoader",
                "details": f"Stored {len(stored_documents)} new documents",
            }
        )
        return context

    def ensure_scorable(self, doc, context):
        """Create or update scorable representation of document for embedding and scoring"""
        if self.embed_full_document:
            embed_text = f"{doc['title']}\n\n{doc.get('text', doc.get('summary', ''))}"
        else:
            embed_text = f"{doc['title']}\n\n{doc.get('summary', '')}"

        doc_id = doc.get("id")
        scorable = Scorable(
            id=doc_id,
            text=embed_text,
            target_type=TargetType.DOCUMENT,
        )
        self.memory.scorable_embeddings.get_or_create(scorable)
        self.memory.pipeline_references.insert(
            {
                "pipeline_run_id": context.get("pipeline_run_id"),
                "scorable_type": TargetType.DOCUMENT,
                "scorable_id": doc_id,
                "relation_type": "inserted",
                "source": self.name,
            }
        )

    def sanitize_filename(self, title: str) -> str:
        """Create a filesystem-safe filename from a document title"""
        return re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", title)[:100]

    def assign_domains_to_document(self, document):
        """Classify document into domain categories and store results"""
        text = document.text
        if text:
            # Get domain classifications
            results = self.domain_classifier.classify(
                text, self.top_k_domains, self.min_classification_score
            )
            # Store each domain classification
            for domain, score in results:
                self.memory.scorable_domains.insert(
                    {
                        "scorable_id": str(document.id),
                        "scorable_type": "document",
                        "domain": domain,
                        "score": score,
                    }
                )
        else:
            self.report(
                {
                    "event": "no_content",
                    "step": "DocumentLoader",
                    "details": f"No content found for document {document.id}",
                }
            )
``n

## File: document_profiler.py

`python
# stephanie/agents/knowledge/document_profiler.py
"""
Document Profiler Agent Module

This module provides the DocumentProfilerAgent class for analyzing and structuring
research documents into standardized sections with domain classification. It transforms
unstructured document text into organized, categorized content for better analysis
and retrieval in the research pipeline.

Key Features:
    - Multi-method document parsing (unstructured parsing + LLM fallback)
    - Section-based document analysis (title, abstract, methods, results, etc.)
    - Domain classification for document sections
    - Content quality evaluation and selection
    - Persistent storage of structured document sections
    - Comprehensive error handling and reporting

Classes:
    DocumentProfilerAgent: Main agent class for document profiling and structuring

Configuration Options:
    - summary_prompt_file: Prompt file for document summarization
    - use_unstructured: Enable/disable unstructured parsing
    - fallback_to_llm: Enable LLM fallback when parsing fails
    - store_inline: Enable inline storage of parsed sections
    - output_sections: List of sections to extract from documents
    - required_sections: Minimum required sections for successful parsing
    - min_chars_per_section: Minimum character threshold for section quality
    - force_domain_update: Force re-classification of existing documents
    - top_k_domains: Number of top domains to assign per section
    - min_classification_score: Minimum confidence score for domain classification

Dependencies:
    - BaseAgent: Core agent functionality and LLM integration
    - ScorableClassifier: Domain classification and scoring
    - DocumentSectionParser: Section extraction from unstructured text

Usage:
    Typically used after document loading to structure and categorize documents
    for further analysis, hypothesis generation, or knowledge extraction.
"""

import re

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.utils.document_section_parser import DocumentSectionParser

# Default sections to extract from documents
DEFAULT_SECTIONS = ["title", "abstract", "methods", "results", "contributions"]
# Minimum required sections for document processing
REQUIRED_SECTIONS = ["title", "summary"]


class DocumentProfilerAgent(BaseAgent):
    """Agent responsible for structuring documents into standardized sections with domain classification"""
    
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        # Configuration parameters
        self.summary_prompt_file = cfg.get("summary_prompt_file", "summarize.txt")
        self.use_unstructured = cfg.get("use_unstructured", True)
        self.fallback_to_llm = cfg.get("fallback_to_llm", False)
        self.store_inline = cfg.get("store_inline", True)
        self.output_sections = cfg.get("output_sections", DEFAULT_SECTIONS)
        self.required_sections = cfg.get("required_sections", REQUIRED_SECTIONS)
        self.min_chars_per_sec = cfg.get("min_chars_per_section", 120)  # quality threshold

        # Domain classification settings
        self.force_domain_update = cfg.get("force_domain_update", False)
        self.top_k_domains = cfg.get("top_k_domains", 3)
        self.min_classification_score = cfg.get("min_classification_score", 0.6)

        # Initialize classifiers and parsers
        self.domain_classifier = ScorableClassifier(
            memory,
            logger,
            cfg.get("domain_seed_config_path", "config/domain/seeds.yaml"),
        )
        self.section_parser = DocumentSectionParser(cfg, logger)

    async def run(self, context: dict) -> dict:
        """Main execution method for document profiling pipeline"""
        documents = context.get(self.input_key, [])
        profiled = []

        # Start profiling process
        self.report({
            "event": "start",
            "step": "DocumentProfiler",
            "details": f"Profiling {len(documents)} documents",
        })

        # Process each document
        for doc in documents:
            try:
                doc_id = doc["id"]
                title = doc.get("title", "")

                # Check if document already profiled
                existing_sections = self.memory.document_sections.get_by_document(doc_id)
                if existing_sections and not self.force_domain_update:
                    self.report({
                        "event": "skipped_existing",
                        "step": "DocumentProfiler",
                        "doc_id": doc_id,
                        "title": title[:80],
                        "details": "Already profiled, skipping.",
                    })
                    continue

                summary = doc.get("summary")
                text = doc.get("content", doc.get("text", ""))

                # STEP 1: Try unstructured parsing first
                unstruct_data = {}
                if self.use_unstructured:
                    unstruct_data = self.section_parser.parse(text)
                    self.report({
                        "event": "parsed_unstructured",
                        "step": "DocumentProfiler",
                        "doc_id": doc_id,
                        "title": title[:80],
                        "sections": list(unstruct_data.keys()),
                    })

                # STEP 2: Use LLM fallback if unstructured parsing is insufficient
                if self.fallback_to_llm and self.needs_fallback(unstruct_data):
                    llm_data = await self.extract_with_prompt(text, context)
                    chosen = self.merge_outputs(unstruct_data, llm_data)
                    self.report({
                        "event": "used_fallback",
                        "step": "DocumentProfiler",
                        "doc_id": doc_id,
                        "title": title[:80],
                        "sections": list(chosen.keys()),
                    })
                else:
                    chosen = unstruct_data

                # Ensure required sections are present
                if title:
                    chosen["title"] = title
                if summary:
                    chosen["summary"] = summary
                else:
                    # Generate summary if missing
                    prompt = self.prompt_loader.from_file(
                        self.summary_prompt_file, self.cfg, context
                    )
                    chosen["summary"] = self.call_llm(prompt, context)

                # STEP 3: Persist sections to memory
                section_summaries = []
                for section, text in chosen.items():
                    existing = self.memory.document_sections.upsert(
                        {
                            "document_id": doc_id,
                            "section_name": section,
                            "section_text": text,
                            "source": "unstructured+llm",
                            "summary": summary,
                        }
                    )

                    # STEP 4: Domain classification for each section
                    section_domains = self.domain_classifier.classify(
                        text, self.top_k_domains, self.min_classification_score
                    )

                    # Store domain classifications
                    for domain, score in section_domains:
                        self.memory.document_section_domains.insert(
                            {
                                "document_section_id": existing.id,
                                "domain": domain,
                                "score": float(score),
                            }
                        )
                    if section_domains:
                        section_summaries.append({
                            "section": section,
                            "domains": [
                                {"domain": d, "score": float(s)} for d, s in section_domains
                            ],
                        })

                # Add to results
                profiled.append(
                    {
                        "id": doc_id,
                        "title": title[:80],
                        "structured_data": chosen,
                    }
                )

                self.report({
                    "event": "profiled",
                    "step": "DocumentProfiler",
                    "doc_id": doc_id,
                    "title": title[:80],
                    "sections": list(chosen.keys()),
                    "classified_domains": section_summaries,
                })

            except Exception as e:
                self.report({
                    "event": "error",
                    "step": "DocumentProfiler",
                    "doc_id": doc.get("id"),
                    "title": doc.get("title", "")[:80],
                    "details": str(e),
                })

        context[self.output_key] = profiled

        # Completion report
        self.report({
            "event": "end",
            "step": "DocumentProfiler",
            "details": f"Profiled {len(profiled)} documents successfully",
        })
        return context

    async def extract_with_prompt(self, text: str, context: dict) -> dict:
        """Extract document sections using LLM prompt-based approach"""
        prompt_ctx = {
            "text": text[: self.cfg.get("llm_max_chars", 12000)],
            "sections": ", ".join(self.output_sections),
        }
        prompt = self.prompt_loader.load_prompt(self.cfg, prompt_ctx)
        raw = self.call_llm(prompt, context)
        headings = self.parse_headings_from_response(raw)

        # Split text into sections based on detected headings
        return self.split_text_by_headings(text, headings)

    def needs_fallback(self, data: dict) -> bool:
        """
        Determine if LLM fallback is needed based on parsing quality
        
        Returns:
            True if any required section is missing or too short
        """
        if not data:
            return True
        for sec in self.required_sections:
            if sec not in data:
                print(f"[FALLBACK NEEDED] Missing section: {sec}")
                return True
            if sec != "title" and len(data[sec]) < self.min_chars_per_sec:
                print(f"[FALLBACK NEEDED] section too small: {sec}")
                return True
        return False

    def evaluate_content_quality(self, text: str) -> float:
        """
        Evaluate content quality using heuristic measures
        
        Args:
            text: Text content to evaluate
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not text:
            return 0.0

        # Calculate basic text metrics
        sentences = text.split(".")
        avg_word_len = (
            sum(len(word) for word in text.split()) / len(text.split())
            if text.split()
            else 0
        )
        sentence_score = len([s for s in sentences if len(s.strip()) > 20]) / max(
            1, len(sentences)
        )

        # Combine metrics into quality score
        score = (
            0.4 * min(1.0, len(text) / 500)  # Normalize length
            + 0.4 * sentence_score
            + 0.2
            * min(1.0, avg_word_len / 8)  # Prefer more complex words up to a point
        )
        return round(score, 2)

    def merge_outputs(self, primary: dict, fallback: dict) -> dict:
        """
        Merge results from different parsing methods, selecting the best version
        
        Args:
            primary: Results from primary parsing method
            fallback: Results from fallback parsing method
            
        Returns:
            Merged results with best version of each section
        """
        merged = {}

        for sec in self.output_sections:
            p_txt = primary.get(sec, "")
            f_txt = fallback.get(sec, "")

            # Skip if neither method found this section
            if not p_txt and not f_txt:
                continue

            # Use available result if only one method found it
            if not p_txt:
                merged[sec] = f_txt
                continue
            if not f_txt:
                merged[sec] = p_txt
                continue

            # Both methods found this section - select the better one
            p_len = len(p_txt)
            f_len = len(f_txt)

            # Check if primary meets minimum length requirement
            if p_len >= self.min_chars_per_sec:
                p_score = self.evaluate_content_quality(p_txt)
                f_score = self.evaluate_content_quality(f_txt)

                # Select version with higher quality score
                if p_score >= f_score:
                    merged[sec] = p_txt
                else:
                    merged[sec] = f_txt
                    print(
                        f"[QUALITY WIN] Fallback used for '{sec}' (P: {p_score}, F: {f_score})"
                    )
            else:
                # Primary doesn't meet threshold - use fallback
                merged[sec] = f_txt

        return merged

    def parse_headings_from_response(self, response: str) -> list[str]:
        """
        Extract headings from LLM response text
        
        Args:
            response: Raw LLM response text
            
        Returns:
            List of cleaned heading strings
        """
        lines = response.strip().splitlines()
        candidates = []

        for line in lines[-20:]:  # Limit to last 20 lines to avoid rambling
            line = line.strip()
            # Match lines that are likely headings
            if line and len(line) < 100:  # reasonable length
                line = re.sub(
                    r"^[\-\*\d\.\)]+\s*", "", line
                )  # remove leading bullets/numbers
                if re.match(r"^[A-Z][\w\s\-]+$", line):  # simple heading pattern
                    candidates.append(line)

        return candidates

    def split_text_by_headings(self, text: str, headings: list[str]) -> dict:
        """
        Split text into sections based on detected headings
        
        Args:
            text: Full document text
            headings: List of section headings
            
        Returns:
            Dictionary of section names to section text
        """
        sections = {}
        current = None
        lines = text.splitlines()

        for line in lines:
            line_stripped = line.strip()

            # Check if this line matches one of the headings
            matched_heading = next(
                (h for h in headings if h.lower() in line_stripped.lower()), None
            )

            if matched_heading:
                current = matched_heading
                sections[current] = []
            elif current:
                sections[current].append(line)

        # Join and trim each section
        return {
            k.lower(): "\n".join(v).strip()
            for k, v in sections.items()
            if len(v) >= 3  # must have at least a few lines
        }
``n

## File: document_reward_scorer.py

`python
# stephanie/agents/knowledge/document_reward_scorer.py
import random
import time
from typing import Any, Dict, List

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_corpus import ScoreCorpus
from stephanie.scoring.calculations.mars_calculator import MARSCalculator
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.scorer.contrastive_ranker_scorer import \
    ContrastiveRankerScorer
from stephanie.scoring.scorer.ebt_scorer import EBTScorer
from stephanie.scoring.scorer.hrm_scorer import HRMScorer
from stephanie.scoring.scorer.mrq_scorer import MRQScorer
from stephanie.scoring.scorer.sicql_scorer import SICQLScorer
from stephanie.scoring.scorer.svm_scorer import SVMScorer


class DocumentRewardScorerAgent(BaseAgent):
    """
    Scores document sections or full documents to assess reward value
    using configured reward model (e.g., SVM-based or regression-based).
    
    Enhanced with MARS (Model Agreement and Reasoning Signal) analysis
    to evaluate consistency across scoring models using the tensor-based architecture.
    """
    
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.dimensions = cfg.get("dimensions", ["helpfulness", "truthfulness", "reasoning_quality"])
        self.include_mars = cfg.get("include_mars", True)
        self.test_mode = cfg.get("test_mode", False)
        self.test_document_count = cfg.get("test_document_count", 100)
        
        # Configure which scorers to use
        self.enabled_scorers = cfg.get("enabled_scorers", [
            "svm", "mrq", "sicql", "ebt", "hrm", "contrastive_ranker"
        ])
        
        # Initialize scorers dynamically
        self.scorers = self._initialize_scorers()
        
        # Initialize MARS calculator with dimension-specific configurations
        dimension_config = cfg.get("dimension_config", {})
        self.mars_calculator = MARSCalculator(dimension_config, self.memory, self.container, self.logger)
        
        self.logger.log("DocumentRewardScorerInitialized", {
            "dimensions": self.dimensions,
            "scorers": self.enabled_scorers,
            "include_mars": self.include_mars,
            "test_mode": self.test_mode
        })

    def _initialize_scorers(self) -> Dict[str, Any]:
        """Initialize all configured scorers"""
        scorers = {}
        
        if "svm" in self.enabled_scorers:
            scorers["svm"] = SVMScorer(self.cfg, memory=self.memory, logger=self.logger)
        if "mrq" in self.enabled_scorers:
            scorers["mrq"] = MRQScorer(self.cfg, memory=self.memory, logger=self.logger)
        if "sicql" in self.enabled_scorers:
            scorers["sicql"] = SICQLScorer(self.cfg, memory=self.memory, logger=self.logger)
        if "ebt" in self.enabled_scorers:
            scorers["ebt"] = EBTScorer(self.cfg, memory=self.memory, logger=self.logger)
        if "hrm" in self.enabled_scorers:
            scorers["hrm"] = HRMScorer(self.cfg, memory=self.memory, logger=self.logger)
        if "contrastive_ranker" in self.enabled_scorers:
            scorers["contrastive_ranker"] = ContrastiveRankerScorer(
                self.cfg, memory=self.memory, logger=self.logger
            )
            
        return scorers

    async def run(self, context: dict) -> dict:
        """Main execution method with optional test mode"""
        start_time = time.time()
        
        # Handle test mode if enabled
        if self.test_mode:
            documents = self._generate_test_documents()
            self.logger.log("TestModeActivated", {
                "document_count": len(documents),
                "dimensions": self.dimensions
            })
        else:
            documents = context.get(self.input_key, [])
            
        if not documents:
            self.logger.log("NoDocumentsFound", {"source": self.input_key})
            return context
            
        # Process all documents and collect ScoreBundles
        all_bundles = {}  # scorable_id -> ScoreBundle
        results = []
        total_documents = len(documents)
        
        # Process documents with progress tracking
        pbar = tqdm(
            documents, 
            desc="Scoring Documents", 
            total=total_documents,
            disable=not self.cfg.get("progress", True)
        )
        
        for idx, doc in enumerate(pbar):
            try:
                # Score document with all scorers
                scoring_start = time.time()
                doc_scores, bundle = self._score_document(context, doc)
                scoring_time = time.time() - scoring_start
                
                # Update progress bar
                pbar.set_postfix({
                    "docs": f"{idx+1}/{total_documents}",
                    "scorers": len(self.scorers)
                })
                
                # Log performance metrics
                if (idx + 1) % 10 == 0 or idx == total_documents - 1:
                    self.logger.log("DocumentScoringProgress", {
                        "processed": idx + 1,
                        "total": total_documents,
                        "avg_time_per_doc": scoring_time,
                        "scorers": len(self.scorers)
                    })
                
                # Store results
                results.append(doc_scores)
                
                # Save bundle for corpus analysis
                all_bundles[doc["id"]] = bundle
                
            except Exception as e:
                self.logger.log("DocumentScoringError", {
                    "document_id": doc.get("id", "unknown"),
                    "error": str(e)
                })
                continue
        
        # Create ScoreCorpus for MARS analysis
        corpus = ScoreCorpus(bundles=all_bundles)
        
        self.logger.log("ScoreCorpusSummary", {
            "dims": corpus.dimensions,
            "scorers": corpus.scorers,
            "shape_example": corpus.get_dimension_matrix(self.dimensions[0]).shape
        })

        # Save corpus to context for potential future analysis
        context["score_corpus"] = corpus.to_dict()
        
        # Run MARS analysis if requested
        mars_results = {}
        if self.include_mars and all_bundles:
            mars_results = self.mars_calculator.calculate(corpus, context=context)
            context["mars_analysis"] = {
                "summary": mars_results,
                "recommendations": self.mars_calculator.generate_recommendations(mars_results)
            }
            self.logger.log("MARSAnalysisCompleted", {
                "document_count": len(all_bundles),
                "dimensions": self.dimensions
            })
        
        # Save results to context
        context[self.output_key] = results
        context["scoring_time"] = time.time() - start_time
        context["total_documents"] = total_documents
        context["scorers_used"] = list(self.scorers.keys())
        
        self.logger.log("DocumentScoringComplete", {
            "total_documents": total_documents,
            "dimensions": self.dimensions,
            "scorers": len(self.scorers),
            "total_time": context["scoring_time"]
        })
        
        return context

    def _score_document(self, context: dict, doc: dict) -> tuple:
        """Score a single document with all configured scorers"""
        doc_id = doc["id"]
        goal = context.get("goal", {"goal_text": ""})
        scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)
        
        # Collect ScoreResults for this document
        score_results = {}
        
        for scorer_name, scorer in self.scorers.items():
            try:
                # Score with this scorer
                score_bundle = scorer.score(
                    context,
                    scorable=scorable,
                    dimensions=self.dimensions,
                )
                
                # Add all results to our collection
                for dim, result in score_bundle.results.items():
                    if dim not in score_results:
                        score_results[dim] = result
            
            except Exception as e:
                self.logger.log("ScorerError", {
                    "scorer": scorer_name,
                    "document_id": doc_id,
                    "error": str(e)
                })
                continue
        
        # Create ScoreBundle for this document
        bundle = ScoreBundle(results=score_results)
        
        eval_id = self.memory.evaluations.save_bundle(
            bundle=bundle,
            scorable=scorable,
            context=context,
            cfg=self.cfg,
            source="document_reward",
            model_name="ensemble",
            evaluator_name=str(self.scorers.keys())
        )
        self.logger.log("EvaluationSaved", {"id": eval_id})
        
        # Prepare results for reporting
        report_scores = {
            dim: {
                "score": result.score,
                "rationale": result.rationale,
                "source": result.source
            } for dim, result in score_results.items()
        }
        
        return {
            "document_id": doc_id,
            "title": doc.get("title", ""),
            "scores": report_scores,
            "goal_text": goal.get("goal_text", "")
        }, bundle

    def _generate_test_documents(self) -> List[Dict]:
        """Generate synthetic documents for testing"""
        self.logger.log("GeneratingTestDocuments", {
            "count": self.test_document_count,
            "dimensions": self.dimensions
        })
        
        documents = []
        for i in range(self.test_document_count):
            # Generate realistic-looking content
            doc_type = random.choice(["article", "research_paper", "blog_post", "technical_doc"])
            length = random.randint(100, 2000)
            
            documents.append({
                "id": f"test_doc_{i}",
                "title": f"Test Document #{i} - {doc_type}",
                "content": " ".join([f"word_{j}" for j in range(length)]),
                "type": doc_type,
                "length": length
            })
            
        return documents
``n

## File: dpo_pair_generator.py

`python
# stephanie/agents/knowledge/dpo_pair_generator.py
from __future__ import annotations

import asyncio
import json
import logging
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.knowledge.casebook_store import CaseBookStore
from stephanie.utils.json_sanitize import safe_json

_logger = logging.getLogger(__name__)

class DPOPairGeneratorAgent(BaseAgent):
    """
    Generates (chosen, rejected) text pairs from scored improvements.
    Designed to feed RL/DPO pipelines.
    """

    def __init__(self, cfg: Dict[str, Any], memory: Any, container, logger: logging.Logger):
        super().__init__(cfg, memory, container, logger)
        self.min_improvement_score = cfg.get("dpo_min_improvement", 0.1)
        self.max_pairs_per_run = cfg.get("dpo_max_pairs", 10)
        self.auto_publish = cfg.get("dpo_auto_publish", True)
        self.output_dir = Path(cfg.get("dpo_output_dir", "data/dpo_pairs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        _logger.info(
            "DPOPairGeneratorAgent initialized | "
            f"output_dir={self.output_dir} | "
            f"min_improvement_score={self.min_improvement_score} | "
            f"auto_publish={self.auto_publish}"
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        generated_pairs = []

        # Pull identifiers from context (fallbacks are last resort)
        casebook_name = context.get("casebook_name") or context.get("casebook") or "default"
        case_id = context.get("case_id")
        if not case_id:
            self.logger.log("DPOGenerationSkipped", {"reason": "missing_case_id", "casebook": casebook_name})
            return context

        try:
            # Retrieve artifacts
            final_text = self._get_final_draft(casebook_name, case_id)
            initial_text = self._get_initial_draft(casebook_name, case_id)
            vpm_row, vpm_meta = self._get_vpm_row(casebook_name, case_id)
            goal_eval = self._get_goal_eval(vpm_meta)  # uses meta.goal if present

            if not final_text or not initial_text:
                self.logger.log("DPOGenerationSkipped", {
                    "reason": "missing_texts",
                    "has_final": bool(final_text),
                    "has_initial": bool(initial_text),
                    "casebook": casebook_name,
                    "case_id": case_id
                })
                return context

            # Compute improvement
            initial_goal_score = float(context.get("initial_goal_score", 0.0))
            final_goal_score = float(goal_eval.get("score", 0.0))
            score_delta = final_goal_score - initial_goal_score

            if score_delta < float(self.min_improvement_score):
                self.logger.log("DPOGenerationSkipped", {
                    "reason": "insufficient_improvement",
                    "delta": score_delta,
                    "threshold": self.min_improvement_score,
                    "initial_goal_score": initial_goal_score,
                    "final_goal_score": final_goal_score
                })
                return context

            # Build DPO pair
            pair_id = f"dpo_{uuid.uuid4().hex[:8]}"
            prompt = (
                context.get("knowledge_plan", {}).get("section_title")
                or context.get("prompt")
                or "Improve this section."
            )

            pair = {
                "id": pair_id,
                "prompt": prompt,
                "chosen": final_text.strip(),
                "rejected": initial_text.strip(),
                "metadata": {
                    "casebook": casebook_name,
                    "case_id": case_id,
                    "improvement_score": score_delta,
                    "final_score": final_goal_score,
                    "initial_score": initial_goal_score,
                    "vpm_row": vpm_row,
                    "goal_template": context.get("goal_template"),
                    "generation_style": context.get("generation_style"),
                    "tags": context.get("tags", []),
                    "source_agent": "TextImproverAgent",
                },
            }

            # Persist
            pair_path = self.output_dir / f"{pair_id}.json"
            with pair_path.open("w", encoding="utf-8") as f:
                json.dump(pair, f, indent=2)

            generated_pairs.append(pair)

            # Publish (works with sync or async bus)
            if self.auto_publish:
                await self._safe_publish("dpo.pair.generated", {
                    "pair_id": pair_id,
                    "path": str(pair_path),
                    "improvement_score": score_delta,
                    "case_id": case_id
                })

            self.logger.log("DPOPairGenerated", {
                "pair_id": pair_id,
                "improvement": score_delta,
                "case_id": case_id
            })

            # Update context
            context["dpo_pair"] = pair
            context["dpo_pair_path"] = str(pair_path)
            return context

        except Exception as e:
            self.logger.log("DPOPairGenerationFailed", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            return context

    # -------------------------- helpers --------------------------

    async def _safe_publish(self, subject: str, payload: Dict[str, Any]) -> None:
        """Publish via whichever bus is available; handle sync/async seamlessly."""
        bus = getattr(self, "kb", None) or getattr(self.memory, "bus", None)
        if not bus:
            return
        try:
            res = bus.publish(subject, payload)
            if asyncio.iscoroutine(res):
                await res
        except Exception as e:
            self.logger.log("DPOBusPublishFailed", {"error": str(e), "subject": subject})

    def _get_final_draft(self, casebook_name: str, case_id: int) -> Optional[str]:
        items = self.memory.casebooks.get_by_case(
            casebook_name=casebook_name,
            case_id=case_id,
            role="text",
            meta_filter={"stage": "final"},
            limit=1,
        )
        return items[0].text if items else None

    def _get_initial_draft(self, casebook_name: str, case_id: int) -> Optional[str]:
        # 1) Prefer DB: role="text", meta.stage="initial"
        items = self.memory.casebooks.get_by_case(
            casebook_name=casebook_name,
            case_id=case_id,
            role="text",
            meta_filter={"stage": "initial"},
            limit=1,
        )
        if items:
            return items[0].text

        # 2) Fallback: local file from run_dir (read from case meta or scorable meta)
        run_dir = self._get_run_dir(casebook_name, case_id)
        initial_path = run_dir / "initial_draft.md" if run_dir else None
        if initial_path and initial_path.exists():
            return initial_path.read_text(encoding="utf-8")

        return None

    def _get_vpm_row(self, casebook_name: str, case_id: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Returns (vpm_row_dict, vpm_meta_dict).
        vpm_row is parsed from DynamicScorable.text (JSON).
        vpm_meta is taken from DynamicScorable.meta (should include 'goal').
        """
        items = self.memory.casebooks.get_by_case(
            casebook_name=casebook_name,
            case_id=case_id,
            role="vpm",
            limit=1,
        )
        if not items:
            return {}, {}

        row = items[0]
        vpm_row = {}
        try:
            vpm_row = json.loads(row.text or "{}")
        except Exception:
            vpm_row = {}

        vpm_meta = row.meta or {}
        return vpm_row, vpm_meta

    def _get_goal_eval(self, vpm_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Extract goal evaluation dict from VPM meta (fallback to empty)."""
        if isinstance(vpm_meta, dict):
            goal = vpm_meta.get("goal")
            if isinstance(goal, dict):
                return goal
        return {"score": 0.0}

    def _get_run_dir(self, casebook_name: str, case_id: int) -> Optional[Path]:
        """
        Try, in order:
          1) Case.meta['run_dir']
          2) Any vpm/text scorable meta with 'run_dir'
        """
        # 1) Case meta
        case = self.memory.casebooks.get_case_by_id(case_id)
        if case and getattr(case, "meta", None):
            rd = case.meta.get("run_dir")
            if rd:
                return Path(rd)

        # 2) Hunt in scorables
        for role in ("vpm", "text"):
            items = self.memory.casebooks.get_by_case(
                casebook_name=casebook_name,
                case_id=case_id,
                role=role,
                limit=5,
            )
            for it in items:
                if getattr(it, "meta", None) and it.meta.get("run_dir"):
                    return Path(it.meta["run_dir"])

        return None
``n

## File: draft_generator.py

`python
# DraftGeneratorAgent — knowledge-first, VPM-powered draft trajectories from paper + chat
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.knowledge.improver import Improver
from stephanie.agents.knowledge.knowledge_fuser import KnowledgeFuser
from stephanie.agents.paper_improver import GoalScorer
from stephanie.agents.paper_improver.vpm_controller import (Signal,
                                                            VPMController,
                                                            VPMRow,
                                                            default_controller)
from stephanie.models.casebook import CaseBookORM
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.utils.json_sanitize import safe_json


class DraftGeneratorAgent(BaseAgent):
    """
    Generates a blog section as an *improvement trajectory*:
      1) Fuse paper text + chat history into a content plan (transient NER + 20 domains)
      2) Iterate drafts via TextImprover → VPM rows
      3) Use VPMController to decide EDIT/RESAMPLE/STOP
      4) Store every step in a CaseBook (for SIS + PACS/CBR)
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.max_steps = cfg.get("max_steps", 5)
        self.goal_template = cfg.get("goal_template", "academic_summary")
        self.section_name_fallback = cfg.get("section_name_fallback", "Blog Section")
        self.vpm: VPMController = default_controller()
        self.goals = GoalScorer()
        self.ti = Improver(cfg, memory=memory, logger=logger, workdir=cfg.get("text_workdir", "./text_runs"))
        self.fuser = KnowledgeFuser(cfg, memory, container, logger)

        # chat ingestion options
        self.chat_max_messages = cfg.get("chat_max_messages", 200)
        self.chat_from_context_key = cfg.get("chat_context_key", "chat_messages")  # optional injection path

    async def run(self, context: dict) -> dict:
        """
        Expects in context:
          - paper: {"id","title","text" or "section_text"}
          - section: {"section_name","section_text"} (optional; will fallback to paper text)
          - goal: GOAL struct
          - chat_messages (optional): [{"role":"user/assistant","text": "...", "ts":...}, ...]
        Produces:
          - casebook_id, case_id
          - plan (fused)
          - trajectory [steps]
          - champion_draft (text), champion_vpm (dict)
        """
        documents = context.get("documents", [])

        paper = documents[0]
        section = context.get("section", {}) or {}

        # --------- Gather inputs ----------
        section_name = section.get("section_name") or self.section_name_fallback
        paper_text = section.get("section_text") or paper.get("text") or paper.get("content") or ""
        chat_messages = self._collect_chat_messages(context)

        self.report({
            "event": "start",
            "step": "DraftGenerator",
            "details": f"Generating trajectory for '{section_name}'",
            "paper_title": paper.get("title", "Unknown"),
            "chat_messages": len(chat_messages)
        })

        # --------- Fuse knowledge (paper + chat) → plan ----------
        plan = await self.fuser.fuse(
            text=paper_text,
            chat_messages=chat_messages,
            section_name=section_name,
            context=context
        )
        # Add explicit goal_template so TextImprover and downstream scorers can use it
        plan["goal_template"] = self.goal_template
        kg_ctx = self.container.get("knowledge_graph").build_context_for_plan(plan, k=5)
        plan["kg"] = kg_ctx  # pass into TextImprover

        # --------- CaseBook to log the trajectory ----------
        casebook = self._ensure_casebook(paper, section_name, plan)
        context["casebook_id"] = casebook.id

        # --------- Trajectory loop ----------
        trajectory: List[Dict[str, Any]] = []
        last_decision: Optional[Signal] = None
        current_vpm_row: Optional[Dict[str, float]] = None
        last_result: Optional[Dict[str, Any]] = None

        for step in range(self.max_steps):
            # Generate/Improve draft
            result = self.ti.improve(plan)  # TextImprover is self-contained; we rerun with the (improved) plan
            last_result = result

            # Extract VPM dims into controller row
            dims = self._vpm_dims_from_text_improver(result)
            vpm_row = VPMRow(
                unit=f"blog:{section_name}",
                kind="text",
                timestamp=time.time(),
                step_idx=step,
                dims=dims,
                meta={"exemplar_id": None}
            )

            # Controller decision
            decision = self.vpm.add(vpm_row, candidate_exemplars=None)
            last_decision = decision.signal

            # Log step to CaseBook
            draft_path = Path(result["final_draft_path"])
            draft_text = draft_path.read_text() if draft_path.exists() else ""
            traj_rec = self._log_step(casebook, plan, result, step, decision, draft_text, context=context)
            trajectory.append(traj_rec)

            # STOP / ESCALATE gates
            if decision.signal in (Signal.STOP, Signal.ESCALATE):
                break

        # --------- Champion selection (best goal score) ----------
        champion = self._select_champion(trajectory)

        draft_scorable = self.memory.dynamic_scorables.add(
            pipeline_run_id=context.get("pipeline_run_id"),
            scorable_type=TargetType.DYNAMIC,
            source=self.name,
            text=draft_text,
            meta={
                "step": step,
                "decision": decision.signal.name,
                "vpm_row": result["vpm_row"],
                "edit_log": result.get("edit_log", []),
            },
        )

        scorable = ScorableFactory.from_orm(draft_scorable)
        goal = context.get("goal", {})
        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id,
            goal_id=goal.get("id"),
            prompt_text=json.dumps({"plan_slice": self._plan_slice(plan)}),
            agent_name="draft_generator",
            scorables=[scorable.to_dict()],
            meta={
                "step": step,
                "decision": decision.signal.name,
                "vpm_row": result["vpm_row"],
            },
        )

        # --------- Reflection ----------
        self._log_reflection(casebook, section_name, trajectory, champion, context=context)

        # --------- Emit context ----------
        context.update({
            "plan": plan,
            "trajectory": trajectory,
            "champion_step": champion["step"],
            "champion_draft": champion["draft_text"],
            "champion_vpm": champion["vpm_row"],
            "champion_goal_score": champion["goal_score"],
            "case_id": champion["case_id"],
        })

        self.report({
            "event": "end",
            "step": "DraftGenerator",
            "details": f"Trajectory steps: {len(trajectory)}",
            "champion_step": champion["step"],
            "casebook_id": casebook.id
        })
        return context

    # ------------- helpers -------------

    def _collect_chat_messages(self, context: dict) -> List[Dict[str, Any]]:
        """
        Return recent chat messages as list of dicts {role, text, ts, id, conversation_id}.
        Priority:
        1) context["chat_messages"] if provided
        2) messages from top conversations in memory
        """
        msgs = context.get(self.chat_from_context_key)
        if isinstance(msgs, list) and msgs:
            return msgs[-self.chat_max_messages:]

        # Otherwise, pull from top conversations
        conversations = self.memory.chats.get_top_conversations(limit=3, by="messages")
        all_msgs = []
        for conv, _ in conversations:
            conv_msgs = self.memory.chats.get_messages(conv.id)
            for m in conv_msgs:
                all_msgs.append({
                    "id": m.id,
                    "conversation_id": m.conversation_id,
                    "role": m.role,
                    "text": m.text,
                    "ts": getattr(m, "created_at", None)
                })

        # Return most recent N across conversations
        all_msgs = sorted(all_msgs, key=lambda m: m.get("ts") or 0)
        return all_msgs[-self.chat_max_messages:]

    def _ensure_casebook(self, paper: dict, section_name: str, plan: dict) -> CaseBookORM:
        casebook_name = f"blog_{paper.get('id','unknown')}_{section_name}_{int(time.time())}"
        meta = {
            "paper_id": paper.get("id"),
            "paper_title": paper.get("title"),
            "section_name": section_name,
            "domains": plan.get("domains", []),
            "knowledge_hash": plan.get("meta", {}).get("knowledge_hash"),
            "transient": True,   # ⚠️ transient domains/NER; not persisted elsewhere
        }
        return self.memory.casebooks.ensure_casebook(
            name=casebook_name,
            description=f"Draft trajectory for '{section_name}' from fused knowledge",
            tag="draft_generator",
            meta=meta
        )

    def _vpm_dims_from_text_improver(self, result: Dict[str, Any]) -> Dict[str, float]:
        row = result.get("vpm_row", {})
        # Normalize keys used by controller
        return {
            "coverage": float(row.get("coverage_final", row.get("coverage", 0.0))),
            "correctness": float(row.get("correctness", 0.0)),
            "coherence": float(row.get("coherence", 0.0)),
            "citation_support": float(row.get("citation_support", 0.0)),
            "entity_consistency": float(row.get("entity_consistency", 0.0)),
            "readability": float(row.get("readability", 10.0)),  # FKGL band, but controller treats as float
            "novelty": float(row.get("novelty", 0.5)),
            # stickiness lives in TextImprover.scores; fall back to 0.5 if absent
            "stickiness": float(result.get("scores", {}).get("stickiness", 0.5)),
        }

    def _log_step(
        self,
        casebook: CaseBookORM,
        plan: dict,
        result: dict,
        step: int,
        decision,
        draft_text: str,
        context: dict,
    ) -> Dict[str, Any]:
        run_id = str(uuid.uuid4())
        pipeline_run_id = context.get("pipeline_run_id")
        goal = context.get("goal", {})

        # --- 1) Create dynamic scorables first ---
        draft_scorable = self.memory.dynamic_scorables.add(
            pipeline_run_id=pipeline_run_id,
            scorable_type=TargetType.DYNAMIC,
            source=self.name,
            text=draft_text,
            meta={
                "step": step,
                "decision": decision.signal.name,
                "vpm_row": result["vpm_row"],
                "edit_log": result.get("edit_log", []),
            },
        )
        scorable = ScorableFactory.from_orm(draft_scorable)

        # --- 2) Create case linking to scorables ---
        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id,
            goal_id=goal.get("id"),
            prompt_text=json.dumps({"plan_slice": self._plan_slice(plan)}),
            agent_name="draft_generator",
            scorables=[scorable.to_dict()],
            meta={
                "step": step,
                "decision": decision.signal.name,
                "vpm_row": result["vpm_row"],
            },
        )

        # --- 3) Goal scoring with fallback ---
        kind = "text"
        goal_text = goal.get("goal_text", "blog_general")
        try:
            goal_score = self.goals.score(kind, goal_text, result["vpm_row"])
        except KeyError:
            # Dynamically register a new GoalTemplate if missing
            from stephanie.agents.paper_improver.goals import GoalTemplate

            self.logger.log("GoalTemplateMissing", {
                "kind": kind,
                "goal": goal_text,
                "message": "Creating dynamic fallback template"
            })

            self.goals.templates[f"{kind}/{goal_text}"] = GoalTemplate(
                name=goal_text,
                dims=list(result["vpm_row"].keys()),  # use all dims present
                thresholds={d: 0.5 for d in result["vpm_row"].keys()},
            )
            goal_score = self.goals.score(kind, goal_text, result["vpm_row"])

        return {
            "step": step,
            "decision": decision.signal.name,
            "vpm_row": result["vpm_row"],
            "goal_score": goal_score,
            "case_id": case.id,
            "draft_len": len(draft_text or ""),
        }

    def _select_champion(self, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trajectory:
            return {"step": 0, "draft_text": "", "vpm_row": {}, "goal_score": {"score": 0.0}, "case_id": None}
        best = max(trajectory, key=lambda t: float(t["goal_score"]["score"]))
        # recover draft for case
        drafts = self.memory.casebooks.list_scorables(best["case_id"], role="draft")  # type: ignore
        draft_text = drafts[-1].text if drafts else ""
        return {
            "step": best["step"],
            "vpm_row": best["vpm_row"],
            "goal_score": best["goal_score"],
            "draft_text": draft_text,
            "case_id": best["case_id"],
        }

    def _log_reflection(self, casebook: CaseBookORM, section_name: str, trajectory: List[Dict[str, Any]], champion: Dict[str, Any], context: dict = {}) -> None:
        # Simple trend vectors (SIS can plot these)
        def series(dim: str) -> List[float]:
            return [float(t["vpm_row"].get(dim, 0.0)) for t in trajectory]

        reflection = {
            "section_name": section_name,
            "steps": len(trajectory),
            "champion_step": champion.get("step"),
            "goal_score_champion": champion.get("goal_score", {}).get("score"),
            "coverage_trend": series("coverage"),
            "citation_trend": series("citation_support"),
            "coherence_trend": series("coherence"),
            "entity_consistency_trend": series("entity_consistency"),
            "novelty_trend": series("novelty"),
            "stickiness_trend": series("stickiness"),
        }
        goal = context.get("goal", {})
        case = self.memory.casebooks.add_case(
            casebook_id=casebook.id,
            goal_id=goal.get("id"),
            prompt_text=f"Reflection for trajectory: {section_name}",
            agent_name="draft_generator",
            meta={"type": "reflection"}
        )
        self.memory.casebooks.add_scorable(
            case.id,
            scorable_type="reflection",
            pipeline_run_id=context.get("pipeline_run_id"),
            text=safe_json(reflection),
            meta={"section": section_name},
            role="reflection"
        )

    def _plan_slice(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "section_title": plan.get("section_title"),
            "units": [{"id": u.get("claim_id"), "claim": u.get("claim")} for u in plan.get("units", [])],
            "abbr": plan.get("entities", {}).get("ABBR", {}),
            "domains": plan.get("domains", [])[:5],
            "knowledge_hash": plan.get("meta", {}).get("knowledge_hash"),
        }
``n

## File: idea_extractor.py

`python
# File: stephanie/agents/knowledge/learnable_idea_extractor.py



from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.builders.belief_cartridge_builder import BeliefCartridgeBuilder
from stephanie.scoring.scorable_factory import ScorableFactory
from stephanie.scoring.scorer.mrq_scorer import \
    MRQScorer  # or wherever your scorer lives
from stephanie.utils.idea_parser import IdeaParser


class LearnableIdeaExtractorAgent(BaseAgent):
    """
    Extracts learnable, actionable ideas from research papers and encodes them
    into belief cartridges — structured cognitive scaffolds that Stephanie can use,
    test, and refine over time.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Components from config
        self.idea_parser = IdeaParser(cfg, logger=logger)
        self.idea_parser.prompt_loader = self.prompt_loader
        self.idea_parser.call_llm = self.call_llm
        self.idea_parser.memory = self.memory
        
        self.cartridge_builder = BeliefCartridgeBuilder(cfg, memory=memory, logger=logger)
        self.domain_classifier = ScorableClassifier(
            memory=memory,
            logger=logger,
            config_path=cfg.get("domain_seed_config_path", "config/domain/seeds.yaml"),
        )
        self.mrq_scorer = MRQScorer(cfg, memory=memory, container=container, logger=logger)  # Replace with your actual scorer

        # Settings
        self.top_k_domains = cfg.get("top_k_domains", 3)
        self.min_classification_score = cfg.get("min_classification_score", 0.6)
        self.max_ideas_per_paper = cfg.get("max_ideas_per_paper", 5)

    async def run(self, context: dict) -> dict:
        """
        Main pipeline:
        1. Load documents from context
        2. Parse each into structured ideas
        3. Classify domains
        4. Encode as belief cartridges
        5. Store results
        """
        documents = context.get(self.input_key, [])
        goal = context.get("goal", {})
        goal_id = goal.get("id")

        cartridges = []

        for doc in documents:
            try:
                doc_id = doc["id"]
                title = doc.get("title")
                summary = doc.get("summary")
                text = doc.get("content", doc.get("text", ""))

                # Skip if already processed
                if self._is_already_processed(doc_id):
                    self.logger.log("DocumentAlreadyProcessed", {"doc_id": doc_id})
                    continue

                # Step 1: Parse paper sections
                parsed_sections = await self._parse_document_sections(text, title, context)
                if not parsed_sections:
                    continue

                # Step 2: Extract candidate ideas
                raw_ideas = self.idea_parser.parse(parsed_sections)
                if not raw_ideas:
                    continue

                # Step 3: Score and rank ideas
                scored_ideas = self._score_ideas(context, raw_ideas)
                top_ideas = sorted(scored_ideas, key=lambda x: x["score"], reverse=True)[
                    : self.max_ideas_per_paper
                ]

                # Step 4: Build cartridges
                for idea in top_ideas:
                    cartridge = self.cartridge_builder.build(
                        title=f"{title}: {idea['title']}",
                        content=idea["description"],
                        source_type="paper",
                        source_id=doc_id,
                        domain_tags=idea.get("tags", []),
                        metadata={
                            "source_paper": title,
                            "source_url": doc.get("url"),
                            "abstract": summary,
                            "scoring": idea.get("scores", {}),
                            "integration_hint": idea.get("hint", ""),
                            "type": idea.get("type", "general"),
                            "goal_id": goal_id,
                        },
                    )
                    if cartridge:
                        cartridges.append(cartridge.to_dict())

                # Step 5: Classify and store domains
                self._classify_and_store_domains(text, doc_id)

            except Exception as e:
                self.logger.log("IdeaExtractionFailed", {"error": str(e), "doc_id": doc_id})

        context[self.output_key] = cartridges
        context["cartridge_ids"] = [c.get("id") for c in cartridges]
        return context

    def _is_already_processed(self, doc_id: int) -> bool:
        """Check if this document has already had ideas extracted."""
        return self.memory.belief_cartridges.exists_by_source(doc_id)

    async def _parse_document_sections(self, text: str, title: str, context: dict) -> dict:
        """Use prompt + parser to extract structured paper content."""
        try:
            # Could also use DocumentProfilerAgent output
            parsed = self.idea_parser.parse(text, title, context)
            return parsed
        except Exception as e:
            self.logger.log("SectionParsingFailed", {"error": str(e)})
            return {}

    def _score_ideas(self, context: dict, ideas: list) -> list:
        """Apply GILD/MRQ-style scoring to prioritize best ideas."""
        scored = []
        for idea in ideas:
            merged = {"idea_description": idea["description"], **context}
            scorable = ScorableFactory.from_text(idea["description"])
            score_bundle = self.mrq_scorer.score(merged, scorable)
            idea["score"] = score_bundle.overall_score()
            idea["scores"] = score_bundle.to_dict()
            scored.append(idea)
        return scored

    def _classify_and_store_domains(self, text: str, doc_id: int):
        """Classify the paper and assign domains."""
        results = self.domain_classifier.classify(text, self.top_k_domains, self.min_classification_score)
        for domain, score in results:
            self.memory.scorable_domains.insert(
                {
                    "document_id": doc_id,
                    "domain": domain,
                    "score": score,
                }
            )
            self.logger.log("DomainAssigned", {"domain": domain, "score": score})
``n

## File: improver.py

`python
# stephanie/agents/knowledge/text_improver.py
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import signal as _signal
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from stephanie.agents.paper_improver.goals import GoalScorer
from stephanie.knowledge.casebook_store import CaseBookStore
from stephanie.knowledge.knowledge_bus import KnowledgeBus
from stephanie.scoring.calibration_manager import CalibrationManager
from stephanie.scoring.scorable_factory import TargetType
from stephanie.utils.json_sanitize import safe_json

from ..paper_improver.faithfulness import FaithfulnessBot

FACTUAL_KWS = (
    "show",
    "prove",
    "result",
    "achiev",
    "increase",
    "decrease",
    "outperform",
    "error",
    "accuracy",
    "loss",
    "significant",
    "statistically",
)


def _supports_alarm() -> bool:
    """Return True if signal.alarm is usable on this platform/thread."""
    return hasattr(_signal, "SIGALRM") and os.name != "nt"


def _timeout_handler(signum, frame):
    raise TimeoutError("TextImprover timed out")


def atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding=encoding)
    tmp.replace(path)


class Improver:
    def __init__(
        self,
        cfg, 
        memory,
        workdir: str = "./data/text_runs",
        timeout: int = 60,
        seed: int = 0,
        faithfulness_topk: int = 5,
        kb: KnowledgeBus | None = None,
        casebooks: CaseBookStore | None = None,
        calibration: CalibrationManager | None = None,  
        logger=None,                                    
    ):
        self.cfg = cfg
        self.memory = memory
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.run_id = 0
        self.timeout = timeout
        self.seed = seed
        self.faithfulness_topk = faithfulness_topk
        self.kb = kb or KnowledgeBus()
        self.casebooks = casebooks or CaseBookStore()

        # NEW: logger + calibration wiring
        self.logger = logger
        self.calibration = calibration or CalibrationManager(
            cfg=self.cfg.get("calibration", {}),  # pass calibration sub-config
            memory=self.memory,
            logger=self.logger,
        )

        self.gs = GoalScorer()

    # --------------------------- public ---------------------------

    def improve(self, content_plan: Dict[str, Any]) -> Dict[str, Any]:
        # Optional timeout (Unix main thread only)
        alarm_installed = False
        if _supports_alarm():
            _signal.signal(_signal.SIGALRM, _timeout_handler)
            _signal.alarm(self.timeout)
            alarm_installed = True

        try:
            return self._improve_inner(content_plan)
        except TimeoutError:
            self._log("TextImproverTimeout", {"timeout_sec": self.timeout})
            result = {
                "error": "timeout",
                "passed": False,
                "scores": {},
                "vpm_row": {},
                "run_dir": "",
                "dpo_pair_path": "",
            }
        except Exception as e:
            self._log(
                "TextImproverUnexpectedError",
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            result = {
                "error": f"unexpected: {str(e)}",
                "traceback": traceback.format_exc(),
                "passed": False,
                "scores": {},
                "vpm_row": {},
                "run_dir": "",
            }
        finally:
            if alarm_installed:
                _signal.alarm(0)

        # Best-effort error artifact
        if result.get("run_dir"):
            try:
                rd = Path(result["run_dir"])
                atomic_write(rd / "ERROR.json", json.dumps(result, indent=2))
            except Exception:
                pass

        return result

    def _improve_inner(self, content_plan: Dict[str, Any]) -> Dict[str, Any]:
        self._seed_everything(self.seed)
        self.run_id += 1

        # 0) Validate & sanitize plan
        plan_norm = self._sanitize_plan(content_plan)

        # 1) Prepare run dir + meta
        plan_hash = hashlib.sha256(
            json.dumps(plan_norm, sort_keys=True).encode()
        ).hexdigest()[:8]
        run_dir = (
            self.workdir
            / f"run_{int(time.time())}_{uuid.uuid4().hex}_{plan_hash}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)

        atomic_write(
            run_dir / "meta.json",
            json.dumps(
                {
                    "plan_sha": plan_hash,
                    "seeds": {"python": self.seed},
                    "timeout": self.timeout,
                    "timestamp": time.time(),
                },
                indent=2,
            ),
        )
        plan_path = run_dir / "plan.json"
        atomic_write(plan_path, json.dumps(plan_norm, indent=2))

        # 2) Casebook + Case
        casebook_name = f"text_{plan_hash}_{(content_plan.get('section_title') or 'section')}"
        cb = self.casebooks.ensure_casebook(
            name=casebook_name,
            tags=["text_improver", "exemplar_text"],
            meta={"plan_sha": plan_hash},
        )
        case = self.casebooks.add_case(
            casebook_name=casebook_name,
            prompt_text=json.dumps(content_plan),
            agent_name="text_improver",
            meta={"run_dir": str(run_dir)},
        )

        # 3) Initial draft
        draft_path = self._generate_draft(plan_norm, run_dir)

        # 4) Score → Edit → Rescore
        initial_scores = self._score_draft(draft_path, plan_norm)
        final_text, edits = self._apply_edit_policy(
            draft_path=draft_path,
            plan=plan_norm,
            max_edits=6,
            trace_path=run_dir / "trace.ndjson",
        )
        final_scores = self._score_draft(draft_path, plan_norm)

        # 5) Build VPM + log
        vpm_row = self._build_vpm_row(initial_scores, final_scores, plan_norm)
        goal_eval = self.gs.score("text", "academic_summary", vpm_row)

        # Log to casebook
        self.casebooks.add_scorable(
            casebook_name=casebook_name,
            case_id=case.id,
            role="vpm",
            text=safe_json(vpm_row),
            meta={"goal": goal_eval},
            scorable_type=TargetType.DYNAMIC,
        )
        self.casebooks.add_scorable(
            casebook_name=casebook_name,
            case_id=case.id,
            role="text",
            text=(run_dir / "draft.md").read_text(),
            meta={"stage": "final"},
            scorable_type=TargetType.DYNAMIC,
        )

        # 6) Optional faithfulness
        faithfulness_score = None
        paper_text = plan_norm.get("paper_text")
        if not paper_text and plan_norm.get("paper_text_path"):
            try:
                p = Path(plan_norm["paper_text_path"])
                if p.exists():
                    paper_text = p.read_text()
            except Exception as e:
                self._log("PaperTextLoadFailed", {"error": str(e)})

        if paper_text and len(paper_text.strip()) > 100:
            try:
                bot = FaithfulnessBot(top_k=self.faithfulness_topk)
                bot.prepare_paper(paper_text)
                claims = [
                    {
                        "claim_id": u.get("claim_id"),
                        "claim": u.get("claim", ""),
                    }
                    for u in plan_norm.get("units", [])
                    if u.get("claim")
                ]
                faithfulness_score = bot.get_faithfulness_score(claims)
                final_scores["faithfulness"] = float(faithfulness_score)
                vpm_row["faithfulness"] = round(float(faithfulness_score), 3)
            except Exception as e:
                self._log("FaithfulnessCheckFailed", {"error": str(e)})

        # 7) DPO pair (persist locally + scorable)
        dpo_pair = {
            "content_plan_slice": self._extract_plan_slice(plan_norm),
            "prompt": "Generate faithful, clear, well-cited prose from this plan.",
            "rejected": (run_dir / "initial_draft.md").read_text(),
            "chosen": final_text,
            "metadata": {
                "run_id": self.run_id,
                "plan_hash": plan_hash,
                "initial_scores": initial_scores,
                "final_scores": final_scores,
                "score_deltas": {
                    k: round(
                        final_scores.get(k, 0.0) - initial_scores.get(k, 0.0),
                        4,
                    )
                    for k in set(initial_scores) | set(final_scores)
                },
                "applied_edits": edits,
            },
        }
        atomic_write(
            run_dir / "text_dpo_pair.json", json.dumps(dpo_pair, indent=2)
        )

        # Pass criteria
        core_ok = all(
            final_scores.get(d, 0.0) >= 0.7
            for d in ("coverage", "correctness", "coherence")
        )
        faithful_ok = (
            True if faithfulness_score is None else (faithfulness_score >= 0.7)
        )

        # Final logging
        self.casebooks.add_scorable(
            casebook_name=casebook_name,
            case_id=case.id,
            role="dpo_pair",
            text=safe_json(dpo_pair),
            meta=dpo_pair["metadata"],
            scorable_type=TargetType.DYNAMIC,
        )


        # --- Log a calibration sample per section/domain ---

        # choose a primary domain (however you’re tagging; fallback to "general")
        primary_domain = (plan_norm.get("domains") or ["general"])[0] if plan_norm.get("domains") else "general"

        # Simple features from your VPM row (extend as needed)
        calib_features = {
            "coverage": vpm_row.get("coverage_final", 0.0),
            "correctness": vpm_row.get("correctness", 0.0),
            "coherence": vpm_row.get("coherence", 0.0),
            "citation_support": vpm_row.get("citation_support", 0.0),
        }

        # Label: pass/fail → 1/0
        label = 1 if (core_ok and faithful_ok) else 0

        # Optional: a single “raw similarity” proxy (you can keep it simple)
        raw_similarity = 0.25*calib_features["coverage"] + 0.25*calib_features["correctness"] + \
                        0.25*calib_features["coherence"] + 0.25*calib_features["citation_support"]

        try:
            # GOOD: use manager's sanitizer
            self.calibration.log_event(
                domain=primary_domain or "general",
                query=(plan_norm.get("section_title") or "")[:2000],
                raw_sim=float(raw_similarity or 0.0),
                is_relevant=bool(label),
                scorable_id=str(case.id),
                scorable_type="text_draft",
                entity_type=None,
            )
        except Exception as e:
            self.logger.warning("CalibrationEventLogFailed", {"error": str(e)})



        self.kb.publish(
            "trajectory.step",
            {
                "casebook": cb.name if hasattr(cb, "name") else casebook_name,
                "case_id": case.id,
                "vpm": vpm_row,
                "goal": goal_eval,
            },
        )

        # 9) Pass criteria
        core_ok = all(
            final_scores.get(d, 0.0) >= 0.7
            for d in ("coverage", "correctness", "coherence")
        )
        faithful_ok = (
            True if faithfulness_score is None else (faithfulness_score >= 0.7)
        )

        return {
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "final_draft_path": str(draft_path),
            "vpm_row": vpm_row,
            "dpo_pair_path": str(run_dir / "text_dpo_pair.json"),
            "scores": final_scores,
            "passed": bool(core_ok and faithful_ok),
        }

    def _sanitize_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure required shape; drop nulls; keep keys we use. Never hard-fail on empty units."""
        if not plan:
            raise ValueError("Plan is None")

        out: Dict[str, Any] = {}
        out["section_title"] = (
            plan.get("section_title") or "Section"
        ).strip() or "Section"

        units_in = plan.get("units") or []
        clean_units: List[Dict[str, Any]] = []
        for u in units_in:
            if not isinstance(u, dict):
                continue
            claim = (u.get("claim") or "").strip()
            evidence = (u.get("evidence") or "See paper").strip()
            cid = u.get("claim_id")
            if not claim:
                continue
            clean_units.append(
                {"claim": claim, "evidence": evidence, "claim_id": cid}
            )

        # Fallback: if no valid units, synthesize a placeholder so pipeline can proceed
        if not clean_units:
            placeholder = f"Overview of {out['section_title']}."
            clean_units = [
                {
                    "claim": placeholder,
                    "evidence": "See paper",
                    "claim_id": None,
                }
            ]

        out["units"] = clean_units

        ents = plan.get("entities") or {}
        out["entities"] = {
            "ABBR": dict(ents.get("ABBR") or {}),
            "REQUIRED": list(ents.get("REQUIRED") or []),
        }

        # Optional extras
        if plan.get("paper_text"):
            out["paper_text"] = plan["paper_text"]
        if plan.get("paper_text_path"):
            out["paper_text_path"] = plan["paper_text_path"]
        if plan.get("outline"):
            out["outline"] = plan["outline"]

        return out

    # --------------------------- generation ---------------------------

    def _generate_draft(self, plan: Dict[str, Any], run_dir: Path) -> Path:
        title = plan.get("section_title", "Section")
        units = plan.get("units", [])
        abbrs = plan.get("entities", {}).get("ABBR", {})

        outline = plan.get("outline") or [
            u.get("claim_id") for u in units if u.get("claim_id")
        ]
        outline = [cid for cid in outline if cid]

        lead = self._lead_paragraph(title, units, abbrs)
        kg_neighbors = (plan.get("kg") or {}).get("neighbors", {})
        seen_ids = set()
        bullets: List[str] = []
        for u in units:
            cid = u.get("claim_id") or ""
            if cid and cid in seen_ids:
                continue
            seen_ids.add(cid)
            claim = u.get("claim", "No claim")
            evidence = u.get("evidence", "See paper")
            tag = f" [#{cid}]" if cid else ""
            cite = " [#]" if evidence and evidence != "See paper" else ""
            bullets.append(f"- {claim}{tag}.\n  *Evidence: {evidence}*{cite}")
            top = (kg_neighbors.get(cid) or [])
            if top:
                src = (top[0].get("sources") or [{}])[0]
                src_hint = src.get("scorable_id") or src.get("source_text") or top[0]["text"]
                if src_hint:
                    evidence = f"KG: {src_hint}"


        draft = f"# {title}\n\n{lead}\n\n"
        if outline:
            draft += (
                "### Outline\n"
                + "\n".join(f"- [{cid}]" for cid in outline)
                + "\n\n"
            )
        draft += "### Details\n" + "\n\n".join(bullets) + "\n"
        cite = " [#]" if evidence and evidence != "See paper" else ""
        bullets.append(f"- {claim}{tag}.\n  *Evidence: {evidence}*{cite}")

        draft_path = run_dir / "draft.md"
        atomic_write(draft_path, self._normalize_ws(draft))
        atomic_write(run_dir / "initial_draft.md", draft_path.read_text())
        return draft_path

    def _lead_paragraph(
        self, title: str, units: List[Dict[str, Any]], abbrs: Dict[str, str]
    ) -> str:
        claims = [u.get("claim", "") for u in units if u.get("claim")]
        head = (
            " ".join([c.rstrip(".") + "." for c in claims[:3]])
            or "This section summarizes key findings and method decisions from the paper."
        )
        for full, abbr in abbrs.items():
            if full in title and abbr not in head:
                head = head.replace(full, f"{full} ({abbr})", 1)
                break
        return head

    # --------------------------- scoring ---------------------------

    def _score_draft(
        self, draft_path: Path, plan: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Heuristic scorer for a draft, returning VPM-style dimensions.
        Guarded against empty/degenerate inputs so the pipeline never crashes.
        """
        # --- Load & normalize ---
        try:
            text = (draft_path.read_text() or "").strip()
        except Exception:
            text = ""
        # Safe default when text is empty
        if not text:
            return {
                "coverage": 0.0,
                "correctness": 0.0,
                "coherence": 0.0,
                "citation_support": 0.0,
                "entity_consistency": 0.0,
                "readability": 10.0,  # neutral band default
                "fkgl_raw": 10.0,
                "novelty": 0.5,
                "stickiness": 0.0,
                "len_chars": 0.0,
                "compactness": 1.0,
            }

        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()
        ]
        units = plan.get("units", [])

        # --- Coverage ---
        ids = [u.get("claim_id") for u in units if u.get("claim_id")]
        covered_ids = (
            sum(1 for cid in ids if f"[#{cid}]" in text) if ids else 0
        )
        if ids:
            coverage = covered_ids / max(1, len(ids))
        else:
            # Fallback: fuzzy overlap between unit claims and text terms
            unit_terms = [
                set(
                    re.findall(
                        r"\b[a-zA-Z]{5,}\b", (u.get("claim") or "").lower()
                    )
                )
                for u in units
            ]
            text_terms = set(re.findall(r"\b[a-zA-Z]{5,}\b", text.lower()))
            if unit_terms:
                hits = sum(
                    1
                    for terms in unit_terms
                    if terms and len(terms & text_terms) / len(terms) >= 0.5
                )
                coverage = hits / max(1, len(unit_terms))
            else:
                coverage = 0.0

        # --- Citation support / correctness proxy ---
        factual = [s for s in sentences if self._is_factual_sentence(s)]
        cited = sum(1 for s in factual if "[#]" in s)
        citation_support = (cited / max(1, len(factual))) if factual else 1.0
        correctness = citation_support  # proxy until abstract-alignment or judge is plugged in

        # --- Entity consistency (ABBR handling) ---
        abbrs = plan.get("entities", {}).get("ABBR", {}) or {}
        entity_consistency = 1.0
        if abbrs:
            for full, abbr in abbrs.items():
                # If neither appears, penalize (missing entity mention)
                if full not in text and abbr not in text:
                    entity_consistency = min(entity_consistency, 0.0)
                # If full term repeats many times without abbreviation after first use, light penalty
                elif text.count(full) > 1:
                    entity_consistency = min(entity_consistency, 0.5)

        # --- Readability (FKGL) ---
        words = re.findall(r"[A-Za-z]+", text)
        num_words = max(1, len(words))
        num_sentences = max(1, len(sentences))
        syllables = sum(self._count_syllables(w) for w in words) or 1
        fkgl_raw = (
            0.39 * (num_words / num_sentences)
            + 11.8 * (syllables / num_words)
            - 15.59
        )
        # Clamp to a sane band for readability score used by edit policy
        readability = float(max(6.0, min(15.0, fkgl_raw)))

        # --- Coherence (adjacency Jaccard) + title drift penalty ---
        coh_scores = []
        for i in range(len(sentences) - 1):
            s1 = set(re.findall(r"\w+", sentences[i].lower()))
            s2 = set(re.findall(r"\w+", sentences[i + 1].lower()))
            denom = len(s1 | s2)
            coh_scores.append((len(s1 & s2) / denom) if denom else 1.0)
        coherence = (
            sum(coh_scores) / max(1, len(coh_scores)) if coh_scores else 1.0
        )

        title_terms = set(
            re.findall(
                r"\b[a-zA-Z]{5,}\b", (plan.get("section_title") or "").lower()
            )
        )
        text_terms = set(re.findall(r"\b[a-zA-Z]{5,}\b", text.lower()))
        drift_penalty = (
            0.0
            if not title_terms
            else max(
                0.0,
                0.2 - len(title_terms & text_terms) / max(1, len(title_terms)),
            )
        )
        coherence = max(0.0, min(1.0, coherence - drift_penalty))

        # --- Stickiness (plan-term retention) ---
        stickiness = self._compute_stickiness(text, plan)

        # --- Compactness / size ---
        len_chars = float(len(text))
        compactness = len(re.sub(r"\s+", " ", text)) / max(
            1, len(text)
        )  # ~1 means minimal extra whitespace

        # --- Novelty (placeholder): prefer some lexical variety over near-duplication ---
        # Ratio of unique sentences to total sentences, softened into [0.3, 1.0]
        uniq_sent_ratio = len(set(sentences)) / max(1, len(sentences))
        novelty = max(0.3, min(1.0, 0.6 + 0.4 * uniq_sent_ratio))


        kg_neighbors = (plan.get("kg") or {}).get("neighbors", {})
        cid_set = {u.get("claim_id") for u in plan.get("units", []) if u.get("claim_id")}
        if kg_neighbors and cid_set:
            # count claims that have a strong neighbor
            strong = 0
            max_sim = 0.0
            for cid in cid_set:
                hits = kg_neighbors.get(cid) or []
                if hits:
                    best = max(h["score"] for h in hits)
                    max_sim = max(max_sim, best)
                    if best >= 0.75:  # tune threshold
                        strong += 1
            kg_support = strong / max(1, len(cid_set))
            novelty = 1.0 - max_sim  # crude novelty proxy
        else:
            kg_support = 0.0
            novelty = 0.6  # prior

        # blend KG support into correctness (optional)
        correctness = 0.5 * citation_support + 0.5 * kg_support


        return {
            "kg_support": float(kg_support), 
            "coverage": float(coverage),
            "correctness": float(correctness),
            "coherence": float(coherence),
            "citation_support": float(citation_support),
            "entity_consistency": float(entity_consistency),
            "readability": float(readability),
            "fkgl_raw": float(fkgl_raw),
            "novelty": float(novelty),
            "stickiness": float(stickiness),
            "len_chars": float(len_chars),
            "compactness": float(compactness),
        }

    def _compute_stickiness(self, text: str, plan: Dict[str, Any]) -> float:
        plan_terms = set()
        for unit in plan.get("units", []):
            claim = unit.get("claim", "") or ""
            for w in re.findall(r"\b[a-zA-Z]{5,}\b", claim.lower()):
                plan_terms.add(w)
        if not plan_terms:
            return 1.0
        text_words = set(re.findall(r"\b[a-zA-Z]{5,}\b", text.lower()))
        return len(plan_terms & text_words) / max(1, len(plan_terms))

    # --------------------------- edits ---------------------------

    def _apply_edit_policy(
        self,
        draft_path: Path,
        plan: Dict[str, Any],
        max_edits: int = 6,
        trace_path: Optional[Path] = None,
    ):
        text = draft_path.read_text()
        edits: List[str] = []

        for i in range(max_edits):
            scores = self._score_draft(draft_path, plan)
            change = False

            if scores["coverage"] < 0.8:
                missing = [
                    u
                    for u in plan.get("units", [])
                    if u.get("claim_id") and f"[#{u['claim_id']}]" not in text
                ]
                if missing:
                    u = missing[0]
                    line = f"- {u.get('claim', 'Claim')} [#{u['claim_id']}].\n  *Evidence: {u.get('evidence', 'See paper')}* [#]\n\n"
                    text = text.rstrip() + "\n" + line
                    edits.append(f"add_claim:{u['claim_id']}")
                    change = True

            if not change and scores["citation_support"] < 0.7:
                sentences = re.split(r"(?<=[.!?])\s+", text)
                for j, s in enumerate(sentences):
                    if self._is_factual_sentence(s) and "[#]" not in s:
                        sentences[j] = s.rstrip() + " [#]"
                        text = " ".join(sentences)
                        edits.append("add_citation_marker")
                        change = True
                        break

            if not change and scores["entity_consistency"] < 1.0:
                abbrs = plan.get("entities", {}).get("ABBR", {})
                for full, abbr in abbrs.items():
                    if full not in text and abbr in text:
                        text = re.sub(
                            rf"\b{re.escape(abbr)}\b",
                            f"{full} ({abbr})",
                            text,
                            count=1,
                        )
                        edits.append(f"expand_abbr:{full}->{abbr}")
                        change = True
                        break
                    if text.count(full) > 1:
                        first = True

                        def _swap(m):
                            nonlocal first
                            if first:
                                first = False
                                return m.group(0)
                            return abbr

                        text = re.sub(rf"\b{re.escape(full)}\b", _swap, text)
                        edits.append(f"abbreviate_repeats:{full}->{abbr}")
                        change = True
                        break

            if not change and not (9.0 <= scores["readability"] <= 11.0):
                if scores["readability"] > 11.0:
                    text = re.sub(r";\s+", ". ", text)
                    text = re.sub(r"\s+and\s+", ". ", text, count=1)
                    edits.append("split_long_sentences")
                else:
                    text = re.sub(r"\.\s+([a-z])", r", \1", text, count=1)
                    edits.append("join_short_sentences")
                change = True

            if not change and scores["coherence"] < 0.7:
                before = text
                text = re.sub(r"\n- ([^.\n]{0,60})\.\n- ", r"\n- \1; ", text)
                if text != before:
                    edits.append("merge_adjacent_bullets")
                    change = True
                else:
                    text = self._regenerate_lead_in(text, plan)
                    edits.append("regen_lead_in")
                    change = True

            if not change and self._has_duplicate_bullets(text):
                text = self._dedup_bullets(text)
                edits.append("dedup_bullets")
                change = True

            if not change:
                break

            atomic_write(draft_path, self._normalize_ws(text))
            if trace_path:
                with trace_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {"edit": i + 1, "scores": scores, "op": edits[-1]}
                        )
                        + "\n"
                    )

        return text, edits

    def _regenerate_lead_in(self, text: str, plan: Dict[str, Any]) -> str:
        m = re.search(r"^# .+\n\n(.+?)\n\n", text, flags=re.DOTALL)
        if not m:
            return text
        start, end = m.span(1)
        title = plan.get("section_title", "Section")
        abbrs = plan.get("entities", {}).get("ABBR", {})
        new_lead = self._lead_paragraph(title, plan.get("units", []), abbrs)
        return text[:start] + new_lead + text[end:]

    # --------------------------- helpers ---------------------------

    def _is_factual_sentence(self, s: str) -> bool:
        s_low = s.lower()
        return any(kw in s_low for kw in FACTUAL_KWS)

    def _build_vpm_row(
        self,
        initial: Dict[str, float],
        final: Dict[str, float],
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "section": plan.get("section_title", "unknown"),
            "coverage_initial": round(initial.get("coverage", 0.0), 3),
            "coverage_final": round(final.get("coverage", 0.0), 3),
            "correctness": round(final.get("correctness", 0.0), 3),
            "coherence": round(final.get("coherence", 0.0), 3),
            "citation_support": round(final.get("citation_support", 0.0), 3),
            "entity_consistency": round(
                final.get("entity_consistency", 0.0), 3
            ),
            "readability": round(final.get("readability", 0.0), 2),
            "fkgl_raw": round(final.get("fkgl_raw", 0.0), 2),
            "novelty": round(final.get("novelty", 0.0), 3),
            "stickiness": round(final.get("stickiness", 0.0), 3),
            "len_chars": int(final.get("len_chars", 0)),
            "compactness": round(final.get("compactness", 0.0), 3),
        }

    def _extract_plan_slice(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "section_title": plan.get("section_title"),
            "claim_count": len(plan.get("units", [])),
            "required_entities": plan.get("entities", {}).get("REQUIRED", []),
            "abbr": plan.get("entities", {}).get("ABBR", {}),
        }

    def _count_syllables(self, word: str) -> int:
        word = (word or "").lower()
        vowels = "aeiouy"
        if not word:
            return 1
        count = 1 if word[0] in vowels else 0
        for idx in range(1, len(word)):
            if word[idx] in vowels and word[idx - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        return max(1, count)

    def _normalize_ws(self, s: str) -> str:
        s = re.sub(r"[ \t]+\n", "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip() + "\n"

    def _has_duplicate_bullets(self, text: str) -> bool:
        bullets = re.findall(r"^- .+$", text, flags=re.MULTILINE)
        return len(bullets) != len(set(bullets))

    def _dedup_bullets(self, text: str) -> str:
        lines = text.splitlines()
        seen = set()
        out = []
        for ln in lines:
            if ln.startswith("- "):
                if ln in seen:
                    continue
                seen.add(ln)
            out.append(ln)
        return "\n".join(out) + ("\n" if not out or out[-1] != "" else "")

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass

    def _log(self, event: str, payload: Dict[str, Any]) -> None:
        if self.logger and hasattr(self.logger, "log"):
            try:
                self.logger.log(event, payload)
                return
            except Exception:
                pass
        # Fallback
        print(f"[{event}] {payload}")

    def _log_calibration_event(
        self,
        *,
        domain: str,
        query: str | None,
        raw_sim: float,
        is_relevant: bool,
        scorable_id: str | None = None,
        scorable_type: str | None = None,
        entity_type: str | None = None,
    ) -> None:
        # Ensure non-null & length-limited strings to satisfy NOT NULL constraints
        q = (query or "").strip()[:2000] or "N/A"
        sid = (scorable_id or "").strip() or "unknown"
        st = (scorable_type or "").strip() or "unknown"
        et = (entity_type or "").strip() or None
        try:
            self.calibration.log_event(
                domain=domain or "general",
                query=q,
                raw_sim=float(raw_sim),
                is_relevant=bool(is_relevant),
                scorable_id=sid,
                scorable_type=st,
                entity_type=et,
            )
        except Exception as e:
            self.logger.error("CalibrationEventLogFailed", {"error": str(e)})
``n

## File: knowledge_db_loader.py

`python
# stephanie/agents/knowledge/knowledge_db_loader.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL, PIPELINE_RUN_ID


class KnowledgeDBLoaderAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.top_k = cfg.get("top_k", 10)
        self.include_full_text = cfg.get("include_full_text", False)
        self.target_type = cfg.get(
            "target_type", "document"
        )  # or "section"
        self.include_ner = cfg.get("include_ner", False)

        self.doc_ids_scoring = cfg.get("doc_ids_scoring", False)
        self.doc_ids = cfg.get("doc_ids", [])

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        goal_text = goal.get("goal_text", "")
        pipeline_run_id = context.get(PIPELINE_RUN_ID)

        # 1. Fetch documents
        if self.doc_ids_scoring:
            if not self.doc_ids:
                self.logger.log(
                    "NoDocumentIdsProvided", "No document ids to score."
                )
                return context

            docs = self.memory.documents.get_by_ids(self.doc_ids)
            if not docs:
                self.logger.log("NoDocumentsFound", {"ids": self.doc_ids})
                return context
            self.logger.log(
                "DocumentsLoadedByIds",
                {"count": len(docs), "ids": self.doc_ids},
            )
            docs = [d.to_dict() for d in docs]
        else:
            docs = self.memory.embedding.search_related_scorables(
                goal_text, top_k=self.top_k, include_ner=self.include_ner, target_type=self.target_type
            )
            self.logger.log(
                "DocumentsSearched",
                {
                    "count": len(docs),
                    "goal_text": goal_text,
                    "top_k": self.top_k,
                },
            )

        # 2. Save retrieved doc dicts into context
        context[self.output_key] = docs

        context["retrieved_ids"] = [d["id"] for d in docs]

        for d in docs:
            self.memory.pipeline_references.insert(
                {
                    "pipeline_run_id": pipeline_run_id,
                    "scorable_type": d["scorable_type"],
                    "scorable_id": d["scorable_id"],
                    "relation_type": "retrieved",
                    "source": self.name,
                }
            )

        self.logger.log(
            "KnowledgeDBLoaded",
            {"count": len(docs), "search_method": self.target_type},
        )

        return context
``n

## File: knowledge_fuser.py

`python
# stephanie/agents/knowledge/knowledge_fuser.py
from __future__ import annotations

import hashlib
import logging
import time
import traceback
from typing import Any, Dict, List

from stephanie.agents.knowledge.chat_knowledge_builder import \
    ChatKnowledgeBuilder
from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.data.knowledge_unit import KnowledgeUnit
from stephanie.services.knowledge_graph_service import KnowledgeGraphService

_logger = logging.getLogger(__name__)


class KnowledgeFuser:
    """
    Advanced knowledge fuser that blends transient chat context with document content
    using AI models: domain classification, entity linking, semantic overlap.

    Produces a structured content plan suitable for draft generation.
    """

    def __init__(
        self, cfg: Dict[str, Any], memory: Any, container, logger: logging.Logger
    ):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        # Core components (already built in standard pattern)
        try:
            self.builder = ChatKnowledgeBuilder(cfg, memory, container, logger)
            self.classifier = ScorableClassifier(
                memory=memory,
                logger=logger,
                config_path=cfg.get(
                    "domain_config", "config/domain/seeds.yaml"
                ),
                metric=cfg.get("domain_metric", "cosine"),
            )
            self.kg_service = self.container.get("knowledge_graph")
            self.kg_service.initialize()  # ensure ready
        except Exception as e:
            _logger.error(
                f"Failed to initialize KnowledgeFuser dependencies: {e}",
                exc_info=True,
            )
            raise

        # Config
        self.top_k_domains = cfg.get("top_k_domains", 5)
        self.min_domain_score = cfg.get("min_classification_score", 0.6)
        self.max_claims = cfg.get("max_claims", 6)
        self.entity_merge_strategy = cfg.get(
            "entity_merge_strategy", "chat_priority"
        )

    async def fuse(
        self,
        *,
        text: str,
        chat_messages: List[Dict[str, Any]],
        section_name: str,
        conversation_id: int = None,
        scorable_id: str = None,
        scorable_type: str = "document_section",
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Fuse chat and paper knowledge into a structured content plan.
        """
        start_time = time.time()
        if not scorable_id:
            scorable_id = (
                f"temp:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
            )

        try:
            # Step 1: Build rich knowledge units
            k = self.builder.build(
                chat_messages=chat_messages,
                paper_text=text,
                conversation_id=conversation_id,
                context=context,
            )
            chat_ku = k["chat"]
            paper_ku = k["paper"]

            # Step 2: Domain alignment (semantic classification)
            try:
                domain_scores = self.classifier.classify(text=text)
                if not isinstance(domain_scores, list):
                    raise ValueError(
                        f"Expected list, got {type(domain_scores)}"
                    )

                # Handle both (str, float) tuples and dicts
                parsed = []
                for item in domain_scores:
                    if isinstance(item, tuple) and len(item) == 2:
                        domain, score = item
                        parsed.append(
                            {"domain": str(domain), "score": float(score)}
                        )
                    elif (
                        isinstance(item, dict)
                        and "domain" in item
                        and "score" in item
                    ):
                        parsed.append(
                            {
                                "domain": str(item["domain"]),
                                "score": float(item["score"]),
                            }
                        )
                    else:
                        _logger.error(
                            f"Unexpected domain score format: {item}"
                        )

                top_domains = [
                    d for d in parsed if d["score"] >= self.min_domain_score
                ][: self.top_k_domains]

            except Exception as e:
                _logger.error(f"Domain classification failed: {e}")
                top_domains = []

            # Step 3: Semantic phrase overlap (not just string match)
            overlap_phrases = self._semantic_overlap(chat_ku, paper_ku)

            # Step 4: Generate claim units from aligned knowledge
            units = self._generate_claims(
                overlap_phrases, paper_ku, scorable_id
            )

            # Step 5: Merge entities with strategy
            merged_entities = self._merge_entities(chat_ku, paper_ku)

            # Step 6: Optionally publish indexing event
            await self._publish_index_request(
                scorable_id, text, paper_ku, top_domains
            )

            # Final plan
            plan = {
                "section_title": section_name, 
                "units": units,
                "entities": merged_entities,
                "domains": top_domains,
                "paper_text": text,
                "scorable_id": scorable_id,
                "scorable_type": scorable_type,
                "meta": {
                    "knowledge_hash": self._hash_content(text, chat_messages),
                    "timestamp": time.time(),
                    "processing_duration_ms": int(
                        (time.time() - start_time) * 1000
                    ),
                    "sources": {
                        "chat_entity_count": sum(
                            len(v) for v in chat_ku.entities.values()
                        ),
                        "paper_entity_count": sum(
                            len(v) for v in paper_ku.entities.values()
                        ),
                        "phrase_overlap": len(overlap_phrases),
                        "used_kg_links": len(paper_ku.linked_kg_nodes),
                    },
                },
            }

            _logger.debug(
                "KnowledgeFusionComplete"
                f"section: {section_name}"
                f"claim_count: {len(units)}"
                f"domain_count: {len(top_domains)}"
                f"entity_count: {len(merged_entities.get('ABBR', {}))} + {len(merged_entities.get('REQUIRED', []))}"
            )

            return plan

        except Exception as e:
            _logger.error(
                "KnowledgeFusionFailed"
                f"section: {section_name}"
                f"error: {str(e)}"
                f"traceback: {traceback.format_exc()}"
            )
            raise

    # ---------------------------
    # Helpers
    # ---------------------------
    def _semantic_overlap(self, chat_ku, paper_ku):
        def normalize_phrases(items):
            return {
                (p["span"] if isinstance(p, dict) else str(p)).lower()
                for p in items if p
            }

        chat_set = normalize_phrases(chat_ku.anchors + chat_ku.phrases)
        paper_set = normalize_phrases(paper_ku.anchors + paper_ku.phrases)

        return list(chat_set & paper_set)

    def _generate_claims(
        self, overlaps: List[str], paper_ku: KnowledgeUnit, scorable_id: str
    ) -> List[Dict[str, Any]]:
        """
        Turn overlapping phrases into claims with evidence hints.
        """
        units = []
        for i, phrase in enumerate(overlaps[: self.max_claims]):
            # Attach supporting entities or KG nodes
            supporting_entities = [
                e["text"]
                for etype, elist in paper_ku.entities.items()
                for e in elist
                if phrase.lower() in e["text"].lower()
            ][:3]

            linked_nodes = [
                n
                for n in paper_ku.linked_kg_nodes
                if n.get("text", "").lower() in phrase.lower()
            ]

            units.append(
                {
                    "claim_id": f"C{i + 1}",
                    "claim": f"{phrase.strip().rstrip('.')}."
                    if not phrase.endswith(".")
                    else phrase,
                    "evidence_hint": "See related work"
                    if supporting_entities
                    else "General context",
                    "supporting_entities": supporting_entities,
                    "kg_links": [n["node_id"] for n in linked_nodes],
                    "confidence": 0.8
                    + 0.1 * min(i, 2),  # prioritize earlier claims
                }
            )
        return units

    def _merge_entities(
        self, chat_ku: KnowledgeUnit, paper_ku: KnowledgeUnit
    ) -> Dict[str, Any]:
        """
        Merge entities with configurable strategy.
        Default: chat wins for ABBR; union for REQUIRED.
        """
        abbr = {}
        required = []

        if self.entity_merge_strategy == "chat_priority":
            # Chat overrides paper on abbreviations
            abbr.update(paper_ku.entities.get("ABBR", {}))
            abbr.update(chat_ku.entities.get("ABBR", {}))  # chat wins

            # Union of required terms
            required_set = set(paper_ku.entities.get("REQUIRED", {})) | set(
                chat_ku.entities.get("REQUIRED", {})
            )
            required = list(required_set)[:12]
        else:
            # Fallback: simple merge
            abbr = {
                **paper_ku.entities.get("ABBR", {}),
                **chat_ku.entities.get("ABBR", {}),
            }
            required = list(
                dict.fromkeys(
                    list(paper_ku.entities.get("REQUIRED", {}).keys())
                    + list(chat_ku.entities.get("REQUIRED", {}).keys())
                )
            )[:12]

        return {"ABBR": abbr, "REQUIRED": required}

    async def _publish_index_request(
        self,
        scorable_id: str,
        text: str,
        ku: KnowledgeUnit,
        domains: List[Dict],
    ):
        """
        Publish async indexing request to KnowledgeBus.
        Enables non-blocking graph updates.
        """
        if not hasattr(self.memory, "bus"):
            return

        relationships = []
        entities = [ent for ents in ku.entities.values() for ent in ents]

        # Simple proximity-based relationships
        for i, e1 in enumerate(entities):
            for j in range(i + 1, len(entities)):
                e2 = entities[j]
                distance = abs(e1["end"] - e2["start"])
                if distance < 100:
                    rel_type = self._infer_relationship_type(e1, e2)
                    confidence = 0.7 + (0.3 / (distance + 1))
                    relationships.append(
                        {
                            "source": f"{scorable_id}:{e1['type']}:{e1['start']}-{e1['end']}",
                            "target": f"{scorable_id}:{e2['type']}:{e2['start']}-{e2['end']}",
                            "type": rel_type,
                            "confidence": min(confidence, 1.0),
                        }
                    )

        event = {
            "event_type": "knowledge_graph.index_request",
            "payload": {
                "scorable_id": scorable_id,
                "scorable_type": "document_section",
                "text": text,
                "entities": entities,
                "domains": domains,
                "relationships": relationships,
                "timestamp": time.time(),
                "source_agent": "KnowledgeFuser",
            },
        }

        try:
            await self.memory.bus.publish(
                subject=event["event_type"],
                payload=event["payload"]
            )

            _logger.debug(
                "IndexRequestPublished"
                f"scorable_id: {scorable_id}"
                f"entity_count: {len(entities)}"
                f"relationship_count: {len(relationships)}"
            )
        except Exception as e:
            _logger.error(f"Failed to publish index request: {e}")

    def _infer_relationship_type(self, e1: Dict, e2: Dict) -> str:
        ordered = e1["end"] < e2["start"]
        first, second = (e1, e2) if ordered else (e2, e1)
        pairs = {
            ("METHOD", "DATASET"): "evaluates",
            ("DATASET", "METRIC"): "measured_by",
            ("MODEL", "TASK"): "performs",
            ("AUTHOR", "PAPER"): "wrote",
            ("PAPER", "METHOD"): "introduces",
        }
        return pairs.get((first["type"], second["type"]), "related_to")

    def _hash_content(self, text: str, chat_messages: List[Dict]) -> str:
        combined = (
            text[:500]
            + "|||"
            + " ".join(
                m.get("text", "") for m in chat_messages if m.get("text")
            )
        )
        return hashlib.sha256(combined.encode()).hexdigest()[:10]
``n

## File: knowledge_fusion.py

`python
# stephanie/agents/knowledge/knowledge_fusion.py
"""
KnowledgeFusionAgent
--------------------
Fuses domains (seed-centroid classifier), entities (NER retriever), and
recent chat interactions into a transient "knowledge plan" per section.

Inputs (context):
- goal: { id, goal_text, ... }   (optional, used to bias domains)
- paper: { id, title, ... }       (optional, for metadata)
- sections: [ { section_name, section_text, paper_id? }, ... ]  <-- required
- chat_corpus: [ { role, text, ts? }, ... ]                     <-- optional; if absent tries memory
- top_domains: int (default=20)
- ner_k: int (default=12)
- ner_min_sim: float (default=0.60)

Outputs (context):
- knowledge_plans: List[dict]  # one per section, transient only
  Each plan contains:
    {
      "section_title": str,
      "paper_id": ...,
      "domains": [{"domain": str, "score": float}, ...],  # top_k (no DB writes)
      "entities": [{"text": str, "type": str, "similar": [..], "source": "paper|chat"} ...],
      "chat_support": [{"snippet": str, "overlap_entities": [...], "sim": float}, ...],
      "claims": [{"claim_id": str, "claim": str, "grounded_entities": [str, ...]}],
      "tags": list[str],  # quick, normalized tags (domains + key entities)
    }

Designed to be piped directly into DraftGeneratorAgent (as 'section plan').
"""

from __future__ import annotations

import logging
import re
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from time import time
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.models.ner_retriever import EntityDetector, NERRetrieverEmbedder
from stephanie.scoring.calibration_manager import CalibrationManager
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import ScorableFactory

_logger = logging.getLogger(__name__)


def _sentences(text: str) -> List[str]:
    if not text:
        return []
    # light sentence split
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 2]


def _unique_keep_order(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


@dataclass
class KFConfig:
    top_domains: int = 20
    min_domain_score: float = 0.0
    ner_k: int = 12
    ner_min_sim: float = 0.60
    ner_min_calibrated_sim: float = (
        0.45  # Critical: use calibrated similarity threshold
    )
    ephemeral_index_dir: str = "/tmp"  # index writes go to tmp; nothing to DB
    max_chat_snippets: int = 12
    max_chunk_size: int = 5000  # For indexing large texts
    entity_detection_fallback: bool = (
        True  # Use heuristic fallback if BERT-NER fails
    )
    enable_chunking: bool = False   

class KnowledgeFusionAgent(BaseAgent):
    """
    Fuse (domains ⨁ entities ⨁ chat overlap) into a transient knowledge plan.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.kfc = KFConfig(
            top_domains=cfg.get("top_domains", 20),
            min_domain_score=cfg.get("min_domain_score", 0.0),
            ner_k=cfg.get("ner_k", 12),
            ner_min_sim=cfg.get("ner_min_sim", 0.60),
            ner_min_calibrated_sim=cfg.get("ner_min_calibrated_sim", 0.45),
            ephemeral_index_dir=cfg.get("ephemeral_index_dir", "/tmp"),
            max_chat_snippets=cfg.get("max_chat_snippets", 12),
            max_chunk_size=cfg.get("max_chunk_size", 5000),
            entity_detection_fallback=cfg.get(
                "entity_detection_fallback", True
            ),
            enable_chunking=cfg.get("enable_chunking", False)
        )
        # Domain backbone (no DB writes of domain tags)
        self.domain_clf = ScorableClassifier(
            memory=self.memory,
            logger=self.logger,
            config_path=cfg.get("domain_config", "config/domain/seeds.yaml"),
            metric=cfg.get("domain_metric", "cosine"),
        )
        # Entity layer (ANN over session-only content)
        self.entity_detector = EntityDetector(device=cfg.get("device", "cuda"))
        self.ner = NERRetrieverEmbedder(
            model_name=cfg.get(
                "ner_model", "meta-llama/Llama-3.2-1B-Instruct"
            ),
            layer=cfg.get("ner_layer", 16), # in paper we seee 17 here the llm has on y 16 layers
            device=cfg.get("device", "cpu"),
            embedding_dim=cfg.get("ner_dim", 2048),
            index_path="data/ner_retriever/index",   # persistent path
            projection_enabled=cfg.get("ner_projection", False),
            projection_dim=cfg.get("ner_projection_dim", 2048),
            projection_dropout=cfg.get("ner_projection_dropout", 0.1),
            logger=self.logger,
            memory=self.memory,
            cfg=cfg,
        )

        from stephanie.scoring.calibration_manager import CalibrationManager
        self.calibration = CalibrationManager(
            cfg=cfg.get("calibration", {}),
            memory=self.memory,
            logger=self.logger
        )
    
        # ADD THIS: Periodic calibration trainer
        self.calibration_trainer = CalibrationTrainer(
            cfg=cfg.get("calibration", {}),
            memory=self.memory,
            logger=self.logger,
            calibration_manager=self.calibration,
        )


    async def run(self, context: dict) -> dict:
        goal = context.get("goal", {})

        chat = self.memory.chats.get_top_conversations(limit=10)
        documents = context.get("documents", []) or []
        for paper in tqdm(
            documents, desc="KnowledgeFusion Papers", unit="paper"
        ):
            sections = (
                self.memory.document_sections.get_by_document(paper.get("id"))
                or []
            )
            sections = [
                s.to_dict()
                for s in sections
                if s.section_text and len(s.section_text) > 20
            ]
            self.calibration_trainer.maybe_train()

            self.report(
                {
                    "event": "start",
                    "step": "KnowledgeFusion",
                    "details": f"Sections: {len(sections)}, Chat msgs: {len(chat)}",
                    "paper_title": paper.get("title"),
                }
            )

            # Build session index
            scorables = self._build_session_scorables(sections, chat)
            if self.kfc.enable_chunking:
                self._index_session_entities_with_chunking(scorables)
            else:
                await self._index_session_entities(scorables)

            # Progress bar for sections
            plans: List[Dict[str, Any]] = []
            for sec in tqdm(
                sections,
                desc=f"Sections of {paper.get('title', 'paper')}",
                unit="sec",
            ):
                try:
                    chat_dicts = [msg[0].to_dict() for msg in chat] if chat else []
                    plan = self._plan_for_section(sec, goal, chat_dicts)
                    plans.append(plan)
                except Exception as e:
                    self.logger.log(
                        "KnowledgeFusionSectionError",{
                            "section": sec.get("section_name"),
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        },
                    )

                # Structured log progress %
                pct = round(100 * len(plans) / max(1, len(sections)), 2)
                self.logger.log(
                    "KnowledgeFusionProgress",
                    {
                        "paper": paper.get("title"),
                        "sections_done": len(plans),
                        "sections_total": len(sections),
                        "pct_complete": pct,
                    },
                )

            context[self.output_key] = plans
            context["knowledge_plans"] = plans

            self.report(
                {
                    "event": "end",
                    "step": "KnowledgeFusion",
                    "details": f"Produced {len(plans)} plans (transient)",
                }
            )
        return context

    # ----------------------------
    # Internals
    # ----------------------------
    def _plan_for_section(self, section: dict, goal: dict, chat: List[Dict[str, str]]) -> Dict[str, Any]:
        sec_name = section.get("section_name", "section")
        text = section.get("section_text", "") or ""
        # 🔴 ADD: Fail fast if no content
        if not text.strip():
            self.logger.log("SectionTextMissing", {
                "section_name": sec_name,
                "paper_id": section.get("paper_id"),
                "warning": "Skipping plan generation due to empty section_text"
            })
            return {
                "section_title": sec_name,
                "paper_id": section.get("paper_id"),
                "error": "empty_section_text",
                "domains": [],
                "entities": [],
                "claims": [],
                "tags": [],
                "meta": {"skipped": True}
            }
        paper_id = section.get("paper_id")
        
        # ✅ CORRECT: Get actual scorable ID and target type
        scorable_id = str(section.get("id", f"temp_{uuid.uuid4().hex}"))
        scorable_type = section.get("target_type", "document_section")
        
        # A) Domains (top_k = 20; no DB writes) - WITH PROPER GOAL CONTEXT
        goal_context = self._build_goal_context(goal)
        dom_matches = self.domain_clf.classify(
            text=text,
            top_k=self.kfc.top_domains,
            min_value=self.kfc.min_domain_score,
            context=goal_context,
        )
        domains = [{"domain": d, "score": float(s)} for d, s in dom_matches]
        
        # B) Entities in the section (surface) + expand w/ nearest neighbors (from session index)
        surface_entities = self._detect_entities(text)
        # Critical: Pass domains to entity expansion for calibration
        expanded_entities = self._expand_entities(surface_entities, domains)
        
        # C) Chat overlap: find chat snippets that share entities or are semantically close
        chat_support = self._chat_overlap(chat, surface_entities, expanded_entities, limit=self.kfc.max_chat_snippets)
        
        # D) Claims with entity grounding
        claims = self._extract_claims_with_entities(text, expanded_entities)
        
        # E) Quick tags for the improver: domains + top entity lemmas
        tags = self._generate_tags(domains, expanded_entities)
        
        plan = {
            "section_title": sec_name,
            "section_name": sec_name,
            "paper_id": paper_id,
            "scorable_id": scorable_id,  
            "scorable_type": scorable_type,  
            "paper_text": text,
            "domains": domains,
            "entities": expanded_entities,
            "chat_support": chat_support,
            "claims": claims,
            "tags": tags,
            "goal_template": self.cfg.get("goal_template", "academic_summary"),
            "generation_style": self.cfg.get("generation_style", "grounded_explanatory"),
            "meta": {
                "knowledge_hash": self._compute_hash(paper_id, text, chat),
                "domain_confidence": self._get_domain_confidence(domains)
            }
        }
        return plan

    def _build_goal_context(self, goal: dict) -> Dict[str, Any]:
        """Build proper goal context for domain classification as per PACS.md"""
        return {
            "goal_text": goal.get("goal_text", ""),
            "goal_id": goal.get("id", ""),
            "strategy": goal.get("strategy", ""),
            "focus_area": goal.get("focus_area", ""),
            "goal_type": goal.get("type", "blog_generation"),
            "audience": goal.get("audience", "academic"),
            "intent": goal.get("intent", "explanation"),
        }

    def _detect_entities(
        self, text: str, source: str = "paper"
    ) -> List[Dict[str, Any]]:
        """Full NER pipeline with BERT-NER + heuristic fallback as per PACS.md"""
        # Primary: BERT-NER
        try:
            results = self.entity_detector.detect_entities(text)
            if results:
                return self._format_entities(results, text, source)
        except Exception as e:
            self.logger.log(
                "NERFallback", {"error": str(e), "method": "bert-ner"}
            )

        # Fallback: Heuristic rules if configured
        if self.kfc.entity_detection_fallback:
            return self._heuristic_entity_detection(text, source)
        return []

    def _format_entities(
        self, results: List[Dict[str, Any]], text: str, source: str
    ) -> List[Dict[str, Any]]:
        """Format entity detector results into standardized structure with calibrated similarity."""
        entities = []
        type_map = {
            "PER": "PERSON",
            "ORG": "ORGANIZATION",
            "LOC": "LOCATION",
            "MISC": "MISC",
            "DATE": "DATE",
            "TIME": "TIME",
            "MONEY": "MONEY",
            "PERCENT": "PERCENT",
            "FAC": "FACILITY",
            "GPE": "GPE",
            "METHOD": "METHOD",
            "METRIC": "METRIC",
            "ACRONYM": "ACRONYM",
        }

        for ent in results:
            start = ent.get("start", 0)
            end = ent.get("end", 0)
            etype = ent.get("type", "UNKNOWN")
            std_type = type_map.get(etype, etype)

            entities.append(
                {
                    "text": ent.get("text", text[start:end]),
                    "type": std_type,
                    "start": start,
                    "end": end,
                    "source": source,
                    "similarity": ent.get("score", 0.9),
                    "calibrated_similarity": ent.get("score", 0.9),
                }
            )

        return entities

    def _heuristic_entity_detection(
        self, text: str, source: str
    ) -> List[Dict[str, Any]]:
        """Heuristic entity detection as fallback per PACS.md."""
        entities = []

        # 1. Acronyms (all-caps words > 2 chars)
        for match in re.finditer(r"\b([A-Z]{2,})\b", text):
            entities.append(
                {
                    "text": match.group(1),
                    "type": "ACRONYM",
                    "start": match.start(),
                    "end": match.end(),
                    "source": source,
                    "similarity": 0.7,
                    "calibrated_similarity": 0.7,
                }
            )

        # 2. Methods (common ML terms)
        method_terms = [
            "MCTS",
            "Chain-of-Thought",
            "RAG",
            "Transformer",
            "AlphaZero",
            "L2Norm",
            "Backprop",
            "Gradient",
            "Attention",
            "Embedding",
        ]
        for term in method_terms:
            for match in re.finditer(
                r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE
            ):
                entities.append(
                    {
                        "text": term,
                        "type": "METHOD",
                        "start": match.start(),
                        "end": match.end(),
                        "source": source,
                        "similarity": 0.8,
                        "calibrated_similarity": 0.8,
                    }
                )

        # 3. Metrics (common ML metrics)
        metric_terms = [
            "accuracy",
            "precision",
            "recall",
            "F1",
            "AUC",
            "RMSE",
            "MAE",
            "BLEU",
            "ROUGE",
            "perplexity",
        ]
        for term in metric_terms:
            for match in re.finditer(
                r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE
            ):
                entities.append(
                    {
                        "text": term,
                        "type": "METRIC",
                        "start": match.start(),
                        "end": match.end(),
                        "source": source,
                        "similarity": 0.75,
                        "calibrated_similarity": 0.75,
                    }
                )

        return entities

    def _expand_entities(
        self,
        surface_entities: List[Dict[str, Any]],
        section_domains: List[Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """Expand entities with domain-aware retrieval and calibrated similarity"""
        expanded = []
        # Get primary domain for calibration (highest scoring)
        primary_domain = (
            section_domains[0]["domain"] if section_domains else None
        )

        for ent in surface_entities:
            # Query with domain calibration
            try:
                # Critical: Use domain for calibration
                sims = self.ner.retrieve_entities(
                    query=ent["text"],
                    k=self.kfc.ner_k,
                    min_similarity=self.kfc.ner_min_sim,
                    domain=primary_domain,
                )
            except Exception as e:
                self.logger.log(
                    "NERQueryError", {"entity": ent["text"], "error": str(e)}
                )
                sims = []

            similar = []
            for s in sims:
                # Always prefer calibrated similarity if available
                raw_sim = s.get("similarity", 0.0)
                calibrated_prob = self.calibration.get_calibrated_probability(
                    domain=primary_domain,
                    raw_sim=raw_sim
                )

                
                # Calculate confidence in this calibration
                calibration_confidence = self.calibration.get_confidence(
                    domain=primary_domain,
                    query=ent["text"]
                )
                
                # Apply confidence-weighted threshold
                effective_threshold = self.kfc.ner_min_calibrated_sim * (0.8 + 0.4 * calibration_confidence)

                effective_threshold = self.kfc.ner_min_calibrated_sim
                if calibrated_prob >= effective_threshold:
                    similar.append(
                        {
                            "entity_text": s.get("entity_text", ""),
                            "similarity": float(raw_sim),
                            "calibrated_similarity": float(calibrated_prob),
                            "calibration_confidence": calibration_confidence,
                            "entity_type": s.get("entity_type", "UNKNOWN"),
                            "source_text": s.get("source_text", "")[:200],
                            "scorable_id": s.get("scorable_id", ""),
                            "scorable_type": s.get("scorable_type", ""),
                            "domain": s.get("domain", primary_domain),
                        }
                    )

                # Log calibration event
                self.calibration.log_event(
                    domain=primary_domain,
                    query=ent["text"],
                    raw_sim=raw_sim,
                    is_relevant=calibrated_prob >= self.kfc.ner_min_calibrated_sim,
                    scorable_id=ent.get("scorable_id", "unknown"),
                    scorable_type=ent.get("scorable_type", "unknown"),
                    entity_type=ent.get("type", None)
                )

           # Only include if we have meaningful similar entities
            if similar or ent.get("calibrated_similarity", 0.0) >= self.kfc.ner_min_calibrated_sim:
                expanded.append(
                    {
                        "text": ent["text"],
                        "type": ent["type"],
                        "source": ent["source"],
                        "similar": similar,
                        "similarity": ent.get("similarity", 0.0),
                        "calibrated_similarity": ent.get("calibrated_similarity", 0.0),
                    }
                )

        # Dedup by head text, prioritize those with better calibrated similarity
        return self._dedup_entities(expanded)

    def _dedup_entities(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate entities while preserving those with highest calibrated similarity"""
        by_head = {}
        for e in entities:
            key = e["text"].lower()
            # If we have calibrated similarity, use that for comparison
            current_sim = by_head.get(key, {}).get(
                "calibrated_similarity", 0.0
            )
            new_sim = e.get("calibrated_similarity", 0.0)

            if key not in by_head or new_sim > current_sim:
                by_head[key] = e

        return list(by_head.values())

    def _extract_claims_with_entities(
        self, text: str, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract claims with entity grounding as per PACS.md."""
        sents = [s for s in _sentences(text) if len(s) > 40]
        claims = [
            {"claim_id": f"C{i + 1}", "claim": s}
            for i, s in enumerate(sents[:5])
        ]

        # Map entities to claims
        entity_map = {e["text"].lower(): e for e in entities}
        for claim in claims:
            claim["grounded_entities"] = []
            for word in claim["claim"].split():
                word_lower = word.lower().strip(".,;:")
                if word_lower in entity_map:
                    claim["grounded_entities"].append(
                        entity_map[word_lower]["text"]
                    )

        return claims

    def _generate_tags(
        self, domains: List[Dict[str, float]], entities: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate tags with domain weighting as per PACS.md."""
        # Top domains (weighted higher)
        domain_tags = [
            d["domain"]
            for d in sorted(domains, key=lambda x: x["score"], reverse=True)[
                :3
            ]
        ]

        # Top entities (weighted lower)
        entity_tags = []
        for entity in entities:
            # Prefer entities with high calibrated similarity
            for similar in sorted(
                entity.get("similar", []),
                key=lambda x: x.get("calibrated_similarity", 0),
                reverse=True,
            ):
                if (
                    similar["calibrated_similarity"] > 0.7
                    and similar["entity_text"] not in entity_tags
                ):
                    entity_tags.append(similar["entity_text"])
            if len(entity_tags) >= 5:
                break

        # Combine with domain weighting
        return _unique_keep_order(domain_tags + entity_tags)

    def _chat_overlap(
        self,
        chat: List[Dict[str, str]],
        surface_entities: List[Dict[str, Any]],
        expanded_entities: List[Dict[str, Any]],
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Rank chat snippets by entity overlap + embedding similarity to section entities.
        No DB writes; uses memory.embedding if available for transient scoring.
        """
        # entity vocabulary from section
        entity_terms = set([e["text"].lower() for e in surface_entities])
        for e in expanded_entities:
            for s in e.get("similar", []):
                t = s.get("entity_text", "").lower()
                if t:
                    entity_terms.add(t)

        results = []
        if not chat:
            return results

        # Prepare an aggregate entity string as query anchor
        query_anchor = (
            ", ".join(sorted(list(entity_terms))[:24]) or "section entities"
        )
        try:
            q_emb = self.memory.embedding.get_or_create(query_anchor)
        except Exception:
            q_emb = None

        for item in chat:
            text = item.get("text", "") or ""
            if not text:
                continue
            # quick overlap
            overlap = [t for t in entity_terms if t in text.lower()]
            overlap_score = (
                min(1.0, len(overlap) / max(1, len(entity_terms)))
                if entity_terms
                else 0.0
            )

            # semantic proximity (optional if embedding infra present)
            sim_score = 0.0
            if q_emb is not None:
                try:
                    c_emb = self.memory.embedding.get_or_create(text[:2000])
                    # cosine similarity
                    num = float((q_emb * c_emb).sum())
                    den = (float((q_emb * q_emb).sum()) ** 0.5) * (
                        float((c_emb * c_emb).sum()) ** 0.5
                    ) + 1e-8
                    sim_score = num / den
                except Exception as e:
                    self.logger.log(
                        "ChatOverlapEmbeddingError",
                        {"error": str(e), "snippet_length": len(text)},
                    )
                    sim_score = 0.0

            score = 0.6 * overlap_score + 0.4 * sim_score
            if score > 0.05:  # keep useful ones
                results.append(
                    {
                        "snippet": text[:500],
                        "overlap_entities": overlap[:8],
                        "sim": round(float(score), 4),
                        "role": item.get("role"),
                        "ts": item.get("ts"),
                    }
                )

        # sort by score desc, truncate
        results.sort(key=lambda r: r["sim"], reverse=True)
        return results[:limit]

    # ----------------------------
    # Session indexing (ephemeral)
    # ----------------------------
    def _build_session_scorables(
        self,
        sections: List[Dict[str, Any]],
        chat: List[Dict[str, str]],
    ) -> List[Scorable]:
        """Build scorables using proper DB IDs (not fabricated composite IDs)."""
        scorables: List[Scorable] = []

        # Paper sections as scorables (entity-bearing units)
        for sec in sections:
            text = sec.get("section_text", "") or ""
            if not text:
                continue
                
            # ✅ CORRECT: Use actual DB ID (as string) and proper target type
            scorable_id = str(sec.get("id", f"temp_{uuid.uuid4().hex}"))
            target_type = sec.get("target_type", "document_section")
            
            scorables.append(Scorable(
                id=scorable_id,
                text=text,
                target_type=target_type
            ))

        # Recent chat messages as scorables (so entities from chat are retrievable)
        for msg in chat or []:
            scorable = ScorableFactory.from_orm(msg[0])
            scorables.append(scorable)

        return scorables

    async def _index_session_entities(self, scorables: List[Scorable]) -> None:
        """
        Instead of indexing directly, publish to KnowledgeBus for async processing.
        No DB writes; no blocking.
        """
        total_queued = 0
        events_published = 0

        for scorable in scorables:
            text = scorable.text.strip()
            if len(text) < 100:
                continue

            try:
                # Classify domains for this scorable
                domain_matches = self.domain_clf.classify(
                    text=text,
                    top_k=self.kfc.top_domains,
                    min_value=self.kfc.min_domain_score
                )
                domains = [{"domain": d, "score": float(s)} for d, s in domain_matches]
                # Detect entities
                results = self.entity_detector.detect_entities(text)  # raw tuples
                entities = self._format_entities(results, text, source="paper")  # normalize to dicts
                filtered_ents = [
                    e for e in entities
                    if e.get("calibrated_similarity", e.get("similarity", 0)) >= self.kfc.ner_min_sim
                ]

                # Build relationships (local heuristic)
                relationships = []
                for i, e1 in enumerate(filtered_ents):
                    for j in range(i + 1, len(filtered_ents)):
                        e2 = filtered_ents[j]
                        distance = abs(e1["end"] - e2["start"])
                        if distance < 100:
                            rel_type = self._infer_relationship_type(e1, e2)
                            confidence = self._calculate_relationship_confidence(e1, e2, distance, domains)
                            if confidence >= 0.75:
                                relationships.append({
                                    "source": f"{scorable.id}:{e1['type']}:{e1['start']}-{e1['end']}",
                                    "target": f"{scorable.id}:{e2['type']}:{e2['start']}-{e2['end']}",
                                    "type": rel_type,
                                    "confidence": confidence
                                })

                # Publish indexing job
                event = {
                    "event_type": "knowledge_graph.index_request",
                    "payload": {
                        "scorable_id": scorable.id,
                        "scorable_type": scorable.target_type,
                        "text": text,
                        "entities": filtered_ents,
                        "domains": domains,
                        "relationships": relationships,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source_agent": "KnowledgeFusionAgent"
                    }
                }

                await self.memory.bus.publish(
                    subject=event["event_type"],
                    payload=event["payload"]
                )

                total_queued += len(filtered_ents)
                events_published += 1

            except Exception as e:
                self.logger.log("KnowledgeFusionIndexEventFailed", {
                    "scorable_id": scorable.id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })

        self.logger.log("KnowledgeFusionIndexEventsPublished", {
            "events_published": events_published,
            "entities_queued": total_queued
        })

    async def _index_session_entities_with_chunking(self, scorables: List[Scorable]) -> None:
        """
        Chunk large texts and publish async indexing requests.
        """
        total_queued = 0
        events_published = 0

        for scorable in scorables:
            text = scorable.text.strip()
            if not text:
                continue

            chunks = []

            if len(text) > self.kfc.max_chunk_size:
                for i in range(0, len(text), self.kfc.max_chunk_size):
                    chunk_text = text[i:i + self.kfc.max_chunk_size]
                    if len(chunk_text.strip()) > 50:
                        chunk_id = f"{scorable.id}_chunk_{i // self.kfc.max_chunk_size}"
                        chunks.append({
                            "id": chunk_id,
                            "text": chunk_text,
                            "parent_id": scorable.id,
                            "offset": i
                        })
            else:
                chunks = [{"id": scorable.id, "text": text}]

            for chunk in chunks:
                try:
                    # Reuse domain classification logic
                    domain_matches = self.domain_clf.classify(
                        text=chunk["text"],
                        top_k=self.kfc.top_domains,
                        min_value=self.kfc.min_domain_score
                    )
                    domains = [{"domain": d, "score": float(s)} for d, s in domain_matches]

                    results = self.entity_detector.detect_entities(text)  # raw tuples
                    entities = self._format_entities(results, text, source="paper")  # normalize to dicts
                    filtered_ents = [
                        e for e in entities
                        if e.get("calibrated_similarity", e.get("similarity", 0)) >= self.kfc.ner_min_sim
                    ]

                    # Adjust entity spans relative to chunk offset
                    offset = chunk.get("offset", 0)
                    for e in filtered_ents:
                        e["start"] += offset
                        e["end"] += offset

                    # Relationships within chunk
                    relationships = []
                    for i, e1 in enumerate(filtered_ents):
                        for j in range(i + 1, len(filtered_ents)):
                            e2 = filtered_ents[j]
                            distance = abs(e1["end"] - e2["start"])
                            if distance < 100:
                                rel_type = self._infer_relationship_type(e1, e2)
                                confidence = self._calculate_relationship_confidence(e1, e2, distance, domains)
                                if confidence >= 0.75:
                                    relationships.append({
                                        "source": f"{chunk['id']}:{e1['type']}:{e1['start']}-{e1['end']}",
                                        "target": f"{chunk['id']}:{e2['type']}:{e2['start']}-{e2['end']}",
                                        "type": rel_type,
                                        "confidence": confidence
                                    })

                    # Publish event
                    event = {
                        "event_type": "knowledge_graph.index_request",
                        "payload": {
                            "scorable_id": chunk["id"],
                            "scorable_type": f"{scorable.target_type}_chunk",
                            "text": chunk["text"],
                            "entities": filtered_ents,
                            "domains": domains,
                            "relationships": relationships,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "source_agent": "KnowledgeFusionAgent",
                            "is_chunk": True,
                            "original_scorable_id": scorable.id
                        }
                    }

                    await self.memory.bus.publish(
                        subject=event["event_type"],
                        payload=event["payload"]
                    )
                    total_queued += len(filtered_ents)
                    events_published += 1

                except Exception as e:
                    self.logger.log("KnowledgeFusionChunkIndexEventFailed", {
                        "chunk_id": chunk.get("id"),
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })

        self.logger.log("KnowledgeFusionChunkedIndexEventsPublished", {
            "chunks_queued": events_published,
            "entities_queued": total_queued
        })

    # ----------------------------
    # Utility methods
    # ----------------------------
    def _compute_hash(
        self, paper_id: Optional[Any], text: str, chat: List[Dict]
    ) -> str:
        """Compute hash of knowledge state for versioning"""
        import hashlib

        combined = f"{paper_id or 'unknown'}|||{text[:1000]}|||{len(chat)}"
        return hashlib.sha256(combined.encode()).hexdigest()[:8]

    def _get_domain_confidence(self, domains: List[Dict[str, float]]) -> float:
        """Estimate confidence in domain classification"""
        if not domains:
            return 0.0
        # Weighted average of scores
        total = sum(d["score"] for d in domains)
        return min(1.0, total / len(domains))

    def _fallback_chat_corpus(self, context: dict) -> List[Dict[str, str]]:
        """
        Best-effort: if the caller didn't pass chat_corpus, try to pull recent
        conversational text from memory (safe, read-only). If nothing exists,
        return an empty list.
        """
        try:
            # If you maintain conversations/casebooks for chats, adapt here:
            # e.g., self.memory.conversations.latest(n=200)
            return context.get("recent_messages", [])
        except Exception as e:
            self.logger.log("ChatFallbackError", {"error": str(e)})
            return []

    def _get_domain_with_fallbacks(self, domain: str) -> str:
        """Get domain with hierarchical fallbacks (specific → parent → general → identity)."""
        # Try specific domain first
        if self.calibration.has_calibration(domain):
            return domain
            
        # Try parent domain (e.g., "computer_vision" → "ai")
        parent_domain = self._get_parent_domain(domain)
        if parent_domain and self.calibration.has_calibration(parent_domain):
            return parent_domain
            
        # Try general domain
        if self.calibration.has_calibration("general"):
            return "general"
            
        # No calibration available - use identity function
        return "identity"

    def _get_parent_domain(self, domain: str) -> Optional[str]:
        """Get parent domain from hierarchy configuration."""
        domain_hierarchy = self.cfg.get("domain_hierarchy", {})
        return domain_hierarchy.get(domain)
    
    def _infer_relationship_type(self, e1: Dict, e2: Dict) -> str:
        ordered = e1["end"] < e2["start"]
        first, second = (e1, e2) if ordered else (e2, e1)
        type_pairs = {
            ("METHOD", "DATASET"): "evaluates",
            ("DATASET", "METRIC"): "measured_by",
            ("MODEL", "TASK"): "performs",
            ("AUTHOR", "PAPER"): "wrote",
            ("PAPER", "METHOD"): "introduces"
        }
        return type_pairs.get((first["type"], second["type"]), "related_to")

    def _calculate_relationship_confidence(self, e1: Dict, e2: Dict, distance: int, domains: List[Dict]) -> float:
        base_score = 1.0 - (distance / 100)
        domain_bonus = 0.1 if any(d["domain"] in {"ml", "nlp"} for d in domains) else 0.0
        proximity_bonus = 0.1 if distance < 20 else 0.0
        return max(min(base_score + domain_bonus + proximity_bonus, 1.0), 0.0)


class CalibrationTrainer:
    """Handles periodic training of calibration models from collected data."""
    
    def __init__(self, cfg: Dict, memory, logger: Any, calibration_manager: 'CalibrationManager'):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.calibration = calibration_manager
        self.last_train = 0
        self.train_interval = cfg.get("calibration_train_interval", 3600)  # Default: 1 hour
        self.lookback_hours = cfg.get("calibration_lookback_hours", 24)

    def maybe_train(self) -> bool:
        now = time()
        if now - self.last_train < self.train_interval:
            return False

        trained_any = False
        for domain in self._get_domains_to_train():
            pos, neg, total = self.calibration.domain_counts(domain)   # implement below
            self.logger.info("CalibrationTrainer: label mix",
                            extra={"domain": domain, "pos": pos, "neg": neg, "total": total})

            MIN_POS = self.cfg.get("calibration", {}).get("min_pos", 10)
            MIN_NEG = self.cfg.get("calibration", {}).get("min_neg", 10)

            if pos == 0 and neg == 0:
                self.logger.info("CalibrationTrainer: no samples yet", extra={"domain": domain})
                continue

            # Train with fallback if single class; require balance for full model
            if pos < 1 or neg < 1:
                self.logger.warning("CalibrationTrainer: one-class data — using fallback",
                                    extra={"domain": domain, "pos": pos, "neg": neg})
                trained_any |= self.calibration.train_model(domain, allow_fallback=True)
                continue

            if pos < MIN_POS or neg < MIN_NEG:
                self.logger.warning("CalibrationTrainer: skipping — insufficient class balance",
                                    extra={"domain": domain, "pos": pos, "neg": neg,
                                        "need_pos": MIN_POS, "need_neg": MIN_NEG})
                continue

            trained_any |= self.calibration.train_model(domain, allow_fallback=False)

        if trained_any:
            self.last_train = now
        return trained_any
    
    def _get_domains_to_train(self) -> List[str]:
        """Get domains that need retraining."""
        # Configured domains
        configured = self.cfg.get("domains", ["general"])
        
        # Recently active domains
        recent = self._get_recent_domains(self.lookback_hours)
        
        # Deduplicate + preserve order
        return list(dict.fromkeys(configured + recent))

    def _get_recent_domains(self, hours: int = 24) -> List[str]:
        """Get domains with recent calibration activity (fallback: general)."""
        try:
            # If CalibrationManager has a store:
            if hasattr(self.calibration, "memory") and hasattr(self.calibration.memory, "calibration_events"):
                since = datetime.now() - timedelta(hours=hours)
                return self.memory.calibration_events.get_recent_domains(since=since)
        except Exception as e:
            self.logger.warning(f"CalibrationTrainer: failed to fetch recent domains: {e}")

        return ["general"]
``n

## File: knowledge_loader.py

`python
# stephanie/agents/knowledge/knowledge_loader.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL


class KnowledgeLoaderAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.domain_seeds = cfg.get("domain_seeds", {})
        self.top_k = cfg.get("top_k", 3)
        self.threshold = cfg.get("domain_threshold", 0.0)
        self.include_full_text = cfg.get("include_full_text", False)
        self.prefer_sections = cfg.get("prefer_sections", True)
        self.knowledge_type = cfg.get("knowledge_type", "documents")  # or 'cartridges'

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        goal_text = goal.get("goal_text", "")
        documents = context.get("documents", [])

        if not goal_text or not documents:
            self.report({
                "event": "skipped",
                "step": "KnowledgeLoader",
                "reason": "Missing goal or documents",
            })
            return context

        self.report({
            "event": "start",
            "step": "KnowledgeLoader",
            "goal_snippet": goal_text[:120],
            "doc_count": len(documents),
        })

        # Step 1: Assign domain to the goal
        goal_vector = self.memory.embedding.get_or_create(goal_text)
        domain_vectors = {
            domain: np.mean(
                [self.memory.embedding.get_or_create(ex) for ex in examples], axis=0
            )
            for domain, examples in self.domain_seeds.items()
        }

        goal_domain, goal_domain_score = None, -1
        domain_scores = []

        for domain, vec in domain_vectors.items():
            score = float(cosine_similarity([goal_vector], [vec])[0][0])
            domain_scores.append((domain, score))
            if score > goal_domain_score:
                goal_domain, goal_domain_score = domain, score

        context["goal_domain"] = goal_domain
        context["goal_domain_score"] = goal_domain_score

        self.report({
            "event": "goal_domain_assigned",
            "step": "KnowledgeLoader",
            "goal_domain": goal_domain,
            "score": round(goal_domain_score, 4),
            "all_scores": {d: round(s, 4) for d, s in domain_scores},
        })

        # Step 2: Filter documents and/or sections
        filtered = []

        for doc in documents:
            doc_id = doc["id"]
            doc_summary = doc.get("summary", "")
            doc_text = doc.get("text", "")
            doc_title = doc.get("title", "")

            doc_domains = self.memory.scorable_domains.get_domains(str(doc_id), self.knowledge_type) 

            for dom in doc_domains[: self.top_k]:
                if dom.domain == goal_domain and dom.score >= self.threshold:
                    selected_content = (
                        doc_text if self.include_full_text else doc_summary
                    )
                    filtered.append(
                        {
                            "id": doc_id,
                            "title": doc_title,
                            "domain": dom.domain,
                            "score": dom.score,
                            "content": selected_content,
                            "source": "document",
                        }
                    )
                    self.report({
                        "event": "doc_selected",
                        "step": "KnowledgeLoader",
                        "doc_id": doc_id,
                        "title": doc_title[:80],
                        "domain": dom.domain,
                        "score": round(dom.score, 4),
                    })
                    break

            for section in doc.get("sections", []):
                section_id = section["id"]
                section_name = section.get("section_name", "Unknown")
                section_text = section.get("section_text", "")
                section_summary = section.get("summary", "")

                section_domains = self.memory.section_domains.get_domains(section_id)
                for sec_dom in section_domains[: self.top_k]:
                    if sec_dom.domain == goal_domain and sec_dom.score >= self.threshold:
                        selected_content = (
                            section_text if self.include_full_text else section_summary
                        )
                        filtered.append(
                            {
                                "id": doc_id,
                                "section_id": section_id,
                                "title": f"{doc_title} - {section_name}",
                                "domain": sec_dom.domain,
                                "score": sec_dom.score,
                                "content": selected_content,
                                "source": "section",
                            }
                        )
                        self.report({
                            "event": "section_selected",
                            "step": "KnowledgeLoader",
                            "doc_id": doc_id,
                            "section": section_name,
                            "domain": sec_dom.domain,
                            "score": round(sec_dom.score, 4),
                        })
                        break

        context[self.output_key] = filtered
        context["filtered_document_ids"] = list({doc["id"] for doc in filtered})

        self.report({
            "event": "end",
            "step": "KnowledgeLoader",
            "filtered_count": len(filtered),
            "unique_docs": len(context["filtered_document_ids"]),
        })

        return context
``n

## File: knowledge_retriever.py

`python
# stephanie/agents/knowledge/knowledge_retriever.py
from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent


class KnowledgeRetriever(BaseAgent):
    """
    Thin wrapper over KnowledgeGraphService that builds a knowledge_tree for the
    provided text and returns a compact 'relevant_knowledge' bundle.

    Design choices:
    - **Single code path**: assumes a KnowledgeGraphService with `.build_tree(...)`.
    - **No fancy branching**: if the service is missing, returns a tiny heuristic bundle.
    - **PlanTrace-friendly**: always returns JSON-serializable dicts.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        # one dependency: the KG service
        try:
            self.kg = container.get("knowledge_graph")
        except Exception:
            self.kg = None

        # knobs
        self.top_k_claims = int(cfg.get("knowledge_retriever", {}).get("top_k_claims", 8))
        self.top_k_insights = int(cfg.get("knowledge_retriever", {}).get("top_k_insights", 8))
        self.include_vpm = bool(cfg.get("knowledge_retriever", {}).get("include_vpm_payload", True))

    # ---------- public API ----------

    def retrieve(
        self,
        *,
        query: str,
        context: str,
        chat_corpus: Optional[List[Dict[str, Any]]] = None,
        trajectories: Optional[List[Dict[str, Any]]] = None,
        domains: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Build and return relevant knowledge for a (query, context) pair.

        Returns:
            {
              "query": str,
              "paper_id": str,
              "knowledge_tree": {...},            # from KG service
              "highlights": { "claims": [...], "insights": [...], "entities": [...] },
              "vpm_payload": {...} | None,
              "stats": {...}
            }
        """
        t0 = time.time()
        chat_corpus = chat_corpus or []
        trajectories = trajectories or []
        domains = domains or []

        paper_id = _stable_id(query or context)

        # Preferred path: rely *only* on the service's build_tree
        if self.kg and hasattr(self.kg, "build_tree"):
            self.report({"event": "KRStart", "paper_id": paper_id, "len_context": len(context)})
            tree = self.kg.build_tree(
                paper_text=context or (query or ""),
                paper_id=paper_id,
                chat_corpus=chat_corpus,
                trajectories=trajectories,
                domains=domains,
            ) or {}

            # Small, useful highlights for PlanTrace viewing
            claims = [c.get("text", "") for c in (tree.get("claims") or [])][: self.top_k_claims]
            insights = [i.get("text", "") for i in (tree.get("insights") or [])][: self.top_k_insights]

            # `entities` shape can vary; normalize to strings when possible
            ents_raw = tree.get("entities") or []
            if ents_raw and isinstance(ents_raw[0], dict):
                entities = [e.get("text", "") for e in ents_raw if e.get("text")]
            else:
                # service may return a list[str]
                entities = [str(e) for e in ents_raw]

            vpm_payload = None
            if self.include_vpm and hasattr(self.kg, "export_for_vpm"):
                try:
                    vpm_payload = self.kg.export_for_vpm(tree)
                except Exception:
                    vpm_payload = None

            bundle = {
                "query": query,
                "paper_id": paper_id,
                "knowledge_tree": tree,
                "highlights": {
                    "claims": claims,
                    "insights": insights,
                    "entities": entities[: self.top_k_claims],
                },
                "vpm_payload": vpm_payload,
                "stats": {
                    "build_ms": int((time.time() - t0) * 1000),
                    "counts": {
                        "claims": len(tree.get("claims") or []),
                        "insights": len(tree.get("insights") or []),
                        "entities": len(tree.get("entities") or []),
                        "relationships": len(tree.get("relationships") or []),
                    },
                },
            }
            self.report({"event": "KRComplete", "paper_id": paper_id, **bundle["stats"]["counts"]})
            return bundle

        # Fallback (service missing): a tiny heuristic bundle
        self.report({"event": "KRServiceMissing", "paper_id": paper_id})
        claims = _heuristic_claims(context, k=self.top_k_claims)
        entities = _heuristic_entities(context)
        bundle = {
            "query": query,
            "paper_id": paper_id,
            "knowledge_tree": {
                "paper_id": paper_id,
                "section_name": "Full Paper",
                "claims": [{"id": f"c{i+1}", "text": c, "confidence": 0.6} for i, c in enumerate(claims)],
                "insights": [],
                "entities": [{"id": f"e{i+1}", "text": e, "type": "TERM"} for i, e in enumerate(entities)],
                "relationships": [],
                "claim_coverage": 0.0,
                "evidence_strength": 0.0,
            },
            "highlights": {"claims": claims, "insights": [], "entities": entities},
            "vpm_payload": None,
            "stats": {"build_ms": int((time.time() - t0) * 1000), "counts": {"claims": len(claims), "insights": 0, "entities": len(entities), "relationships": 0}},
        }
        return bundle

    # ---------- optional: health probe ----------
    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "ok" if (self.kg and hasattr(self.kg, "build_tree")) else "degraded",
            "has_service": bool(self.kg),
        }


# ---------- small helpers (no external deps) ----------

def _stable_id(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "kr:empty"
    import hashlib
    return "kr:" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def _heuristic_claims(text: str, k: int = 8) -> List[str]:
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    picks: List[str] = []
    for s in sents:
        sl = s.lower()
        if any(w in sl for w in ("we ", "our ", "this paper", "results", "achiev", "improv", "conclud", "show", "demonstrat")):
            picks.append(s.strip())
        if len(picks) >= k:
            break
    return picks or [x.strip() for x in sents[:k]]

def _heuristic_entities(text: str) -> List[str]:
    # crude: capitalized tokens & acronyms
    if not text:
        return []
    toks = re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b", text)
    uniq = []
    seen = set()
    for t in toks:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq[:12]
``n

## File: knowledge_summarizer_agent.py

`python
# stephanie/agents/knowledge_summarizer_agent.py
import re
import uuid
from datetime import datetime
from stephanie.models.plan_trace import ExecutionStep, PlanTrace
from stephanie.agents.base_agent import BaseAgent

class KnowledgeSummarizerAgent(BaseAgent):
    """
    4th Agent: Summarization using knowledge retrieval + CARE/UR² logic.
    """

    def __init__(self, llm, knowledge_store, scorer=None, logger=None):
        super().__init__(logger=logger)
        self.llm = llm
        self.knowledge_store = knowledge_store  # e.g., CaseBooks/Knowledge Graph
        self.scorer = scorer

    async def run(self, goal: str, paper_section: str, meta=None) -> PlanTrace:
        trace_id = str(uuid.uuid4())
        steps = []

        # Step 1: Difficulty-aware trigger
        if self._needs_retrieval(paper_section):
            retrievals = self.knowledge_store.search(paper_section, top_k=5)
        else:
            retrievals = []

        # Step 2: Construct prompt (CARE: enforce retrieval tags)
        retrieval_context = "\n".join(r["text"] for r in retrievals)
        system_prompt = (
            "Summarize the paper section with reasoning inside <think> tags. "
            "If using prior knowledge, wrap retrieved spans inside <retrieval> tags. "
            "Use evidence from the context below when necessary."
        )
        prompt = f"Goal: {goal}\n\nSection:\n{paper_section}\n\nKnowledge:\n{retrieval_context}\n\n"

        response = await self.llm.ainvoke(system_prompt + prompt)

        reasoning = self._extract_tagged(response, "think")
        answer = self._extract_answer(response)
        used_retrievals = self._extract_all_tags(reasoning, "retrieval")

        step = ExecutionStep(
            id=str(uuid.uuid4()),
            trace_id=trace_id,
            goal=goal,
            reasoning=reasoning,
            answer=answer,
            retrievals=used_retrievals,
            created_at=datetime.now(),
            meta={"raw": response, "knowledge_refs": retrievals},
        )
        steps.append(step)

        trace = PlanTrace(id=trace_id, goal=goal, steps=steps, created_at=datetime.now(), meta=meta or {})

        if self.scorer:
            trace = await self.scorer.score_trace(trace, context=paper_section)

        return trace

    def _needs_retrieval(self, section: str) -> bool:
        return len(section.split()) > 150  # toy heuristic: long sections = hard → retrieve

    def _extract_tagged(self, text: str, tag: str) -> str:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    def _extract_all_tags(self, text: str, tag: str):
        return re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)

    def _extract_answer(self, text: str):
        m = re.search(r"Answer:\s*(.*)", text, re.DOTALL)
        return m.group(1).strip() if m else ""
``n

## File: knowledge_tree_builder.py

`python
# stephanie/agents/knowledge/knowledge_tree_builder.py
"""
KnowledgeTreeBuilderAgent (Integrated)
----------------------------
Builds a verifiable knowledge tree for paper sections from:
- Paper claims with entity grounding
- Critical conversation messages with confidence scores
- Entity-based connections with calibrated similarity
- Domain-aware relationship types
- Trajectory evidence from conversation paths

This integrated implementation:
- Uses your full NER system from KnowledgeFusionAgent
- Incorporates calibrated similarity scores
- Adds domain-aware relationship types
- Includes trajectory evidence from conversation paths
- Structures the tree for optimal verification
- Supports distributed verification via NATS/KV

Input context:
  - paper_section: { section_name, section_text, paper_id, domain? }
  - critical_messages: [ { text, score, reason, ... } ]
  - conversation_trajectories: [ { start_idx, end_idx, messages, score, goal_achieved } ]
  - domains: [ { domain: str, score: float } ]  # From KnowledgeFusion
  - fusion_entities: { surface_entities, expanded_entities }  # Optional from KnowledgeFusion

Output context:
  - knowledge_tree: {
        root: str,
        paper_id: str,
        section_name: str,
        section_text: str,
        domains: [ { domain: str, score: float } ],
        claims: [ { id, text, source, confidence, entities: [entity_ids] } ],
        insights: [ { id, text, confidence, timestamp, entities: [entity_ids], trajectory_id? } ],
        entities: [ { id, text, type, source, calibrated_similarity } ],
        relationships: [ 
          { 
            id: str,
            from: str,  # claim_id or insight_id
            to: str,    # claim_id or insight_id or entity_id
            type: str,  # "supports", "contradicts", "extends", etc.
            confidence: float,
            evidence: [ { section_span, trajectory_span, strength } ]
          }
        ],
        trajectory_paths: [ { id, messages: [insight_ids], confidence } ]
    }
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.knowledge.casebook_store import CaseBookStore
from stephanie.models.ner_retriever import EntityDetector, NERRetrieverEmbedder

# Constants from the new analysis
_MIN_INSIGHT_SCORE = 0.70
_MAX_CLAIMS = 10
_MIN_CONN_OVERLAP = 0.06

def _sentences(t: str, max_sents: int = 80) -> List[str]:
    """Split text into sentences with optional max limit."""
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t.strip())
    sentences = [p.strip() for p in parts if len(p.strip()) > 2]
    return sentences[:max_sents] if max_sents and len(sentences) > max_sents else sentences

def _extract_claim_sents(text: str, max_claims: int = _MAX_CLAIMS) -> List[str]:
    """Extract claim sentences with improved heuristic."""
    sents = _sentences(text)
    claimish = [
        s for s in sents 
        if len(s) > 40 and re.search(
            r"(show|demonstrat|result|achiev|improv|evidence|increase|decrease|prove|find|conclude)", 
            s, 
            re.I
        )
    ]
    return (claimish or sents)[:max_claims]

def _norm_token_set(t: str) -> Set[str]:
    """Normalize text to token set for Jaccard similarity."""
    return set(re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", (t or "").lower()))

def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Calculate Jaccard similarity between two token sets."""
    if not a or not b: 
        return 0.0
    i = len(a & b)
    u = len(a | b)
    return i / max(1, u)

def _soft_dedup(items: List[Dict[str, Any]], key="text", thr=0.8) -> List[Dict[str, Any]]:
    """Soft deduplication based on Jaccard similarity."""
    out: List[Dict[str, Any]] = []
    for it in items:
        tw = _norm_token_set(it.get(key, ""))
        if any(_jaccard(tw, _norm_token_set(o.get(key,""))) >= thr for o in out):
            continue
        out.append(it)
    return out

@dataclass
class KTConfig:
    """Configuration for KnowledgeTreeBuilderAgent."""
    # Entity extraction
    ner_k: int = 12
    ner_min_sim: float = 0.60
    ner_min_calibrated_sim: float = 0.45
    
    # Claim extraction
    min_claim_length: int = 40
    claim_keywords: List[str] = None
    max_claims: int = _MAX_CLAIMS
    
    # Relationship building
    entity_overlap_threshold: float = _MIN_CONN_OVERLAP
    relationship_threshold: float = 0.75
    max_hops: int = 3
    
    # Domain awareness
    domain_aware: bool = True
    domain_boost: float = 0.15
    
    # Output
    include_trajectory_paths: bool = True
    include_evidence_spans: bool = True
    max_evidence_spans: int = 3
    max_connections: int = 200
    
    def __post_init__(self):
        if self.claim_keywords is None:
            self.claim_keywords = [
                "show", "demonstrat", "result", "achiev", 
                "improv", "evidence", "prove", "find", "conclude"
            ]


class KnowledgeTreeBuilderAgent(BaseAgent):
    """
    Integrated KnowledgeTreeBuilderAgent that creates a verifiable knowledge tree
    with entity-based connections, calibrated similarity, and domain awareness.
    
    This agent:
    - Uses your existing NER system for precise entity extraction
    - Builds connections based on entity overlap and calibrated similarity
    - Incorporates domain information into relationship types
    - Structures the tree for optimal verification by downstream agents
    - Supports distributed verification via NATS/KV
    
    Designed to run after ConversationFilterAgent and before VerifiedSectionGenerator.
    """

    def __init__(self, cfg: Dict[str, Any], memory: Any, container: Any, logger: logging.Logger):
        super().__init__(cfg, memory, container=container, logger=logger)
        
        # Configuration
        self.kt_cfg = KTConfig(**cfg.get("knowledge_graph", {}))

        # Components
        self.casebooks: CaseBookStore = cfg.get("casebooks") or CaseBookStore()
        self.classifier: Optional[ScorableClassifier] = cfg.get("classifier")
        self.bus = getattr(self.memory, "bus", None)
        
        # Entity extraction components (reuse from KnowledgeFusion)
        self._entity_detector = None
        self._retriever = None
        
        # Stats
        self.stats = {
            "total_sections": 0,
            "total_claims": 0,
            "total_insights": 0,
            "total_entities": 0,
            "total_relationships": 0,
            "domain_aware_connections": 0,
            "processing_time": 0.0
        }
        
        self.logger.info("KnowledgeTreeBuilderAgent initialized", {
            "config": {k: v for k, v in asdict(self.kt_cfg).items() if k != "claim_keywords"},
            "domain_aware": self.kt_cfg.domain_aware,
            "message": "Ready to build knowledge trees"
        })
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Builds a knowledge tree from paper section and critical conversation messages.
        
        Expected context:
          - paper_section: { section_name, section_text, paper_id, domain? }
          - critical_messages: [ { text, score, reason, ... } ]
          - conversation_trajectories: [ { ... } ] (optional)
          - domains: [ { domain: str, score: float } ] (from KnowledgeFusion)
          - fusion_entities: { surface_entities, expanded_entities } (optional from KnowledgeFusion)
          
        Returns context with:
          - knowledge_tree: structured knowledge tree for verification
        """
        start_time = time.time()
        
        paper_section = context.get("paper_section")
        critical_messages = context.get("critical_messages", [])
        conversation_trajectories = context.get("conversation_trajectories", [])
        domains = context.get("domains", [])
        fusion_entities = context.get("fusion_entities", {}) or {}
        
        if not paper_section:
            self.logger.log("KnowledgeTreeBuilderSkipped", {
                "reason": "missing_section",
                "has_section": bool(paper_section),
                "has_messages": bool(critical_messages)
            })
            return context
            
        # Clean and prepare data
        section_text = paper_section.get("section_text", "")
        section_name = paper_section.get("section_name", "Unknown")
        paper_id = paper_section.get("paper_id")
        
        if not section_text or len(section_text.strip()) < 10:
            self.logger.log("KnowledgeTreeBuilderSkipped", {
                "reason": "empty_section",
                "section": section_name
            })
            return context
            
        self.stats["total_sections"] += 1
        
        try:
            # Initialize entity extraction if needed
            if not self._entity_detector or not self._retriever:
                self._init_entity_extraction()
            
            # Build the knowledge tree
            knowledge_graph = self._build_knowledge_graph(
                paper_section, 
                critical_messages,
                conversation_trajectories,
                domains,
                fusion_entities
            )
            
            # Update context
            context["knowledge_graph"] = knowledge_graph
            
            # Log results
            processing_time = time.time() - start_time
            self.stats["processing_time"] = processing_time
            
            self.logger.log("KnowledgeTreeBuilt", {
                "section": section_name,
                "paper_id": paper_id,
                "claims": len(knowledge_graph["claims"]),
                "insights": len(knowledge_graph["insights"]),
                "entities": len(knowledge_graph["entities"]),
                "relationships": len(knowledge_graph["relationships"]),
                "processing_time": f"{processing_time:.2f}s"
            })
            
            # Optional: Publish tree to bus for verification
            if self.bus and context.get("publish_tree", True):
                await self._publish_tree(knowledge_graph)
                
            return context
            
        except Exception as e:
            self.logger.log("KnowledgeTreeBuildError", {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "section": section_name,
                "paper_id": paper_id
            })
            context["tree_build_error"] = str(e)
            return context
    
    def _init_entity_extraction(self):
        """Initialize entity extraction components if not already set up."""
        try:
            # Reuse the same components as KnowledgeFusionAgent
            self._entity_detector = EntityDetector(
                device=self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            )
            self._retriever = NERRetrieverEmbedder(
                model_name=self.cfg.get("ner_model", "dslim/bert-base-NER"),
                device=self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            )
            self.logger.info("Entity extraction components initialized")
        except Exception as e:
            self.logger.warning("EntityExtractionInitFailed", {
                "error": str(e),
                "message": "Falling back to heuristic entity extraction"
            })
            # Will use heuristic fallback in _extract_entities
    
    def _build_knowledge_graph(self,
                             paper_section: Dict[str, Any],
                             critical_messages: List[Dict[str, Any]],
                             conversation_trajectories: List[Dict[str, Any]],
                             domains: List[Dict[str, float]],
                             fusion_entities: Dict[str, Any]) -> Dict[str, Any]:
        """Build the complete knowledge tree structure."""
        section_text = paper_section["section_text"]
        
        # 1. Extract claims from paper section (using improved heuristic)
        claims = self._extract_claims(section_text)
        
        # 2. Process critical messages into insights (with soft deduplication)
        insights = self._process_insights(critical_messages, conversation_trajectories)
        
        # 3. Use fusion entities if available, otherwise extract our own
        paper_entities, insight_entities = self._get_entities(
            section_text, 
            insights, 
            fusion_entities
        )
        
        # 4. Build combined entity set (deduplicated)
        all_entities = self._merge_entities(paper_entities, insight_entities)
        
        # 5. Build relationships between claims, insights, and entities
        relationships = self._build_relationships(
            claims, 
            insights, 
            all_entities, 
            domains,
            section_text
        )
        
        # 6. Build trajectory paths if enabled
        trajectory_paths = []
        if self.kt_cfg.include_trajectory_paths and conversation_trajectories:
            trajectory_paths = self._build_trajectory_paths(
                conversation_trajectories, 
                insights
            )
        
        # 7. Structure the final tree
        return {
            "root": paper_section.get("section_name", "section"),
            "paper_id": paper_section.get("paper_id"),
            "section_name": paper_section.get("section_name", "Unknown"),
            "section_text": section_text[:500] + "..." 
                if len(section_text) > 500 
                else section_text,
            "domains": domains,
            "claims": claims,
            "insights": insights,
            "entities": all_entities,
            "relationships": relationships,
            "trajectory_paths": trajectory_paths,
            "metadata": {
                "build_time": datetime.now(timezone.utc).isoformat(),
                "domain_aware": self.kt_cfg.domain_aware,
                "relationship_threshold": self.kt_cfg.relationship_threshold
            }
        }
    

    def _merge_entities(
        self, 
        paper_entities: List[Dict[str, Any]], 
        insight_entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate paper + insight entities into one unified set.
        Deduplication is done by normalized text match (case/punct stripped).
        """
        all_entities: List[Dict[str, Any]] = []
        seen: Dict[str, str] = {}  # normalized_text -> entity_id

        for ent in paper_entities + insight_entities:
            norm_text = self._normalize_entity(ent.get("text", ""))
            if not norm_text:
                continue

            if norm_text in seen:
                # merge meta into the already-existing entity
                existing = next((e for e in all_entities if e["id"] == seen[norm_text]), None)
                if existing:
                    # optional: merge sources or add provenance
                    existing_sources = set(existing.get("source", "").split(","))
                    new_source = ent.get("source", "")
                    if new_source and new_source not in existing_sources:
                        existing["source"] = ",".join(existing_sources | {new_source})

                    # merge calibrated similarity (take max)
                    existing["calibrated_similarity"] = max(
                        existing.get("calibrated_similarity", 0.0),
                        ent.get("calibrated_similarity", 0.0)
                    )

                    # merge embeddings if both exist (average them)
                    if existing.get("embedding") is not None and ent.get("embedding") is not None:
                        try:
                            import numpy as np
                            e1 = np.array(existing["embedding"])
                            e2 = np.array(ent["embedding"])
                            existing["embedding"] = ((e1 + e2) / 2.0).tolist()
                        except Exception:
                            pass
            else:
                # new entity
                eid = ent.get("id") or f"ent_{uuid.uuid4().hex[:8]}"
                ent["id"] = eid
                all_entities.append(ent)
                seen[norm_text] = eid

        self.stats["total_entities"] += len(all_entities)
        return all_entities

    def _get_entities(self,
                     section_text: str,
                     insights: List[Dict[str, Any]],
                     fusion_entities: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Get entities from fusion entities if available, otherwise extract."""
        paper_entities = []
        insight_entities = []
        
        # Try to use fusion entities first
        if fusion_entities:
            surface_entities = fusion_entities.get("surface_entities", [])
            expanded_entities = fusion_entities.get("expanded_entities", [])
            
            # Convert surface entities to our format
            for i, ent in enumerate(surface_entities):
                paper_entities.append({
                    "id": f"ent_{i}_{uuid.uuid4().hex[:4]}",
                    "text": ent["text"],
                    "type": ent["type"],
                    "start": ent.get("start", 0),
                    "end": ent.get("end", 0),
                    "source": "paper",
                    "similarity": ent.get("similarity", 0.6),
                    "calibrated_similarity": ent.get("calibrated_similarity", 0.6)
                })
            
            # Convert expanded entities to our format
            for i, ent in enumerate(expanded_entities):
                insight_entities.append({
                    "id": f"ent_exp_{i}_{uuid.uuid4().hex[:4]}",
                    "text": ent["text"],
                    "type": ent["type"],
                    "start": ent.get("start", 0),
                    "end": ent.get("end", 0),
                    "source": "insight",
                    "similarity": ent.get("similarity", 0.6),
                    "calibrated_similarity": ent.get("calibrated_similarity", 0.6)
                })
                
            return paper_entities, insight_entities
        
        # Fall back to our own extraction
        paper_entities = self._extract_entities(section_text)
        
        for insight in insights:
            entities = self._extract_entities(insight["text"])
            for entity in entities:
                # Add insight reference to entity
                entity["source_insights"] = entity.get("source_insights", []) + [insight["id"]]
            insight_entities.extend(entities)
            
        return paper_entities, insight_entities
    
    def _extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract claims from section text with entity grounding."""
        # Use improved claim extraction from the new analysis
        claim_texts = _extract_claim_sents(text)
        self.stats["total_claims"] += len(claim_texts)
        
        claims = []
        for i, claim_text in enumerate(claim_texts):
            claims.append({
                "id": f"claim_{i+1}",
                "text": claim_text,
                "source": "paper",
                "confidence": 1.0,
                "position": i / max(1, len(claim_texts))
            })
        
        # Add entity grounding to claims
        for claim in claims:
            entities = self._extract_entities(claim["text"])
            claim["entities"] = [e["id"] for e in entities]
            
        return claims
    
    def _process_insights(self, 
                         critical_messages: List[Dict[str, Any]],
                         conversation_trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process critical messages into structured insights with soft deduplication."""
        # Convert to our insight format
        raw_insights = []
        trajectory_map = {
            traj["trajectory_id"]: traj 
            for traj in conversation_trajectories
        } if conversation_trajectories else {}
        
        for i, msg in enumerate(critical_messages):
            # Only include high-score insights
            if float(msg["score"]) < _MIN_INSIGHT_SCORE:
                continue
                
            # Create insight structure
            insight = {
                "id": f"ins_{uuid.uuid4().hex[:8]}",
                "text": msg["text"],
                "confidence": float(msg["score"]),
                "timestamp": msg.get("timestamp"),
                "vpm_dims": msg.get("vpm_dims", {}),
                "similarity": msg.get("similarity", 0.0),
                "reason": msg.get("reason", "Relevance-based filtering"),
                "source": "conversation"
            }
            
            # Add trajectory information if available
            if msg.get("trajectory_id") and msg["trajectory_id"] in trajectory_map:
                traj = trajectory_map[msg["trajectory_id"]]
                insight["trajectory_id"] = traj["trajectory_id"]
                insight["trajectory_score"] = traj["score"]
                insight["in_trajectory"] = True
            
            raw_insights.append(insight)
        
        # Soft deduplicate insights
        insights = _soft_dedup(raw_insights, key="text", thr=0.8)
        self.stats["total_insights"] += len(insights)
        
        return insights
    
    def _build_relationships(self,
                            claims: List[Dict[str, Any]],
                            insights: List[Dict[str, Any]],
                            entities: List[Dict[str, Any]],
                            domains: List[Dict[str, float]],
                            section_text: str) -> List[Dict[str, Any]]:
        """Build relationships between claims, insights, and entities."""
        relationships = []
        ent_lexicon = {e["text"].lower() for e in entities}
        
        # 1. Claim-Insight relationships (using Jaccard + entity ping)
        for claim in claims:
            cw = _norm_token_set(claim["text"])
            for insight in insights:
                iw = _norm_token_set(insight["text"])
                # lexical connection
                lex = _jaccard(cw, iw)
                # entity ping: any known entity inside insight?
                ent_hit = 1 if any(ent in insight["text"].lower() for ent in ent_lexicon) else 0
                # simple blend
                strength = min(1.0, 0.7*lex + 0.3*ent_hit)
                
                if strength >= self.kt_cfg.entity_overlap_threshold:
                    # Determine relationship type
                    rel_type = self._determine_claim_insight_relationship(claim, insight)
                    
                    relationships.append({
                        "id": f"rel_{uuid.uuid4().hex[:6]}",
                        "from": claim["id"],
                        "to": insight["id"],
                        "type": rel_type,
                        "confidence": strength,
                        "strength": strength,
                        "lex": float(lex),
                        "ent": int(ent_hit),
                        "evidence": self._extract_evidence_spans(claim["text"], insight["text"], section_text)
                    })
        
        # 2. Claim-Entity relationships (if we have entity grounding)
        for claim in claims:
            for entity_id in claim.get("entities", []):
                # Find the entity object
                entity = next((e for e in entities if e["id"] == entity_id), None)
                if not entity:
                    continue
                    
                # Calculate relationship confidence
                confidence = self._calculate_relationship_confidence(
                    claim["text"], 
                    entity["text"], 
                    0,  # Distance is 0 for direct mentions
                    domains
                )
                
                if confidence >= self.kt_cfg.relationship_threshold:
                    relationships.append({
                        "id": f"rel_{uuid.uuid4().hex[:6]}",
                        "from": claim["id"],
                        "to": entity["id"],
                        "type": "mentions",
                        "confidence": confidence,
                        "evidence": self._extract_evidence_spans(claim["text"], entity["text"], section_text)
                    })
        
        # 3. Insight-Entity relationships
        for insight in insights:
            # Extract entities from insight text
            insight_entities = self._extract_entities(insight["text"])
            for entity in insight_entities:
                # Find matching entity in our entity list
                norm_text = self._normalize_entity(entity["text"])
                matched_entity = next(
                    (e for e in entities if self._normalize_entity(e["text"]) == norm_text), 
                    None
                )
                
                if not matched_entity:
                    continue
                    
                # Calculate relationship confidence
                confidence = self._calculate_relationship_confidence(
                    insight["text"],
                    matched_entity["text"],
                    0,
                    domains
                )
                
                if confidence >= self.kt_cfg.relationship_threshold:
                    relationships.append({
                        "id": f"rel_{uuid.uuid4().hex[:6]}",
                        "from": insight["id"],
                        "to": matched_entity["id"],
                        "type": "references",
                        "confidence": confidence,
                        "evidence": self._extract_evidence_spans(insight["text"], matched_entity["text"], section_text)
                    })
        
        # Sort by strength and limit connections
        relationships.sort(key=lambda x: (-x.get("strength", 0), x["from"]))
        return relationships[:self.kt_cfg.max_connections]
    
    def _calculate_relationship_confidence(self,
                                         text1: str,
                                         text2: str,
                                         distance: int,
                                         domains: List[Dict[str, float]]) -> float:
        """Calculate confidence score for a relationship."""
        # Base score based on lexical overlap
        base_score = self._lexical_overlap(text1, text2)
        
        # Distance penalty
        distance_penalty = min(1.0, distance / 100)
        base_score = base_score * (1 - distance_penalty * 0.3)
        
        # Domain boost if relevant domains are present
        domain_boost = 0.0
        if self.kt_cfg.domain_aware:
            domain_names = {d["domain"] for d in domains}
            # ML/NLP domains get higher boost
            if any(d in domain_names for d in {"ml", "nlp", "machine learning", "natural language processing"}):
                domain_boost = self.kt_cfg.domain_boost
                
        # Proximity bonus for close entities
        proximity_bonus = 0.1 if distance < 20 else 0.0
        
        # Final confidence
        confidence = min(1.0, base_score + domain_boost + proximity_bonus)
        return max(0.0, confidence)
    
    def _determine_claim_insight_relationship(self, claim: Dict[str, Any], insight: Dict[str, Any]) -> str:
        """Determine the type of relationship between a claim and insight."""
        insight_text = insight["text"].lower()
        
        # Check for supporting language
        if any(kw in insight_text for kw in ["supports", "confirms", "validates", "evidence for"]):
            return "supports"
        if any(kw in insight_text for kw in ["extends", "builds on", "enhances"]):
            return "extends"
        if any(kw in insight_text for kw in ["contradicts", "challenges", "disproves"]):
            return "contradicts"
        if any(kw in insight_text for kw in ["clarifies", "explains", "elaborates"]):
            return "clarifies"
            
        # Default relationship type based on confidence
        if insight["confidence"] > 0.8:
            return "supports"
        elif insight["confidence"] < 0.5:
            return "questionable"
        else:
            return "relates_to"
    
    def _build_trajectory_paths(self,
                               trajectories: List[Dict[str, Any]],
                               insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build structured trajectory paths from conversation trajectories."""
        paths = []
        
        # Map insight IDs for quick lookup
        insight_map = {ins["id"]: ins for ins in insights}
        
        for i, traj in enumerate(trajectories):
            # Get insight IDs in this trajectory
            trajectory_insights = []
            for msg in traj["messages"]:
                # Find matching insight
                insight = next(
                    (ins for ins in insights if ins["text"] == msg["text"]), 
                    None
                )
                if insight:
                    trajectory_insights.append(insight["id"])
            
            if not trajectory_insights:
                continue
                
            # Calculate trajectory confidence (average of insight confidences)
            confidences = [
                insight_map[ins_id]["confidence"] 
                for ins_id in trajectory_insights 
                if ins_id in insight_map
            ]
            trajectory_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
            paths.append({
                "id": f"path_{i+1}",
                "trajectory_id": traj.get("trajectory_id", f"traj_{i+1}"),
                "insight_ids": trajectory_insights,
                "confidence": trajectory_confidence,
                "length": len(trajectory_insights),
                "goal_achieved": traj.get("goal_achieved", False),
                "start_idx": traj.get("start_idx", 0),
                "end_idx": traj.get("end_idx", 0)
            })
            
        return paths
    

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text using the configured detector/retriever.
        Falls back to regex-based heuristic if components not initialized.
        """
        entities: List[Dict[str, Any]] = []
        if not text or not text.strip():
            return entities

        try:
            # Ensure detector/retriever initialized
            if not self._entity_detector or not self._retriever:
                self._init_entity_extraction()

            # Detect entities
            dets = self._entity_detector.detect_entities(text)
            for i, d in enumerate(dets):
                span = (d.get("start", 0), d.get("end", len(d["text"])))

                # Embed returns a tensor (vector), not a dict
                emb = self._retriever.embed_entity(d["text"], span)
                if hasattr(emb, "detach"):  # torch.Tensor
                    emb = emb.detach().cpu().numpy()

                # Simple calibration heuristic: normalize length of vector
                calibrated_sim = float(d.get("score", 0.0))
                if emb is not None and len(emb) > 0:
                    norm = float(np.linalg.norm(emb))
                    if norm > 0:
                        calibrated_sim = min(1.0, calibrated_sim + (1.0 / norm))

                entities.append({
                    "id": f"ent_{uuid.uuid4().hex[:8]}",
                    "text": d["text"],
                    "type": d.get("label", "UNK"),
                    "start": d.get("start", -1),
                    "end": d.get("end", -1),
                    "source": "auto",
                    "similarity": float(d.get("score", 0.0)),
                    "calibrated_similarity": calibrated_sim,
                    "embedding": emb.tolist() if emb is not None else None
                })
        except Exception as e:
            # fallback: regex heuristic for proper nouns/acronyms
            self.logger.warning("EntityExtractionFallback", {"error": str(e)})
            matches = re.findall(r"\b([A-Z][A-Za-z0-9\-]{2,})\b", text)
            for m in set(matches):
                entities.append({
                    "id": f"ent_{uuid.uuid4().hex[:8]}",
                    "text": m,
                    "type": "Heuristic",
                    "source": "fallback",
                    "similarity": 0.5,
                    "calibrated_similarity": 0.5
                })

        return entities

    def _extract_evidence_spans(self, 
                              text1: str, 
                              text2: str, 
                              full_text: str) -> List[Dict[str, Any]]:
        """Extract evidence spans showing connection between two texts."""
        if not self.kt_cfg.include_evidence_spans:
            return []
            
        evidence = []
        
        # Find where text2 appears in full_text
        text2_pos = full_text.find(text2)
        if text2_pos >= 0:
            # Get surrounding context
            start = max(0, text2_pos - 50)
            end = min(len(full_text), text2_pos + len(text2) + 50)
            section_span = full_text[start:end]
            
            # Find similar span in text1
            best_match = self._find_best_match_span(text1, section_span)
            if best_match and best_match["strength"] > 0.3:
                evidence.append({
                    "section_span": section_span,
                    "trajectory_span": best_match["span"],
                    "strength": best_match["strength"]
                })
                
        return evidence[:self.kt_cfg.max_evidence_spans]
    
    def _find_best_match_span(self, source: str, target: str) -> Dict[str, Any]:
        """Find the best matching span in source for the target text."""
        source_sents = _sentences(source)
        best_score = 0.0
        best_span = ""
        
        for sent in source_sents:
            score = _jaccard(_norm_token_set(sent), _norm_token_set(target))
            if score > best_score:
                best_score = score
                best_span = sent
                
        return {
            "span": best_span,
            "strength": best_score
        }
    
    def _lexical_overlap(self, a: str, b: str) -> float:
        """Calculate lexical overlap between two texts."""
        if not a or not b:
            return 0.0
            
        a_words = set(re.findall(r"\b\w+\b", a.lower()))
        b_words = set(re.findall(r"\b\w+\b", b.lower()))
        
        if not a_words:
            return 0.0
            
        return len(a_words & b_words) / len(a_words)
    
    def _sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if len(p.strip()) > 2]
    
    def _normalize_entity(self, text: str) -> str:
        """Normalize entity text for deduplication."""
        # Convert to lowercase
        text = text.lower()
        # Remove articles and common stopwords
        text = re.sub(r"\b(the|a|an|this|that|these|those)\b", "", text)
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    async def _publish_tree(self, tree: Dict[str, Any]):
        """Publish knowledge tree to the bus for verification."""
        try:
            # Create a job ID based on tree content
            tree_hash = hashlib.sha256(json.dumps(tree).encode("utf-8")).hexdigest()[:16]
            
            # Create a verification job
            job = {
                "job_id": tree_hash,
                "tree": tree,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Store the full job in KV (to avoid MaxPayloadError)
            if hasattr(self.bus, "get_kv"):
                try:
                    kv = self.bus.get_kv(
                        bucket="vsec.jobs", 
                        description="Verification jobs",
                        max_age_seconds=3600
                    )
                    kv.put(tree_hash, json.dumps(job).encode("utf-8"))
                except Exception as e:
                    self.logger.warning("VerificationKVStoreError", {
                        "error": str(e),
                        "job_id": tree_hash
                    })
            
            # Publish a small job envelope
            envelope = {
                "job_id": tree_hash,
                "section": tree["section_name"],
                "paper_id": tree["paper_id"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await self.bus.publish("verify.section", envelope)
            
            self.logger.log("VerificationJobEnqueued", {
                "job_id": tree_hash,
                "section": tree["section_name"],
                "paper_id": tree["paper_id"]
            })
        except Exception as e:
            self.logger.warning("TreePublishFailed", {
                "error": str(e),
                "section": tree.get("section_name")
            })
    
    def health_check(self) -> Dict[str, Any]:
        """Return health status and metrics for the tree builder agent."""
        return {
            "status": "healthy",
            "stats": self.stats,
            "config": {k: v for k, v in asdict(self.kt_cfg).items() if k != "claim_keywords"},
            "message": "Knowledge tree builder operational"
        }
``n

## File: learning_evidence.py

`python
# stephanie/agents/meta/learning_evidence_agent.py
from __future__ import annotations
import os, json, glob
from typing import Dict, Any, List
from stephanie.agents.base_agent import BaseAgent
from stephanie.models.learning_evidence import LearningEvidenceORM

class LearningEvidenceAgent(BaseAgent):
    """
    Meta-agent that proves 'learning from learning':
      • collects Track A/B/C metrics
      • parses strategy evolution events
      • writes LearningEvidenceORM + CSV/MD
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.report_dir = str(cfg.get("report_dir", "reports/learning_evidence"))
        os.makedirs(self.report_dir, exist_ok=True)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        docs = context.get("documents", [])
        evidence_rows: List[Dict[str, Any]] = []

        for doc in docs:
            doc_id = str(doc.get("id") or doc.get("paper_id"))

            # 1) Collect Track A/B/C scores
            summary_v2 = (context.get("summary_v2") or {}).get(doc_id, {})
            metrics_c = summary_v2.get("metrics", {})
            guardrails = summary_v2.get("guardrail_details", {})

            # 2) Extract strategy evolution logs (already persisted in context/logs)
            strategy_events = context.get("strategy_events", {}).get(doc_id, [])

            for ev in strategy_events:
                row = {
                    "doc_id": doc_id,
                    "strategy_version": ev.get("new_version"),
                    "old_threshold": ev.get("old_thr"),
                    "new_threshold": ev.get("new_thr"),
                    "old_weights": ev.get("old_weights"),
                    "new_weights": ev.get("new_weights"),
                    "avg_gain": ev.get("avg_gain"),
                    "metrics": metrics_c,
                    "guardrails": guardrails,
                }
                evidence_rows.append(row)

                # persist to DB
                self.memory.session.add(LearningEvidenceORM(
                    doc_id=row["doc_id"],
                    strategy_version=row["strategy_version"],
                    old_threshold=row["old_threshold"],
                    new_threshold=row["new_threshold"],
                    old_weights=row["old_weights"],
                    new_weights=row["new_weights"],
                    avg_gain=row["avg_gain"],
                    meta={"metrics": metrics_c, "guardrails": guardrails}
                ))

        self.memory.session.commit()

        # 3) Export CSV for dashboards
        csv_path = os.path.join(self.report_dir, "learning_evidence.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("doc_id,strategy_version,old_thr,new_thr,avg_gain\n")
            for r in evidence_rows:
                f.write(f"{r['doc_id']},{r['strategy_version']},{r['old_threshold']},{r['new_threshold']},{r['avg_gain']}\n")

        # 4) Push evidence summary to context for SIS dashboards
        context.setdefault("learning_evidence", [])
        context["learning_evidence"].extend(evidence_rows)

        return context
``n

## File: literature.py

`python
# stephanie/agents/knowledge/literature.py

import re

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL
from stephanie.tools import WebSearchTool
from stephanie.utils.file_utils import write_text_to_file


class LiteratureAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.strategy = cfg.get("strategy", "query_and_summarize")
        self.preferences = cfg.get("preferences", ["goal_consistency", "novelty"])
        self.max_results = cfg.get("max_results", 5)
        self.web_search_tool = WebSearchTool(cfg.get("web_search", {}), self.logger)

        self.logger.log(
            "LiteratureAgentInit",
            {
                "strategy": self.strategy,
                "preferences": self.preferences,
                "max_results": self.max_results,
            },
        )

    async def run(self, context: dict) -> dict:
        self.logger.log("LiteratureQuery", {"context": context})
        goal = self.extract_goal_text(context.get(GOAL))

        # Step 1: Generate search query using LLM
        search_query = self._generate_search_query(context)
        if not search_query:
            self.logger.log("LiteratureQueryFailed", {"goal": goal})
            return context

        self.logger.log("SearchingWeb", {"query": search_query, "goal": goal})

        # Step 2: Perform web search
        results = await self.web_search_tool.search(
            search_query, max_results=self.max_results
        )

        if not results:
            self.logger.log(
                "NoResultsFromWebSearch",
                {
                    "goal_snippet": goal[:60],
                    "search_query": search_query,
                },
            )
            return context

        self.logger.log("SearchResult", {"results": results})

        # Step 3: Parse each result with LLM
        parsed_results = []
        for result in results:
            summary_context = {
                **{
                    "title": result.get("title", "no Title"),
                    "link": result.get("url", ""),
                    "snippet": result.get("snippet", ""),
                    "page": result.get("page", ""),
                },
                **context,
            }

            summary = self._summarize_result(summary_context)

            if summary.strip():
                parsed_results.append(f"""
                    [Title: {result["title"]}]({result["url"]})\n
                    Summary: {summary}
                """)

        self.logger.log(
            "LiteratureSearchCompleted",
            {
                "total_results": len(parsed_results),
                "goal": goal,
                "search_query": search_query,
            },
        )

        context["literature"] = parsed_results

        return context

    def _generate_search_query(self, context: dict) -> str:
        try:
            prompt = self.prompt_loader.load_prompt(self.cfg, context)
            self.logger.log(
                "LLMPromptGenerated_SearchQuery", {"prompt_snippet": prompt[:200]}
            )

            response = self.call_llm(prompt, context)
            self.logger.log(
                "LLMResponseReceived_SearchQuery", {"response_snippet": response[:200]}
            )

            # Structured format
            match = re.search(r"search query:<([^>]+)>", response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

            # Fallback format
            match = re.search(
                r"(?:query|search)[:\s]+\"([^\"]+)\"", response, re.IGNORECASE
            )
            if match:
                query = match.group(1).strip()
                self.logger.log("SearchQuery", {"Search Query": query})
                return query

            # Fallback to goal
            goal = self.extract_goal_text(context.get(GOAL))
            self.logger.log("FallingBackToGoalAsQuery", {"goal": goal})
            return f"{goal} productivity study"

        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            self.logger.log("LiteratureQueryGenerationFailed", {"error": str(e)})
            return f"{context.get('goal', '')} remote work meta-analysis"

    def _summarize_result(self, context: dict) -> str:
        try:
            prompt = self.prompt_loader.from_file(
                self.cfg.get("parse_prompt", "parse.txt"), self.cfg, context
            )
            self.logger.log(
                "LLMPromptGenerated_Summarize",
                {"title": context.get("title", ""), "prompt_snippet": prompt[:200]},
            )

            raw_summary = self.call_llm(prompt, context)
            self.logger.log(
                "LLMResponseReceived_Summarize",
                {
                    "title": context.get("title", ""),
                    "response_snippet": raw_summary[:200],
                },
            )

            # Try extracting "Summary" section
            summary_match = re.search(
                r"Summary\s*\n(?:.*\n)*?\s*(.+?)(?=\n#|\Z)",
                raw_summary,
                re.DOTALL | re.IGNORECASE,
            )
            if summary_match:
                return summary_match.group(1).strip()

            # Fallback: first paragraph of sufficient length
            lines = raw_summary.splitlines()
            for line in lines:
                if len(line.strip()) > 50:
                    return line.strip()

            return ""

        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            self.logger.log("FailedToParseLiterature", {"error": str(e)})
            return ""
``n

## File: paper_score.py

`python
# stephanie/agents/knowledge/paper_score.py

import logging
import time
from typing import Dict

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_corpus import ScoreCorpus
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType

_logger = logging.getLogger(__name__)

class PaperScoreAgent(BaseAgent):
    """
    Scores academic papers (e.g. from Arxiv) across multiple scorers.
    Similar design to DocumentRewardScorer, but specialized for research papers.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.dimensions = cfg.get(
            "dimensions",
            ["novelty", "clarity", "relevance", "implementability", "alignment"],
        )
        self.include_mars = cfg.get("include_mars", True)
        self.enabled_scorers = cfg.get(
            "enabled_scorers",
            ["svm", "mrq", "sicql", "ebt", "hrm", "contrastive_ranker"],
        )

        # Initialize MARS calculator
        self.enabled_scorers = cfg.get("enabled_scorers", [])

        _logger.debug(
            "PaperScoreAgentInitialized:"
            f"dimensions={self.dimensions}, "
            f"scorers={self.enabled_scorers}, "
            f"include_mars={self.include_mars}"
        )

    async def run(self, context: Dict) -> Dict:
        """Score all papers in the context"""
        start_time = time.time()
        documents = context.get(self.input_key, [])

        if not documents:
            self.logger.log("NoPapersFound", {"source": self.input_key})
            return context

        self.report({"event": "start", "step": "PaperScoring", "details": f"{len(documents)} papers"})

        all_bundles = {}
        results = []

        pbar = tqdm(documents, desc="Scoring Papers", total=len(documents), disable=not self.cfg.get("progress", True))

        for idx, doc in enumerate(pbar):
            try:
                doc_scores, bundle = self._score_paper(context, doc)
                results.append(doc_scores)
                all_bundles[doc["id"]] = bundle
            except Exception as e:
                self.logger.log("PaperScoringError", {"doc_id": doc.get("id"), "error": str(e)})
                continue

        # Run MARS analysis
        if self.include_mars and all_bundles:
            corpus = ScoreCorpus(bundles=all_bundles)
            self.logger.log("ScoreCorpusSummary", {
                "dims": corpus.dimensions,
                "scorers": corpus.scorers,
                "shape_example": corpus.get_dimension_matrix(self.dimensions[0]).shape
            })

            mars_results = self.mars_calculator.calculate(corpus, context=context)
            context["mars_analysis"] = {
                "summary": mars_results,
                "recommendations": self.mars_calculator.generate_recommendations(mars_results),
            }

        context[self.output_key] = results
        context["scoring_time"] = time.time() - start_time
        context["total_documents"] = len(documents)

        self.report({"event": "end", "step": "PaperScoring", "details": f"Scored {len(documents)} papers"})
        return context

    def _score_paper(self, context: dict, doc: dict) -> tuple:
        """Score one paper with all scorers"""
        doc_id = doc["id"]
        goal = context.get("goal", {"goal_text": ""})
        scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)

        score_results = {}

        for scorer_name in self.enabled_scorers:
            try:
                bundle = self.container.get("scoring").score(
                    scorer_name,
                    context=context,
                    scorable=scorable,
                    dimensions=self.dimensions
                )
                for dim, result in bundle.results.items():
                    # ensure the result carries its dimension and source
                    if not getattr(result, "dimension", None):
                        result.dimension = dim
                    if not getattr(result, "source", None):
                        result.source = scorer_name  # fallback if scorer didn't set it

                    # use a composite key to avoid overwriting, but keep result.dimension == dim
                    key = f"{dim}::{result.source}"
                    score_results[key] = result
            except Exception as e:
                self.logger.log("ScorerError", {"scorer": scorer_name, "doc_id": doc_id, "error": str(e)})
                continue

        bundle = ScoreBundle(results=dict(score_results))

        eval_id = self.memory.evaluations.save_bundle(
            bundle=bundle,
            scorable=scorable,
            context=context,
            cfg=self.cfg,
            source="paper_score",
            model_name="ensemble",
            evaluator_name=str(self.enabled_scorers)
        )
        self.logger.log("EvaluationSaved", {"id": eval_id})


        report_scores = {
            dim: {"score": result.score, "rationale": result.rationale, "source": result.source}
            for dim, result in score_results.items()
        }

        return {
            "document_id": doc_id,
            "title": doc.get("title", ""),
            "scores": report_scores,
            "goal_text": goal.get("goal_text", ""),
        }, bundle
``n

## File: planner_reuse.py

`python
# stephanie/agents/planning/planner_reuse.py
import re

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import PLAN_TRACE_ID
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.scoring.scorer.scorable_ranker import ScorableRanker


class PlannerReuseAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.ranker = ScorableRanker(cfg, memory, container, logger)
        self.top_k = cfg.get("top_k", 100)
        self.min_hrm = cfg.get("min_hrm", 0.6)
        self.use_db_knn = cfg.get("use_db_knn", True)
        self.rerank_with_scorable_ranker = cfg.get(
            "rerank_with_scorable_ranker", False
        )
        self.dimensions = cfg.get("dimensions", ["alignment"])
        self.hrm_scorer = "hrm"
        self.hrm_by_id = {}
        self.knn_by_id = {}

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text", "")

        # --- 1. Retrieve candidate past traces ---
        candidates = []

        filtered_traces = []

        # get ids of matching traces
        related_scorables = self.memory.embedding.search_related_scorables(
            goal_text, TargetType.PLAN_TRACE, self.top_k
        )
        for scorable in related_scorables:
            pt = self.memory.plan_traces.get_by_trace_id(scorable.get("id"))
            if not pt:
                continue
            self.knn_by_id[str(pt.trace_id)] = float(
                scorable.get("score", 0.0)
            )  # keep KNN sim Wait stop this this is
            to_score = ScorableFactory.from_plan_trace(pt, goal_text=goal_text)
            bundle = self.container.get("scoring").score(self.hrm_scorer, context=context, scorable=to_score, dimensions=self.dimensions)
            score = bundle.aggregate()
            self.logger.log(
                "PlannerReuseHRMScore",
                {"score": score, "trace_id": pt.trace_id},
            )
            hrm_score = bundle.aggregate()
            if hrm_score >= self.min_hrm:
                filtered_traces.append(pt)
                self.hrm_by_id[str(pt.trace_id)] = float(hrm_score)
            else:
                self.logger.log(
                    "PlannerReuseFilteredTrace",
                    {"trace_id": pt.trace_id, "hrm_score": hrm_score},
                )

        # filter list based upon hrm score

        pbar = tqdm(
            filtered_traces,
            desc="Embedding Candidates",
            disable=not self.cfg.get("progress", True),
        )
        for idx, pt in enumerate(pbar, start=1):
            related_scorables = ScorableFactory.from_plan_trace(
                pt, goal_text=goal_text
            )
            embed_id = self.memory.scorable_embeddings.get_or_create(
                related_scorables
            )
            self.logger.log(
                "PlannerReuseCandidate",
                {
                    "scorable_id": related_scorables.id,
                    "embedding_id": embed_id,
                },
            )

            # Build hybrid text: goal + final output (+ step outputs if wanted)
            trace_text_parts = [
                pt.goal.goal_text if pt.goal else "",
                pt.final_output_text,
            ]
            if self.cfg.get("include_steps", False):
                trace_text_parts.extend(
                    [s.output_text for s in pt.execution_steps]
                )

            candidate_text = "\n".join([t for t in trace_text_parts if t])
            candidates.append(
                Scorable(
                    id=pt.trace_id,
                    text=candidate_text,
                    target_type="plan_trace",
                )
            )
            pbar.set_postfix({"candidates": f"{idx}/{len(filtered_traces)}"})

        if not candidates: 
            self.logger.log(
                "PlannerReuseNoCandidates", {"goal_text": goal_text}
            )
            self.report(
                {
                    "event": "planner_reuse",
                    "step": "PlannerReuse",
                    "details": "No past traces available for reuse",
                    "goal_text": goal_text,
                }
            )
            return context

        # --- 2. Rank candidates ---
        query_scorable = Scorable(
            id="current_goal", text=goal_text, target_type="goal"
        )
        ranked = self.ranker.rank(
            query=query_scorable, candidates=candidates, context=context
        )

        # --- 3. Report (convert ORM → dict for SYS)
        ranked_dicts = [ev.to_dict() for ev in ranked]
        self.report(self.ranker.to_report_dict(query_scorable, ranked_dicts))

        top = ranked[: self.top_k]  # take top k

        # --- 2. Gather top examples ---
        examples = []
        pbar = tqdm(
            zip(top, candidates),
            total=len(top),
            desc="Collecting Top Examples",
            disable=not self.cfg.get("progress", True),
        )
        for bundle, cand in pbar:
            # Match bundles back to the original candidate
            # (since rank() processed them in the same order)
            pt = self.memory.plan_traces.get_by_trace_id(cand.id)
            goal_text = self.memory.plan_traces.get_goal_text(cand.id)
            if pt:
                example = {
                    "trace_id": pt.trace_id,
                    "goal": goal_text,
                    "plan": pt.plan_signature,
                    "knn_score": self.knn_by_id.get(str(pt.trace_id)),
                    "hrm": self.hrm_by_id.get(str(pt.trace_id)),
                    "rank_score": bundle.results["rank_score"].score,
                }
                self._save_example(
                    cand, example, bundle, context
                )  # persist evaluation
                examples.append(example)
                pbar.set_postfix({"examples": f"{len(examples)}/{len(top)}"})

        self.report(
            {
                "event": "planner_reuse",
                "step": "PlannerReuse",
                "details": f"Retrieved {len(examples)} past traces for reuse",
                "examples": examples,
                "goal_text": goal_text,
            }
        )

        # --- 3. Adaptation step (LLM) ---
        prompt = self.prompt_loader.load_prompt(self.cfg, context)
        merged = {"examples": examples, **context}
        response = self.call_llm(prompt, context=merged)
        parsed = self._extract_plan_from_response(response)

        # --- 4. Update context ---
        context[self.output_key] = parsed["plan"]
        
        context["examples"] = examples
        context[f"{self.output_key}_meta"] = {
            "rationale": parsed["rationale"],
            "confidence_score": parsed["score"],
        }

        self.logger.log(
            "PlannerReuseGenerated",
            {
                "goal_text": goal_text,
                "plan": parsed["plan"],
                "confidence_score": parsed["score"],
            },
        )

        self.report(
            {
                "event": "planner_reuse",
                "step": "PlannerReuse",
                "details": "New plan adapted from past traces",
                "goal_text": goal_text,
                "plan": parsed["plan"],
                "confidence_score": parsed["score"],
            }
        )

        # --- 5. Record reuse links in DB ---
        try:
            # We need the trace_id of the *new* plan being generated.
            # Assume Supervisor/Monitor has already created a PlanTrace in memory.
            new_trace_id = context.get(PLAN_TRACE_ID)

            if new_trace_id:
                for ex in examples:
                    parent_trace_id = ex.get("trace_id") or None
                    if parent_trace_id:
                        self.memory.plan_traces.add_reuse_link(
                            parent_trace_id=parent_trace_id,
                            child_trace_id=new_trace_id,
                        )
                self.logger.log(
                    "PlannerReuseLinksCreated",
                    {
                        "child": new_trace_id,
                        "parents": [
                            ex.get("trace_id")
                            for ex in examples
                            if ex.get("trace_id")
                        ],
                    },
                )
        except Exception as e:
            self.logger.log("PlannerReuseLinkError", {"error": str(e)})

        self.logger.log(
            "PlannerReuseGenerated",
            {
                "goal_text": goal_text,
                "plan": parsed["plan"],
                "confidence_score": parsed["score"],
            },
        )

        self.report(
            {
                "event": "planner_reuse",
                "step": "PlannerReuse",
                "details": "New plan adapted from past traces",
                "goal_text": goal_text,
                "plan": parsed["plan"],
                "confidence_score": parsed["score"],
            }
        )

        return context

    def _extract_plan_from_response(self, response: str) -> dict:
        """
        Parse rationale, score, and plan steps from LLM response.
        Returns: {"rationale": str, "score": float, "plan": list[str]}
        """
        result = {"rationale": "", "score": None, "plan": []}

        # Rationale
        rationale_match = re.search(
            r"##\s*rationale:\s*(.*)", response, re.IGNORECASE
        )
        if rationale_match:
            result["rationale"] = rationale_match.group(1).strip()

        # Score
        score_match = re.search(
            r"##\s*score:\s*(\d+)", response, re.IGNORECASE
        )
        if score_match:
            try:
                result["score"] = float(score_match.group(1))
            except ValueError:
                result["score"] = None

        # Plan steps (lines after "## plan:")
        plan_match = re.split(r"##\s*plan:", response, flags=re.IGNORECASE)
        if len(plan_match) > 1:
            plan_block = plan_match[1]
            steps = re.findall(
                r"^\s*\d+\.\s*(.+)$", plan_block, flags=re.MULTILINE
            )
            result["plan"] = [s.strip() for s in steps if s.strip()]

        return result

    def _save_example(
        self, cand: Scorable, example: dict, bundle: ScoreBundle, context
    ) -> None:
        """
        Save an example to the database.
        """
        try:
            # Persist an evaluation bundle per example with both rank_score and hrm
            # (attributes carry the full rank_components + knn_score, tools, embed id)
            ex_rank = bundle.results["rank_score"].score
            reuse_bundle = ScoreBundle(
                results={
                    "rank_score": ScoreResult(
                        dimension="rank_score",
                        score=float(ex_rank),
                        weight=1.0,
                        source="planner_reuse",
                        rationale="PlannerReuse selection rank",
                        attributes={
                            "hrm": self.hrm_by_id.get(
                                str(example.get("trace_id"))
                            ),
                            "knn_score": self.knn_by_id.get(
                                str(example.get("trace_id"))
                            ),
                        },
                    ),
                    "hrm": ScoreResult(
                        dimension="hrm",
                        score=float(
                            self.hrm_by_id.get(str(example.get("trace_id")))
                            or 0.0
                        ),
                        weight=1.0,
                        source="planner_reuse",
                        rationale="HRM gate value",
                        attributes={
                            "knn_score": self.knn_by_id.get(
                                str(example.get("trace_id"))
                            )
                        },
                    ),
                }
            )

            # Persist against the candidate scorable itself
            self.memory.evaluations.save_bundle(
                bundle=reuse_bundle,
                scorable=cand,  # the Scorable you already built
                context=context,
                cfg=self.cfg,
                source="planner_reuse_examples",
                embedding_type=self.memory.embedding.name,
            )
        except Exception as e:
            self.logger.log(
                "PlannerReuseExamplePersistError",
                {"trace_id": example.get("trace_id"), "error": str(e)},
            )
``n

## File: planner_revise.py

`python
# stephanie/agents/quality/plan_revise_agent.py
import re
from typing import Any, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import PLAN_TRACE_ID
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.scorable import Scorable


class PlannerReviseAgent(BaseAgent):
    """
    Post-plan 'Revise' pass:
      - Critique candidate plan
      - Optionally propose a revised plan
      - Produce revise_score (0–1), plus optional sub-dimensions
      - Persist all signals; adopt revised plan if above threshold
    """

    def __init__(self, cfg, memory, container, logger, full_cfg=None):
        super().__init__(cfg, memory, container, logger)
        self.min_revise_score = cfg.get("min_revise_score", 0.65)
        self.enable_edit = cfg.get("enable_edit", True)
        self.revise_dimensions = cfg.get(
            "revise_dimensions",
            ["coherence", "feasibility", "completeness"]  # optional sub-scores
        )
        # Optional: custom prompt template via PromptLoader; fallback text if not set
        self.prompt_key = cfg.get("revise_prompt_template", "revise_plan_prompt.txt")
        self.judge_on_low = cfg.get("judge_on_low", True)          # enable pairwise judge when score is low
        self.judge_scorer = cfg.get("judge_scorer", "reward")      # 'reward' or 'llm_judge' (registered in ScoringService)
        self.judge_dimensions = cfg.get("judge_dimensions", None)  # optional dimension list for judge
        self.judge_margin = cfg.get("judge_margin", 0.05)          # tie-break margin
        self.on_fail = cfg.get("on_fail", "replan")                # 'replan' | 'retry' | 'none' | 'ask_human'
        self.max_revise_attempts = cfg.get("max_revise_attempts", 1)


    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text", "")

        attempt = int(context.get("revise_attempts", 0))
        context["revise_attempts"] = attempt + 1

        # Input: pull plan from either configured input_key or the conventional 'plan'
        candidate_plan: List[str] = context.get(self.input_key)

        examples = context.get("examples", [])  # from planner_reuse
        if not candidate_plan:
            self.logger.log("PlanReviseNoPlan", {"detail": "No plan found in context"})
            return context

        self.report({
            "event": "revise_start",
            "step": "PlanRevise",
            "goal": goal_text[:300],
            "num_examples": len(examples),
        })

        # Build prompt context and call LLM
        merged = {
            **context,
            "goal_text": goal_text,
            "candidate_plan": candidate_plan,
            "examples": examples,
            "enable_edit": self.enable_edit,
        }

        try:
            prompt = self.prompt_loader.load_prompt(self.cfg, context=merged)
            response = self.call_llm(prompt, context=merged)
            parsed = self._extract_revise_from_response(response)
        except Exception as e:
            self.logger.log("PlanReviseLLMError", {"error": str(e)})
            # non-fatal; keep original plan
            parsed = {"rationale": "", "score": None, "revised_plan": []}

        # Decide adoption
        revise_score = parsed.get("score")
        revised_plan = parsed.get("revised_plan", []) if self.enable_edit else []
        adopted = False
        final_plan = candidate_plan
        judge_res = None

        # Optional judge when score is low but we HAVE a revision to compare
        if (self.judge_on_low 
            and self.enable_edit 
            and revised_plan 
            and (not isinstance(revise_score, (int, float)) or revise_score < self.min_revise_score)):

            try:
                a_text = "\n".join(candidate_plan)
                b_text = "\n".join(revised_plan)
                judge_res = self.scoring.compare_pair(
                    scorer_name=self.judge_scorer,
                    context=context,  # carries goal + prefs, etc.
                    a=Scorable(id="candidate", text=a_text, target_type="plan"),
                    b=Scorable(id="revised",  text=b_text, target_type="plan"),
                    dimensions=self.judge_dimensions,
                    margin=self.judge_margin,
                )
                self.logger.log("PlanReviseJudgeResult", judge_res)
            except Exception as e:
                self.logger.log("PlanReviseJudgeError", {"error": str(e)})
                judge_res = None

        adopt_reason = None
        if self.enable_edit and revised_plan:
            if isinstance(revise_score, (int, float)) and revise_score >= self.min_revise_score:
                final_plan = revised_plan
                adopted = True
                adopt_reason = "revise_score"
            elif judge_res and judge_res.get("winner") == "b":
                # Revised beats original by the pairwise judge
                final_plan = revised_plan
                adopted = True
                adopt_reason = f"judge:{self.judge_scorer}"

        # If we didn’t adopt, set recall/next-action flags for the pipeline
        if not adopted:
            context["revise_outcome"] = {
                "adopted": False,
                "reason": ("low_score" if not isinstance(revise_score, (int, float)) or revise_score < self.min_revise_score else "no_revision"),
                "action": self.on_fail,
                "attempt": context["revise_attempts"],
            }
            if self.on_fail == "replan":
                context["recall_planner"] = True
            elif self.on_fail == "ask_human":
                context["needs_review"] = True

        # Persist evaluation + scores
        try:
            self._persist_revise_scores(context, candidate_plan, parsed, adopted)
        except Exception as e:
            self.logger.log("PlanRevisePersistError", {"error": str(e)})

        # Update context
        context[self.output_key] = final_plan
        context[f"{self.output_key}_meta"] = {
            "source": "revise_agent",
            "adopted": adopted,
            "adopt_reason": adopt_reason,
            "revise_score": revise_score,
            "issues_count": len(parsed.get("issues", [])),
        }

        # Reporting
        self.report({
            "event": "revise_done",
            "step": "PlanRevise",
            "adopted": adopted,
            "revise_score": revise_score,
            "orig_len": len(candidate_plan),
            "revised_len": len(revised_plan) if revised_plan else 0,
        })

        return context

    def _format_examples_for_prompt(self, examples: List[dict], max_n: int = 3) -> str:
        if not examples:
            return "None"
        out = []
        for e in examples[:max_n]:
            out.append(
                f"- trace_id={e.get('trace_id')}, hrm={e.get('hrm')}, rank={e.get('rank_score')}, knn={e.get('knn_score')}\n"
                f"  goal: {str(e.get('goal') or '')[:160]}"
            )
        return "\n".join(out)

    def _extract_revise_from_response(self, response: str) -> dict:
        """
        Parse a revise response.
        Supports:
          1) JSON with keys: rationale, score, revised_plan (list[str]), issues (list[str]), subscores{...}
          2) Markdown with headers (## rationale, ## revised plan, ## score: x)
        """
        res = {"rationale": "", "score": None, "revised_plan": [], "issues": [], "subscores": {}}
        txt = (response or "").strip()

        # 1) Try JSON first (fenced or plain)
        try:
            t = txt
            if t.startswith("```"):
                import re as _re
                t = _re.sub(r"^```[^\n]*\n", "", t)
                t = _re.sub(r"\n?```$", "", t).strip()
            if t.startswith("{"):
                import json
                data = json.loads(t)
                res["rationale"] = str(data.get("rationale", ""))[:4000]
                if isinstance(data.get("revised_plan"), list):
                    res["revised_plan"] = [str(s).strip() for s in data["revised_plan"] if str(s).strip()]
                if isinstance(data.get("issues"), list):
                    res["issues"] = [str(s).strip() for s in data["issues"] if str(s).strip()]
                if isinstance(data.get("subscores"), dict):
                    res["subscores"] = {k: float(v) for k, v in data["subscores"].items() if self._is_num(v)}
                sc = data.get("score")
                if self._is_num(sc):
                    res["score"] = float(sc)
                return res
        except Exception:
            pass

        # 2) Markdown fallback
        m_rat = re.search(r"##\s*rationale:\s*(.*)", txt, re.IGNORECASE)
        if m_rat:
            res["rationale"] = m_rat.group(1).strip()

        # issues: bullet lines after ## issues:
        m_issues = re.split(r"##\s*issues:\s*", txt, flags=re.IGNORECASE)
        if len(m_issues) > 1:
            block = m_issues[1]
            bullets = re.findall(r"^\s*[-*]\s*(.+)$", block, flags=re.MULTILINE)
            res["issues"] = [b.strip() for b in bullets if b.strip()]

        # revised plan: numbered steps
        m_plan = re.split(r"##\s*revised\s*plan:\s*", txt, flags=re.IGNORECASE)
        if len(m_plan) > 1:
            block = m_plan[1]
            steps = re.findall(r"^\s*\d+\.\s*(.+)$", block, flags=re.MULTILINE)
            res["revised_plan"] = [s.strip() for s in steps if s.strip()]

        # score:
        m_score = re.search(r"##\s*score:\s*([0-9]+(\.[0-9]+)?)", txt, re.IGNORECASE)
        if m_score:
            try:
                res["score"] = float(m_score.group(1))
            except Exception:
                res["score"] = None

        return res

    def _is_num(self, x: Any) -> bool:
        try:
            float(x)
            return True
        except Exception:
            return False

    def _to_float(self, x, default=None):
        try: 
            if x is None:
                return default
            # allow numeric strings like "0.82" and ints
            return float(x)
        except (TypeError, ValueError):
            return default


    def _persist_revise_scores(self, context: dict, orig_plan: list[str], parsed: dict, adopted: bool) -> None:
        """
        Persist revise_score (+ optional subscores) directly on the *current* plan_trace.
        Assumes PLAN_TRACE_ID is present in context (as you confirmed).
        """
        trace_id = str(context.get(PLAN_TRACE_ID))
        if not trace_id:
            # Just in case, but per your setup this should never hit
            self.logger.log("PlanReviseMissingTraceId", {"note": "PLAN_TRACE_ID absent; skipping persist"})
            return

        # Compact scorable text: original plan (first ~50 steps)
        scorable_text = "\n".join(orig_plan[:50])

        scorable = Scorable(
            id=trace_id,
            text=scorable_text,
            target_type="plan_trace",
        )

        # Main revise score (0–1)
        results = {
            "revise_score": ScoreResult(
                dimension="revise_score",
                score=float(parsed.get("score") or 0.0),
                weight=1.0,
                source="plan_revise",
                rationale=(parsed.get("rationale") or "")[:1000],
                attributes={
                    "adopted": bool(adopted),
                    "issues_count": len(parsed.get("issues") or []),
                    "orig_len": len(orig_plan),
                    "revised_len": len(parsed.get("revised_plan") or []),
                    "plan_trace_id": trace_id,
                    "pipeline_run_id": context.get("pipeline_run_id"),
                    "goal_id": (context.get("goal") or {}).get("id"),
                },
            )
        }

        # Optional sub-dimensions (if you included them in the prompt/response JSON)
        subs = parsed.get("subscores") or {}
        if not isinstance(subs, dict):
            subs = {}

        # Prefer configured keys, but also accept any other numeric subscores returned
        wanted_dims = list(self.revise_dimensions or [])
        extra_dims = [d for d in subs.keys() if d not in wanted_dims]
        all_dims = wanted_dims + extra_dims

        for dim in all_dims:
            raw = subs.get(dim, None)
            v = self._to_float(raw, default=None)
            if v is None:
                # Log once per missing/invalid subscore, but don’t crash
                self.logger.log("PlanReviseSubscoreSkipped", {
                    "dimension": dim,
                    "raw_value": raw,
                    "reason": "missing_or_non_numeric"
                })
                continue

            results[dim] = ScoreResult(
                dimension=dim,
                score=v,
                weight=1.0,
                source="plan_revise",
                rationale=f"Subscore: {dim}",
            )

        bundle = ScoreBundle(results=results)

        # This will:
        # - create an EvaluationORM tied to (scorable_id=trace_id, scorable_type="plan_trace")
        # - attach ScoreORM rows for each dimension
        # - auto-link to RuleApplication(s) via pipeline_run_id / goal_id (your save_bundle handles this)
        self.memory.evaluations.save_bundle(
            bundle=bundle,
            scorable=scorable,
            context=context,          # carries pipeline_run_id + goal
            cfg=self.cfg,
            source="plan_revise",
            embedding_type=self.memory.embedding.name,
        )

        self.logger.log("PlanRevisePersisted", {
            "plan_trace_id": trace_id,
            "revise_score": results["revise_score"].score,
            "subscores": {k: v.score for k, v in results.items() if k != "revise_score"}
        })

``n

## File: search_orchestrator.py

`python
# stephanie/agents/knowledge/search_orchestrator.py
from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.knowledge.automind_knowledge_collector import \
    AutoMindKnowledgeCollector
from stephanie.constants import GOAL
from stephanie.tools import WebSearchTool
from stephanie.tools.arxiv_tool import search_arxiv
from stephanie.tools.cos_sim_tool import get_top_k_similar
from stephanie.tools.huggingface_tool import (recommend_similar_papers,
                                              search_huggingface_datasets)
from stephanie.tools.wikipedia_tool import WikipediaTool


class SearchOrchestratorAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.web_search_tool = WebSearchTool(cfg.get("web_search", {}), self.logger)
        self.wikipedia_tool = WikipediaTool(self.memory, self.logger)
        self.max_results = cfg.get("max_results", 5)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        queries = context.get("search_queries", [])
        goal_id = goal.get("id")
        results = []

        for search_query in queries:
            source = self.route_query(goal, search_query)
            try:
                if source == "arxiv":
                    hits = await search_arxiv([search_query])
                elif source == "huggingface":
                    hits = await search_huggingface_datasets([search_query])
                elif source == "wikipedia":
                    hits = self.wikipedia_tool.find_similar(search_query)
                elif source == "web":
                    hits = await self.web_search_tool.search(
                        search_query, max_results=self.max_results
                    )
                elif source == "similar_papers":
                    hits = recommend_similar_papers(search_query)
                elif source == "automind":
                    collector = AutoMindKnowledgeCollector(self)
                    task_description = context.get("task_description", "AI agent task")
                    knowledge = await collector.retrieve_knowledge(task_description)
                    # Now you can pass this knowledge to planner or tree search agent
                    context["knowledge"] = knowledge
                else:
                    continue

                enriched_hits = [
                    {
                        "query": search_query,
                        "source": source,
                        "result_type": hit.get("type", "unknown"),
                        "title": hit.get("title", hit.get("name", "")),
                        "summary": hit.get("snippet", hit.get("description", "")),
                        "url": hit.get("url", ""),
                        "goal_id": goal_id,
                        "parent_goal": goal.get("goal_text"),
                        "strategy": goal.get("strategy"),
                        "focus_area": goal.get("focus_area"),
                        "extra_data": {"source_specific": hit},
                    }
                    for hit in hits
                ]

                # Store results in DB
                stored_results = self.memory.search_results.bulk_add_results(
                    enriched_hits
                )
                results.extend(stored_results)

            except Exception as e:
                self.logger.log(
                    "SearchToolFailed",
                    {"query": search_query, "tool": source, "error": str(e)},
                )

        # Save result IDs or ORM objects back to context
        context["search_result_ids"] = [r.id for r in results]
        context["search_results"] = [r.to_dict() for r in results]
        return context

    def route_query(self, goal, query: str) -> str:
        """
        Decide which source to use based on query content.
        """
        query_lower = query.lower()

        # Try fast metadata path first
        source = self.fast_metadata_routing(goal, query_lower)
        if source:
            return source

        # Fallback to semantic similarity
        return self.semantic_fallback_routing(query)

    def fast_metadata_routing(self, goal, query_lower):
        focus_area = goal.get("focus_area", "").lower()
        goal_type = goal.get("goal_type", "").lower()

        if goal_type == "automind":
            return "automind"
        if goal_type == "similar_papers":
            return "similar_papers"
        if goal_type == "data_search" or "dataset" in query_lower:
            return "huggingface"
        if goal_type == "model_review" or "model" in query_lower:
            return "arxiv"
        if goal_type == "background" or any(
            k in query_lower for k in ["overview", "definition"]
        ):
            return "wikipedia"
        if focus_area in ["nlp", "cv", "graph learning"] and "baseline" in query_lower:
            return "arxiv"

        return None

    def semantic_fallback_routing(self, query: str) -> str:
        intent_map = {
            "arxiv": [
                "find research paper_score",
                "latest ML study",
                "scientific method",
            ],
            "huggingface": ["find dataset", "huggingface model", "nlp corpus"],
            "wikipedia": ["define concept", "what is", "overview of topic"],
            "web": ["general info", "random search", "link to resource"],
        }

        candidates = [
            (intent, phrase)
            for intent, phrases in intent_map.items()
            for phrase in phrases
        ]
        phrases = [p for _, p in candidates]

        top = get_top_k_similar(query, phrases, self.memory, top_k=1)
        best_phrase = top[0][0]

        for intent, phrase in candidates:
            if phrase == best_phrase:
                return intent

        return "web"
``n

## File: section_prioritizer.py

`python
"""
SectionPrioritizer
------------------
Ranks sections to process first. Methods/Results-heavy sections first,
then Discussion/Conclusion, then Intro/Related Work.

Signals:
- Title heuristics (priority by kind)
- Density of evidence (Figures/Tables)
- Technique keywords (method/algorithm/ablation/experiment)
"""

from __future__ import annotations

import re
from typing import Dict, List

TITLE_PRI = [
    ("method", 1.0),
    ("approach", 0.95),
    ("model", 0.92),
    ("experiment", 0.90),
    ("results", 0.90),
    ("ablation", 0.88),
    ("evaluation", 0.86),
    ("analysis", 0.84),
    ("discussion", 0.80),
    ("conclusion", 0.78),
    ("introduction", 0.60),
    ("related work", 0.55),
    ("background", 0.50),
]

TECH_HINTS = ("transformer", "adapter", "loss", "optimization", "pipeline", "retrieval", "graph", "policy", "reward")

def score_section(sec: Dict) -> float:
    title = (sec.get("section_name") or "").lower()
    text  = sec.get("section_text") or ""

    # 1) title heuristic
    base = 0.5
    for key, w in TITLE_PRI:
        if key in title:
            base = max(base, w)

    # 2) evidence density
    ev = len(re.findall(r"\b(fig\.|figure|table|tbl\.)\b", text.lower()))
    ev_score = min(0.15, 0.03 * ev)

    # 3) technique keywords
    tech = sum(1 for k in TECH_HINTS if k in text.lower())
    tech_score = min(0.2, 0.02 * tech)

    # length sanity
    length_score = min(0.15, 0.00002 * len(text))

    return float(min(1.0, base + ev_score + tech_score + length_score))

def prioritize_sections(sections: List[Dict], top_n: int = None) -> List[Dict]:
    ranked = sorted(sections, key=score_section, reverse=True)
    return ranked[:top_n] if top_n else ranked
``n

## File: survey.py

`python
# stephanie/agents/knowledge/survey.py
import re

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL


class SurveyAgent(BaseAgent):
    """
    The Survey Agent generates adaptive search queries for literature exploration.

    From the paper_score:
    > 'The Survey Agent deconstructs the research task into multiple keyword combinations'
    > 'It supports two distinct modes: literature review mode and deep research mode'
    > 'Each idea is mapped to testable components before being executed'
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.max_queries = cfg.get("max_queries", 5)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, {})
        goal_type = goal.get("goal_type", "survey")
        context["search_strategy"] = self.strategy

        if goal_type == "similar_papers":
            # Skip survey agent if goal is to find similar papers
            self.logger.log("SurveyAgentSkipped", {"reason": "similar_papers_goal"})
            context["search_queries"] = [goal.get("goal_text", "")]
            return context

        if not goal:
            self.logger.log("NoGoalProvided", {"reason": "survey_agent_skipped"})
            return context

        # Generate new queries based on goal + baseline + preferences
        prompt_context = {
            "goal_text": goal.get("goal_text"),
            "focus_area": goal.get("focus_area"),
            "baseline_method": context.get("baseline_method", ""),
            "preferences": context.get("preferences", ["novelty", "feasibility"]),
            "previous_ideas": context.get("ideas", []),
        }
        merged = {**self.cfg, **prompt_context}

        prompt = self.prompt_loader.load_prompt(self.cfg, merged)

        raw_output = self.call_llm(prompt, context)
        formatted_output = self.remove_think_blocks(raw_output)
        queries = self._parse_query_response(goal, formatted_output)

        # Store in context for SearchOrchestratorAgent
        context["search_queries"] = queries

        self.logger.log(
            "SurveyQueriesGenerated",
            {
                "queries": queries,
                "strategy_used": self.strategy,
                "pipeline_stage": context.get("pipeline_stage"),
            },
        )

        return context

    def _parse_query_response(self, goal, response: str) -> list:
        """Parse LLM output into clean list of search queries"""
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        if not lines:
            # Fallback strategy
            return [
                f"{goal.get('focus_area')} machine learning",
                f"{goal.get('goal_text')}",
            ]
        return lines[: self.max_queries]

    def expand_queries_to_goals(self, queries: list, base_goal: dict) -> list:
        """
        Convert queries into sub-goals for future pipeline stages

        Args:
            queries (list): Generated search strings
            base_goal (dict): Original goal

        Returns:
            list: List of structured sub-goals
        """
        return [
            {
                "goal_text": q,
                "parent_goal": base_goal.get("goal_text"),
                "focus_area": base_goal.get("focus_area"),
                "strategy": base_goal.get("strategy"),
                "source": "survey_agent",
            }
            for q in queries
        ]

    def remove_think_blocks(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
``n

## File: text_improver.py

`python
# stephanie/agents/knowledge/text_improver.py
"""
TextImproverAgent — Advanced refinement using policy-driven edits and VPM feedback.
Leverages full ecosystem: KnowledgeBus, CasebookStore, CalibrationManager, GoalScorer.
"""

from __future__ import annotations

import json
import os
import re
import signal as _signal
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.paper_improver.goals import GoalScorer
from stephanie.knowledge.casebook_store import CaseBookStore
from stephanie.knowledge.knowledge_bus import KnowledgeBus
from stephanie.scoring.calibration_manager import CalibrationManager
from stephanie.scoring.scorable_factory import TargetType
from stephanie.utils.json_sanitize import safe_json


def _supports_alarm() -> bool:
    """Return True if signal.alarm is usable on this platform/thread."""
    return hasattr(_signal, "SIGALRM") and os.name != "nt"


def _timeout_handler(signum, frame):
    raise TimeoutError("TextImprover timed out")


def atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding=encoding)
    tmp.replace(path)


class TextImproverAgent(BaseAgent):
    """
    Production-ready text improver that applies targeted edits based on VPM gaps.
    Fully integrated with casebook, calibration, and event bus.
    """

    def __init__(self, cfg: Dict[str, Any], memory: Any, container: Any, logger: Any):
        super().__init__(cfg, memory, container, logger)

        # Core components
        self.workdir = Path(cfg.get("text_improve_workdir", "./data/text_runs"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.timeout = cfg.get("improve_timeout", 60)
        self.seed = cfg.get("seed", 0)
        self.faithfulness_topk = cfg.get("faithfulness_topk", 5)

        # Optional services
        self.kb = cfg.get("kb") or KnowledgeBus()
        self.casebooks = cfg.get("casebooks") or CaseBookStore()
        self.calibration = cfg.get("calibration") or CalibrationManager(
            cfg=cfg.get("calibration", {}),
            memory=memory,
            logger=logger
        )
        self.gs = GoalScorer(logger=logger)

        # State
        self.run_id = 0

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:
          - knowledge_plan: dict
          - draft_text: str
          - goal_template: str
          - initial_goal_score: float
          - session_id: str
          - casebook_name: str
          - case_id: int

        Output:
          - improved_draft: str
          - improvement_trajectory / edit_log: list
          - final_vpm_row: dict
          - final_goal_score: dict
          - run_dir: Path
          - initial_scores, final_scores: raw scoring breakdowns
        """
        plan = context.get("knowledge_plan")
        draft_text = context.get("draft_text", "").strip()
        goal_template = context.get("goal_template", "academic_summary")
        initial_score = context.get("initial_goal_score", 0.0)
        session_id = context.get("session_id", "unknown")
        casebook_name = context.get("casebook_name", "default")
        case_id = context.get("case_id")

        if not plan or not draft_text:
            self.logger.log("TextImproverSkipped", {
                "reason": "missing_input",
                "has_plan": bool(plan),
                "has_draft": bool(draft_text)
            })
            return context

        # Create run directory
        self.run_id += 1
        run_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        run_dir = self.workdir / run_id
        run_dir.mkdir(exist_ok=True)

        try:
            # Optional timeout guard (POSIX only)
            alarm_installed = False
            if _supports_alarm():
                _signal.signal(_signal.SIGALRM, _timeout_handler)
                _signal.alarm(int(self.timeout))
                alarm_installed = True

            # Normalize plan
            plan_norm = self._normalize_plan(plan, run_dir)

            # Save initial draft
            draft_path = run_dir / "draft.md"
            atomic_write(draft_path, draft_text)
            atomic_write(run_dir / "initial_draft.md", draft_text)  # traceable

            # Score BEFORE edits
            initial_scores = self._score_draft(draft_path, plan_norm)

            # Apply edit policy (writes back to draft_path)
            final_text, edits = self._apply_edit_policy(
                draft_path=draft_path,
                plan=plan_norm,
                max_edits=int(context.get("max_edits", 6)),
                trace_path=run_dir / "trace.ndjson"
            )

            # Score AFTER edits
            final_scores = self._score_draft(draft_path, plan_norm)
            vpm_row = self._build_vpm_row(initial_scores, final_scores, plan_norm)

            # Robust goal scoring with fallback
            try:
                goal_eval = self.gs.score("text", goal_template, vpm_row)
            except KeyError:
                self.logger.log("GoalTemplateFallback", {
                    "requested": goal_template,
                    "fallback": "academic_summary"
                })
                goal_eval = self.gs.score("text", "academic_summary", vpm_row)
            label = bool(goal_eval.get("score", 0.0) >= 0.7)

            # Log to casebook
            self.casebooks.add_scorable(
                casebook_name=casebook_name,
                case_id=case_id,
                role="vpm",
                text=safe_json(vpm_row),
                meta={"goal": goal_eval},
                scorable_type=TargetType.DYNAMIC
            )
            self.casebooks.add_scorable(
                casebook_name=casebook_name,
                case_id=case_id,
                role="text",
                text=final_text,
                meta={"stage": "final"},
                scorable_type=TargetType.DYNAMIC
            )
            # Persist final draft explicitly too
            atomic_write(run_dir / "final_draft.md", final_text)

            # Publish trajectory event
            self.kb.publish("trajectory.step", {
                "casebook": casebook_name,
                "case_id": case_id,
                "vpm": vpm_row,
                "goal": goal_eval,
            })

            # Update context
            context.update({
                "improved_draft": final_text,
                "improvement_trajectory": edits,
                "edit_log": edits,  # alias for downstream consumers
                "final_vpm_row": vpm_row,
                "final_goal_score": goal_eval,
                "run_dir": str(run_dir),
                "draft_path": str(draft_path),
                "initial_scores": initial_scores,
                "final_scores": final_scores,
            })

            self.report({
                "event": "end",
                "step": "TextImprover",
                "details": f"Scored {goal_eval['score']:.3f} → {'PASS' if label else 'FAIL'}"
            })

            return context

        except TimeoutError:
            self.logger.log("TextImproverTimeout", {"timeout_sec": self.timeout})
            context["error"] = "timeout"
            return context

        except Exception as e:
            self.logger.log("TextImproverUnexpectedError", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            context["error"] = f"unexpected: {str(e)}"
            return context

        finally:
            if _supports_alarm():
                _signal.alarm(0)  # clear any pending alarms

    def _normalize_plan(self, plan: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        """Ensure plan has consistent structure."""
        out = {}
        out["section_title"] = (plan.get("section_title") or "Section").strip() or "Section"

        units_in = plan.get("units") or []
        clean_units = []
        for u in units_in:
            if not isinstance(u, dict):
                continue
            claim = (u.get("claim") or "").strip()
            evidence = (u.get("evidence") or "See paper").strip()
            cid = u.get("claim_id")
            if not claim:
                continue
            clean_units.append({"claim": claim, "evidence": evidence, "claim_id": cid})

        if not clean_units:
            clean_units = [{"claim": f"Overview of {out['section_title']}.", "evidence": "", "claim_id": "C1"}]

        out["units"] = clean_units
        out["entities"] = plan.get("entities", {})
        out["paper_text"] = plan.get("paper_text", "")
        out["domains"] = plan.get("domains", [])
        out["tags"] = plan.get("tags", [])

        # Save for debugging
        atomic_write(run_dir / "plan.json", json.dumps(out, indent=2))
        return out

    def _score_draft(self, draft_path: Path, plan: Dict[str, Any]) -> Dict[str, float]:
        """Compute fine-grained scores for edit policy."""
        if not draft_path.exists():
            return {}

        text = draft_path.read_text()

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        units = plan.get("units", [])

        # Coverage
        ids = [u.get("claim_id") for u in units if u.get("claim_id")]
        covered_ids = sum(1 for cid in ids if f"[#{cid}]" in text) if ids else 0
        coverage = covered_ids / max(1, len(ids)) if ids else 0.0

        # Citation support / correctness proxy
        factual = [s for s in sentences if self._is_factual_sentence(s)]
        cited = sum(1 for s in factual if "[#]" in s)
        citation_support = (cited / max(1, len(factual))) if factual else 1.0
        correctness = citation_support

        # Entity consistency (ABBR handling)
        abbrs = plan.get("entities", {}).get("ABBR", {})
        entity_consistency = self._compute_abbr_consistency(text, abbrs)

        # Readability (FKGL)
        words = text.split()
        num_words = len(words)
        num_sentences = len(sentences)
        syllables = sum(self._count_syllables(w) for w in words) or 1
        fkgl_raw = (
            0.39 * (num_words / max(1, num_sentences)) +
            11.8 * (syllables / max(1, num_words)) -
            15.59
        )
        readability = float(max(6.0, min(15.0, fkgl_raw)))

        # Coherence
        coh_scores = []
        for i in range(len(sentences) - 1):
            s1 = set(re.findall(r"\w+", sentences[i].lower()))
            s2 = set(re.findall(r"\w+", sentences[i + 1].lower()))
            denom = len(s1 | s2)
            coh_scores.append((len(s1 & s2) / denom) if denom else 1.0)
        coherence = sum(coh_scores) / len(coh_scores) if coh_scores else 1.0

        # Novelty & Stickiness
        novelty = 0.5  # placeholder
        stickiness = self._compute_stickiness(text, plan)

        len_chars = len(text)
        compactness = len(re.sub(r"\s+", " ", text)) / max(1, len(text))

        return {
            "coverage": coverage,
            "correctness": correctness,
            "coherence": coherence,
            "citation_support": citation_support,
            "entity_consistency": entity_consistency,
            "readability": readability,
            "novelty": novelty,
            "stickiness": stickiness,
            "len_chars": float(len_chars),
            "compactness": float(compactness),
            "fkgl_raw": float(fkgl_raw),
        }

    def _apply_edit_policy(self, draft_path: Path, plan: Dict[str, Any], max_edits: int = 6, trace_path: Optional[Path] = None) -> tuple[str, List[str]]:
        text = draft_path.read_text()
        edits = []

        for i in range(max_edits):
            scores = self._score_draft(draft_path, plan)
            change = False

            if scores["coverage"] < 0.8:
                missing = [
                    u for u in plan.get("units", [])
                    if u.get("claim_id") and f"[#{u['claim_id']}]" not in text
                ]
                if missing:
                    u = missing[0]
                    line = (
                        f"- {u.get('claim','Claim')} [#{u['claim_id']}].\n"
                        f"  *Evidence: {u.get('evidence','See paper')}* [#]\n"
                    )
                    text = text.rstrip() + "\n\n" + line
                    edits.append(f"add_claim:{u['claim_id']}")
                    change = True

            if not change and scores["citation_support"] < 0.9:
                lines = text.splitlines()
                new_lines = []
                for line in lines:
                    if any(kw in line.lower() for kw in ["show","prove","result","increase","decrease","outperform","statistically"]) and "[#]" not in line:
                        line += " [#]"
                    new_lines.append(line)
                new_text = "\n".join(new_lines)
                if new_text != text:
                    text = new_text
                    edits.append("add_citation_placeholders")
                    change = True

            if not change and not (9.0 <= scores["readability"] <= 11.0):
                before = text
                if scores["readability"] > 11.0:
                    text = re.sub(r";\s+", ". ", text)
                    text = re.sub(r"\s+and\s+", ". ", text, count=1)
                    edits.append("split_long_sentences")
                else:
                    text = re.sub(r"\.\s+([a-z])", r", \1", text, count=1)
                    edits.append("join_short_sentences")
                change = True

            if not change and scores["coherence"] < 0.7:
                before = text
                # Merge short adjacent bullets: "- aaa.\n- bbb." -> "- aaa; bbb."
                text = re.sub(r"\n- ([^.\n]{0,60})\.\s*\n- ", r"\n- \1; ", text)
                if text != before:
                    edits.append("merge_adjacent_bullets")
                else:
                    text = self._regenerate_lead_in(text, plan)
                    edits.append("regen_lead_in")
                change = True

            if not change and self._has_duplicate_bullets(text):
                text = self._dedup_bullets(text)
                edits.append("dedup_bullets")
                change = True

            if not change:
                break

            atomic_write(draft_path, self._normalize_ws(text))
            if trace_path:
                with trace_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"edit": i + 1, "scores": scores, "op": edits[-1]}) + "\n")

        return text, edits

    def _build_vpm_row(self, initial: Dict, final: Dict, plan: Dict) -> Dict[str, float]:
        return {
            "coverage": final["coverage"],
            "correctness": final["correctness"],
            "coherence": final["coherence"],
            "citation_support": final["citation_support"],
            "entity_consistency": final["entity_consistency"],
            "readability": final["readability"],
            "fkgl_raw": final["fkgl_raw"],
            "novelty": final["novelty"],
            "stickiness": final["stickiness"],
            "len_chars": final["len_chars"],
            "compactness": final["compactness"]
        }

    def _is_factual_sentence(self, s: str) -> bool:
        return any(kw in s.lower() for kw in ("show", "prove", "result", "achiev", "increase", "decrease", "outperform", "error", "accuracy", "loss", "significant", "statistically"))

    def _count_syllables(self, word: str) -> int:
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        if word and word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count = 1
        return count

    def _compute_abbr_consistency(self, text: str, abbrs: Dict[str, str]) -> float:
        if not abbrs:
            return 1.0
        matches = 0
        for full, abbr in abbrs.items():
            instances = len(re.findall(rf"\b{re.escape(full)}\b", text, re.IGNORECASE))
            abbreviations = len(re.findall(rf"\b{re.escape(abbr)}\b", text, re.IGNORECASE))
            if instances > 1:
                matches += 1 if abbreviations >= 1 else 0
            else:
                matches += 1 if abbreviations == 0 else 0
        return matches / max(1, len(abbrs))

    def _compute_stickiness(self, text: str, plan: Dict[str, Any]) -> float:
        plan_terms = set()
        for unit in plan.get("units", []):
            claim = unit.get("claim", "") or ""
            for w in re.findall(r"\b[a-zA-Z]{5,}\b", claim.lower()):
                plan_terms.add(w)
        if not plan_terms:
            return 1.0
        text_words = set(re.findall(r"\b[a-zA-Z]{5,}\b", text.lower()))
        return len(plan_terms & text_words) / max(1, len(plan_terms))

    def _normalize_ws(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip() + "\n"

    def _regenerate_lead_in(self, text: str, plan: Dict[str, Any]) -> str:
        lead = f"This section covers {plan['section_title'].lower()} with key insights."
        bullets = text.splitlines()
        return lead + "\n\n" + "\n".join(b for b in bullets if b.strip())

    def _has_duplicate_bullets(self, text: str) -> bool:
        lines = [b.strip() for b in text.splitlines() if b.strip()]
        seen = set()
        for line in lines:
            if line in seen:
                return True
            seen.add(line)
        return False

    def _dedup_bullets(self, text: str) -> str:
        lines = text.splitlines()
        seen = set()
        unique = []
        for line in lines:
            if line.strip() not in seen:
                seen.add(line.strip())
                unique.append(line)
        return "\n".join(unique)
``n

## File: verified_section_generator.py

`python
# stephanie/agents/knowledge/verified_section_generator.py
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.paper_improver.goals import GoalScorer


def _hash(x: str) -> str: return hashlib.sha256(x.encode("utf-8")).hexdigest()[:12]
def _toklen(s: str) -> int: return max(1, len(s)//4)

@dataclass
class VConfig:
    beam: int = 6
    max_depth: int = 6
    branch_per_node: int = 6
    target_quality: float = 0.95
    max_bundle_tokens: int = 1800
    keep_top_k_bundles: int = 8
    # weights for committee
    w_goal: float = 0.25
    w_evidence: float = 0.25
    w_traj: float = 0.15
    w_coherence: float = 0.15
    w_style: float = 0.10
    w_novelty: float = 0.10

class VerificationScorer:
    """
    Reuses GoalScorer + simple evidence/trajectory checks + calibration.
    Assumes 'knowledge_tree' is provided.
    """
    def __init__(self, memory, logger, goal_template="blog_section", weights: Dict[str, float] = None):
        self.memory = memory
        self.logger = logger
        self.goal = goal_template
        self.gs = GoalScorer(logger=logger)
        self.w = weights or {}

    def _overlap(self, a: str, b: str) -> float:
        aw = set(a.lower().split()); bw = set(b.lower().split())
        return len(aw & bw)/max(1, len(aw))

    def score(self, text: str, tree: Dict[str, Any], neighbors: Tuple[str, str] = ("","")) -> Tuple[float, Dict[str,float]]:
        claims = tree.get("claims", [])
        insights = tree.get("insights", [])
        conns = tree.get("connections", [])

        # goal fit
        goal_fit = self.gs.score(kind="text", goal=self.goal, vpm_row={
            "correctness": 0.6, "progress": 0.6, "evidence": 0.6, "novelty": 0.6
        }).get("score", 0.6)  # we don’t have per-text vpm here; keep neutral baseline

        # evidence support: claim & insight coverage
        claim_cov = 0.0
        if claims:
            hits = sum(1 for c in claims if self._overlap(text, c["text"]) > 0.05)
            claim_cov = hits / len(claims)
        insight_cov = 0.0
        if insights:
            hits = sum(1 for i in insights if self._overlap(text, i["text"]) > 0.05)
            insight_cov = hits / len(insights)
        evidence = 0.6 * claim_cov + 0.4 * insight_cov

        # trajectory alignment ~ rely on connections density
        traj = min(1.0, 0.5 + 0.5 * evidence)

        # coherence with neighbors
        prev, nxt = neighbors
        coh = 0.5
        if prev: coh = max(coh, 0.5 + 0.5*self._overlap(text, prev))
        if nxt:  coh = max(coh, 0.5 + 0.5*self._overlap(text, nxt))

        # novelty vs raw paper section text: prefer synthesis not copy
        # crude proxy: penalize very high overlap with any single claim
        max_claim_overlap = max([self._overlap(text, c["text"]) for c in claims] or [0.0])
        novelty = max(0.0, 1.0 - max_claim_overlap)

        # style: short, clear, cites figures if relevant
        style = 0.5 + 0.2*("figure" in text.lower() or "table" in text.lower())

        parts = {
            "goal": goal_fit, "evidence": evidence, "traj": traj,
            "coherence": coh, "novelty": novelty, "style": style
        }
        # weighted sum (calibration hook)
        w = self.w or {"goal":0.25,"evidence":0.25,"traj":0.15,"coherence":0.15,"novelty":0.10,"style":0.10}
        raw = sum(parts[k]*w[k] for k in w)
        # simple calibration fallback (identity)
        calibrated = min(1.0, max(0.0, raw))
        return calibrated, parts

class VerifiedSectionGeneratorAgent(BaseAgent):
    """
    LATS/beam search over candidates from tile bundles, verified against knowledge tree.
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        vcfg = cfg.get("verified_section", {})
        self.c = VConfig(**vcfg) if vcfg else VConfig()
        self.scorer = VerificationScorer(memory, logger, goal_template="blog_section")

    def _tiles_from_context(self, sec: Dict[str,Any], msgs: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        tiles = []
        # paper tiles: first/middle/last spans
        st = (sec.get("section_text") or "").strip()
        if st:
            cuts = [st[:600], st[len(st)//2: len(st)//2 + 600], st[-600:]]
            for i, ct in enumerate(cuts):
                tiles.append({"id": f"ps_{i}", "kind": "paper", "text": ct.strip()})
        # conversation tiles
        for m in msgs:
            tiles.append({"id": f"cv_{_hash(m['text'][:80])}", "kind": "conv", "text": m["text"][:600].strip()})
        return tiles

    def _bundle(self, tiles: List[Dict[str,Any]]) -> List[List[Dict[str,Any]]]:
        # simple greedy packing by token budget + diversity
        random.shuffle(tiles)
        bundles, cur, cur_tok = [], [], 0
        for t in tiles:
            ln = _toklen(t["text"])
            if cur_tok + ln > self.c.max_bundle_tokens and cur:
                bundles.append(cur); cur, cur_tok = [], 0
            cur.append(t); cur_tok += ln
        if cur: bundles.append(cur)
        # keep top-N diverse by kind balance
        def score_bundle(b): 
            kinds = [x["kind"] for x in b]
            bal = 1.0 - abs(kinds.count("paper") - kinds.count("conv"))/max(1,len(b))
            return 0.7*bal + 0.3*min(1.0, len(b)/10)
        bundles.sort(key=score_bundle, reverse=True)
        return bundles[:self.c.keep_top_k_bundles]

    def _prompt(self, section_name: str, bundle: List[Dict[str,Any]], tree: Dict[str,Any]) -> str:
        ctx = "\n\n".join([f"[{t['kind'].upper()}] {t['text']}" for t in bundle])
        claims = "\n".join([f"- {c['text']}" for c in tree.get("claims",[])[:6]])
        return (
            f"You are drafting a crisp blog section titled: {section_name}.\n"
            f"Ground yourself ONLY in the following context tiles and do not invent facts.\n"
            f"Context:\n{ctx}\n\n"
            f"Key claims to cover:\n{claims}\n\n"
            f"Write 2–4 short paragraphs with inline figure/table references when appropriate."
        )

    def _gen(self, prompt: str) -> str:
        # minimal LLM wrapper using your memory.embedding or existing gen; replace with your generator
        # Here we assume a simple synchronous call via memory.embedding.llm if you have it; otherwise stub.
        # For now, just echo prompt tail as a placeholder (replace in your stack).
        return prompt.split("Context:\n")[-1][:1000]  # <<< REPLACE with real LLM call

    async def run(self, context: Dict[str,Any]) -> Dict[str,Any]:
        sec = context.get("paper_section") or {}
        tree = context.get("knowledge_graph") or {}
        crit = context.get("critical_messages", [])
        if not sec or not tree:
            self.logger.log("VerifiedSectionSkipped", {"reason":"missing_inputs"})
            return context

        tiles = self._tiles_from_context(sec, crit)
        bundles = self._bundle(tiles)
        section_name = sec.get("section_name", "Section")

        # Beam search
        Node = Tuple[str, float, Dict[str,float], List[str]]  # (text, score, parts, bundle_ids)
        beam: List[Node] = []
        # seed from bundles
        for b in bundles:
            prompt = self._prompt(section_name, b, tree)
            cand = self._gen(prompt)
            s, parts = self.scorer.score(cand, tree)
            beam.append((cand, s, parts, [t["id"] for t in b]))
        beam.sort(key=lambda x: x[1], reverse=True)
        beam = beam[:self.c.beam]

        depth = 1
        while depth < self.c.max_depth:
            # stopping if strong enough and stable
            top = beam[0]
            if top[1] >= self.c.target_quality:
                break

            # expand
            cands: List[Node] = []
            for (text, score, parts, used) in beam:
                # propose variations: rewrite / swap bundle / focused improvement
                for b in random.sample(bundles, k=min(self.c.branch_per_node, len(bundles))):
                    prompt = self._prompt(section_name, b, tree) + "\n\nRevise to improve coverage and coherence."
                    cand = self._gen(prompt + "\n\nDraft:\n" + text)
                    s, pr = self.scorer.score(cand, tree)
                    cands.append((cand, s, pr, [t["id"] for t in b]))

            cands.sort(key=lambda x: x[1], reverse=True)
            beam = cands[:self.c.beam]
            depth += 1

        best = max(beam, key=lambda x: x[1])
        context["verified_section"] = best[0]
        context["verification_trace"] = {
            "score": best[1], "parts": best[2], "bundles": best[3], "depth": depth
        }
        context["quality_confidence"] = best[1]
        self.logger.log("VerifiedSectionDone", {"confidence": best[1], "depth": depth})
        return context
``n
