# co_ai/agents/document/document_loader.py
"""
DocumentLoaderAgent module for Co AI

This module defines an agent responsible for retrieving, parsing, and storing research documents
based on search results. It supports PDF downloading, text extraction, optional summarization
using LLMs, and persistent storage in the document database.

Typically used after a search orchestrator agent in the pipeline to prepare documents for scoring,
ranking, or hypothesis generation.
"""

import os

import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

from co_ai.agents.base_agent import BaseAgent
from co_ai.constants import GOAL
from co_ai.tools.arxiv_tool import fetch_arxiv_metadata
from co_ai.tools.pdf_tools import PDFConverter
from co_ai.analysis.domain_classifier import DomainClassifier

def guess_title_from_text(text: str) -> str:
    lines = text.strip().split("\n")
    lines = [line.strip() for line in lines if line.strip()]
    candidates = [line for line in lines[:15] if len(line.split()) >= 4]
    return candidates[0] if candidates else None


class DocumentLoaderAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.max_chars_for_summary = cfg.get("max_chars_for_summary", 8000)
        self.summarize_documents = cfg.get("summarize_documents", False)
        self.domain_seeds = cfg.get("domain_seeds", {})
        self.top_k_domains = cfg.get("top_k_domains", 3)
        self.domain_classifier = DomainClassifier(
            memory=self.memory,
            logger=self.logger,
            config_path=cfg.get("domain_seed_config_path", "config/domain/seeds.yaml"),
        )



    async def run(self, context: dict) -> dict:
        search_results = context.get(self.input_key, [])
        goal = context.get(GOAL, {})
        goal_id = goal.get("id")

        stored_documents = []
        document_domains = []

        for result in search_results:
            try:
                url = result.get("url")
                external_id = result.get("title") # A quirk of the search we store the id as the title
                title = result.get("title")
                summary = result.get("summary")

                existing = self.memory.document.get_by_url(url)
                if existing:
                    self.logger.log("DocumentAlreadyExists", {"url": url})
                    stored_documents.append(existing.to_dict())
                    if not self.memory.document_domains.get_domains(existing.id):
                        text = existing.text or ""
                        domain = self.domain_classifier.classify(text)
                        document_domains.append({"document_id": existing.id, "domain": domain})

                        self.memory.document_domains.insert(
                            {
                                "document_id": existing.id,
                                "domain": domain,
                                "score": 1.0,  # assume max score for existing match
                            }
                        )

                        self.logger.log(
                            "DomainAssigned",
                            {
                                "title": existing.title[:60],
                                "domain": domain,
                                "score": 1.0,
                            },
                        )
                    continue

                # Download PDF
                response = requests.get(url)
                if response.status_code != 200:
                    self.logger.log(
                        "DocumentLoadFailed",
                        {"url": url, "error": f"HTTP {response.status_code}"}
                    )
                    continue

                # Save to temporary file
                pdf_path = f"/tmp/{title}.pdf"
                with open(pdf_path, "wb") as f:
                    f.write(response.content)

                # Extract text
                text = PDFConverter.pdf_to_text(pdf_path)
                os.remove(pdf_path)

                if self.summarize_documents:
                    meta_data = fetch_arxiv_metadata(title)
                    if meta_data:
                        title = meta_data["title"]
                        summary = meta_data["summary"]
                    else:
                        merged = {"document_text": text, **context}
                        prompt_text = self.prompt_loader.load_prompt(self.cfg, merged)
                        summary = self.call_llm(prompt_text, context)
                        guessed_title = guess_title_from_text(text)
                        if guessed_title:
                            title = guessed_title

                # Store as DocumentORM
                doc = {
                    "goal_id": goal_id,
                    "title": title,
                    "external_id": external_id,
                    "summary": summary,
                    "source": self.name,
                    "text": text,
                    "url": url,
                }

                # Save embedding
                embed_text = f"{doc['title']}\n\n{doc.get('summary', '')}"
                self.memory.embedding.get_or_create(embed_text)

                # Save to DB
                stored = self.memory.document.add_document(doc)
                stored_documents.append(stored.to_dict())

                # Assign + store domain
                text = stored.text or ""
                domain = self.domain_classifier.classify(text)
                document_domains.append({"document_id": stored.id, "domain": domain})

                self.memory.document_domains.insert(
                    {
                        "document_id": stored.id,
                        "domain": domain,
                        "score": 1.0,  # assuming single best match
                    }
                )

                self.logger.log(
                    "DomainAssigned",
                    {
                        "title": stored.title[:60],
                        "domain": domain,
                        "score": 1.0,
                    },
                )


            except Exception as e:
                self.logger.log(
                    "DocumentLoadFailed", {"url": result.get("url"), "error": str(e)}
                )

        context[self.output_key] = stored_documents
        context["document_ids"] = [doc.get("id") for doc in stored_documents]
        context["document_domains"] = document_domains
        return context

