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

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.domain_classifier import DomainClassifier
from stephanie.constants import GOAL
from stephanie.scoring.scorable_factory import TargetType
from stephanie.tools.arxiv_tool import fetch_arxiv_metadata
from stephanie.tools.pdf_tools import PDFConverter


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
        self.force_domain_update = cfg.get("force_domain_update", False)
        self.top_k_domains = cfg.get("top_k_domains", 3)
        self.download_directory = cfg.get("download_directory", "/tmp")
        self.min_classification_score = cfg.get(
            "min_classification_score", 0.6
        )
        self.embed_full_document = cfg.get("embed_full_document", True)
        self.domain_classifier = DomainClassifier(
            memory=self.memory,
            logger=self.logger,
            config_path=cfg.get(
                "domain_seed_config_path", "config/domain/seeds.yaml"
            ),
        )

    async def run(self, context: dict) -> dict:
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
        pipeline_run_id = context.get("pipeline_run_id")
        for result in search_results:
            try:
                url = result.get("url")
                title = result.get("title")
                summary = result.get("summary")

                # Skip existing
                existing = self.memory.document.get_by_url(url)
                if existing:
                    self.report(
                        {
                            "event": "skipped_existing",
                            "step": "DocumentLoader",
                            "details": f"Document already exists: {title}",
                            "url": url,
                        }
                    )
                    stored_documents.append(existing.to_dict())
                    self.memory.pipeline_references.insert(
                        {
                            "pipeline_run_id": pipeline_run_id,
                            "target_type": TargetType.DOCUMENT,
                            "target_id": existing.id,
                            "relation_type": "existing",
                            "source": self.name,
                        }
                    )
                    # Assign domains if needed
                    if (
                        not self.memory.document_domains.has_domains(existing.id)
                        or self.force_domain_update
                    ):
                        self.assign_domains_to_document(existing)

                    continue

                # Download PDF
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

                file_name = (
                    result.get("pid")
                    or result.get("arxiv_id")
                    or self.sanitize_filename(title)
                    or "document"
                )
                pdf_path = f"{self.download_directory}/{file_name}"

                with open(pdf_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

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

                text = PDFConverter.pdf_to_text(pdf_path)
                os.remove(pdf_path)

                # Summarize (optional)
                if self.summarize_documents:
                    pid = result.get("pid") or result.get("arxiv_id")
                    meta_data = fetch_arxiv_metadata(pid)
                    if meta_data:
                        title = meta_data["title"]
                        summary = meta_data["summary"]
                    else:
                        merged = {"document_text": text, **context}
                        prompt_text = self.prompt_loader.load_prompt(
                            self.cfg, merged
                        )
                        summary = self.call_llm(prompt_text, context)
                        guessed_title = guess_title_from_text(text)
                        if guessed_title:
                            title = guessed_title

                # Store document
                doc = {
                    "goal_id": goal_id,
                    "title": title,
                    "external_id": result.get("title"),
                    "summary": summary,
                    "source": self.name,
                    "text": text,
                    "url": url,
                }
                stored = self.memory.document.add_document(doc)
                doc_id = stored.id

                if self.embed_full_document:
                    embed_text = f"{doc['title']}\n\n{doc.get('text', doc.get('summary', ''))}"
                else:
                    embed_text = f"{doc['title']}\n\n{doc.get('summary', '')}"

                embedding_vector = self.memory.embedding.get_or_create(
                    embed_text
                )
                embedding_id = self.memory.embedding.get_id_for_text(
                    embed_text
                )

                self.memory.document_embeddings.insert(
                    {
                        "document_id": doc_id,
                        "document_type": TargetType.DOCUMENT,
                        "embedding_id": embedding_id,
                        "embedding_type": self.memory.embedding.name,
                    }
                )
                self.memory.pipeline_references.insert(
                    {
                        "pipeline_run_id": pipeline_run_id,
                        "target_type": TargetType.DOCUMENT,
                        "target_id": doc_id,
                        "relation_type": "inserted",
                        "source": self.name,
                    }
                )

                stored_documents.append(stored.to_dict())

                self.report(
                    {
                        "event": "stored",
                        "step": "DocumentLoader",
                        "details": f"Stored document: {title}",
                        "doc_id": doc_id,
                        "url": url,
                    }
                )

                # Assign domains
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

    def sanitize_filename(self, title: str) -> str:
        return re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", title)[:100]

    def assign_domains_to_document(self, document):
        text = document.text
        if text:
            results = self.domain_classifier.classify(
                text, self.top_k_domains, self.min_classification_score
            )
            for domain, score in results:
                self.memory.document_domains.insert(
                    {
                        "document_id": document.id,
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
