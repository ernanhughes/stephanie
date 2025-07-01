# stephanie/agents/document/document_loader.py
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
        self.min_classification_score = cfg.get("min_classification_score", 0.6)
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
                external_id = result.get(
                    "title"
                )  # A quirk of the search we store the id as the title
                title = result.get("title")
                summary = result.get("summary")

                existing = self.memory.document.get_by_url(url)
                if existing:
                    self.logger.log("DocumentAlreadyExists", {"url": url})
                    stored_documents.append(existing.to_dict())
                    if not self.memory.document_domains.get_domains(existing.id) or self.force_domain_update:
                        self.assign_domains_to_document(existing)
                        continue

                # Download PDF
                response = requests.get(url, stream=True)
                if response.status_code != 200:
                    self.logger.log(
                        "DocumentRequestFailed",
                        {"url": url, "error": f"HTTP {response.status_code}"},
                    )
                    continue

                file_name = result.get("pid") or result.get("arxiv_id")
                if not file_name:
                    file_name = self.sanitize_filename(title) or "document"        
                # Save to temporary file
                pdf_path = f"{self.download_directory}/{file_name}.pdf"
                with open(pdf_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                # Extract text
                if not PDFConverter.validate_pdf(pdf_path):
                    self.logger.log(
                        "DocumentLoadFailed",
                        {"url": url, "error": "Invalid PDF format 0x00 byte found"},
                    )
                    os.remove(pdf_path)
                    continue
                text = PDFConverter.pdf_to_text(pdf_path)
                os.remove(pdf_path)

                pid = result.get("pid") or result.get("arxiv_id")
                if self.summarize_documents:
                    meta_data = fetch_arxiv_metadata(pid)
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
                self.assign_domains_to_document(stored)

            except Exception as e:
                self.logger.log(
                    "DocumentLoadFailed", {"url": result.get("url"), "error": str(e)}
                )

        context[self.output_key] = stored_documents
        context["document_ids"] = [doc.get("id") for doc in stored_documents]
        context["document_domains"] = document_domains
        return context


    def sanitize_filename(self, title: str) -> str:
        return re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", title)[:100]  # truncate to 100 chars

    def assign_domains_to_document(self, document):
        """
        Classifies the document text into one or more domains,
        and stores results in the document_domains table.
        """
        content = document.content
        if content:
            results = self.domain_classifier.classify(content, self.top_k_domains, self.min_classification_score)
            for domain, score in results:
                self.memory.document_domains.insert({
                    "document_id": document.id,
                    "domain": domain,
                    "score": score,
                })
                self.logger.log("DomainAssigned", {
                    "title": document.title[:60] if document.title else "",
                    "domain": domain,
                    "score": score,
                })
        else:
            self.logger.log("DocumentNoContent", {
                "document_id": document.id,
                "title": document.title[:60] if document.title else "", })
