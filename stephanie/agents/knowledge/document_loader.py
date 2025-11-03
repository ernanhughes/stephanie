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
from __future__ import annotations

import os
import re

import requests
from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.constants import GOAL
from stephanie.scoring.scorable import Scorable, ScorableType
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
                    pid = result.get("pid") or result.get("arxiv_id") or result.get("title")
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
                    "external_id": result.get("pid") or result.get("title"),
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
            target_type=ScorableType.DOCUMENT,
        )
        self.memory.scorable_embeddings.get_or_create(scorable)
        self.memory.pipeline_references.insert(
            {
                "pipeline_run_id": context.get("pipeline_run_id"),
                "scorable_type": ScorableType.DOCUMENT,
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