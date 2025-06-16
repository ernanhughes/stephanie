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
import requests
from co_ai.agents.base_agent import BaseAgent
from co_ai.tools.pdf_tools import PDFConverter
from co_ai.tools.arxiv_tool import fetch_arxiv_metadata
from co_ai.constants import GOAL

class DocumentLoaderAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.max_chars_for_summary = cfg.get("max_chars_for_summary", 8000)
        self.summarize_documents = cfg.get("summarize_documents", False)

    async def run(self, context: dict) -> dict:
        search_results = context.get(self.input_key, [])
        goal = context.get(GOAL, {})
        goal_id = goal.get("id")

        stored_documents = []

        for result in search_results:
            try:
                url = result.get("url")
                title = result.get("title")
                summary = result.get("summary")

                existing = self.memory.document.get_by_url(url)
                if existing:
                    self.logger.log("DocumentAlreadyExists", {"url": url})
                    stored_documents.append(existing)
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
                        guessed_title = self.guess_title_from_text(text)
                        if guessed_title:
                            title = guessed_title

                # Store as DocumentORM
                doc = {
                    "goal_id": goal_id,
                    "title": title,
                    "summary": summary,
                    "source": self.name,
                    "text": text,
                    "url": url,
                }

                stored = self.memory.document.add_document(doc)
                stored_documents.append(stored)

            except Exception as e:
                self.logger.log(
                    "DocumentLoadFailed", {"url": result.get("url"), "error": str(e)}
                )

        context["document_ids"] = [doc.id for doc in stored_documents]
        context[self.output_key] = [doc.to_dict() for doc in stored_documents]
        return context

    def guess_title_from_text(self, text: str) -> str:
        lines = text.strip().split("\n")
        lines = [line.strip() for line in lines if line.strip()]
        candidates = [line for line in lines[:15] if len(line.split()) >= 4]
        return candidates[0] if candidates else None