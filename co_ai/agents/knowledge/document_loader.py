# co_ai/agents/document/document_loader_agent.py

import os
import requests
from co_ai.agents.base_agent import BaseAgent
from co_ai.tools.pdf_tools import PDFConverter
from co_ai.models.document import DocumentORM
from co_ai.memoryI .document_store import DocumentStore


class DocumentLoaderAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.storage = DocumentStore(self.memory)

    async def run(self, context: dict) -> dict:
        search_results = context.get("search_results", [])
        goal = context.get("goal", {})
        goal_id = goal.get("id")

        stored_documents = []

        for result in search_results:
            try:
                url = result.get("url")
                title = result.get("title")
                summary = result.get("summary", "")
                source = result.get("source", "unknown")

                # Download PDF
                response = requests.get(url)
                if response.status_code != 200:
                    raise Exception(f"Failed to fetch PDF: {url}")

                # Save to temporary file
                pdf_path = f"/tmp/{title}.pdf"
                with open(pdf_path, "wb") as f:
                    f.write(response.content)

                # Extract text
                text = PDFConverter.extract_text_from_pdf(pdf_path)
                os.remove(pdf_path)

                # Store as DocumentORM
                doc = {
                    "goal_id": goal_id,
                    "source": source,
                    "title": title,
                    "summary": summary,
                    "text": text,
                    "url": url,
                }

                stored = self.storage.add_document(doc)
                stored_documents.append(stored)

            except Exception as e:
                self.logger.log(
                    "DocumentLoadFailed", {"url": result.get("url"), "error": str(e)}
                )

        context["document_ids"] = [doc.id for doc in stored_documents]
        context["documents"] = [doc.to_dict() for doc in stored_documents]
        return context
