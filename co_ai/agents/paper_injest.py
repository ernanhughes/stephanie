from co_ai.agents.base import BaseAgent
from co_ai.models.document import DocumentORM
from co_ai.utils.pdf_tools import extract_text_from_url  # You should have a PDF extraction util


class PaperIngestAgent(BaseAgent):
    def __init__(self, cfg: dict, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    def ingest(self, title: str, url: str, source: str = "arxiv", external_id: str = None):
        self.logger(f"📥 Ingesting paper: {title}")

        # Check if document already exists
        existing = (
            self.session.query(DocumentORM)
            .filter_by(url=url)
            .first()
        )
        if existing:
            self.logger(f"⚠️ Document already ingested: {url}")
            return existing

        # Extract text (from PDF or HTML)
        try:
            content = extract_text_from_url(url)
        except Exception as e:
            self.logger(f"❌ Failed to extract content from {url}: {e}")
            content = None

        # Save to DB
        doc = DocumentORM(
            title=title,
            url=url,
            source=source,
            external_id=external_id,
            content=content,
        )
        self.session.add(doc)
        self.session.commit()
        self.logger(f"✅ Paper saved: {title} ({url})")
        return doc

    def batch_ingest_from_list(self, paper_list: list[dict]):
        for paper in paper_list:
            self.ingest(
                title=paper.get("title"),
                url=paper.get("url"),
                source=paper.get("source", "arxiv"),
                external_id=paper.get("external_id")
            )
