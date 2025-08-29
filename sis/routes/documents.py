# sis/routes/documents.py
"""
Routes for browsing and viewing documents in SIS.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/documents", response_class=HTMLResponse)
def list_documents(request: Request):
    """
    Show a list of all documents stored in the system.
    """
    memory = request.app.state.memory
    templates = request.app.state.templates

    logger.info("[SIS] Fetching documents for listing")

    docs = memory.documents.get_all(limit=50)  # <-- implement in store if not present

    return templates.TemplateResponse(
        "documents/list.html",
        {
            "request": request,
            "title": "📄 Documents",
            "documents": [doc.to_dict() for doc in docs],
        },
    )


@router.get("/documents/{doc_id}", response_class=HTMLResponse)
def document_detail(request: Request, doc_id: int):
    """
    Show details for a single document.
    """
    memory = request.app.state.memory
    templates = request.app.state.templates

    logger.info(f"[SIS] Fetching document detail: {doc_id}")

    doc = memory.documents.get_by_id(doc_id)
    if not doc:
        return templates.TemplateResponse(
            "errors/not_found.html",
            {"request": request, "title": "Document Not Found", "id": doc_id},
            status_code=404,
        )

    # domains + embeddings if available
    domain_list = memory.scorable_domains.get_by_scorable(str(doc.id), "document")
    domains = []
    for d in domain_list:
        domains.append(d.to_dict()) 
    emb_list = memory.scorable_embeddings.get_by_scorable(str(doc.id), "document", memory.embedding.name)
    emb = None
    if emb_list and len(emb_list) > 0:
       emb = emb_list[0].to_dict()

    return templates.TemplateResponse(
        "documents/detail.html",
        {
            "request": request,
            "title": f"📄 Document: {doc.title}",
            "document": doc.to_dict(),
            "domains": domains,
            "embedding": emb,
        },
    )


@router.get("/documents/{doc_id}/profile", response_class=HTMLResponse)
def document_profile(doc_id: int, request: Request):
    """
    Show structured sections of a document (profiled by DocumentProfilerAgent).
    """
    memory = request.app.state.memory
    templates = request.app.state.templates

    logger.info(f"[SIS] Fetching document profile: {doc_id}")

    doc = memory.documents.get_by_id(doc_id)
    if not doc:
        return templates.TemplateResponse(
            "errors/not_found.html",
            {"request": request, "title": "Document Not Found", "id": doc_id},
            status_code=404,
        )

    # Fetch profiled sections
    sections = memory.document_sections.get_by_document(doc_id)
    sections = [
        {
            **s.to_dict(),
            "domains": [d.domain if hasattr(d, "domain") else d["domain"] for d in s.domains] if s.domains else []
        }
        for s in sections
    ]

    return templates.TemplateResponse(
        "documents/profile.html",
        {
            "request": request,
            "document": doc,
            "sections": sections,
        },
    )
