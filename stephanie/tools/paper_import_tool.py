from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from stephanie.components.information.data import PaperNode, PaperReferenceRecord
from stephanie.memory.paper_store import PaperStore
from stephanie.tools.arxiv_tool import fetch_arxiv_metadata
from stephanie.tools.pdf_tool import PDFConverter
from stephanie.utils.hash_utils import hash_text

log = logging.getLogger(__name__)

# Matches:
#   - "arXiv:2506.21734"
#   - "arxiv 2506.21734"
#   - "https://arxiv.org/abs/2506.21734"
ARXIV_INLINE_PATTERN = re.compile(
    r"(?:arxiv[:\s]*)(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
ARXIV_ABS_PATTERN = re.compile(
    r"https?://arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)


@dataclass
class PaperImportResult:
    node: PaperNode
    text: str
    text_hash: str
    pdf_path: Optional[str] = None
    references_path: Optional[str] = None
    references: Optional[List[PaperReferenceRecord]] = None
    raw: Optional[Dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None


class PaperImportTool:
    """
    Reusable importer:
      - download/read PDF
      - convert PDF -> text
      - fetch arXiv metadata (optional)
      - extract + persist references.json (optional)
      - (optionally) upsert into PaperStore / PaperReferenceStore if present in memory
    """

    name = "paper_import"

    def __init__(self, cfg: Dict[str, Any], memory, container, logger=None):
        self.cfg = cfg or {}
        self.memory = memory
        self.container = container
        self.logger = logger

        self.papers_root = Path(self.cfg.get("papers_root", "data/papers"))
        self.papers_root.mkdir(parents=True, exist_ok=True)

        self.max_refs = int(self.cfg.get("max_refs", 256))
        self.timeout_s = float(self.cfg.get("http_timeout_s", 30.0))
        self.user_agent = str(self.cfg.get("user_agent", "stephanie-paper-import/1.0"))
        self.persist_references_json = bool(self.cfg.get("persist_references_json", True))
        self.enable_metadata = bool(self.cfg.get("enable_metadata", True))

        # Optional DB stores (if wired into memory)
        self.paper_store: PaperStore = memory.papers


    def _new_run_id(self) -> str:
        return uuid.uuid4().hex  # swap to UniversalID if you want

    def _pdf_url(self, arxiv_id: str) -> str:
        return f"https://arxiv.org/abs/{arxiv_id}"
    
    async def import_paper(
        self,
        *,
        arxiv_id: Optional[str] = None,
        url: Optional[str] = None,
        local_pdf_path: str | Path = "data/papers",
        role: Optional[str] = "root",
        source: str = "arxiv",
        force: bool = True,
        force_references: bool = False,
        max_refs: int = 100,
    ) -> PaperImportResult:
        """
        Main entrypoint.

        Rules:
          - If arxiv_id is provided, it is the stable primary id.
          - If local_pdf_path provided, we use it (no download).
          - Else we download from url (or default arxiv PDF url).
          - If PaperStore exists and has text (and not force), we reuse it.
        """
        if not arxiv_id and url:
            arxiv_id = self._infer_arxiv_id(url)

        store: PaperStore = self.memory.papers  # <-- single interface

        # 1) Fast path: already in DB (skip import unless forced)
        existing = store.get_by_id(arxiv_id)
        if existing and getattr(existing, "text", None) and not force:
            store.create_run(
                paper_id=arxiv_id,
                run_type="paper_import",
                variant="cache_hit",
                stats={"cached": True},
            )
            # Return a minimal, "cached" result; caller can re-load full node/text if needed
            return PaperImportResult(
                success=True,
                error=None,
            )

        paper_id = arxiv_id or self._slugify(
            url or (str(local_pdf_path) if local_pdf_path else "paper")
        )
        paper_dir = self.papers_root / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)

        # Stable guess of a PDF URL if not provided
        pdf_url = url
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # 1) DB cache shortcut (text)
        cached_text, cached_hash = self._maybe_load_text_from_store(
            paper_id, force=force
        )
        if cached_text and cached_hash:
            text = cached_text
            text_hash = cached_hash
            pdf_path: Optional[Path] = (
                paper_dir / "paper.pdf" if (paper_dir / "paper.pdf").exists() else None
            )
        else:
            # 2) Resolve PDF path (local or downloaded)
            pdf_path = self._resolve_pdf_path(
                paper_dir=paper_dir,
                paper_id=paper_id,
                pdf_url=pdf_url,
                local_pdf_path=local_pdf_path,
                force=force,
            )

            if pdf_path is None:
                # Could not get a PDF – log and continue with empty text
                log.warning(
                    "PDF could not be resolved for paper_id=%s url=%s; "
                    "continuing with metadata only.",
                    paper_id,
                    pdf_url,
                )
                text = ""
                text_hash = hash_text(text)
            else:
                # 3) Convert PDF -> text
                text = PDFConverter.pdf_to_text(pdf_path)
                if not text.strip():
                    log.warning(
                        "Empty extracted text for paper_id=%s pdf=%s; "
                        "continuing with empty text.",
                        paper_id,
                        pdf_path,
                    )
                    text = ""
                text_hash = hash_text(text)

        if not text.strip():
            log.warning(
                "No text extracted for paper_id=%s; skipping reference extraction.",
                paper_id,
            )
            return PaperImportResult(
                node = PaperNode(
                    arxiv_id=arxiv_id,
                    role="unknown",
                    source=source,
                    url=pdf_url,
                    pdf_path=(
                        str(local_pdf_path)
                        if local_pdf_path
                        else (str(paper_dir / "paper.pdf") if (paper_dir / "paper.pdf").exists() else None)
                    ),
                    meta={
                        "paper_id": paper_id,
                        "text_hash": text_hash,
                    },
                ),
                text="",
                text_hash=text_hash,
                pdf_path=str(pdf_path) if pdf_path else None,
                references=[],
                success=False,
                error=None,
            )

        # 4) Metadata (optional)
        meta: Dict[str, Any] = {}
        if self.enable_metadata and arxiv_id:
            try:
                meta_data = fetch_arxiv_metadata(arxiv_id)
                if meta_data:
                    meta["arxiv"] = meta_data
            except Exception as e:
                log.warning(
                    "fetch_arxiv_metadata failed arxiv_id=%s err=%s", arxiv_id, e
                )

        # 5) References
        refs: List[PaperReferenceRecord] = []
        references_path: Optional[str] = None

        if self.persist_references_json:
            # Load from DB if possible unless forced
            if self.paper_store and arxiv_id and not (force or force_references):
                try:
                    refs = self._load_references_from_store(arxiv_id)
                except Exception as e:
                    log.warning(
                        "ReferenceStore load failed arxiv_id=%s err=%s",
                        arxiv_id,
                        e,
                    )

        n_refs = 0
        reference_orms = []
        # Only try to extract references when we actually have text
        if (not refs or (force or force_references)) and text.strip():
            try:
                refs = self._extract_reference_records(
                    text,
                    max_refs=max_refs or self.max_refs,
                    require_arxiv_id=True,
                )
                references_path = str(paper_dir / "references.json")

                # Write JSON from the raw refs (dataclasses)
                self._write_references_json(Path(references_path), refs)

                # Prepare payload for the store (dicts)
                refs_payload = [
                    asdict(r) if is_dataclass(r) else r
                    for r in (refs or [])
                ]
                reference_orms = store.replace_references(arxiv_id, refs_payload)
                n_refs = len(reference_orms)

            except Exception as e:
                log.warning(
                    "Reference extraction failed for paper_id=%s err=%s",
                    paper_id,
                    e,
                )
                refs = []
                references_path = None
                n_refs = 0
        else:
            if not text.strip():
                log.warning(
                    "Skipping reference extraction for paper_id=%s: no text.",
                    paper_id,
                )

        # 2) Create run row early (so we can attach events/errors)
        paper_run = store.create_run(
            paper_id=arxiv_id,
            run_type="paper_import",
            variant="fresh",
            stats={"cached": False},
            config={},
        )
        store.add_event(
            run_id=paper_run.id,
            stage=self.name,
            event_type="info",
            message="starting import",
        )

        # 6) Construct node
        node = PaperNode(
            arxiv_id=arxiv_id or paper_id,
            role=role or "unknown",
            source=source,
            url=pdf_url,
            pdf_path=(
                str(local_pdf_path)
                if local_pdf_path
                else (str(paper_dir / "paper.pdf") if (paper_dir / "paper.pdf").exists() else None)
            ),
            meta={
                "paper_id": paper_id,
                "text_hash": text_hash,
                "references_path": references_path,
                "reference_count": n_refs,
                "ref_arxiv_ids": [
                    r.arxiv_id
                    for r in (refs or [])
                    if getattr(r, "arxiv_id", None)
                ],
                **meta,
            },
        )

        raw = {
            "paper_id": paper_id,
            "arxiv_id": arxiv_id,
            "pdf_url": pdf_url,
            "pdf_path": (
                str(paper_dir / "paper.pdf")
                if (paper_dir / "paper.pdf").exists()
                else None
            ),
            "text_hash": text_hash,
            "reference_count": n_refs,
        }

        store.add_event(
            run_id=paper_run.id,
            stage="paper_import",
            event_type="info",
            message=f"import complete: {len(text)} chars, {n_refs} refs",
            data={
                "chars": len(text),
                "n_refs": n_refs,
                "refs": [r.to_dict() for r in reference_orms],
                "pdf_path": str(pdf_path) if pdf_path is not None else None,
            },
        )

        return PaperImportResult(
            node=node,
            text=text,
            text_hash=text_hash,
            pdf_path=raw["pdf_path"],
            references_path=references_path,
            references=refs,
            raw=raw,
            success=True,
            error=None, 
        )

    # -------------------------
    # Storage helpers
    # -------------------------

    def _maybe_load_text_from_store(self, paper_id: str, *, force: bool) -> Tuple[Optional[str], Optional[str]]:
        if force or not self.paper_store:
            return None, None
        try:
            p = self.paper_store.get_by_id(paper_id)
            if p and getattr(p, "text", None) and getattr(p, "text_hash", None):
                return p.text, p.text_hash
        except Exception as e:
            log.warning("PaperStore get_by_id failed paper_id=%s err=%s", paper_id, e)
        return None, None

    def _load_references_from_store(self, arxiv_id: str) -> List[PaperReferenceRecord]:
        refs: List[PaperReferenceRecord] = []
        rows = self.paper_store.get_references(arxiv_id)  # expects your PaperReferenceStore API
        for r in rows or []:
            refs.append(
                PaperReferenceRecord(
                    arxiv_id=getattr(r, "ref_arxiv_id", None),
                    doi=getattr(r, "doi", None),
                    title=getattr(r, "title", None),
                    year=getattr(r, "year", None),
                    url=getattr(r, "url", None),
                    raw_citation=getattr(r, "raw_citation", None) or "",
                )
            )
        return refs

    def _maybe_upsert_to_store(
        self,
        *,
        paper_id: str,
        arxiv_id: Optional[str],
        pdf_url: Optional[str],
        pdf_path: Optional[str],
        text: str,
        text_hash: str,
        meta: Dict[str, Any],
        references: List[PaperReferenceRecord],
        force: bool,
    ) -> None:
        if not self.paper_store:
            return

        # Upsert paper text + meta
        try:
            fields = {
                "id": paper_id,
                "external_id": arxiv_id or paper_id,
                "url": pdf_url,
                "pdf_path": pdf_path,
                "text": text,
                "text_hash": text_hash,
                "meta": meta,
            }
            self.paper_store.upsert_paper(paper_id, fields)
        except Exception as e:
            log.warning("PaperStore upsert failed paper_id=%s err=%s", paper_id, e)

        # Upsert references
        if arxiv_id and references:
            try:
                ref_dicts = []
                for idx, rr in enumerate(references):
                    ref_dicts.append(
                        {
                            "order_idx": idx,
                            "ref_arxiv_id": rr.get("arxiv_id"),
                            "doi": rr.get("doi"),
                            "title": rr.get("title"),
                            "year": rr.get("year"),
                            "url": rr.get("url"),
                            "raw_citation": rr.get("raw_citation"),
                            "source": "paper_import",
                            "raw": rr,  # JSON-safe
                        }
                    )
                self.paper_store.replace_references(arxiv_id, ref_dicts)
            except Exception as e:
                log.warning("PaperReferenceStore replace_for_paper failed arxiv_id=%s err=%s", arxiv_id, e)

    # -------------------------
    # PDF + parsing
    # -------------------------

    def _resolve_pdf_path(
        self,
        *,
        paper_dir: Path,
        paper_id: str,
        pdf_url: Optional[str],
        local_pdf_path: Optional[str | Path],
        force: bool,
    ) -> str:
        if local_pdf_path:
            p = Path(local_pdf_path)
            if not p.exists():
                raise FileNotFoundError(f"local_pdf_path not found: {p}")
            return str(p)

        pdf_local_path = paper_dir / "paper.pdf"
        if pdf_local_path.exists() and pdf_local_path.stat().st_size > 0 and not force:
            return str(pdf_local_path)

        if not pdf_url:
            raise ValueError(f"No pdf_url and no local_pdf_path for paper_id={paper_id}")

        self._download_pdf(pdf_url, pdf_local_path)
        return str(pdf_local_path)

    def _download_pdf(self, url: str, out_path: Path) -> bool:
        """
        Try to download a PDF to out_path.

        Returns
        -------
        bool
            True  -> download succeeded and file exists at out_path
            False -> download failed (404, 5xx, network error, etc.)
        """
        tmp = out_path.with_suffix(".pdf.part")
        headers = {"User-Agent": self.user_agent}

        try:
            with requests.get(url, stream=True, timeout=self.timeout_s, headers=headers) as r:
                status = r.status_code

                if status == 404:
                    log.warning("PDF not found (404) at %s; skipping.", url)
                    return False

                if status != 200:
                    log.warning(
                        "Unexpected status %s when downloading PDF from %s; skipping.",
                        status,
                        url,
                    )
                    return False

                tmp.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 128):
                        if chunk:
                            f.write(chunk)

            tmp.replace(out_path)
            return True

        except requests.RequestException as e:
            # Network / timeout / connection errors
            log.warning("Error downloading PDF from %s: %s. Skipping.", url, e)
            return False

    def _extract_text(self, pdf_path: str) -> str:
        converter = PDFConverter()
        return converter.convert_pdf_to_text(Path(pdf_path))

    # -------------------------
    # References
    # -------------------------

    def _write_references_json(self, path: Path, refs: list[Any]) -> None:
        """
        Write references to JSON. Accepts either PaperReferenceRecord dataclasses
        or plain dicts.
        """
        payload = []
        for rr in refs or []:
            if is_dataclass(rr):
                payload.append(asdict(rr))
            elif isinstance(rr, dict):
                payload.append(rr)
            else:
                log.warning(
                    "Unexpected reference type %r when writing references JSON; skipping",
                    type(rr),
                )
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _extract_reference_records(self, text: str, *, max_refs: int, require_arxiv_id: bool = False) -> List[PaperReferenceRecord]:
        block = self._slice_references_block(text)
        if not block:
            return []

        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        refs: List[PaperReferenceRecord] = []

        for ln in lines:
            # grab an arxiv id if present
            aid = None
            m = ARXIV_INLINE_PATTERN.search(ln)
            if m:
                aid = m.group(1)
            else:
                m2 = ARXIV_ABS_PATTERN.search(ln)
                if m2:
                    aid = m2.group(1)
            if require_arxiv_id and not aid:
                continue
            else:
                meta_data = fetch_arxiv_metadata(aid)
            idx = len(refs) + 1
            refs.append(
                PaperReferenceRecord(
                    paper_id=aid,
                    arxiv_id=aid,
                    doi=meta_data.get("doi"),
                    order_idx=idx,
                    title=meta_data.get("title"),
                    year=meta_data.get("published_year"),
                    url=meta_data.get("url"),
                    summary=meta_data.get("summary"),
                    authors=meta_data.get("authors"),
                    raw_citation=ln,
                )
            )
            if len(refs) >= max_refs:
                break

        return refs

    def _slice_references_block(self, text: str) -> str:
        """
        Heuristic: take the last chunk of the paper and look for a REFERENCES-like header.
        """
        if not text:
            return ""

        tail = text[-200_000:] if len(text) > 200_000 else text
        # common headings
        for header in ["references", "bibliography", "reference"]:
            idx = tail.lower().rfind(header)
            if idx != -1:
                return tail[idx:]
        # fallback: last ~15%
        n = max(2000, int(len(tail) * 0.15))
        return tail[-n:]

    # -------------------------
    # Id helpers
    # -------------------------

    def _infer_arxiv_id(self, url: str) -> Optional[str]:
        m = ARXIV_ABS_PATTERN.search(url or "")
        return m.group(1) if m else None

    def _slugify(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[^a-z0-9._-]+", "-", s)
        s = re.sub(r"-{2,}", "-", s).strip("-")
        return s or "paper"
