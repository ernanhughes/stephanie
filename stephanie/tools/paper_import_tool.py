from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass
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
        run_id = self._new_run_id()

        # 1) Fast path: already in DB (skip import unless forced)
        existing = store.get_by_id(arxiv_id)
        if existing and getattr(existing, "text", None) and not force:
            store.create_run(
                paper_id=arxiv_id,
                run_type="paper_import",
                variant="cache_hit",
                stats={"cached": True},
            )
            return PaperImportResult(paper_id=arxiv_id, run_id=run_id)

 
        paper_id = arxiv_id or self._slugify(url or (str(local_pdf_path) if local_pdf_path else "paper"))
        paper_dir = self.papers_root / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)

        # Stable guess of a PDF URL if not provided
        pdf_url = url
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # 1) DB cache shortcut (text)
        cached_text, cached_hash = self._maybe_load_text_from_store(paper_id, force=force)
        if cached_text and cached_hash:
            text = cached_text
            text_hash = cached_hash
            pdf_path = str(paper_dir / "paper.pdf") if (paper_dir / "paper.pdf").exists() else None
        else:
            # 2) Resolve PDF path (local or downloaded)
            pdf_path = self._resolve_pdf_path(
                paper_dir=paper_dir,
                paper_id=paper_id,
                pdf_url=pdf_url,
                local_pdf_path=local_pdf_path,
                force=force,
            )

            # 3) Convert PDF -> text
            text = PDFConverter.pdf_to_text(pdf_path)
            if not text.strip():
                raise ValueError(f"Empty extracted text for paper_id={paper_id} pdf={pdf_path}")

            text_hash = hash_text(text)

        # 4) Metadata (optional)
        meta: Dict[str, Any] = {}
        if self.enable_metadata and arxiv_id:
            try:
                meta_data = fetch_arxiv_metadata(arxiv_id)
                if meta_data:
                    meta["arxiv"] = meta_data
            except Exception as e:
                log.warning("fetch_arxiv_metadata failed arxiv_id=%s err=%s", arxiv_id, e)

        # 5) References (only meaningful for root papers usually, but caller decides)
        refs: List[PaperReferenceRecord] = []
        references_path: Optional[str] = None
        if self.persist_references_json:
            # Load from DB if possible unless forced
            if self.paper_store and arxiv_id and not (force or force_references):
                try:
                    refs = self._load_references_from_store(arxiv_id)
                except Exception as e:
                    log.warning("ReferenceStore load failed arxiv_id=%s err=%s", arxiv_id, e)

 
 
        store.upsert_paper(
            arxiv_id,
            {
                "id": arxiv_id,
                "external_id": arxiv_id,
                "url": pdf_url,
                "pdf_path": pdf_path,
                "text": text,
                "text_hash": text_hash,
                "meta": meta,
            }
        )

        n_refs = 0
        if not refs or (force or force_references):
            refs = self._extract_reference_records(text, max_refs=max_refs or self.max_refs, 
                                                    require_arxiv_id=True)
            references_path = str(paper_dir / "references.json")
            self._write_references_json(Path(references_path), refs)
            refs = [asdict(r) for r in refs]
            n_refs = store.replace_references(arxiv_id, refs)

        # 2) Create run row early (so we can attach events/errors)
        paper_run = store.create_run(
            paper_id=arxiv_id,
            run_type="paper_import",
            variant="fresh",
            stats={"cached": False},
            config={}
        )
        store.add_event(run_id=paper_run.id, stage=self.name, event_type="info", message="starting import")

       # 6) Construct node
        node = PaperNode(
            arxiv_id=arxiv_id or paper_id,
            role=role or "unknown",
            source=source,
            url=pdf_url,
            pdf_path=str(local_pdf_path) if local_pdf_path else (str(paper_dir / "paper.pdf") if (paper_dir / "paper.pdf").exists() else None),
            meta={
                "paper_id": paper_id,
                "text_hash": text_hash,
                "references_path": references_path,
                "reference_count": len(refs),
                "ref_arxiv_ids": [r.get("arxiv_id") for r in refs if r.get("arxiv_id") ],
                **meta,
            },
        )

        raw = {
            "paper_id": paper_id,
            "arxiv_id": arxiv_id,
            "pdf_url": pdf_url,
            "pdf_path": str(paper_dir / "paper.pdf") if (paper_dir / "paper.pdf").exists() else None,
            "text_hash": text_hash,
            "reference_count": len(refs),
        }

        # 5) Upsert references (v0 placeholder)

        store.add_event(
            run_id=paper_run.id,
            stage="paper_import",
            event_type="info",
            message=f"import complete: {len(text)} chars, {len(refs)} refs",
            data={"chars": len(text), "n_refs": len(refs), "pdf_path": str(pdf_path)},
        )
        return PaperImportResult(
            node=node,
            text=text,
            text_hash=text_hash,
            pdf_path=raw["pdf_path"],
            references_path=references_path,
            references=refs,
            raw=raw,
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
            doc_dict = {
                "id": paper_id,
                "external_id": arxiv_id or paper_id,
                "url": pdf_url,
                "pdf_path": pdf_path,
                "text": text,
                "text_hash": text_hash,
                "meta": meta,
            }
            self.paper_store.upsert_paper(paper_id, doc_dict)
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

    def _download_pdf(self, url: str, out_path: Path) -> None:
        tmp = out_path.with_suffix(".pdf.part")
        headers = {"User-Agent": self.user_agent}
        with requests.get(url, stream=True, timeout=self.timeout_s, headers=headers) as r:
            r.raise_for_status()
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        f.write(chunk)
        tmp.replace(out_path)

    def _extract_text(self, pdf_path: str) -> str:
        converter = PDFConverter()
        return converter.convert_pdf_to_text(Path(pdf_path))

    # -------------------------
    # References
    # -------------------------

    def _write_references_json(self, path: Path, refs: List[PaperReferenceRecord]) -> None:
        # Provider compatibility: "arxiv_id" key
        items = []
        for rr in refs:
            d = asdict(rr)
            # ensure key exists and stringy
            d["arxiv_id"] = d.get("arxiv_id") or None
            items.append(d)
        path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

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
