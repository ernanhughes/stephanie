# stephanie/components/information/tasks/paper_import_task.py
from __future__ import annotations

import logging
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

import requests

from stephanie.components.information.data import PaperNode, ReferenceRecord
from stephanie.utils.hash_utils import hash_text
from stephanie.tools.pdf_tool import PDFConverter
from stephanie.tools.arxiv_tool import fetch_arxiv_metadata
from dataclasses import asdict

log = logging.getLogger(__name__)

ARXIV_INLINE_PATTERN = re.compile(
    r"arxiv[: ]\s*(\d{4}\.\d{4,5})",
    re.IGNORECASE,
)

ARXIV_URL_PATTERN = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})",
    re.IGNORECASE,
)

@dataclass
class PaperImportResult:
    node: PaperNode
    text: str
    raw: Dict[str, Any]
    references: Optional[List[ReferenceRecord]] = None


class PaperImportTask:
    """
    Paper import task that works like your DocumentLoader:

    - If given a URL or arxiv_id:
        * download the PDF into papers_root/<key>/paper.pdf
        * validate it with PDFConverter.validate_pdf
        * extract text with PDFConverter.pdf_to_text
        * (if we know the arxiv_id) fetch metadata from arXiv API

    - If given a local pdf_path:
        * validate + extract text directly from that path

    It returns:
        - PaperNode (with arxiv_id/key, role, pdf_path, text_hash, metadata)
        - text (full extracted text)
        - raw (debug metadata dict)
        - references (list[ReferenceRecord])
    """

    def __init__(
        self,
        papers_root: Path,
    ) -> None:
        self.papers_root = Path(papers_root)
        self.max_refs = 256

    # ------------------------------------------------------------------ #
    async def run(
        self,
        *,
        arxiv_id: Optional[str] = None,
        url: Optional[str] = None,
        pdf_path: Optional[Path] = None,
        role: str = "root",
    ) -> PaperImportResult:
        """
        Import a single paper and return (PaperNode, text, references).

        You can call this with:
            - arxiv_id="2501.01234"
            - url="https://arxiv.org/pdf/2501.01234.pdf"
            - pdf_path=Path("/path/to/local.pdf")
        """

        if not (arxiv_id or url or pdf_path):
            raise ValueError("One of arxiv_id, url, pdf_path must be provided")

        # Decide the "key" we use on disk and in the graph
        if arxiv_id:
            key = arxiv_id
        elif pdf_path is not None:
            key = pdf_path.stem
        else:
            key = self._derive_arxiv_id_from_url(url) or self._sanitize_filename(
                url or "paper"
            )

        paper_dir = self.papers_root / (key or "unknown")
        paper_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # 1) Resolve or download the PDF to a local path
        # ------------------------------------------------------------------
        if pdf_path is not None:
            # Use the provided local file
            pdf_local_path = Path(pdf_path)
            source_url = url  # may be None
        else:
            # Build a URL to download
            if url is None and arxiv_id:
                # Default arXiv PDF URL pattern
                url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            source_url = url
            if not source_url:
                raise ValueError("Could not determine URL for paper import")

            pdf_local_path = paper_dir / "paper.pdf"
            self._download_pdf(source_url, pdf_local_path)

        log.info("[PaperImportTask] Importing PDF %s for key=%s", pdf_local_path, key)

        # ------------------------------------------------------------------
        # 2) Validate and extract text (DocumentLoader-style)
        # ------------------------------------------------------------------
        if not PDFConverter.validate_pdf(str(pdf_local_path)):
            raise RuntimeError(f"Invalid PDF format for {pdf_local_path}")

        text = PDFConverter.pdf_to_text(str(pdf_local_path))
        text_hash = hash_text(text or "")
        
        # ------------------------------------------------------------------
        # 2b) Extract references (block + arXiv IDs)
        # ------------------------------------------------------------------
        references = []
        ref_lines = []
        ref_arxiv_ids = []
        raw = []

        if role == "root":
            ref_info = self._extract_references_from_text(
                text=text,
                root_key=key,
                max_refs=int(self.cfg.get("max_refs", 256)) if hasattr(self, "cfg") else 256,
            )
            references = ref_info["records"]
            ref_lines = ref_info["lines"]
            ref_arxiv_ids = ref_info["arxiv_ids"]
            raw = [asdict(r) for r in ref_info["records"]]

            log.info(
                "[PaperImportTask] Extracted %d references for root paper %s",
                len(references),
                key,
            )
        else:
            log.info(
                "[PaperImportTask] Skipping reference extraction for non-root paper %s (role=%s)",
                key,
                role,
            )
        # ------------------------------------------------------------------
        # 3) Optional: fetch arXiv metadata (title, summary, authors)
        # ------------------------------------------------------------------
        arxiv_meta: Optional[Dict[str, Any]] = None
        meta_id: Optional[str] = None

        # Prefer explicit arxiv_id; else try to derive from URL.
        if arxiv_id:
            meta_id = arxiv_id
        elif source_url:
            meta_id = self._derive_arxiv_id_from_url(source_url)

        if meta_id:
            try:
                arxiv_meta = fetch_arxiv_metadata(meta_id)
            except Exception as e:
                log.warning(
                    "[PaperImportTask] Failed to fetch arXiv metadata for %s: %s",
                    meta_id,
                    e,
                )

        # ------------------------------------------------------------------
        # 4) Extract references from the full text and persist them
        # ------------------------------------------------------------------
        references: List[ReferenceRecord] = self._extract_reference_records(
            text or "", key, max_refs=100
        )
        references_path: Optional[Path] = None
        if references:
            references_path = self._persist_references_json(paper_dir, references)

        # ------------------------------------------------------------------
        # 5) Build PaperNode + raw metadata
        # ------------------------------------------------------------------
        metadata: Dict[str, Any] = {
            "source_url": source_url,
            "key": key,
            "reference_arxiv_ids": ref_arxiv_ids,
            "reference_count": len(references),
        }

        if arxiv_meta:
            metadata["arxiv"] = arxiv_meta
        if references_path is not None:
            metadata["references_path"] = str(references_path)
            metadata["references_count"] = len(references)


        ref_arxiv_ids = sorted(
            {r.arxiv_id for r in references if r.arxiv_id}
        )
        node = PaperNode(
            arxiv_id=key,
            role=role,
            title=arxiv_meta["title"] if arxiv_meta else None,
            summary=arxiv_meta["summary"] if arxiv_meta else None,
            published_date=arxiv_meta["published"] if arxiv_meta else None,
            authors=arxiv_meta["authors"] if arxiv_meta else [],
            url=(arxiv_meta.get("url") if arxiv_meta else source_url),
            pdf_path=pdf_local_path,
            text_hash=text_hash,
            meta={
                "pdf_metadata": metadata,
                "references_block": raw,
                "references_lines": ref_lines,
                "reference_arxiv_ids": ref_arxiv_ids,
                "reference_count": len(references),
            },
        )

        raw = {
            "pdf_path": str(pdf_local_path),
            "source_url": source_url,
            "arxiv_metadata": arxiv_meta,
            "references": references,        # List[ReferenceRecord]
            "reference_lines": ref_lines,    # For debugging / display
            "arxiv_ids": ref_arxiv_ids,
        }

        log.info(
            "[PaperImportTask] Imported paper key=%s, text_length=%d, references=%d",
            key,
            len(text or ""),
            len(references),
        )

        return PaperImportResult(node=node, text=text, raw=raw, references=references)

    # ------------------------------------------------------------------ #
    def _download_pdf(self, url: str, dest_path: Path) -> None:
        """
        Download a PDF from URL to dest_path, similar in spirit to
        DocumentLoaderAgent (requests.get(. stream=True)).
        """
        # Check if file already exists and has reasonable size (> 1KB)
        if dest_path.exists() and dest_path.stat().st_size > 1024:
            log.info("[PaperImportTask] File already exists with reasonable size, skipping download: %s", dest_path)
            return
            
        log.info("[PaperImportTask] Downloading %s -> %s", url, dest_path)
        resp = requests.get(url, stream=True, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code} while downloading {url}")

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    
    # ------------------------------------------------------------------ #
    def _derive_arxiv_id_from_url(self, url: Optional[str]) -> Optional[str]:
        """
        Pull an arxiv-like id out of a typical arxiv PDF URL, if possible.
        Examples:
            https://arxiv.org/pdf/2505.08827.pdf  -> 2505.08827
            https://arxiv.org/pdf/2505.08827      -> 2505.08827
        """
        if not url:
            return None
        m = re.search(r"/(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?$", url)
        return m.group(1) if m else None

    def _sanitize_filename(self, name: str) -> str:
        """
        Create a filesystem-safe filename from a URL or title.
        Similar to DocumentLoaderAgent.sanitize_filename.
        """
        # Strip protocol if present
        name = re.sub(r"^https?://", "", name)
        # Replace forbidden characters
        name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
        if not name:
            name = "document"
        return name[:100]


    def _persist_references_json(
        self,
        paper_dir: Path,
        references: List[ReferenceRecord],
    ) -> Path:
        """
        Save references as JSON so a ReferenceProvider can load them later.

        File: papers_root/<key>/references.json
        """
        refs_path = paper_dir / "references.json"
        try:
            payload = [
                {
                    "arxiv_id": r.arxiv_id,
                    "doi": r.doi,
                    "title": r.title,
                    "year": r.year,
                    "url": r.url,
                    "raw_citation": r.raw_citation,
                }
                for r in references
            ]
            refs_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            log.warning(
                "[PaperImportTask] Failed to write references.json in %s: %s",
                paper_dir,
                e,
            )
        return refs_path

    # ------------------------------------------------------------------ #
    # Reference extraction
    # ------------------------------------------------------------------ #

    def _slice_references_block(self, text: str) -> Optional[str]:
        """
        Try to slice out the References/Bibliography section.

        This is a *best effort* heuristic. If it fails, we fall back to a
        global arXiv-ID regex sweep.
        """
        if not text:
            return None

        lower = text.lower()

        # Look for common headings near the end of the document.
        # We use rfind() so we get the *last* occurrence.
        headings = [
            "\nreferences\n",
            "\nreferences\r\n",
            "\n# references",
            "\n## references",
            "\nbibliography\n",
            "\nreferences\n\n",
        ]

        start_idx = -1
        for h in headings:
            idx = lower.rfind(h)
            if idx != -1:
                start_idx = idx + len(h)
                break

        if start_idx == -1:
            return None

        # Optionally trim off appendices if present
        tail = text[start_idx:]
        for stopper in ["\nappendix", "\nacknowledgments", "\nacknowledgements"]:
            stop_idx = tail.lower().find(stopper)
            if stop_idx != -1:
                tail = tail[:stop_idx]
                break

        return tail.strip() or None

    def _extract_arxiv_ids_from_text(self, text: str) -> list[str]:
        """
        Fallback: scan the *entire* document for arXiv IDs like
        'arxiv:2504.21318' or 'https://arxiv.org/abs/2110.14168'.

        Returns a sorted, de-duplicated list of bare IDs like '2504.21318'.
        """
        if not text:
            return []

        ids: set[str] = set()

        for pat in (ARXIV_INLINE_PATTERN, ARXIV_URL_PATTERN):
            for m in pat.findall(text):
                ids.add(m.strip())

        if not ids:
            return []

        # Log once so you can see this kicking in
        log.info(
            "[PaperImportTask] Fallback arXiv-ID extraction found %d ids: %s",
            len(ids),
            ", ".join(sorted(ids)),
        )

        return sorted(ids)


    def _extract_reference_records(
        self,
        text: str,
        root_key: str,
        max_refs: int = 256,
    ) -> list["ReferenceRecord"]:
        """
        Unified reference extractor.

        Strategy:
          1) Try to find a 'References' / 'Bibliography' heading near the end and
             parse numbered citations under it into ReferenceRecord objects.
          2) If that fails, fall back to a global arXiv-ID regex sweep and build
             minimal ReferenceRecord entries from those IDs.
        """
        if not text:
            return []

        lower = text.lower()
        start_search = int(len(lower) * 0.66)
        block = lower[start_search:]

        heading_match = None
        # Look for the last occurrence of a references-like heading
        for pattern in (r"(?mi)^\s*references\s*$", r"(?mi)^\s*bibliography\s*$"):
            matches = list(re.finditer(pattern, block))
            if matches:
                heading_match = matches[-1]
                break

        if heading_match:
            # --- 1) References block path ---
            abs_start = start_search + heading_match.end()
            refs_block = text[abs_start:]

            lines = [ln.rstrip() for ln in refs_block.splitlines()]
            entries: list[str] = []
            buf: list[str] = []

            def flush_buf():
                if buf:
                    merged = " ".join(buf).strip()
                    if len(merged) > 30:  # drop tiny junk lines
                        entries.append(merged)
                buf.clear()

            # Group lines into individual citations
            for ln in lines:
                stripped = ln.strip()
                if not stripped:
                    flush_buf()
                    continue

                # Start of a new numbered ref: [n], "n." or "n)"
                if re.match(r"^\s*(\[\d+\]|\d+[\.\)]\s+)", stripped) and buf:
                    flush_buf()
                    buf.append(stripped)
                else:
                    buf.append(stripped)

            flush_buf()

            if len(entries) > max_refs:
                entries = entries[:max_refs]

            records: list[ReferenceRecord] = []
            for entry in entries:
                # arXiv ID inside the citation
                arxiv_match = re.search(
                    r"(?:arxiv[: ]*)?(\d{4}\.\d{4,5})(?:v\d+)?",
                    entry,
                    flags=re.IGNORECASE,
                )
                arxiv_id = arxiv_match.group(1) if arxiv_match else None

                # DOI
                doi_match = re.search(r"\b10\.\d{4,9}/\S+\b", entry)
                doi = doi_match.group(0) if doi_match else None

                # Year
                year_match = re.search(r"\b(19|20)\d{2}\b", entry)
                year = int(year_match.group(0)) if year_match else None

                # URL
                url_match = re.search(r"https?://\S+", entry)
                url = url_match.group(0) if url_match else None
                if not url and arxiv_id:
                    url = f"https://arxiv.org/abs/{arxiv_id}"

                records.append(
                    ReferenceRecord(
                        arxiv_id=arxiv_id,
                        doi=doi,
                        title=None,
                        year=year,
                        url=url,
                        raw_citation=entry,
                        source="unknown",
                        raw={},
                    )
                )

            log.info(
                "[PaperImportTask] Extracted %d references for key=%s via References block",
                len(records),
                root_key,
            )
            return records

        # --- 2) Fallback: no explicit references block, use arXiv-ID sweep ---
        arxiv_ids = self._extract_arxiv_ids_from_text(text)
        if not arxiv_ids:
            log.info(
                "[PaperImportTask] No references found for key=%s (no block, no arxiv IDs)",
                root_key,
            )
            return []

        records: list[ReferenceRecord] = []
        for arxiv_id in arxiv_ids[:max_refs]:
            url = f"https://arxiv.org/abs/{arxiv_id}"
            records.append(
                ReferenceRecord(
                    arxiv_id=arxiv_id,
                    doi=None,
                    title=None,
                    year=None,
                    url=url,
                    raw_citation=f"arXiv:{arxiv_id}",
                    source="unknown",
                    raw={},
                )
            )

        log.info(
            "[PaperImportTask] Extracted %d references for key=%s via arxiv-ID fallback",
            len(records),
            root_key,
        )
        return records

    def _extract_references_from_text(
        self,
        text: str,
        root_key: str,
        max_refs: int = 256,
    ) -> Dict[str, Any]:
        """
        Backwards-compatible wrapper around `_extract_reference_records`.

        Returns:
          {
            "records": List[ReferenceRecord],
            "lines":   List[str],       # raw citations
            "arxiv_ids": List[str],     # unique arxiv IDs
          }
        """
        records = self._extract_reference_records(text, root_key, max_refs=max_refs)

        return {
            "records": records,
            "lines": [r.raw_citation for r in records if r.raw_citation],
            "arxiv_ids": [r.arxiv_id for r in records if r.arxiv_id],
        }
