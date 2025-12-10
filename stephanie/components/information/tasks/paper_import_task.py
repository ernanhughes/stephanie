# stephanie/components/information/tasks/paper_import_task.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from stephanie.components.information.data import PaperNode
from stephanie.utils.hash_utils import hash_text
from stephanie.tools.pdf_tool import PDFConverter
from stephanie.tools.arxiv_tool import fetch_arxiv_metadata  # NEW

log = logging.getLogger(__name__)


@dataclass
class PaperImportResult:
    node: PaperNode
    text: str
    raw: Dict[str, Any]


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
    """

    def __init__(
        self,
        papers_root: Path,
    ) -> None:
        self.papers_root = Path(papers_root)

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
        Import a single paper and return (PaperNode, text).

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
            key = self._derive_arxiv_id_from_url(url) or self._sanitize_filename(url or "paper")

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
            await self._download_pdf(source_url, pdf_local_path)

        log.info("[PaperImportTask] Importing PDF %s for key=%s", pdf_local_path, key)

        # ------------------------------------------------------------------
        # 2) Validate and extract text (DocumentLoader-style)
        # ------------------------------------------------------------------
        if not PDFConverter.validate_pdf(str(pdf_local_path)):
            raise RuntimeError(f"Invalid PDF format for {pdf_local_path}")

        text = PDFConverter.pdf_to_text(str(pdf_local_path))
        text_hash = hash_text(text or "")

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
        # 4) Build PaperNode + raw metadata
        # ------------------------------------------------------------------
        metadata: Dict[str, Any] = {
            "source_url": source_url,
            "key": key,
        }

        if arxiv_meta:
            metadata["arxiv"] = arxiv_meta

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
            },
        )

        raw = {
            "pdf_path": str(pdf_local_path),
            "source_url": source_url,
            "arxiv_metadata": arxiv_meta,
        }

        return PaperImportResult(node=node, text=text, raw=raw)

    # ------------------------------------------------------------------ #
    def _download_pdf(self, url: str, dest_path: Path) -> None:
        """
        Download a PDF from URL to dest_path, similar in spirit to
        DocumentLoaderAgent (requests.get(..., stream=True)).
        """
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
