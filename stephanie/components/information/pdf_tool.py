from __future__ import annotations

import os
import io
import tempfile
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable
from urllib.parse import urlparse

import requests  # if you don't want this, you can inject a downloader

from stephanie.tools.pdf_tool import PDFConverter

log = logging.getLogger(__name__)


# ---- Interfaces / DTOs ------------------------------------------------------


@dataclass
class PdfSource:
    """Represents a single PDF to be processed."""
    id: str
    path: Path                # local path to the PDF file
    origin: str               # "file" | "directory" | "url"
    original_input: str       # the original path/url the user passed


@dataclass
class PdfImportResult:
    """Result of importing a single PDF via your PDF tool."""
    source: PdfSource
    text: str                 # full text or concatenated pages
    metadata: Dict[str, Any]  # whatever your PDF tool exposes


@dataclass
class PdfComparisonResult:
    """Comparison between PDF text and some reference text."""
    source: PdfSource
    pdf_text_len: int
    reference_text_len: int
    overlap_ratio: float      # very simple similarity
    notes: str


# ---- PDF Tool Adapter (hook to your existing tool) -------------------------


class PdfTool:
    """
    Thin wrapper around your existing PDF import tool.
    Replace the internals of `import_pdf` to call your real code.
    """

    def import_pdf(self, path: Path) -> PdfImportResult:
        # TODO: integrate with your real PDF importer.
        #
        # For now, placeholder that just reads bytes (if text-based)
        # and returns dummy metadata.
        try:
            text = PDFConverter.pdf_to_text(path)
        except UnicodeDecodeError:
            text = ""

        source = PdfSource(
            id=str(path),
            path=path,
            origin="file",
            original_input=str(path),
        )

        return PdfImportResult(
            source=source,
            text=text,
            metadata={"dummy": True},
        )


# ---- First Task: Import + Compare ------------------------------------------


class ImportPdfTask:
    """
    Task 1:
    - Accept directory, file, or URL
    - Resolve to one or more local PDF files
    - Import each via PDF tool
    - Optionally compare imported text to a reference text
    """

    def __init__(self, pdf_tool: Optional[PdfTool] = None) -> None:
        self.pdf_tool = pdf_tool or PdfTool()

    # --- Public entrypoint ---------------------------------------------------

    def run(
        self,
        input_path_or_url: str,
        reference_text: Optional[str] = None,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """
        Main entrypoint for the task.

        Returns:
            {
              "imports": [PdfImportResult, ...],
              "comparisons": [PdfComparisonResult, ...]  # if reference_text provided
            }
        """
        pdf_sources = list(self._resolve_sources(input_path_or_url, recursive=recursive))
        log.info("Resolved %d PDF source(s) from %r", len(pdf_sources), input_path_or_url)

        imports: List[PdfImportResult] = []
        comparisons: List[PdfComparisonResult] = []

        for src in pdf_sources:
            result = self._import_single(src)
            imports.append(result)

            if reference_text is not None:
                comparison = self._compare_to_text(result, reference_text)
                comparisons.append(comparison)

        return {
            "imports": imports,
            "comparisons": comparisons,
        }

    # --- Source resolution ---------------------------------------------------

    def _resolve_sources(self, input_path_or_url: str, recursive: bool) -> Iterable[PdfSource]:
        """
        Detect whether input is a directory, file, or URL, and yield PdfSource objects.
        """
        # Heuristic: treat as URL if it has a scheme
        parsed = urlparse(input_path_or_url)
        is_url = bool(parsed.scheme and parsed.netloc)

        if is_url:
            yield self._download_url(input_path_or_url)
            return

        path = Path(input_path_or_url)

        if path.is_dir():
            yield from self._iter_dir(path, recursive=recursive, original_input=input_path_or_url)
        elif path.is_file():
            if path.suffix.lower() == ".pdf":
                yield PdfSource(
                    id=str(path),
                    path=path,
                    origin="file",
                    original_input=input_path_or_url,
                )
            else:
                log.warning("Input file %s is not a PDF; skipping.", path)
        else:
            raise FileNotFoundError(f"Input path does not exist: {input_path_or_url}")

    def _iter_dir(self, directory: Path, recursive: bool, original_input: str) -> Iterable[PdfSource]:
        if recursive:
            for p in directory.rglob("*.pdf"):
                yield PdfSource(
                    id=str(p),
                    path=p,
                    origin="directory",
                    original_input=original_input,
                )
        else:
            for p in directory.glob("*.pdf"):
                yield PdfSource(
                    id=str(p),
                    path=p,
                    origin="directory",
                    original_input=original_input,
                )

    def _download_url(self, url: str) -> PdfSource:
        """
        Download a PDF from URL to a temp file and return as PdfSource.
        """
        log.info("Downloading PDF from URL: %s", url)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower():
            log.warning("URL does not look like a PDF (Content-Type=%s)", content_type)

        fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="imported_")
        os.close(fd)

        with open(tmp_path, "wb") as f:
            f.write(resp.content)

        path = Path(tmp_path)
        return PdfSource(
            id=str(path),
            path=path,
            origin="url",
            original_input=url,
        )

    # --- Import + Compare ----------------------------------------------------

    def _import_single(self, src: PdfSource) -> PdfImportResult:
        """
        Wraps the PDF tool to keep the PdfSource info.
        """
        log.info("Importing PDF: %s", src.path)
        result = self.pdf_tool.import_pdf(src.path)

        # Make sure the result carries the original source metadata
        result.source = src
        return result

    def _compare_to_text(self, pdf_result: PdfImportResult, reference_text: str) -> PdfComparisonResult:
        """
        Simple baseline comparison between PDF text and provided reference text.
        Replace this with MRQ/EBT or more advanced metrics later.
        """
        pdf_text = pdf_result.text or ""
        ref = reference_text or ""

        pdf_len = len(pdf_text)
        ref_len = len(ref)

        if not pdf_text or not ref:
            overlap_ratio = 0.0
            notes = "One of the texts is empty; overlap_ratio set to 0.0."
        else:
            # Naive overlap: fraction of shared tokens
            pdf_tokens = set(pdf_text.split()) 
            ref_tokens = set(ref.split())
            common = pdf_tokens & ref_tokens
            union = pdf_tokens | ref_tokens
            overlap_ratio = len(common) / max(1, len(union))
            notes = f"Overlap computed on token sets: |common|={len(common)}, |union|={len(union)}."

        return PdfComparisonResult(
            source=pdf_result.source,
            pdf_text_len=pdf_len,
            reference_text_len=ref_len,
            overlap_ratio=overlap_ratio,
            notes=notes,
        )
