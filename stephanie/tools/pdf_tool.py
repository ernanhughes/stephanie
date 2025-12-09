# stephanie/tools/pdf_tool.py
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union
from urllib.parse import urlparse

import requests
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class PdfSource:
    """
    Represents a single PDF to be processed.
    This is the *origin* metadata only.
    """
    id: str                # unique id for this source (you can swap to your UniversalID later)
    path: Path             # local path to the PDF file
    origin: str            # "file" | "directory" | "url" | "blob"
    original_input: str    # the original path/url/blob-id the user passed


@dataclass
class PdfImportResult:
    """
    Result of importing a single PDF.

    - `text` is the full extracted text.
    - `metadata` is free-form; you can extend later.
    """
    source: PdfSource
    text: str
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Low-level: validation + text extraction
# ---------------------------------------------------------------------------

class PDFConverter:
    """
    Thin wrapper around pdfminer.
    """

    @staticmethod
    def validate_pdf(file_path: Union[str, Path]) -> bool:
        """
        Uses `pdfminer` to validate that the file is parsable as a PDF.

        :param file_path: Path to the PDF file
        :return: True if parsable PDF, False otherwise
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.stat().st_size < 1000:
            # Tiny files are often junk; heuristically reject
            log.warning("[validate_pdf] File %s is very small; treating as invalid PDF.", path)
            return False

        try:
            # Just try reading the first page
            _ = extract_text(str(path), maxpages=1)
            return True
        except PDFSyntaxError:
            return False
        except Exception as e:
            log.exception("[validate_pdf] Exception while parsing %s: %s", file_path, e)
            return False

    @staticmethod
    def pdf_to_text(file_path: Union[str, Path]) -> str:
        """
        Extracts plain text from a PDF file.

        :param file_path: Path to the PDF file
        :return: Extracted text (cleaned)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            text = extract_text(str(file_path))
            # Remove null characters and trim
            clean_text = text.replace("\x00", "")
            return clean_text.strip()
        except PDFSyntaxError as e:
            raise ValueError(f"Error parsing PDF file: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF {file_path}: {e}") from e


_converter = PDFConverter()  # module-level singleton; easy to swap in tests


# ---------------------------------------------------------------------------
# Helper: URL / blob handling
# ---------------------------------------------------------------------------

def _is_url(s: str) -> bool:
    try:
        parsed = urlparse(s)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False


def _download_pdf(url: str) -> Path:
    """
    Download a PDF from URL to a temp file and return the local path.
    """
    log.info("[pdf_tool] Downloading PDF from URL: %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower():
        log.warning("[pdf_tool] URL %s does not look like a PDF (Content-Type=%s)", url, content_type)

    fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="imported_")
    os.close(fd)

    with open(tmp_path, "wb") as f:
        f.write(resp.content)

    return Path(tmp_path)


# If/when you support blobs or DB-backed PDFs, add helpers like:
#
# def _load_pdf_from_blob(blob_id: str, db_session) -> Path:
#     ...
#     return Path(tmp_path)


# ---------------------------------------------------------------------------
# Public API: single PDF
# ---------------------------------------------------------------------------

def import_pdf_from_path(file_path: Union[str, Path]) -> PdfImportResult:
    """
    Import a single *local* PDF file:

        - validates it
        - extracts text
        - returns PdfImportResult

    Raises if the file is missing or unreadable.
    """
    path = Path(file_path)

    if not _converter.validate_pdf(path):
        raise ValueError(f"File {path} is not a valid PDF or cannot be parsed.")

    text = _converter.pdf_to_text(path)
    source = PdfSource(
        id=str(path),              # you can swap this for UniversalID later
        path=path,
        origin="file",
        original_input=str(path),
    )
    meta = {
        "size_bytes": path.stat().st_size,
    }
    return PdfImportResult(source=source, text=text, metadata=meta)


def import_pdf_from_url(url: str) -> PdfImportResult:
    """
    Import a single PDF from a URL:

        - downloads it to a temp file
        - validates + extracts text
        - returns PdfImportResult
    """
    tmp_path = _download_pdf(url)

    if not _converter.validate_pdf(tmp_path):
        raise ValueError(f"Downloaded URL {url} is not a valid PDF.")

    text = _converter.pdf_to_text(tmp_path)
    source = PdfSource(
        id=str(tmp_path),
        path=tmp_path,
        origin="url",
        original_input=url,
    )
    meta = {
        "size_bytes": tmp_path.stat().st_size,
    }
    return PdfImportResult(source=source, text=text, metadata=meta)


def import_pdf(input_path_or_url: Union[str, Path]) -> PdfImportResult:
    """
    Convenience wrapper:

      - If `input_path_or_url` looks like a URL -> treat as URL.
      - Else -> treat as local file path.

    Returns a single PdfImportResult.

    This is your "import from a file or URL" primitive.
    """
    s = str(input_path_or_url)
    if _is_url(s):
        return import_pdf_from_url(s)
    return import_pdf_from_path(s)


# ---------------------------------------------------------------------------
# Public API: directory of PDFs
# ---------------------------------------------------------------------------

def import_pdfs_from_directory(
    directory: Union[str, Path],
    recursive: bool = True,
) -> List[PdfImportResult]:
    """
    Import all PDFs in a directory (optionally recursive).

    Returns:
        List[PdfImportResult] (one per PDF that successfully parsed).
        Invalid PDFs are logged and skipped.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    pdf_paths: List[Path] = []
    if recursive:
        pdf_paths = list(dir_path.rglob("*.pdf"))
    else:
        pdf_paths = list(dir_path.glob("*.pdf"))

    results: List[PdfImportResult] = []
    for p in pdf_paths:
        try:
            result = import_pdf_from_path(p)
            # Adjust origin/original_input to reflect directory import
            result.source.origin = "directory"
            result.source.original_input = str(dir_path)
            results.append(result)
        except Exception as e:
            log.warning("[pdf_tool] Skipping PDF %s due to error: %s", p, e)

    return results


# ---------------------------------------------------------------------------
# Public API: generic entrypoints (what agents / scripts should use)
# ---------------------------------------------------------------------------

def import_any(
    input_path_or_url: Union[str, Path],
    recursive: bool = True,
) -> List[PdfImportResult]:
    """
    High-level entrypoint used by agents/scripts:

      - If `input_path_or_url` is a directory:
            -> import all PDFs (honouring `recursive`)
      - If it's a file:
            -> import that one PDF
      - If it looks like a URL:
            -> import from URL

    Always returns a list of PdfImportResult (0, 1, or many).
    """
    s = str(input_path_or_url)

    # URL case
    if _is_url(s):
        return [import_pdf_from_url(s)]

    path = Path(s)

    if path.is_dir():
        return import_pdfs_from_directory(path, recursive=recursive)
    elif path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File {path} is not a PDF.")
        return [import_pdf_from_path(path)]
    else:
        raise FileNotFoundError(f"Input path does not exist or is not a URL: {s}")
