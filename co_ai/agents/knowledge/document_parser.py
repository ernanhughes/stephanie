# co_ai/agents/document_parser.py
import fitz  # PyMuPDF for PDF handling
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
from loguru import logger


class DocumentParser:
    """
    A lightweight agent that parses academic papers (PDFs) into structured data.
    """

    def __init__(self):
        pass

    def parse(self, file_path: Union[str, Path]) -> Dict:
        """
        Main entry point to parse a document. Currently supports PDFs only.
        Returns structured content including text, figures, and metadata.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() == ".pdf":
            return self._parse_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _parse_pdf(self, path: Path) -> Dict:
        """
        Parses a PDF file into raw text and metadata like title, authors, abstract.
        Also extracts figures as pixmap objects (can be saved later).
        """
        try:
            doc = fitz.open(path)
        except Exception as e:
            logger.error(f"Failed to open PDF: {path} | Error: {e}")
            raise

        text_blocks = []
        figures = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_blocks.append(page.get_text())
            pix = page.get_pixmap()
            figures.append(pix)

        full_text = "\n".join(text_blocks).strip()

        return {
            "full_text": full_text,
            "figures": figures,
            "metadata": self._extract_metadata(full_text),
            "source_file": str(path),
        }

    def _extract_metadata(self, text: str) -> Dict:
        """
        Extracts basic metadata from the first few pages of the paper.
        This is a heuristic approach and can be replaced by more robust parsing.
        """
        title_match = re.search(r"^\s*(.+?)\n\s*\n", text[:500], flags=re.MULTILINE)
        title = title_match.group(1).strip() if title_match else None

        authors_match = re.search(
            r"(?:Authors?|Author[s:])(.*?)(?=\n\n)", text[:1000], re.DOTALL
        )
        authors = (
            [a.strip() for a in authors_match.group(1).split(",")]
            if authors_match
            else []
        )

        abstract_match = re.search(
            r"(Abstract|ABSTRACT).*?\n\n(.*?)\n\n", text, re.DOTALL
        )
        abstract = abstract_match.group(2).strip() if abstract_match else None

        return {
            "title": title,
            "authors": authors,
            "abstract": abstract,
        }