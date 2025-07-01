# pdf_tools.py
import os
from pathlib import Path
from typing import Union

from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError


class PDFConverter:
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
        if path.stat().st_size < 1000:  # You can keep this check
            return False

        try:
            _ = extract_text(str(path), maxpages=1)  # Just try reading the first page
            return True
        except PDFSyntaxError:
            return False
        except Exception as e:
            # Log and return False instead of crashing the pipeline
            print(f"[validate_pdf] Exception while parsing {file_path}: {e}")
            return False
        
    @staticmethod
    def pdf_to_text(file_path: Union[str, Path]) -> str:
        """
        Extracts plain text from a PDF file.

        :param file_path: Path to the PDF file
        :return: Extracted text
        :raises FileNotFoundError: if the file doesn't exist
        :raises ValueError: if the PDF is malformed
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            text = extract_text(str(file_path))
            clean_text = text.replace('\x00', '') 
            return clean_text.strip()
        except PDFSyntaxError as e:
            raise ValueError(f"Error parsing PDF file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {e}")
