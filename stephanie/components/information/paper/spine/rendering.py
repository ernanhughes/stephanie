from pathlib import Path
from typing import List

from stephanie.tools.pdf_tool import PDFConverter


def render_pdf_to_images(
    pdf_path: Path,
    *,
    output_dir: Path,
    cfg: dict,
) -> List[Path]:
    """
    Render PDF pages to images using shared PDF tool.
    """
    converter = PDFConverter(cfg, memory=None, container=None, logger=None)
    return converter.render_to_images(str(pdf_path), output_dir=output_dir)
