from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from stephanie.tools.base_tool import BaseTool
from stephanie.scoring.scorable import Scorable

from stephanie.components.information.data import DocumentElement
from stephanie.tools.pdf_tool import PDFConverter  # your existing PDF utility
from stephanie.tools.paddle_ocr_tool import PaddleOcrTool  # optional backend

log = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None


@dataclass
class PdfElementsToolCfg:
    enable_figures: bool = True
    enable_ocr: bool = False
    ocr_on_low_text_ratio: float = 0.15  # trigger OCR if too many pages have little text
    min_image_area_frac: float = 0.02    # ignore tiny icons/logos


class PdfElementsTool(BaseTool):
    """
    Extracts DocumentElements from PDFs:
      - figures/images via PyMuPDF
      - optional OCR-derived text blocks via PaddleOcrTool
    """

    name = "pdf_elements"

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.pdf = PDFConverter(cfg.get("pdf", {}), memory, container, logger)

        self.cfg_obj = PdfElementsToolCfg(
            enable_figures=bool(cfg.get("enable_figures", True)),
            enable_ocr=bool(cfg.get("enable_ocr", False)),
            ocr_on_low_text_ratio=float(cfg.get("ocr_on_low_text_ratio", 0.15)),
            min_image_area_frac=float(cfg.get("min_image_area_frac", 0.02)),
        )

        # Optional OCR backend
        self.ocr_backend = None
        if self.cfg_obj.enable_ocr:
            self.ocr_backend = PaddleOcrTool(cfg.get("paddle_ocr", {}), memory, container, logger)

    async def apply(self, scorable: Scorable, context: Dict[str, Any]) -> Scorable:
        pdf_path = context.get("pdf_path") or (scorable.meta or {}).get("pdf_path") or scorable.text
        if not pdf_path:
            return scorable

        elements = await self.extract_elements(pdf_path=str(pdf_path), context=context)

        meta = scorable.meta
        meta.setdefault("elements", {})
        meta["elements"][self.name] = [e.__dict__ if hasattr(e, "__dict__") else e for e in elements]
        return scorable

    async def extract_elements(self, *, pdf_path: str, context: Dict[str, Any]) -> List[DocumentElement]:
        elements: List[DocumentElement] = []

        # 1) figures/images
        if self.cfg_obj.enable_figures and fitz is not None:
            try:
                elements.extend(self._extract_figures(pdf_path))
            except Exception:
                log.warning("PdfElementsTool figure extraction failed: %s", pdf_path, exc_info=True)

        # 2) optional OCR (gated)
        if self.cfg_obj.enable_ocr and self.ocr_backend is not None:
            if await self._should_run_ocr(pdf_path):
                page_images = self.pdf.render_to_images(pdf_path)  # reuse your existing renderer
                for img_path in page_images:
                    s = Scorable(text="", meta={"image_path": img_path})
                    s = await self.ocr_backend.apply(s, {"image_path": img_path})
                    lines = ((s.meta or {}).get("ocr") or {}).get(self.ocr_backend.name, []) or []
                    elements.extend(self._elements_from_ocr_lines(lines, image_path=img_path))

        return elements

    async def _should_run_ocr(self, pdf_path: str) -> bool:
        # Use per-page text density as cheap proxy
        try:
            page_texts = self.pdf.extract_page_texts(pdf_path)
            if not page_texts:
                return True
            low = 0
            for t in page_texts:
                if not t or len(t.strip()) < 30:
                    low += 1
            ratio = low / max(1, len(page_texts))
            return ratio >= self.cfg_obj.ocr_on_low_text_ratio
        except Exception:
            return True

    def _extract_figures(self, pdf_path: str) -> List[DocumentElement]:
        doc = fitz.open(pdf_path)
        out: List[DocumentElement] = []
        for page_index in range(len(doc)):
            page = doc[page_index]
            rect = page.rect
            page_area = float(rect.width * rect.height) if rect else 1.0

            # get_images(full=True) gives image xrefs
            for img in page.get_images(full=True) or []:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                w, h = pix.width, pix.height
                area_frac = float(w * h) / max(page_area, 1.0)

                if area_frac < self.cfg_obj.min_image_area_frac:
                    continue

                # Persisting image bytes is optional; you likely want paths.
                # For now, store metadata only (or wire to your file_utils to save).
                out.append(
                    DocumentElement(
                        element_type="figure",
                        page_start=page_index + 1,
                        page_end=page_index + 1,
                        text="",
                        bbox=None,
                        meta={"xref": xref, "width": w, "height": h, "area_frac": area_frac},
                    )
                )
        doc.close()
        return out

    def _elements_from_ocr_lines(self, lines: List[Dict[str, Any]], *, image_path: str) -> List[DocumentElement]:
        out: List[DocumentElement] = []
        for ln in lines:
            txt = (ln.get("text") or "").strip()
            if not txt:
                continue
            out.append(
                DocumentElement(
                    element_type="ocr_text",
                    page_start=None,
                    page_end=None,
                    text=txt,
                    bbox=ln.get("bbox"),
                    meta={"score": ln.get("score"), "source": ln.get("source"), "image_path": image_path},
                )
            )
        return out
