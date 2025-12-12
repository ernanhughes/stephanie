# stephanie/components/information/agents/paper_spine_builder.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable
from stephanie.components.information.data import (
    attach_elements_to_sections,
    DocumentElement,
    BoundingBox,
)

log = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF: pip install pymupdf
except ImportError:  # pragma: no cover
    fitz = None
    log.warning(
        "PaperSpineBuilderAgent: PyMuPDF (fitz) is not installed; "
        "PDF → image rendering will be disabled."
    )


class PaperSpineBuilderAgent(BaseAgent):
    """
    v2: build a spine *and* extract page-level figures/images from the PDF.

    Inputs (context):
      - paper_sections: List[DocumentSection]          (required)
      - paper_elements: List[DocumentElement]          (optional, will be extended)
      - paper_pdf_path: str | Path                     (optional if arxiv_id given)
      - arxiv_id | paper_arxiv_id | root_arxiv_id      (for default PDF path)

    Outputs (context):
      - paper_elements: List[DocumentElement] (existing + figures)
      - paper_spine:    attached sections with elements
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg=cfg, memory=memory, container=container, logger=logger)
        self.papers_root = Path(self.cfg.get("papers_root", "data/papers"))
        self.page_image_root = Path(
            self.cfg.get("page_image_root", f"runs/paper_blogs/{self.run_id}")
        )
        self.figures_root = Path(self.cfg.get("figures_root", "runs/paper_figures"))
        self.max_pages: Optional[int] = self.cfg.get("max_pages")

        # minimum fraction of page area for us to consider an image a "figure"
        # (filters tiny logos / icons)
        self.min_figure_area_frac: float = float(self.cfg.get("min_figure_area_frac", 0.02))

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        sections = context.get("paper_sections") or []
        if not sections:
            log.warning("PaperSpineBuilderAgent: no paper_sections in context.")
            context["paper_spine"] = []
            return context

        # 1) Start from any elements already present (NER, metrics, etc.)
        elements: List[DocumentElement] = context.get("paper_elements") or []

        # 2) Try to extract figures/images from the PDF
        new_figures: List[DocumentElement] = self._extract_figures_from_pdf(context)
        if new_figures:
            log.info(
                "PaperSpineBuilderAgent: extracted %d figure elements from PDF.",
                len(new_figures),
            )
            elements.extend(new_figures)

        # 3) Update context and build spine
        context["paper_elements"] = elements
        spine = attach_elements_to_sections(sections=sections, elements=elements)
        context["paper_spine"] = spine

        log.info(
            "PaperSpineBuilderAgent: built spine with %d sections and %d total elements.",
            len(sections),
            len(elements),
        )
        return context

    # ----------------------------------------------------------------- helpers ---

    def _extract_figures_from_pdf(self, context: Dict[str, Any]) -> List[DocumentElement]:
        """
        Use PyMuPDF to pull out embedded images from each page, save them,
        and wrap them as DocumentElement(type='figure').

        This does NOT use OCR at all – it's purely structural extraction.
        """
        if fitz is None:
            return []

        arxiv_id = (
            context.get("arxiv_id")
            or context.get("paper_arxiv_id")
            or context.get("root_arxiv_id")
            or "unknown"
        )

        pdf_path = context.get("paper_pdf_path")
        if pdf_path:
            pdf_path = Path(pdf_path)
        else:
            pdf_path = self.papers_root / f"{arxiv_id}.pdf"

        if not pdf_path.exists():
            log.warning(
                "PaperSpineBuilderAgent: PDF not found for %s at %s",
                arxiv_id,
                pdf_path,
            )
            return []

        doc = fitz.open(str(pdf_path))
        figures_dir = self.figures_root / str(arxiv_id)
        figures_dir.mkdir(parents=True, exist_ok=True)

        elements: List[DocumentElement] = []

        num_pages = len(doc)
        if self.max_pages is not None:
            num_pages = min(num_pages, int(self.max_pages))

        for page_index in range(num_pages):
            page = doc.load_page(page_index)
            page_rect = page.rect
            page_area = float(page_rect.width * page_rect.height)
            page_num = page_index + 1

            # full=True gets all images, not just a subset
            images = page.get_images(full=True)
            if not images:
                continue

            for img_idx, img_info in enumerate(images):
                xref = img_info[0]  # image reference ID in the PDF

                # Get the rectangles where this image is drawn on the page
                rects = page.get_image_rects(xref)
                if not rects:
                    continue

                # Extract the raw image bytes
                img = doc.extract_image(xref)
                img_bytes = img.get("image")
                img_ext = img.get("ext", "png")

                # We'll create one element per rect, reusing the same image file.
                # Save image once:
                file_stem = f"{arxiv_id}_p{page_num:03d}_img{img_idx:02d}"
                img_filename = f"{file_stem}.{img_ext}"
                img_path = figures_dir / img_filename

                # Avoid re-writing if it already exists
                if not img_path.exists():
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)

                for rect_i, r in enumerate(rects):
                    # Filter out tiny images (icons / logos)
                    rect_area = float(r.width * r.height)
                    if page_area > 0 and rect_area / page_area < self.min_figure_area_frac:
                        continue

                    bbox = BoundingBox(
                        x1=float(r.x0),
                        y1=float(r.y0),
                        x2=float(r.x1),
                        y2=float(r.y1),
                    )
                    elem_id = f"{arxiv_id}:p{page_num}:img{img_idx:02d}:{rect_i}"

                    elements.append(
                        DocumentElement(
                            id=elem_id,
                            paper_id=str(arxiv_id),
                            page=page_num,
                            type="figure",          # v1: treat all as figures
                            bbox=bbox,
                            text=None,
                            latex=None,
                            markdown_table=None,
                            image_path=str(img_path),
                            caption=None,          # we can add caption later from text
                            meta={
                                "xref": xref,
                                "page_area": page_area,
                                "rect_area": rect_area,
                            },
                        )
                    )

        return elements


    # ------------------------------------------------------------ PDF → images ---

    def _render_pdf_to_images(self, arxiv_id: str, pdf_path: Path) -> List[Path]:
        """
        Render each page of the PDF into a PNG image and return list of paths.

        If PyMuPDF is not installed or the PDF doesn't exist, returns [].
        """
        if fitz is None:
            log.warning(
                "PaperSpineBuilderAgent: cannot render PDF '%s' → images "
                "(PyMuPDF not installed).",
                pdf_path,
            )
            return []

        if not pdf_path.exists():
            log.warning(
                "PaperSpineBuilderAgent: PDF not found for %s at %s",
                arxiv_id,
                pdf_path,
            )
            return []

        out_dir = self.page_image_root / arxiv_id
        out_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(pdf_path))
        page_paths: List[Path] = []

        num_pages = len(doc)
        if self.max_pages is not None:
            num_pages = min(num_pages, int(self.max_pages))

        for idx in range(num_pages):
            page = doc.load_page(idx)
            pix = page.get_pixmap()
            img_path = out_dir / f"{arxiv_id}_p{idx + 1:03d}.png"
            pix.save(str(img_path))
            page_paths.append(img_path)

        log.info(
            "PaperSpineBuilderAgent: rendered %d pages of %s to %s",
            len(page_paths),
            pdf_path,
            out_dir,
        )
        return page_paths

    # ------------------------------------------------------- OCR → elements -----

    async def _ocr_pages_to_elements(
        self,
        arxiv_id: str,
        page_image_paths: List[Path],
    ) -> List[DocumentElement]:
        """
        For each page image:
          - create a Scorable,
          - call PaddleOcrTool.apply(...),
          - read scorable.meta["ocr"]["paddle_ocr"],
          - convert each line to a DocumentElement(type="text_block").
        """

        if self.ocr_tool is None:
            return []

        elements: List[DocumentElement] = []

        for page_idx, img_path in enumerate(page_image_paths, start=1):
            # Build a minimal Scorable for the tool
            scorable = Scorable(
                id=f"{arxiv_id}:p{page_idx}",
                text=str(img_path),
                target_type="document_page",
                meta={"image_path": str(img_path)},
            )

            # Context for the tool: explicitly pass image_path
            tool_context: Dict[str, Any] = {"image_path": str(img_path)}

            # Run the tool (async)
            scorable = await self.ocr_tool.apply(scorable, tool_context)

            ocr_meta = scorable.meta.get("ocr", {})
            lines = ocr_meta.get(getattr(self.ocr_tool, "name", "paddle_ocr"), [])

            for line_idx, line in enumerate(lines):
                bbox_poly = line.get("bbox") or [[0.0, 0.0]] * 4
                xs = [float(p[0]) for p in bbox_poly]
                ys = [float(p[1]) for p in bbox_poly]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                elem_id = f"{arxiv_id}:p{page_idx}:line{line_idx}"
                text = line.get("text") or ""
                score = float(line.get("score", 1.0))

                elements.append(
                    DocumentElement(
                        id=elem_id,
                        paper_id=str(arxiv_id),
                        page=page_idx,
                        type="text_block",  # everything from OCR is a text_block v1
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        text=text,
                        meta={"score": score, "source": "paddleocr"},
                    )
                )

        log.info(
            "PaperSpineBuilderAgent: created %d text_block elements from %d pages.",
            len(elements),
            len(page_image_paths),
        )
        return elements
