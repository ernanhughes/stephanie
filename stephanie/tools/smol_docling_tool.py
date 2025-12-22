# stephanie/tools/smol_docling_tool.py
from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from stephanie.tools.base_tool import BaseTool
from stephanie.tools.pdf_tool import PDFConverter

log = logging.getLogger(__name__)

class SmolDoclingTool(BaseTool):
    name = "smol_docling"

    def __init__(self, cfg: Dict[str, Any], memory, container, logger) -> None:
        super().__init__(cfg, memory, container, logger)

        self.model_name = cfg.get("model_name", "ds4sd/SmolDocling-256M-preview")
        self.prompt = cfg.get("prompt", "Convert to Docling.")
        self.dpi = int(cfg.get("dpi", 144))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 4096))
        self.store_pages = bool(cfg.get("store_pages", True))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("[SmolDoclingTool] device=%s model=%s", self.device, self.model_name)

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        if self.device == "cuda":
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name, torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(self.model_name).to(self.device)

        self.model.eval()

    async def apply(self, scorable, context: Dict[str, Any]):
        paper_id = getattr(scorable, "id", None) or getattr(scorable, "paper_id", None)
        if not paper_id:
            raise ValueError("SmolDoclingTool requires scorable.id or scorable.paper_id")

        paper = self.memory.papers.get_by_id(paper_id)
        if not paper or not paper.pdf_path:
            return scorable

        run_id = context.get("run_id")  # you likely create PaperRun outside; reuse if present

        # Render PDF pages to images (you’ll add render_to_images() to PDFConverter)
        page_paths = PDFConverter.render_to_images(paper.pdf_path, dpi=self.dpi)

        pages_payload: List[Dict[str, Any]] = []
        for i, img_path in enumerate(page_paths, start=1):
            img = Image.open(img_path).convert("RGB")

            inputs = self.processor(images=img, text=self.prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

            doctags = self.processor.batch_decode(out, skip_special_tokens=True)[0]

            pages_payload.append({
                "page_num": i,
                "doctags": doctags,
                "meta": {
                    "source": "smol_docling",
                    "prompt": self.prompt,
                    "dpi": self.dpi,
                    "model_name": self.model_name,
                }
            })

        if self.store_pages:
            # Requires your PaperStore additions: upsert_docling_pages(...)
            self.memory.papers.upsert_docling_pages(
                paper_id=paper_id,
                run_id=run_id,
                pages=pages_payload,
                model_name=self.model_name,
                dpi=self.dpi,
            )

        # Optionally: attach to scorable.meta for downstream steps
        scorable.meta.setdefault("docling", {})["pages"] = pages_payload
        return scorable


    async def _infer_doctags(self, image_path: str) -> tuple[str, Dict[str, Any]]:
        """
        Wire this to your inference runtime:
        - local HTTP service
        - transformers
        - vLLM
        """
        raise NotImplementedError("Implement SmolDocling inference backend here.")

    def _doctags_to_sections(self, *, paper_id: str, run_id: str, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Minimal block sectioning: one section per page (safe baseline).
        Later, you replace this with a real DocTags parser that yields blocks/headings/captions/tables.
        """
        sections: List[Dict[str, Any]] = []
        sec_idx = 0
        for p in pages:
            page_num = int(p["page_num"])
            sections.append(
                {
                    "id": f"{paper_id}::{run_id}::page::{page_num}",
                    "paper_id": paper_id,
                    "run_id": run_id,
                    "parent_id": None,
                    "level": 0,
                    "path": str(page_num),
                    "section_index": sec_idx,
                    "start_page": page_num,
                    "end_page": page_num,
                    "title": f"Page {page_num}",
                    "text": None,  # leave blank; downstream can render from doctags
                    "meta": {
                        "source": "docling",
                        "block_type": "page",
                        "page_num": page_num,
                        "model_name": p.get("model_name"),
                        "dpi": p.get("dpi"),
                        "doctags": p.get("doctags"),
                    },
                }
            )
            sec_idx += 1
        return sections
