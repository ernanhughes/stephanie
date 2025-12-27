# stephanie/tools/smol_docling_tool.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import re
import html
from inspect import Parameter, signature

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from stephanie.components.information.data import PaperSection
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

    @staticmethod
    def build_semantic_sections(
        *,
        arxiv_id: str,
        paper_role: str,
        pages: List[Dict[str, Any]],  # [{"page_num": 1, "doctags": "...", ...}, ...]
        run_id: Optional[str] = None,
        heading_tags: Optional[List[str]] = None,
        text_tags: Optional[List[str]] = None,
        min_heading_chars: int = 3,
        merge_small_sections_to_prev_if_under_chars: int = 200,
        fallback_to_page_sections: bool = True,
    ) -> List[PaperSection]:
        """
        Build semantic-ish sections from Docling doctags.

        Output: List[PaperSection] where each section has start_page/end_page,
        and (usually) text aggregated from doctags.

        v1 heuristic:
        - parse doctags in-order for heading tags and text tags
        - start a new section when heading tag encountered
        - accumulate text blocks under current section
        - fallback: one section per page (if no headings found)
        """
        from stephanie.components.information.data import PaperSection  # local import to avoid cycles

        heading_tags = heading_tags or ["section_header", "heading", "title"]
        text_tags = text_tags or ["text", "paragraph", "p"]

        # compile regex once (order-preserving)
        tag_union = "|".join(re.escape(t) for t in (heading_tags + text_tags + ["caption"]))
        block_re = re.compile(
            rf"<(?P<tag>{tag_union})\b[^>]*>(?P<content>.*?)</(?P=tag)>",
            flags=re.IGNORECASE | re.DOTALL,
        )

        def _clean(s: str) -> str:
            s = html.unescape(s or "")
            # collapse whitespace
            s = re.sub(r"[ \t]+\n", "\n", s)
            s = re.sub(r"\n{3,}", "\n\n", s)
            s = re.sub(r"[ \t]{2,}", " ", s)
            return s.strip()

        def _strip_tags(s: str) -> str:
            s = html.unescape(s or "")
            s = re.sub(r"<[^>]+>", " ", s)
            s = re.sub(r"[ \t]{2,}", " ", s)
            s = re.sub(r"\n{3,}", "\n\n", s)
            return s.strip()

        def _new_section(**kwargs) -> "PaperSection":
            # Create PaperSection using signature introspection (like your PaperPipeline does)
            sig = signature(PaperSection)
            param_names = [
                p.name
                for p in sig.parameters.values()
                if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
            ]
            ctor_kwargs = {k: v for k, v in kwargs.items() if k in param_names and v is not None}
            sec = PaperSection(**ctor_kwargs)

            # best-effort attribute patching (for fields not in ctor)
            for k, v in kwargs.items():
                if v is None:
                    continue
                if not hasattr(sec, k):
                    continue
                try:
                    setattr(sec, k, v)
                except Exception:
                    pass
            return sec

        # ---- build sections -------------------------------------------------
        built: List["PaperSection"] = []
        cur_title: Optional[str] = None
        cur_text_parts: List[str] = []
        cur_start_page: Optional[int] = None
        cur_end_page: Optional[int] = None

        def _flush():
            nonlocal cur_title, cur_text_parts, cur_start_page, cur_end_page
            if cur_start_page is None or cur_end_page is None:
                return
            text = _clean("\n\n".join(cur_text_parts))
            title = (cur_title or "").strip() or f"Pages {cur_start_page}-{cur_end_page}"

            idx = len(built)
            sec_id = f"{arxiv_id}::docling::{run_id or 'run'}::sec-{idx:03d}"

            meta = {
                "source": "smol_docling",
                "kind": "semantic_section_v1",
                "start_page": cur_start_page,
                "end_page": cur_end_page,
                "run_id": run_id,
            }

            sec = _new_section(
                id=sec_id,
                paper_arxiv_id=arxiv_id,
                paper_role=paper_role,
                section_index=idx,
                title=title,
                text=text,
                meta=meta,
                start_page=cur_start_page,
                end_page=cur_end_page,
                pages=list(range(cur_start_page, cur_end_page + 1)),
            )
            built.append(sec)

            cur_title = None
            cur_text_parts = []
            cur_start_page = None
            cur_end_page = None

        saw_any_heading = False

        for p in pages:
            page_num = int(p.get("page_num") or 0)
            doctags = p.get("doctags") or ""
            if page_num <= 0:
                continue

            blocks = list(block_re.finditer(doctags))
            if not blocks:
                # no structured tags found — fallback to plain text for this page
                page_text = _strip_tags(doctags)
                if page_text:
                    if cur_start_page is None:
                        cur_start_page = page_num
                    cur_end_page = page_num
                    cur_text_parts.append(page_text)
                continue

            for m in blocks:
                tag = (m.group("tag") or "").lower()
                content = _clean(m.group("content") or "")
                if not content:
                    continue

                if tag in (t.lower() for t in heading_tags):
                    # Start new section on heading
                    if len(content) >= min_heading_chars:
                        saw_any_heading = True
                        # close current section before starting another
                        if cur_start_page is not None:
                            _flush()
                        cur_title = content
                        cur_start_page = page_num
                        cur_end_page = page_num
                    continue

                if tag in (t.lower() for t in text_tags):
                    if cur_start_page is None:
                        cur_start_page = page_num
                    cur_end_page = page_num
                    cur_text_parts.append(content)

                # captions are optional; include if you want
                # elif tag == "caption": ...

            # keep page range correct even if the page had only headings
            if cur_start_page is not None:
                cur_end_page = page_num

        # flush last
        if cur_start_page is not None:
            _flush()

        # ---- fallback: no headings -> per-page sections ---------------------
        if fallback_to_page_sections and (not built or not saw_any_heading):
            built = []
            for p in pages:
                page_num = int(p.get("page_num") or 0)
                doctags = p.get("doctags") or ""
                if page_num <= 0:
                    continue
                text = _strip_tags(doctags)

                idx = len(built)
                sec_id = f"{arxiv_id}::docling::{run_id or 'run'}::page-{page_num:03d}"
                meta = {
                    "source": "smol_docling",
                    "kind": "page_section_fallback",
                    "start_page": page_num,
                    "end_page": page_num,
                    "run_id": run_id,
                }

                built.append(
                    _new_section(
                        id=sec_id,
                        paper_arxiv_id=arxiv_id,
                        paper_role=paper_role,
                        section_index=idx,
                        title=f"Page {page_num}",
                        text=text,
                        meta=meta,
                        start_page=page_num,
                        end_page=page_num,
                        pages=[page_num],
                    )
                )

        # ---- merge tiny sections into previous (optional) -------------------
        if merge_small_sections_to_prev_if_under_chars and built:
            merged: List[PaperSection] = []
            for sec in built:
                t = (getattr(sec, "text", None) or "").strip()
                if merged and len(t) < merge_small_sections_to_prev_if_under_chars:
                    prev = merged[-1]
                    prev_text = (getattr(prev, "text", None) or "").rstrip()
                    join = "\n\n" if prev_text and t else ""
                    try:
                        setattr(prev, "text", f"{prev_text}{join}{t}".strip())
                    except Exception:
                        pass
                    # extend end_page if possible
                    try:
                        prev_end = getattr(prev, "end_page", None)
                        sec_end = getattr(sec, "end_page", None)
                        if prev_end is not None and sec_end is not None:
                            setattr(prev, "end_page", max(int(prev_end), int(sec_end)))
                    except Exception:
                        pass
                else:
                    merged.append(sec)
            built = merged

        # reindex section_index
        for i, sec in enumerate(built):
            try:
                setattr(sec, "section_index", i)
            except Exception:
                pass

        return built
