# stephanie/components/information/agents/paper_spine_builder.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stephanie.components.information.utils.spine_dump import SpineDumper
from stephanie.agents.base_agent import BaseAgent
from stephanie.components.information.data import (
    BoundingBox,
    DocumentElement,
    PaperSection,
    attach_elements_to_sections,
)
from stephanie.scoring.scorable import Scorable
from stephanie.tools.pdf_tool import extract_page_texts

log = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF: pip install pymupdf
except ImportError:  # pragma: no cover
    fitz = None
    log.warning(
        "PaperSpineBuilderAgent: PyMuPDF (fitz) is not installed; "
        "PDF → image rendering will be disabled."
    )

# Optional HF deps for SmolDocling processor (only imported if enabled)
try:  # pragma: no cover
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForVision2Seq
except Exception:  # pragma: no cover
    torch = None
    Image = None
    AutoProcessor = None
    AutoModelForVision2Seq = None


@dataclass
class ProcessorResult:
    name: str
    enabled: bool
    ran: bool
    added_elements: int = 0
    error: Optional[str] = None
    stats: Dict[str, Any] = None


class PaperSpineBuilderAgent(BaseAgent):
    """
    v3: processor-driven spine builder.

    Inputs (context):
      - paper_sections: List[DocumentSection]          (required)
      - paper_elements: List[DocumentElement]          (optional, will be extended)
      - paper_pdf_path: str | Path                     (optional if arxiv_id given)
      - arxiv_id | paper_arxiv_id | root_arxiv_id      (for default PDF path)

    Outputs (context):
      - paper_elements: List[DocumentElement] (existing + extracted)
      - paper_spine:    attached sections with elements
      - paper_processing_signals: dict (optional debug + metrics)
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg=cfg, memory=memory, container=container, logger=logger)

        self.papers_root = Path(self.cfg.get("papers_root", "data/papers"))
        self.page_image_root = Path(self.cfg.get("page_image_root", f"runs/paper_blogs/{self.run_id}"))
        self.figures_root = Path(self.cfg.get("figures_root", f"runs/paper_blogs/{self.run_id}/paper_figures"))

        # Existing max_pages still supported (used by pdf_figures and rendering)
        self.max_pages: Optional[int] = self.cfg.get("max_pages")

        # minimum fraction of page area for us to consider an image a "figure"
        self.min_figure_area_frac: float = float(self.cfg.get("min_figure_area_frac", 0.02))

        # processor config
        self.processor_cfgs: List[Dict[str, Any]] = list(self.cfg.get("processors", []) or [])
        self.signals_cfg: Dict[str, Any] = dict(self.cfg.get("signals", {}) or {})
        self.routing_cfg: Dict[str, Any] = dict(self.cfg.get("routing", {}) or {})

        # SmolDocling cached runtime (loaded lazily only if enabled)
        self._docling_loaded: bool = False
        self._docling_model = None
        self._docling_processor = None
        self._docling_device = "cpu"

        # inside __init__ of PaperSpineBuilderAgent
        self.dump_cfg: Dict[str, Any] = dict(self.cfg.get("dump", {}) or {})
        self.run_dir = f"runs/paper_blogs/{self.run_id}"
        self._spine_dumper = SpineDumper(
            run_dir=self.run_dir,
            enabled=bool(self.dump_cfg.get("enabled", True)),
            max_text_chars=int(self.dump_cfg.get("max_text_chars", 240)),
        )

        log.info(
            "PaperSpineBuilderAgent:init run_id=%s papers_root=%s max_pages=%s processors=%s",
            getattr(self, "run_id", None),
            self.papers_root,
            self.max_pages,
            [p.get("name") for p in self.processor_cfgs],
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        sections = context.get("paper_sections") or []

        arxiv_id, pdf_path = self._resolve_paper_identity(context)

        log.info(
            "PaperSpineBuilderAgent: start arxiv_id=%s sections=%d pdf_path=%s",
            arxiv_id,
            len(sections),
            str(pdf_path) if pdf_path else None,
        )

        # Start from any elements already present
        elements: List[DocumentElement] = context.get("paper_elements") or []
        log.info("PaperSpineBuilderAgent: initial elements=%d", len(elements))

        # Run processors (NEW)
        proc_results: List[ProcessorResult] = []
        if self.processor_cfgs:
            log.info("PaperSpineBuilderAgent: running %d processors", len(self.processor_cfgs))
            elements, proc_results = await self._run_processors(
                context=context,
                arxiv_id=arxiv_id,
                pdf_path=pdf_path,
                elements=elements,
            )
        else:
            # Backwards-compatible behavior: do what v2 did (pdf_figures only)
            log.info("PaperSpineBuilderAgent: no processors configured; running legacy pdf_figures")
            new_figures = self._extract_figures_from_pdf(context)
            if new_figures:
                log.info("PaperSpineBuilderAgent: extracted %d figure elements from PDF.", len(new_figures))
                elements.extend(new_figures)

        # Update context and build spine
        context["paper_elements"] = elements
        spine_sections = list(sections)

        if _needs_page_fallback(spine_sections):
            # best source of page count is the rendered page PNGs you already write:
            # e.g. runs/.../2506.21734/2506.21734_p001.png ... _p024.png
            pages_dir = Path(self.page_image_root) / arxiv_id
            page_pngs = sorted(pages_dir.glob(f"{arxiv_id}_p*.png"))
            num_pages = len(page_pngs) or max((e.page for e in elements), default=0)

            page_texts = extract_page_texts(str(pdf_path))

            spine_sections = _make_page_sections(arxiv_id=arxiv_id, paper_role="root", num_pages=num_pages, page_text_by_page=page_texts)

        spine = attach_elements_to_sections(spine_sections, elements)

        context["paper_sections"] = sections          # keep real sections for NLP pipeline
        context["paper_spine_sections"] = spine_sections  # optional: so the dumper can show them
        context["paper_spine"] = spine


        spine = attach_elements_to_sections(sections=sections, elements=elements)
        context["paper_spine"] = spine

        # near the end of run(), AFTER context["paper_spine"]=spine and signals emitted
        try:
            dumped = self._spine_dumper.dump(
                arxiv_id=arxiv_id,
                sections=sections,
                elements=elements,
                spine=spine,
                proc_results=proc_results,
            )
            context.setdefault("paper_processing_signals", {}).setdefault("spine_dump", {})["files"] = dumped
            log.info("PaperSpineBuilderAgent: spine dumped files=%s", dumped)
        except Exception as e:
            log.exception("PaperSpineBuilderAgent: spine dump failed: %s", e)

        log.info(
            "PaperSpineBuilderAgent: built spine sections=%d total_elements=%d processors_ran=%s",
            len(sections),
            len(elements),
            [r.name for r in proc_results if r.ran],
        )

        # Emit signals into context for report/judge (optional)
        self._emit_processing_signals(context=context, proc_results=proc_results, sections=sections, elements=elements)

        return context

    # ----------------------------------------------------------------- processors ---

    async def _run_processors(
        self,
        *,
        context: Dict[str, Any],
        arxiv_id: str,
        pdf_path: Optional[Path],
        elements: List[DocumentElement],
    ) -> Tuple[List[DocumentElement], List[ProcessorResult]]:
        results: List[ProcessorResult] = []

        # Routing heuristics: currently only applied to smol_docling (cheap probe)
        routing_enabled = bool(self.routing_cfg.get("enabled", False))
        use_heuristics = bool(self.routing_cfg.get("use_heuristics", True))

        for idx, pcfg in enumerate(self.processor_cfgs):
            name = (pcfg.get("name") or "").strip()
            enabled = bool(pcfg.get("enabled", True))
            r = ProcessorResult(name=name, enabled=enabled, ran=False, stats={})
            results.append(r)

            if not enabled:
                log.info("PaperSpineBuilderAgent: processor[%d]=%s disabled", idx, name)
                continue

            try:
                before = len(elements)

                if name == "pdf_figures":
                    log.info("PaperSpineBuilderAgent: processor[%d]=pdf_figures begin", idx)
                    new_elems = self._extract_figures_from_pdf(context)
                    elements.extend(new_elems)
                    r.ran = True
                    r.added_elements = len(elements) - before
                    r.stats = {"extracted_figures": r.added_elements}
                    log.info(
                        "PaperSpineBuilderAgent: processor[%d]=pdf_figures done added=%d total=%d",
                        idx,
                        r.added_elements,
                        len(elements),
                    )

                elif name == "smol_docling":
                    if pdf_path is None:
                        log.warning("PaperSpineBuilderAgent: smol_docling skipped (no pdf_path)")
                        continue

                    log.info("PaperSpineBuilderAgent: processor[%d]=smol_docling begin", idx)

                    # Decide how many pages to run
                    max_pages = pcfg.get("max_pages", self.max_pages)
                    max_pages = int(max_pages) if max_pages is not None else None
                    probe_pages = int(self.routing_cfg.get("docling_probe_pages", 3)) if routing_enabled else 0

                    # Run probe if heuristics on
                    if routing_enabled and use_heuristics and probe_pages > 0:
                        log.info(
                            "PaperSpineBuilderAgent: smol_docling probe_pages=%d (heuristics enabled)",
                            probe_pages,
                        )
                        probe_elements, tag_counts = await self._smol_docling_extract_elements(
                            arxiv_id=arxiv_id,
                            pdf_path=pdf_path,
                            pcfg=pcfg,
                            max_pages=probe_pages,
                        )

                        # Decide continue
                        cont = self._docling_should_continue(tag_counts)
                        log.info(
                            "PaperSpineBuilderAgent: smol_docling probe tag_counts=%s continue=%s",
                            tag_counts,
                            cont,
                        )

                        # Always keep probe results
                        elements.extend(probe_elements)

                        if not cont:
                            r.ran = True
                            r.added_elements = len(elements) - before
                            r.stats = {"probe_only": True, "tag_counts": tag_counts}
                            log.info(
                                "PaperSpineBuilderAgent: processor[%d]=smol_docling stopped after probe added=%d",
                                idx,
                                r.added_elements,
                            )
                            continue

                        # Continue beyond probe to max_pages (minus what we already did)
                        remaining = None if max_pages is None else max(0, max_pages - probe_pages)
                        if remaining == 0:
                            r.ran = True
                            r.added_elements = len(elements) - before
                            r.stats = {"probe_only": True, "tag_counts": tag_counts}
                            log.info(
                                "PaperSpineBuilderAgent: processor[%d]=smol_docling max_pages==probe_pages; done",
                                idx,
                            )
                            continue

                        cont_elements, cont_tag_counts = await self._smol_docling_extract_elements(
                            arxiv_id=arxiv_id,
                            pdf_path=pdf_path,
                            pcfg=pcfg,
                            max_pages=None if remaining is None else remaining,
                            start_page=probe_pages + 1,
                        )
                        elements.extend(cont_elements)

                        # Merge tag counts
                        for k, v in cont_tag_counts.items():
                            tag_counts[k] = tag_counts.get(k, 0) + v

                        r.ran = True
                        r.added_elements = len(elements) - before
                        r.stats = {"probe_only": False, "tag_counts": tag_counts}

                        log.info(
                            "PaperSpineBuilderAgent: processor[%d]=smol_docling done added=%d total=%d tag_counts=%s",
                            idx,
                            r.added_elements,
                            len(elements),
                            tag_counts,
                        )

                    else:
                        # No heuristics: run up to max_pages
                        new_elems, tag_counts = await self._smol_docling_extract_elements(
                            arxiv_id=arxiv_id,
                            pdf_path=pdf_path,
                            pcfg=pcfg,
                            max_pages=max_pages,
                        )
                        elements.extend(new_elems)
                        r.ran = True
                        r.added_elements = len(elements) - before
                        r.stats = {"tag_counts": tag_counts}

                        log.info(
                            "PaperSpineBuilderAgent: processor[%d]=smol_docling done added=%d total=%d tag_counts=%s",
                            idx,
                            r.added_elements,
                            len(elements),
                            tag_counts,
                        )

                else:
                    log.warning(
                        "PaperSpineBuilderAgent: unknown processor '%s' (enabled but no implementation)",
                        name,
                    )

            except Exception as e:
                r.error = f"{type(e).__name__}: {e}"
                log.exception(
                    "PaperSpineBuilderAgent: processor '%s' failed: %s",
                    name,
                    r.error,
                )

        return elements, results

    def _docling_should_continue(self, tag_counts: Dict[str, int]) -> bool:
        # thresholds from routing config
        cont_cfg = dict(self.routing_cfg.get("docling_continue_if", {}) or {})
        min_table = int(cont_cfg.get("min_table_tags", 1))
        min_code = int(cont_cfg.get("min_code_tags", 1))
        min_formula = int(cont_cfg.get("min_formula_tags", 1))

        # tag keys we count below
        if tag_counts.get("table", 0) >= min_table:
            return True
        if tag_counts.get("code", 0) >= min_code:
            return True
        if tag_counts.get("formula", 0) >= min_formula:
            return True
        return False

    # ----------------------------------------------------------------- docling processor impl ---

    async def _smol_docling_extract_elements(
        self,
        *,
        arxiv_id: str,
        pdf_path: Path,
        pcfg: Dict[str, Any],
        max_pages: Optional[int],
        start_page: int = 1,
    ) -> Tuple[List[DocumentElement], Dict[str, int]]:
        """
        Render pages -> run SmolDocling (HF) -> parse DocTags into DocumentElements.

        This returns elements + simple tag counts (tables/code/formula/caption).
        """
        # Render to images first (we already have a renderer)
        all_page_paths = self._render_pdf_to_images(arxiv_id, pdf_path)
        if not all_page_paths:
            log.warning("PaperSpineBuilderAgent: smol_docling no page images available.")
            return [], {}

        # Apply paging window
        # start_page is 1-based; list is 0-based
        page_paths = all_page_paths[start_page - 1 :]
        if max_pages is not None:
            page_paths = page_paths[: int(max_pages)]

        log.info(
            "PaperSpineBuilderAgent: smol_docling rendering window start_page=%d pages=%d",
            start_page,
            len(page_paths),
        )

        # Load docling model lazily
        self._ensure_docling_loaded(pcfg)

        extract_cfg = dict((pcfg.get("extract") or {}) if isinstance(pcfg.get("extract"), dict) else {})
        want_tables = bool(extract_cfg.get("tables", True))
        want_code = bool(extract_cfg.get("code", True))
        want_equations = bool(extract_cfg.get("equations", True))
        want_captions = bool(extract_cfg.get("captions", True))
        want_headings = bool(extract_cfg.get("headings", True))
        want_paragraphs = bool(extract_cfg.get("paragraphs", False))

        prompt = pcfg.get("prompt", "Convert to Docling.")
        max_new_tokens = int(pcfg.get("max_new_tokens", 4096))

        elements: List[DocumentElement] = []
        tag_counts = {"table": 0, "code": 0, "formula": 0, "caption": 0, "heading": 0, "text": 0}

        for page_idx, img_path in enumerate(page_paths, start=start_page):
            doctags = self._docling_generate_doctags(image_path=img_path, prompt=prompt, max_new_tokens=max_new_tokens)
            counts = self._count_doctags(doctags)
            for k, v in counts.items():
                tag_counts[k] = tag_counts.get(k, 0) + v

            log.info(
                "PaperSpineBuilderAgent: smol_docling page=%d doctags_len=%d counts=%s img=%s",
                page_idx,
                len(doctags),
                counts,
                img_path.name,
            )

            # Parse doctags into elements (minimal v1 parser)
            elems = self._doctags_to_elements(
                arxiv_id=arxiv_id,
                page_num=page_idx,
                doctags=doctags,
                want_tables=want_tables,
                want_code=want_code,
                want_equations=want_equations,
                want_captions=want_captions,
                want_headings=want_headings,
                want_paragraphs=want_paragraphs,
            )
            elements.extend(elems)

        log.info(
            "PaperSpineBuilderAgent: smol_docling extracted elements=%d tag_counts=%s",
            len(elements),
            tag_counts,
        )
        return elements, tag_counts

    def _ensure_docling_loaded(self, pcfg: Dict[str, Any]) -> None:
        if self._docling_loaded:
            return

        if torch is None or AutoProcessor is None or AutoModelForVision2Seq is None or Image is None:
            raise RuntimeError(
                "SmolDocling dependencies missing. Install: torch, transformers, pillow "
                "(and ensure transformers supports AutoModelForVision2Seq)."
            )

        model_name = pcfg.get("model_name", "ds4sd/SmolDocling-256M-preview")

        # device selection similar to your summarizer tool
        self._docling_device = "cuda" if torch.cuda.is_available() else "cpu"

        log.info(
            "PaperSpineBuilderAgent: loading SmolDocling model=%s device=%s",
            model_name,
            self._docling_device,
        )

        self._docling_processor = AutoProcessor.from_pretrained(model_name)

        if self._docling_device == "cuda":
            self._docling_model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            ).to(self._docling_device)
        else:
            self._docling_model = AutoModelForVision2Seq.from_pretrained(model_name).to(self._docling_device)

        self._docling_model.eval()
        self._docling_loaded = True

        log.info("PaperSpineBuilderAgent: SmolDocling loaded successfully")

    def _docling_generate_doctags(self, *, image_path: Path, prompt: str, max_new_tokens: int) -> str:
        img = Image.open(image_path).convert("RGB")

        # best-effort: use model-defined token if available, otherwise "<image>"
        image_token = "<image>"
        tok = getattr(self._docling_processor, "tokenizer", None)
        if tok is not None and getattr(tok, "image_token", None):
            image_token = tok.image_token

        base_prompt = (prompt or "").strip() or "Convert this page to docling."

        # ensure exactly ONE image token for ONE image
        if image_token not in base_prompt:
            text_in = f"{image_token}\n{base_prompt}"
        else:
            # collapse duplicates if someone accidentally injected multiple
            first = base_prompt.find(image_token)
            after = base_prompt[first + len(image_token):].replace(image_token, "")
            text_in = base_prompt[: first + len(image_token)] + after

        inputs = self._docling_processor(
            images=[img],
            text=[text_in],
            return_tensors="pt",
        ).to(self._docling_device)

        with torch.no_grad():
            out = self._docling_model.generate(**inputs, max_new_tokens=max_new_tokens)

        return self._docling_processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def _count_doctags(self, doctags: str) -> Dict[str, int]:
        """
        Very simple counts for routing + logging.
        DocTags vocabulary varies; these are common signals.
        """
        # use broad patterns; tighten later once you see real outputs
        return {
            "table": len(re.findall(r"<otsl>|<table>", doctags)),
            "code": len(re.findall(r"<code>", doctags)),
            "formula": len(re.findall(r"<formula>|<equation>", doctags)),
            "caption": len(re.findall(r"<caption>", doctags)),
            "heading": len(re.findall(r"<section_header>|<title>", doctags)),
            "text": len(re.findall(r"<text>", doctags)),
        }

    def _doctags_to_elements(
        self,
        *,
        arxiv_id: str,
        page_num: int,
        doctags: str,
        want_tables: bool,
        want_code: bool,
        want_equations: bool,
        want_captions: bool,
        want_headings: bool,
        want_paragraphs: bool,
    ) -> List[DocumentElement]:
        """
        Minimal parser: extracts tagged blocks by regex.
        This is v1 and intentionally conservative.
        Later you can replace with a real DocTags parser.
        """
        elems: List[DocumentElement] = []

        def add_elem(tag: str, idx: int, text: str, elem_type: str):
            elem_id = f"{arxiv_id}:p{page_num}:{tag}{idx}"
            elems.append(
                DocumentElement(
                    id=elem_id,
                    paper_id=str(arxiv_id),
                    page=page_num,
                    type=elem_type,
                    bbox=None,
                    text=text if elem_type not in ("table",) else None,
                    markdown_table=text if elem_type == "table" else None,
                    latex=text if elem_type == "equation" else None,
                    image_path=None,
                    caption=text if elem_type == "caption" else None,
                    meta={"source": "smol_docling", "tag": tag},
                )
            )

        # Tables
        if want_tables:
            for i, m in enumerate(re.finditer(r"<otsl>(.*?)</otsl>", doctags, flags=re.DOTALL), start=1):
                add_elem("otsl", i, m.group(1).strip(), "table")

        # Code
        if want_code:
            for i, m in enumerate(re.finditer(r"<code>(.*?)</code>", doctags, flags=re.DOTALL), start=1):
                add_elem("code", i, m.group(1).strip(), "code")

        # Equations / formulas
        if want_equations:
            for i, m in enumerate(re.finditer(r"<formula>(.*?)</formula>", doctags, flags=re.DOTALL), start=1):
                add_elem("formula", i, m.group(1).strip(), "equation")

        # Captions
        if want_captions:
            for i, m in enumerate(re.finditer(r"<caption>(.*?)</caption>", doctags, flags=re.DOTALL), start=1):
                add_elem("caption", i, m.group(1).strip(), "caption")

        # Headings
        if want_headings:
            for i, m in enumerate(re.finditer(r"<section_header>(.*?)</section_header>", doctags, flags=re.DOTALL), start=1):
                add_elem("section_header", i, m.group(1).strip(), "heading")
            for i, m in enumerate(re.finditer(r"<title>(.*?)</title>", doctags, flags=re.DOTALL), start=1):
                add_elem("title", i, m.group(1).strip(), "heading")

        # Paragraphs (off by default to avoid duplicating your existing text)
        if want_paragraphs:
            for i, m in enumerate(re.finditer(r"<text>(.*?)</text>", doctags, flags=re.DOTALL), start=1):
                add_elem("text", i, m.group(1).strip(), "text_block")

        log.info(
            "PaperSpineBuilderAgent: doctags_to_elements page=%d produced=%d (tables=%s code=%s eq=%s cap=%s head=%s text=%s)",
            page_num,
            len(elems),
            want_tables,
            want_code,
            want_equations,
            want_captions,
            want_headings,
            want_paragraphs,
        )
        return elems

    # ----------------------------------------------------------------- signals / logging ---

    def _emit_processing_signals(
        self,
        *,
        context: Dict[str, Any],
        proc_results: List[ProcessorResult],
        sections: List[Any],
        elements: List[DocumentElement],
    ) -> None:
        if not bool(self.signals_cfg.get("enabled", True)):
            return

        key = self.signals_cfg.get("context_key", "paper_processing_signals")

        ran = [r.name for r in proc_results if r.ran]
        errors = {r.name: r.error for r in proc_results if r.error}
        added = {r.name: r.added_elements for r in proc_results if r.ran}
        stats = {r.name: (r.stats or {}) for r in proc_results if r.ran}

        # Basic counts by element.type
        counts_by_type: Dict[str, int] = {}
        for e in elements:
            counts_by_type[e.type] = counts_by_type.get(e.type, 0) + 1

        payload = {
            "processors": {
                "configured": [r.name for r in proc_results],
                "ran": ran,
                "added_elements": added,
                "errors": errors,
                "stats": stats,
            },
            "counts_by_type": counts_by_type,
            "total_elements": len(elements),
            "total_sections": len(sections),
        }

        context[key] = payload

        log.info("PaperSpineBuilderAgent: signals_key=%s payload=%s", key, payload)

    # ----------------------------------------------------------------- existing helpers (mostly unchanged) ---

    def _resolve_paper_identity(self, context: Dict[str, Any]) -> Tuple[str, Optional[Path]]:
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
            log.warning("PaperSpineBuilderAgent: PDF not found for %s at %s", arxiv_id, pdf_path)
            return str(arxiv_id), None

        return str(arxiv_id), pdf_path

    # ----------------------------------------------------------------- figures extractor (unchanged) ---

    def _extract_figures_from_pdf(self, context: Dict[str, Any]) -> List[DocumentElement]:
        """
        Use PyMuPDF to pull out embedded images from each page, save them,
        and wrap them as DocumentElement(type='figure').

        This does NOT use OCR at all – it's purely structural extraction.
        """
        if fitz is None:
            return []

        arxiv_id, pdf_path = self._resolve_paper_identity(context)
        if pdf_path is None:
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

            images = page.get_images(full=True)
            if not images:
                continue

            for img_idx, img_info in enumerate(images):
                xref = img_info[0]
                rects = page.get_image_rects(xref)
                if not rects:
                    continue

                # Filter rects BEFORE extracting/saving bytes (prevents tons of tiny files)
                min_side_pts = float(self.cfg.get("min_figure_side_pts", 30.0))
                keep: list[tuple[int, Any, float]] = []
                for rect_i, r in enumerate(rects):
                    rect_area = float(r.width * r.height)
                    if page_area > 0 and rect_area / page_area < self.min_figure_area_frac:
                        continue
                    # Optional: drop very thin strips (gridlines, tick marks)
                    if min(float(r.width), float(r.height)) < min_side_pts:
                        continue
                    keep.append((rect_i, r, rect_area))

                if not keep:
                    continue

                img = doc.extract_image(xref)
                img_bytes = img.get("image")
                img_ext = img.get("ext", "png")

                # Use xref in filename so it’s stable and debuggable
                img_path = figures_dir / f"{arxiv_id}_p{page_num:03d}_xref{xref}.{img_ext}"
                if not img_path.exists():
                    img_path.write_bytes(img_bytes)

                for rect_i, r, rect_area in keep:
                    bbox = BoundingBox(
                         x1=float(r.x0), y1=float(r.y0), x2=float(r.x1), y2=float(r.y1)
                     )
                    elem_id = f"{arxiv_id}:p{page_num}:xref{xref}:{rect_i}"


                    bbox = BoundingBox(
                        x1=float(r.x0), y1=float(r.y0), x2=float(r.x1), y2=float(r.y1)
                    )
                    elem_id = f"{arxiv_id}:p{page_num}:img{img_idx:02d}:{rect_i}"

                    elements.append(
                        DocumentElement(
                            id=elem_id,
                            paper_id=str(arxiv_id),
                            page=page_num,
                            type="figure",
                            bbox=bbox,
                            text=None,
                            latex=None,
                            markdown_table=None,
                            image_path=str(img_path),
                            caption=None,
                            meta={"xref": xref, "page_area": page_area, "rect_area": rect_area},
                        )
                    )

        log.info("PaperSpineBuilderAgent: pdf_figures extracted=%d", len(elements))
        return elements

    # ----------------------------------------------------------------- PDF → images (unchanged, but used by docling) ---

    def _render_pdf_to_images(self, arxiv_id: str, pdf_path: Path) -> List[Path]:
        """
        Render each page of the PDF into a PNG image and return list of paths.
        """
        if fitz is None:
            log.warning("PaperSpineBuilderAgent: cannot render PDF '%s' → images (PyMuPDF not installed).", pdf_path)
            return []

        if not pdf_path.exists():
            log.warning("PaperSpineBuilderAgent: PDF not found for %s at %s", arxiv_id, pdf_path)
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

        log.info("PaperSpineBuilderAgent: rendered pages=%d pdf=%s out_dir=%s", len(page_paths), pdf_path, out_dir)
        return page_paths

    # ----------------------------------------------------------------- OCR helper (unchanged) ---

    async def _ocr_pages_to_elements(self, arxiv_id: str, page_image_paths: List[Path]) -> List[DocumentElement]:
        if getattr(self, "ocr_tool", None) is None:
            return []

        elements: List[DocumentElement] = []
        for page_idx, img_path in enumerate(page_image_paths, start=1):
            scorable = Scorable(
                id=f"{arxiv_id}:p{page_idx}",
                text=str(img_path),
                target_type="document_page",
                meta={"image_path": str(img_path)},
            )
            tool_context: Dict[str, Any] = {"image_path": str(img_path)}
            scorable = await self.ocr_tool.apply(scorable, tool_context)

            ocr_meta = scorable.meta.get("ocr", {})
            lines = ocr_meta.get(getattr(self.ocr_tool, "name", "paddle_ocr"), [])

            for line_idx, line in enumerate(lines):
                bbox_poly = line.get("bbox") or [[0.0, 0.0]] *  None4
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
                        type="text_block",
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        text=text,
                        meta={"score": score, "source": "paddleocr"},
                    )
                )

        log.info("PaperSpineBuilderAgent: OCR created=%d elements pages=%d", len(elements), len(page_image_paths))
        return elements

def _needs_page_fallback(sections: list) -> bool:
    for s in sections:
        meta = getattr(s, "meta", None) or {}
        if getattr(s, "start_page", None) is not None or getattr(s, "end_page", None) is not None:
            return False
        if hasattr(s, "pages") and getattr(s, "pages", None):
            return False
        if isinstance(meta, dict) and (meta.get("start_page") is not None or meta.get("pages") is not None):
            return False
    return True

def _make_page_sections(
    *,
    arxiv_id: str,
    paper_role: str,
    num_pages: int,
    page_text_by_page: dict[int, str] | None = None,
) -> list[PaperSection]:
    out = []
    page_text_by_page = page_text_by_page or {}

    for p in range(1, num_pages + 1):
        out.append(
            PaperSection(
                id=f"{arxiv_id}::page-{p:03d}",
                paper_arxiv_id=arxiv_id,
                paper_role=paper_role,
                section_index=p - 1,
                text=page_text_by_page.get(p, ""),  # ✅ page text if available
                title=f"Page {p}",
                meta={"start_page": p, "end_page": p, "kind": "spine_page"},
            )
        )
    return out
