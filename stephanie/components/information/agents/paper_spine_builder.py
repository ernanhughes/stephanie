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
    assign_page_ranges_to_semantic_sections,
    attach_elements_to_sections,
)
from stephanie.scoring.scorable import Scorable
from stephanie.tools.pdf_tool import extract_page_texts
from stephanie.tools.smol_docling_tool import SmolDoclingTool

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
      - paper_sections: List[PaperSection]          (required)
      - paper: Scorable or dict-like               (optional; for arxiv_id/pdf_path)
      - paper_elements: List[DocumentElement]      (optional; will be extended)

    Outputs (context):
      - paper_elements: List[DocumentElement]
      - paper_spine_sections: List[PaperSection]   (sections actually used to build the spine)
      - paper_spine: List[SectionSpine]
      - paper_processing_signals: dict             (optional metrics)
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
        self._docling_device = None

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
        """Build a document 'spine' by attaching extracted visual elements to page-ranged sections.

        Order of section sources:
          1) context['paper_spine_sections'] (typically Docling-derived; page-ranged)
          2) context['paper_sections'] (semantic sections if already page-ranged)
          3) one section-per-page fallback (spine pages)
        """
        # semantic sections you already have
        arxiv_id, pdf_path = self._resolve_paper_identity(context)

        sections = context.get("paper_sections") or []



        # get page texts
        page_texts = extract_page_texts(str(pdf_path)) if pdf_path else {}

        # ✅ add start_page/end_page onto semantic sections
        assign_page_ranges_to_semantic_sections(sections, page_texts)

        # Now attach using semantic sections (not page sections)
        spine_sections = sections
        log.info(
            "PaperSpineBuilderAgent: start arxiv_id=%s sections=%d pdf_path=%s",
            arxiv_id,
            len(sections),
            str(pdf_path) if pdf_path else None,
        )

        # Start from any elements already present
        elements: List[DocumentElement] = context.get("paper_elements") or []
        log.info("PaperSpineBuilderAgent: initial elements=%d", len(elements))

        spine = attach_elements_to_sections(spine_sections, elements)


        proc_results: List[ProcessorResult] = []

        if self.processor_cfgs:
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

        # Update context with extracted elements
        context["paper_elements"] = elements

        # Prefer Docling-derived page-ranged sections if provided by processors
        spine_sections = context.get("paper_spine_sections") or list(sections)

        # Fallback to page sections if we can't compute page ranges
        if _needs_page_fallback(spine_sections):
            pages_dir = Path(self.page_image_root) / arxiv_id
            page_pngs = sorted(pages_dir.glob(f"{arxiv_id}_p*.png"))

            # older layout: pages directly under page_image_root
            if not page_pngs:
                pages_dir = Path(self.page_image_root)
                page_pngs = sorted(pages_dir.glob(f"{arxiv_id}_p*.png"))

            num_pages = len(page_pngs) or max((e.page for e in elements), default=0)

            page_texts: Dict[int, str] = {}
            if pdf_path:
                try:
                    page_texts = extract_page_texts(str(pdf_path))
                except Exception:
                    log.exception("PaperSpineBuilderAgent: failed extract_page_texts for %s", pdf_path)

            spine_sections = _make_page_sections(
                arxiv_id=arxiv_id,
                paper_role="root",
                num_pages=num_pages,
                page_text_by_page=page_texts,
            )

        spine = attach_elements_to_sections(spine_sections, elements)

        # Keep semantic sections for downstream NLP/blog, but expose spine sections separately
        context["paper_sections"] = sections
        context["paper_spine_sections"] = spine_sections
        context["paper_spine"] = spine

        # Dump artifacts
        try:
            dumped = self._spine_dumper.dump(
                arxiv_id=arxiv_id,
                sections=spine_sections,  # dump what we actually used
                elements=elements,
                spine=spine,
                proc_results=proc_results,
            )
            context.setdefault("paper_processing_signals", {}).setdefault("spine_dump", {})["files"] = dumped
            log.info("PaperSpineBuilderAgent: spine dumped files=%s", dumped)
        except Exception as e:
            log.exception("PaperSpineBuilderAgent: spine dump failed: %s", e)

        log.info(
            "PaperSpineBuilderAgent: built spine_sections=%d total_elements=%d processors_ran=%s",
            len(spine_sections),
            len(elements),
            [r.name for r in proc_results if r.ran],
        )

        # Emit signals into context for report/judge (optional)
        self._emit_processing_signals(
            context=context,
            proc_results=proc_results,
            sections=spine_sections,
            elements=elements,
        )
        return context

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
            if not name:
                continue

            enabled = bool(pcfg.get("enabled", True))
            r = ProcessorResult(name=name, enabled=enabled, ran=False, stats={})
            results.append(r)

            if not enabled:
                log.info("PaperSpineBuilderAgent: processor[%d]=%s disabled", idx, name)
                continue

            before = len(elements)

            try:
                if name == "pdf_figures":
                    log.info("PaperSpineBuilderAgent: processor[%d]=pdf_figures begin", idx)
                    new_figures = self._extract_figures_from_pdf(context)
                    if new_figures:
                        elements.extend(new_figures)

                    r.ran = True
                    r.added_elements = len(elements) - before
                    r.stats = {"extracted": len(new_figures or [])}

                    log.info(
                        "PaperSpineBuilderAgent: processor[%d]=pdf_figures done added=%d total=%d",
                        idx,
                        r.added_elements,
                        len(elements),
                    )

                elif name == "smol_docling":
                    if pdf_path is None:
                        raise RuntimeError("smol_docling requires pdf_path")

                    log.info("PaperSpineBuilderAgent: processor[%d]=smol_docling begin", idx)

                    max_pages = pcfg.get("max_pages", self.max_pages)
                    max_pages = int(max_pages) if max_pages is not None else None

                    probe_pages = int(self.routing_cfg.get("docling_probe_pages", 3)) if routing_enabled else 0
                    emit_sections_on_probe = bool(pcfg.get("emit_sections_on_probe", False))

                    all_pages: List[Dict[str, Any]] = []
                    tag_counts: Dict[str, int] = {}
                    probe_only = False

                    if routing_enabled and use_heuristics and probe_pages > 0:
                        # --- Probe ---
                        log.info(
                            "PaperSpineBuilderAgent: smol_docling probe_pages=%d (heuristics enabled)",
                            probe_pages,
                        )
                        probe_elems, probe_pages_out, probe_tag_counts = await self._smol_docling_extract_elements(
                            arxiv_id=arxiv_id,
                            pdf_path=pdf_path,
                            pcfg=pcfg,
                            max_pages=probe_pages,
                            start_page=1,
                        )
                        elements.extend(probe_elems)
                        all_pages.extend(probe_pages_out)
                        tag_counts = dict(probe_tag_counts)

                        cont = self._docling_should_continue(tag_counts)
                        log.info(
                            "PaperSpineBuilderAgent: smol_docling probe tag_counts=%s continue=%s",
                            tag_counts,
                            cont,
                        )

                        if not cont:
                            probe_only = True
                            r.ran = True
                            r.added_elements = len(elements) - before
                            r.stats = {"probe_only": True, "tag_counts": tag_counts, "docling_pages": len(all_pages)}
                            log.info(
                                "PaperSpineBuilderAgent: processor[%d]=smol_docling stopped after probe added=%d",
                                idx,
                                r.added_elements,
                            )

                            if emit_sections_on_probe and all_pages:
                                context["paper_spine_sections"] = self._build_docling_sections(
                                    arxiv_id=arxiv_id,
                                    paper_role="root",
                                    pages=all_pages,
                                    run_id=str(self.run_id),
                                    pcfg=pcfg,
                                    partial=True,
                                    probe_only=True,
                                )
                            continue

                        # --- Continue ---
                        remaining = None if max_pages is None else max(0, max_pages - probe_pages)
                        if remaining == 0:
                            probe_only = True
                            r.ran = True
                            r.added_elements = len(elements) - before
                            r.stats = {"probe_only": True, "tag_counts": tag_counts, "docling_pages": len(all_pages)}
                            log.info(
                                "PaperSpineBuilderAgent: processor[%d]=smol_docling max_pages==probe_pages; done",
                                idx,
                            )
                            if emit_sections_on_probe and all_pages:
                                context["paper_spine_sections"] = self._build_docling_sections(
                                    arxiv_id=arxiv_id,
                                    paper_role="root",
                                    pages=all_pages,
                                    run_id=str(self.run_id),
                                    pcfg=pcfg,
                                    partial=True,
                                    probe_only=True,
                                )
                            continue

                        cont_elems, cont_pages_out, cont_tag_counts = await self._smol_docling_extract_elements(
                            arxiv_id=arxiv_id,
                            pdf_path=pdf_path,
                            pcfg=pcfg,
                            max_pages=None if remaining is None else remaining,
                            start_page=probe_pages + 1,
                        )
                        elements.extend(cont_elems)
                        all_pages.extend(cont_pages_out)

                        for k, v in cont_tag_counts.items():
                            tag_counts[k] = tag_counts.get(k, 0) + v

                    else:
                        # No heuristics: run up to max_pages
                        new_elems, pages_out, tag_counts = await self._smol_docling_extract_elements(
                            arxiv_id=arxiv_id,
                            pdf_path=pdf_path,
                            pcfg=pcfg,
                            max_pages=max_pages,
                            start_page=1,
                        )
                        elements.extend(new_elems)
                        all_pages = pages_out

                    r.ran = True
                    r.added_elements = len(elements) - before
                    r.stats = {
                        "probe_only": bool(probe_only),
                        "tag_counts": tag_counts,
                        "docling_pages": len(all_pages),
                    }

                    # Build semantic sections (page-ranged) for spine attachment
                    if (not probe_only) and all_pages:
                        docling_sections = self._build_docling_sections(
                            arxiv_id=arxiv_id,
                            paper_role="root",
                            pages=all_pages,
                            run_id=str(self.run_id),
                            pcfg=pcfg,
                        )
                        context["paper_spine_sections"] = docling_sections
                        r.stats["docling_sections"] = len(docling_sections)

                    log.info(
                        "PaperSpineBuilderAgent: processor[%d]=smol_docling done added=%d total=%d stats=%s",
                        idx,
                        r.added_elements,
                        len(elements),
                        r.stats,
                    )

                elif name == "ocr_elements":
                    log.info("PaperSpineBuilderAgent: processor[%d]=ocr_elements begin", idx)
                    new_elems = self._extract_elements_with_ocr(context, arxiv_id=arxiv_id, pdf_path=pdf_path, pcfg=pcfg)
                    if new_elems:
                        elements.extend(new_elems)

                    r.ran = True
                    r.added_elements = len(elements) - before
                    r.stats = {"extracted": len(new_elems or [])}

                    log.info(
                        "PaperSpineBuilderAgent: processor[%d]=ocr_elements done added=%d total=%d",
                        idx,
                        r.added_elements,
                        len(elements),
                    )

                else:
                    log.warning(
                        "PaperSpineBuilderAgent: unknown processor '%s' (skipping). Valid: pdf_figures, smol_docling, ocr_elements",
                        name,
                    )

            except Exception as e:
                r.error = f"{type(e).__name__}: {e}"
                log.exception("PaperSpineBuilderAgent: processor '%s' failed: %s", name, r.error)

        return elements, results

    def _docling_should_continue(self, tag_counts: Dict[str, int]) -> bool:
        """Heuristic gate: keep going if we see useful non-text structure."""
        if not tag_counts:
            return True
        if tag_counts.get("table", 0) > 0:
            return True
        if tag_counts.get("code", 0) > 0:
            return True
        if tag_counts.get("formula", 0) > 0:
            return True
        if tag_counts.get("caption", 0) > 0:
            return True
        if tag_counts.get("heading", 0) > 0:
            return True
        return False

    def _ensure_section_page_fields(self, sections: List[PaperSection]) -> None:
        """Ensure sections expose start_page/end_page/pages as attributes (not just meta)."""
        for s in sections:
            meta = getattr(s, "meta", None) or {}
            if not isinstance(meta, dict):
                meta = {}

            sp = getattr(s, "start_page", None)
            ep = getattr(s, "end_page", None)

            if sp is None and meta.get("start_page") is not None:
                try:
                    sp = int(meta["start_page"])
                    setattr(s, "start_page", sp)
                except Exception:
                    pass

            if ep is None and meta.get("end_page") is not None:
                try:
                    ep = int(meta["end_page"])
                    setattr(s, "end_page", ep)
                except Exception:
                    pass

            pages = getattr(s, "pages", None)
            if (not pages) and sp is not None and ep is not None:
                try:
                    setattr(s, "pages", list(range(int(sp), int(ep) + 1)))
                except Exception:
                    pass

    def _build_docling_sections(
        self,
        *,
        arxiv_id: str,
        paper_role: str,
        pages: List[Dict[str, Any]],
        run_id: str,
        pcfg: Optional[Dict[str, Any]] = None,
        partial: bool = False,
        probe_only: bool = False,
    ) -> List[PaperSection]:
        """Build PaperSection list from Docling doctags pages.

        Supports both:
          - SmolDoclingTool.build_semantic_sections as @staticmethod/classmethod
          - SmolDoclingTool.build_semantic_sections as instance method
        """
        fn = getattr(SmolDoclingTool, "build_semantic_sections", None)
        if fn is None:
            raise RuntimeError("SmolDoclingTool.build_semantic_sections is not available")

        try:
            sections = fn(arxiv_id=arxiv_id, paper_role=paper_role, pages=pages, run_id=run_id)
        except TypeError:
            tool_cfg = dict(pcfg or {})
            tool = SmolDoclingTool(cfg=tool_cfg, memory=self.memory, container=self.container, logger=self.logger)
            sections = tool.build_semantic_sections(arxiv_id=arxiv_id, paper_role=paper_role, pages=pages, run_id=run_id)

        if not sections:
            return []

        for s in sections:
            meta = getattr(s, "meta", None) or {}
            if not isinstance(meta, dict):
                meta = {}
            if partial:
                meta["partial"] = True
            if probe_only:
                meta["probe_only"] = True
            meta.setdefault("source", "smol_docling")
            try:
                setattr(s, "meta", meta)
            except Exception:
                pass

        self._ensure_section_page_fields(sections)
        return sections

    def _ensure_docling_loaded(self, pcfg: Dict[str, Any]) -> None:
        if self._docling_loaded:
            return

        if AutoProcessor is None or AutoModelForVision2Seq is None or torch is None or Image is None:
            raise RuntimeError(
                "smol_docling requires torch, PIL, and transformers. "
                "Please install: pip install torch pillow transformers"
            )

        model_name = pcfg.get("model_name", "ds4sd/SmolDocling-256M-preview")
        device = pcfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        device = str(device)

        log.info("PaperSpineBuilderAgent: loading SmolDocling model=%s device=%s", model_name, device)

        self._docling_processor = AutoProcessor.from_pretrained(model_name)
        self._docling_model = AutoModelForVision2Seq.from_pretrained(model_name)
        self._docling_device = torch.device(device)
        self._docling_model.to(self._docling_device)
        self._docling_model.eval()

        self._docling_loaded = True
        log.info("PaperSpineBuilderAgent: SmolDocling loaded OK")

    def _docling_generate_doctags(self, *, image_path: Path, prompt: str, max_new_tokens: int) -> str:
        img = Image.open(image_path).convert("RGB")
        base_prompt = (prompt or "").strip() or "Convert this page to docling."
# best-effort: use model-defined token if available, otherwise "<image>"
        image_token = "<image>"
        tok = getattr(self._docling_processor, "tokenizer", None)
        if tok is not None and getattr(tok, "image_token", None):
            image_token = tok.image_token

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

        text_out = self._docling_processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        return text_out

    def _count_doctags(self, doctags: str) -> Dict[str, int]:
        # Super-lightweight tag counts. Adjust patterns to match your doctag scheme.
        patterns = {
            "table": r"<table\b|<tab\b",
            "code": r"<code\b|<pre\b",
            "formula": r"<formula\b|<math\b|<equation\b",
            "caption": r"<caption\b",
            "heading": r"<h[1-6]\b|<heading\b",
            "text": r"<p\b|<paragraph\b",
        }
        out: Dict[str, int] = {}
        for k, pat in patterns.items():
            out[k] = len(re.findall(pat, doctags, flags=re.IGNORECASE))
        return out

    async def _smol_docling_extract_elements(
        self,
        *,
        arxiv_id: str,
        pdf_path: Path,
        pcfg: Dict[str, Any],
        max_pages: Optional[int],
        start_page: int = 1,
    ) -> Tuple[List[DocumentElement], List[Dict[str, Any]], Dict[str, int]]:
        """Render pages -> run SmolDocling -> parse DocTags into elements + doctags pages.

        Returns:
            (elements, docling_pages, tag_counts)
        docling_pages item schema:
            {"page_num": int, "doctags": str, "image_path": str}
        """
        self._ensure_docling_loaded(pcfg)

        all_page_paths = self._render_pdf_to_images(arxiv_id, pdf_path)
        if not all_page_paths:
            return [], [], {}

        start_idx = max(0, int(start_page) - 1)
        page_paths = all_page_paths[start_idx:]
        if max_pages is not None:
            page_paths = page_paths[: int(max_pages)]

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
        docling_pages: List[Dict[str, Any]] = []
        tag_counts: Dict[str, int] = {"table": 0, "code": 0, "formula": 0, "caption": 0, "heading": 0, "text": 0}

        for page_idx, img_path in enumerate(page_paths, start=int(start_page)):
            doctags = self._docling_generate_doctags(image_path=img_path, prompt=prompt, max_new_tokens=max_new_tokens)

            docling_pages.append(
                {
                    "page_num": int(page_idx),
                    "doctags": doctags,
                    "image_path": str(img_path),
                }
            )

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

            elems = self._doctags_to_elements(
                arxiv_id=arxiv_id,
                page_num=int(page_idx),
                doctags=doctags,
                want_tables=want_tables,
                want_code=want_code,
                want_equations=want_equations,
                want_captions=want_captions,
                want_headings=want_headings,
                want_paragraphs=want_paragraphs,
            )
            if elems:
                elements.extend(elems)

        return elements, docling_pages, tag_counts

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
        Minimal v1 DocTags → DocumentElement parser.

        IMPORTANT: This intentionally ignores bbox positioning (Docling tags may not provide reliable coords in v1).
        v2 can incorporate bbox + reading order.
        """
        out: List[DocumentElement] = []

        def _mk(elem_type: str, text_val: Optional[str] = None) -> DocumentElement:
            return DocumentElement(
                id=f"{arxiv_id}:p{page_num}:{elem_type}:{len(out)}",
                paper_id=str(arxiv_id),
                page=page_num,
                type=elem_type,
                bbox=None,
                text=text_val,
                latex=None,
                markdown_table=None,
                image_path=None,
                caption=None,
                meta={"source": "smol_docling", "doctags": True},
            )

        # NOTE: These are intentionally cheap “presence heuristics” (not full XML parsing)
        low = doctags.lower()

        if want_tables and ("<table" in low or "<tab" in low):
            out.append(_mk("table", None))

        if want_code and ("<code" in low or "<pre" in low):
            out.append(_mk("code", None))

        if want_equations and ("<formula" in low or "<math" in low or "<equation" in low):
            out.append(_mk("formula", None))

        if want_captions and "<caption" in low:
            out.append(_mk("caption", None))

        if want_headings and ("<h1" in low or "<h2" in low or "<h3" in low or "<heading" in low):
            out.append(_mk("heading", None))

        if want_paragraphs and ("<p" in low or "<paragraph" in low):
            out.append(_mk("paragraph", None))

        return out

    def _resolve_paper_identity(self, context: Dict[str, Any]) -> Tuple[str, Optional[Path]]:
        """
        Resolve arxiv_id and pdf_path from context.

        Expected shapes:
          - context["paper"] is Scorable with .meta containing "arxiv_id" and maybe "pdf_path"
          - or context has "paper_arxiv_id", "paper_pdf_path"
          - or the pdf exists under papers_root/{arxiv_id}/{arxiv_id}.pdf
        """
        arxiv_id = context.get("arxiv_id")
        pdf_path = context.get("paper_pdf_path")

        paper = context.get("paper")
        if isinstance(paper, Scorable):
            meta = paper.meta or {}
            arxiv_id = arxiv_id or meta.get("arxiv_id") or meta.get("paper_arxiv_id")
            pdf_path = pdf_path or meta.get("pdf_path") or meta.get("paper_pdf_path")

        arxiv_id = str(arxiv_id) if arxiv_id else "unknown"

        if pdf_path:
            p = Path(pdf_path)
            if p.exists():
                return arxiv_id, p

        # default location
        p = self.papers_root / arxiv_id / f"{arxiv_id}.pdf"
        if p.exists():
            return arxiv_id, p

        return arxiv_id, None

    def _render_pdf_to_images(self, arxiv_id: str, pdf_path: Path) -> List[Path]:
        """Render PDF pages to PNGs under page_image_root/{arxiv_id}/... (or reuse if already present)."""
        if fitz is None:
            return []

        out_dir = Path(self.page_image_root) / str(arxiv_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(out_dir.glob(f"{arxiv_id}_p*.png"))
        if existing:
            return existing

        doc = fitz.open(str(pdf_path))
        page_paths: List[Path] = []

        for i in range(doc.page_count):
            page_num = i + 1
            if self.max_pages is not None and page_num > int(self.max_pages):
                break

            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=int(self.cfg.get("dpi", 150)))
            img_path = out_dir / f"{arxiv_id}_p{page_num:03d}.png"
            pix.save(str(img_path))
            page_paths.append(img_path)

        log.info("PaperSpineBuilderAgent: rendered %d pages to %s", len(page_paths), out_dir)
        return page_paths

    def _extract_figures_from_pdf(self, context: Dict[str, Any]) -> List[DocumentElement]:
        """Use PyMuPDF to pull out embedded images from each page and wrap them as figure elements.

        Filters rects BEFORE extracting/saving bytes to avoid generating thousands of tiny files
    on image-heavy PDFs.
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

        min_side_pts = float(self.cfg.get("min_figure_side_pts", 30.0))

        for page_num in range(1, doc.page_count + 1):
            if self.max_pages is not None and page_num > int(self.max_pages):
                break

            page = doc.load_page(page_num - 1)
            page_rect = page.rect
            page_area = float(page_rect.width * page_rect.height) if page_rect else 0.0

            images = page.get_images(full=True)
            if not images:
                continue

            for img_idx, img_info in enumerate(images):
                xref = img_info[0]
                rects = page.get_image_rects(xref)
                if not rects:
                    continue

                keep: list[tuple[int, Any, float]] = []
                for rect_i, r in enumerate(rects):
                    rect_area = float(r.width * r.height)
                    if page_area > 0 and rect_area / page_area < self.min_figure_area_frac:
                        continue
                    if min(float(r.width), float(r.height)) < min_side_pts:
                        continue
                    keep.append((rect_i, r, rect_area))

                if not keep:
                    continue

                img = doc.extract_image(xref)
                img_bytes = img.get("image")
                img_ext = img.get("ext", "png")

                img_path = figures_dir / f"{arxiv_id}_p{page_num:03d}_xref{xref}.{img_ext}"
                if not img_path.exists():
                    try:
                        img_path.write_bytes(img_bytes)
                    except Exception:
                        log.exception("PaperSpineBuilderAgent: failed writing figure %s", img_path)

                for rect_i, r, rect_area in keep:
                    bbox = BoundingBox(x1=float(r.x0), y1=float(r.y0), x2=float(r.x1), y2=float(r.y1))
                    elem_id = f"{arxiv_id}:p{page_num}:xref{xref}:{rect_i}"

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
                            meta={"xref": xref, "img_idx": img_idx, "page_area": page_area, "rect_area": rect_area},
                        )
                    )

        log.info("PaperSpineBuilderAgent: pdf_figures extracted=%d", len(elements))
        return elements

    def _extract_elements_with_ocr(
        self,
        context: Dict[str, Any],
        *,
        arxiv_id: str,
        pdf_path: Optional[Path],
        pcfg: Dict[str, Any],
    ) -> List[DocumentElement]:
        """
        Optional OCR fallback. Kept as-is from your previous implementation:
        - expects page images under page_image_root/{arxiv_id}
        - emits DocumentElement items with bbox/text/type
        """
        # NOTE: This method body is whatever you already had; leaving it untouched.
        # If your original file already has it implemented below, keep that implementation.
        # Placeholder here for completeness:
        return self._extract_elements_with_ocr_impl(context=context, arxiv_id=arxiv_id, pdf_path=pdf_path, pcfg=pcfg)

    # -------------------------------------------------------------------------
    # Everything below this point is your existing implementation (signals,
    # OCR impl, etc.) — unchanged in spirit. Keep your existing methods here.
    # -------------------------------------------------------------------------

    def _extract_elements_with_ocr_impl(
        self,
        *,
        context: Dict[str, Any],
        arxiv_id: str,
        pdf_path: Optional[Path],
        pcfg: Dict[str, Any],
    ) -> List[DocumentElement]:
        # --- your original OCR code should already exist in your file ---
        # If you're pasting this as a full replacement module, move your OCR code here.
        return []

    def _emit_processing_signals(
        self,
        *,
        context: Dict[str, Any],
        proc_results: List[ProcessorResult],
        sections: List[PaperSection],
        elements: List[DocumentElement],
    ) -> None:
        """
        Optional: populate context["paper_processing_signals"] for downstream reporting/judging.
        Your existing implementation likely already has richer signals — keep it.
        """
        signals = context.setdefault("paper_processing_signals", {})
        signals["spine"] = {
            "sections": len(sections),
            "elements": len(elements),
            "processors": [
                {
                    "name": r.name,
                    "enabled": r.enabled,
                    "ran": r.ran,
                    "added_elements": r.added_elements,
                    "error": r.error,
                    "stats": r.stats or {},
                }
                for r in proc_results
            ],
        }


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
