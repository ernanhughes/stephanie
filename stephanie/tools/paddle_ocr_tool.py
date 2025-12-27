# stephanie/tools/paddle_ocr_tool.py

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

from stephanie.scoring.scorable import Scorable
from stephanie.tools.base_tool import BaseTool

log = logging.getLogger(__name__)

try:
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover - environment-specific
    PaddleOCR = None
    log.warning(
        "[PaddleOcrTool] paddleocr is not installed. "
        "Install with `pip install paddleocr` to enable this tool."
    )


class PaddleOcrTool(BaseTool):
    """
    OCR tool wrapping PaddleOCR.

    Responsibilities:
      - Given an image (or page) path, run PaddleOCR.
      - Attach OCR results to scorable.meta["ocr"][self.name].
      - Optionally persist results in memory.scorable_ocr_results.

    Typical usage:
      - scorable.meta["image_path"] points to a PNG/JPG page image, OR
      - context["image_path"] holds the path, OR
      - scorable.text itself is a file path / URL.

    Stored result format (per line):

        {
          "text": <recognized text>,
          "score": <confidence>,
          "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
          "source": "paddleocr",
        }

    Config keys (YAML):

        use_model: bool (default: True)
        persist: bool (default: True)
        hydrate: bool (default: True)
        force: bool (default: False)     # recompute even if DB row exists

        lang: str (default: "en")
        use_angle_cls: bool (default: True)
        det: bool (default: True)
        rec: bool (default: True)
        cls: bool (default: True)

        min_conf: float (default: 0.05)

        store_to_memory: bool (default: True)
        image_root: optional base dir for relative paths
    """

    name = "paddle_ocr"

    def __init__(self, cfg: Dict[str, Any], memory, container, logger) -> None:
        super().__init__(cfg, memory, container, logger)

        self.use_model: bool = bool(cfg.get("use_model", True))
        self.persist: bool = bool(cfg.get("persist", True))
        self.hydrate: bool = bool(cfg.get("hydrate", True))
        self.force: bool = bool(cfg.get("force", False))
        self.min_conf: float = float(cfg.get("min_conf", 0.05))

        self.image_root: Optional[str] = cfg.get("image_root")

        # --- PaddleOCR configuration (tool-side toggles) ---
        # These can still live in cfg, even if we don't pass them all to PaddleOCR.
        self.lang: str = cfg.get("lang", "en")

        # Kept for backwards compatibility / potential future use in _run_ocr,
        # but NOT passed into the new PaddleOCR constructor:
        self.use_angle_cls: bool = bool(cfg.get("use_angle_cls", True))
        self.det: bool = bool(cfg.get("det", True))
        self.rec: bool = bool(cfg.get("rec", True))
        self.cls: bool = bool(cfg.get("cls", True))

        self.store_to_memory: bool = bool(cfg.get("store_to_memory", True))

        # New-API relevant flags
        self.use_doc_orientation_classify = cfg.get("use_doc_orientation_classify", None)
        self.use_doc_unwarping = cfg.get("use_doc_unwarping", None)
        self.use_textline_orientation = cfg.get("use_textline_orientation", None)

        # Optional fine-tuning knobs for detection / recognition
        self.ocr_version = cfg.get("ocr_version", None)
        self.text_det_limit_side_len = cfg.get("text_det_limit_side_len", None)
        self.text_det_limit_type = cfg.get("text_det_limit_type", None)
        self.text_det_thresh = cfg.get("text_det_thresh", None)
        self.text_det_box_thresh = cfg.get("text_det_box_thresh", None)
        self.text_det_unclip_ratio = cfg.get("text_det_unclip_ratio", None)
        self.text_det_input_shape = cfg.get("text_det_input_shape", None)
        self.text_rec_score_thresh = cfg.get("text_rec_score_thresh", None)
        self.text_rec_input_shape = cfg.get("text_rec_input_shape", None)
        self.return_word_box = cfg.get("return_word_box", None)

        # Initialize PaddleOCR (if available)
        self.ocr = None
        if self.use_model:
            if PaddleOCR is None:
                log.error(
                    "[PaddleOcrTool] PaddleOCR backend unavailable (import failed); "
                    "tool will no-op."
                )
            else:
                # Only pass params that the new PaddleOCR __init__ actually accepts.
                ocr_kwargs: Dict[str, Any] = {
                    "lang": self.lang,
                    "ocr_version": self.ocr_version,
                    "use_doc_orientation_classify": self.use_doc_orientation_classify,
                    "use_doc_unwarping": self.use_doc_unwarping,
                    "use_textline_orientation": self.use_textline_orientation,
                    "text_det_limit_side_len": self.text_det_limit_side_len,
                    "text_det_limit_type": self.text_det_limit_type,
                    "text_det_thresh": self.text_det_thresh,
                    "text_det_box_thresh": self.text_det_box_thresh,
                    "text_det_unclip_ratio": self.text_det_unclip_ratio,
                    "text_det_input_shape": self.text_det_input_shape,
                    "text_rec_score_thresh": self.text_rec_score_thresh,
                    "text_rec_input_shape": self.text_rec_input_shape,
                    "return_word_box": self.return_word_box,
                }

                # Strip out Nones so we don't override defaults unnecessarily
                ocr_kwargs = {k: v for k, v in ocr_kwargs.items() if v is not None}

                self.ocr = PaddleOCR(**ocr_kwargs)
                log.info(
                    "[PaddleOcrTool] Initialized PaddleOCR(lang=%s, ocr_version=%s, "
                    "doc_orient=%s, doc_unwarp=%s, textline_orient=%s)",
                    self.lang,
                    self.ocr_version,
                    self.use_doc_orientation_classify,
                    self.use_doc_unwarping,
                    self.use_textline_orientation,
                )

    # ------------------------------------------------------------------ Tool API --

    async def apply(self, scorable: Scorable, context: Dict[str, Any]) -> Scorable:
        """
        Main entry point, matching BaseTool interface.

        - Determine image/page path.
        - If hydrate+not force: try to load OCR results from DB.
        - Else run PaddleOCR, filter, persist, and attach to scorable.meta["ocr"].
        """
        image_path = self._resolve_image_path(scorable, context)
        if not image_path:
            # Nothing to do; leave scorable unchanged
            return scorable

        # Ensure meta["ocr"] exists
        meta: Dict[str, Any] = scorable.meta
        ocr_meta: Dict[str, Any] = meta.setdefault("ocr", {})

        # If we already have OCR for this tool and not forcing, bail early
        if not self.force and self.name in ocr_meta and not self.hydrate:
            return scorable

        # 1. DB hydration (optional)
        if self.hydrate and not self.force:
            hydrated = self._load_from_db(scorable)
            if hydrated is not None:
                ocr_meta[self.name] = hydrated
                return scorable

        # 2. Run OCR model if allowed
        if self.use_model and self.ocr is not None:
            try:
                ocr_results = await self._run_ocr(image_path)
            except Exception as e:
                log.exception(
                    "[PaddleOcrTool] OCR inference failed for %r: %s", image_path, e
                )
                return scorable
        else:
            ocr_results = []

        # 3. Persist to DB if configured
        if self.persist and ocr_results:
            self._persist_ocr(scorable, ocr_results)

        # 4. Attach to scorable.meta
        ocr_meta[self.name] = ocr_results
        return scorable

    # ------------------------------------------------------------------ helpers ---

    def _resolve_image_path(
        self,
        scorable: Scorable,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """
        Resolve the input image (or page) path from:
          - context["image_path"]
          - scorable.meta["image_path"]
          - scorable.text (if it looks like a path/URL)

        v1: treat everything as a single image file. For PDF page images,
        you can generate one PNG per page and keep the path here.
        """
        # Highest priority: explicit context
        path = context.get("image_path")

        # Next: metadata on the scorable (common in your pipelines)
        if not path:
            meta = scorable.meta or {}
            path = meta.get("image_path") or meta.get("page_image_path")

        # Fallback: scorable.text looks like a path?
        if not path:
            text = (scorable.text or "").strip()
            if text and (os.path.exists(text) or text.lower().startswith(("http://", "https://"))):
                path = text

        if not path:
            return None

        # Resolve relative paths via image_root if configured
        if self.image_root and not path.lower().startswith(("http://", "https://")):
            if not os.path.isabs(path):
                path = os.path.join(self.image_root, path)

        return path

    def _load_from_db(self, scorable: Scorable) -> Optional[List[Dict[str, Any]]]:
        """
        Hydrate OCR results from a scorable_ocr_results table if present.

        Expected schema (you'll create the table):
          - scorable_id
          - scorable_type
          - text
          - score
          - x1, y1, x2, y2, x3, y3, x4, y4 (polygon bbox)
        """
        try:
            store = getattr(self.memory, "scorable_ocr_results", None)
            if store is None:
                return None

            rows = store.find(
                scorable_id=str(scorable.id),
                scorable_type=scorable.target_type,
            )
            if not rows:
                return None

            ocr_results: List[Dict[str, Any]] = []
            for r in rows:
                bbox = [
                    [r.get("x1"), r.get("y1")],
                    [r.get("x2"), r.get("y2")],
                    [r.get("x3"), r.get("y3")],
                    [r.get("x4"), r.get("y4")],
                ]
                ocr_results.append(
                    {
                        "text": r.get("text") or "",
                        "score": float(r.get("score", 1.0)),
                        "bbox": bbox,
                        "source": "db",
                    }
                )

            return ocr_results

        except Exception as e:
            log.error(
                "[PaddleOcrTool] DB hydration failed for %s: %s",
                getattr(scorable, "id", None),
                e,
            )
            return None

    async def _run_ocr(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Run PaddleOCR on the given image path and convert results into
        a standardized list[dict].

        We use the classic `ocr.ocr` API here; if you want to switch to the
        newer `predict()` API you pasted, you can adapt this function.
        """
        if self.ocr is None:
            return []

        # PaddleOCR.ocr returns a nested list. For a single image:
        #   result = [[ [box, (text, score)], ... ]]
        import asyncio

        def _sync_call() -> List[Dict[str, Any]]:
            raw = self.ocr.ocr(image_path, cls=self.cls) or []
            if not raw:
                return []

            # For a single image, we care about raw[0]
            lines = raw[0]
            out: List[Dict[str, Any]] = []
            for line in lines:
                try:
                    box, (text, score) = line
                    score = float(score)
                    if score < self.min_conf:
                        continue
                    # box is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    out.append(
                        {
                            "text": text,
                            "score": score,
                            "bbox": box,
                            "source": "paddleocr",
                        }
                    )
                except Exception as e:
                    log.debug("[PaddleOcrTool] Failed to parse line %r: %s", line, e)
                    continue

            return out

        return await asyncio.to_thread(_sync_call)

    def _persist_ocr(self, scorable: Scorable, ocr_results: List[Dict[str, Any]]) -> None:
        """
        Persist OCR lines into a scorable_ocr_results store, if available.
        """
        try:
            store = getattr(self.memory, "scorable_ocr_results", None)
            if store is None:
                return

            for r in ocr_results:
                bbox = r.get("bbox") or [[None, None]] * 4
                try:
                    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox
                except Exception:
                    x1 = y1 = x2 = y2 = x3 = y3 = x4 = y4 = None

                row = {
                    "scorable_id": scorable.id,
                    "scorable_type": scorable.target_type,
                    "text": r.get("text") or "",
                    "score": float(r.get("score", 1.0)),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "x3": x3,
                    "y3": y3,
                    "x4": x4,
                    "y4": y4,
                }
                try:
                    store.insert(row)
                except Exception as e:
                    log.error(
                        "[PaddleOcrTool] persist failed for %s: %s",
                        scorable.id,
                        e,
                    )

        except Exception as e:
            log.error(
                "[PaddleOcrTool] persist_ocr unexpected failure for %s: %s",
                getattr(scorable, "id", None),
                e,
            )

    # ---------------------------------------------------------------- convenience --

    def ocr_image(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Convenience sync wrapper, similar to SummarizationTool.summarize_text:
        runs OCR on an arbitrary image path and returns the standardized list
        of OCR lines, without needing a Scorable.

        (Use in quick scripts / debugging.)
        """
        if not image_path or self.ocr is None:
            return []

        try:
            raw = self.ocr.ocr(image_path, cls=self.cls) or []
        except Exception as e:
            log.exception("[PaddleOcrTool] ocr_image failed for %r: %s", image_path, e)
            return []

        if not raw:
            return []

        lines = raw[0]
        out: List[Dict[str, Any]] = []
        for line in lines:
            try:
                box, (text, score) = line
                score = float(score)
                if score < self.min_conf:
                    continue
                out.append(
                    {
                        "text": text,
                        "score": score,
                        "bbox": box,
                        "source": "paddleocr",
                    }
                )
            except Exception as e:
                log.debug("[PaddleOcrTool] Failed to parse line %r: %s", line, e)
                continue

        return out
