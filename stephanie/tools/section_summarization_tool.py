from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from stephanie.tools.base_tool import BaseTool  # same interface as EmbeddingTool

log = logging.getLogger(__name__)


class SectionSummarizationTool(BaseTool):
    """
    Generate a short section heading + a technical summary for scorable.text
    using Hugging Face summarization models.

    - Uses two encoder-decoder models:
        * one for the main summary (longer),
        * one for the ultra-short "title" / heading.
    - Results are attached to scorable.meta["summaries"][<tool-name>].
    """

    name = "section_summarizer"

    def __init__(self, cfg: Dict[str, Any], memory, container, logger) -> None:
        super().__init__(cfg, memory, container, logger)

        # Config with sensible defaults for your “CNN + distilled” stack.
        self.summary_model_name: str = cfg.get(
            "summary_model_name",
            "google/pegasus-cnn_dailymail",  # heavier, good for technical-ish summaries
        )
        self.title_model_name: str = cfg.get(
            "title_model_name",
            "sshleifer/distilbart-cnn-12-6",  # lighter, good for short headline-style output
        )

        # Token + length settings
        self.max_input_tokens: int = int(cfg.get("max_input_tokens", 1024))

        self.max_summary_tokens: int = int(cfg.get("max_summary_tokens", 160))
        self.min_summary_tokens: int = int(cfg.get("min_summary_tokens", 40))

        self.max_title_tokens: int = int(cfg.get("max_title_tokens", 16))
        self.min_title_tokens: int = int(cfg.get("min_title_tokens", 3))

        # Optional flag to skip if already summarized
        self.force: bool = bool(cfg.get("force", False))

        # Device selection
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(
            "[SectionSummarizationTool] Using device=%s, summary_model=%s, title_model=%s",
            self.device,
            self.summary_model_name,
            self.title_model_name,
        )

        # Load models/tokenizers once and reuse
        self._summary_tok, self._summary_model = self._load_model(
            self.summary_model_name
        )

        if self.title_model_name == self.summary_model_name:
            # Share weights if same model requested
            self._title_tok = self._summary_tok
            self._title_model = self._summary_model
        else:
            self._title_tok, self._title_model = self._load_model(
                self.title_model_name
            )

    # ------------------------------------------------------------------
    # HF helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_name: str):
        """
        Load a Seq2Seq model + tokenizer.

        We use half-precision on CUDA for memory/speed; full precision on CPU.
        """
        log.info("[SectionSummarizationTool] Loading model %s", model_name)

        tok = AutoTokenizer.from_pretrained(model_name)

        if self.device == "cuda":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            ).to(self.device)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        model.eval()
        return tok, model

    def _encode(self, tokenizer, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize + move to device, truncating long sections.
        """
        return tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_input_tokens,
            truncation=True,
        ).to(self.device)

    def _generate_seq2seq(
        self,
        model,
        tokenizer,
        text: str,
        max_len: int,
        min_len: int,
    ) -> str:
        """
        Generic helper for encoder-decoder summarization.
        """
        if not text.strip():
            return ""

        inputs = self._encode(tokenizer, text)

        # Basic, deterministic beam search (tune later if you like)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                num_beams=4,
                max_length=max_len,
                min_length=min_len,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        text_out = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        return text_out

    def _generate_summary(self, text: str) -> str:
        return self._generate_seq2seq(
            self._summary_model,
            self._summary_tok,
            text,
            max_len=self.max_summary_tokens,
            min_len=self.min_summary_tokens,
        )

    def _generate_title(self, text: str) -> str:
        """
        Ultra-short "title". We keep it very short; for now we simply
        ask the summarization model for a tiny summary, which tends to
        be headline-like on technical text.
        """
        return self._generate_seq2seq(
            self._title_model,
            self._title_tok,
            text,
            max_len=self.max_title_tokens,
            min_len=self.min_title_tokens,
        )

    # ------------------------------------------------------------------
    # Tool API
    # ------------------------------------------------------------------

    async def apply(self, scorable, context: dict):
        """
        Main entry point, matching BaseTool / EmbeddingTool interface.

        - Reads scorable.text
        - Writes scorable.meta["summaries"][self.name] = {title, summary, ...}
        """
        text: str = getattr(scorable, "text", "") or ""
        if not text.strip():
            return scorable

        # Ensure meta exists
        meta: Dict[str, Any] = getattr(scorable, "meta", None) or {}
        setattr(scorable, "meta", meta)

        summaries: Dict[str, Any] = meta.setdefault("summaries", {})

        if not self.force and self.name in summaries:
            # Already summarized; skip unless forced
            log.debug(
                "[SectionSummarizationTool] Summary already exists for scorable %r",
                getattr(scorable, "id", None),
            )
            return scorable

        # Optional: allow caller to override device via context
        device_override: Optional[str] = context.get("device")
        if device_override and device_override != self.device:
            log.warning(
                "[SectionSummarizationTool] Ignoring device override %s (tool is already on %s)",
                device_override,
                self.device,
            )

        try:
            # Generate summary + title
            section_summary = self._generate_summary(text)
            section_title = self._generate_title(text)

            summaries[self.name] = {
                "title": section_title,
                "summary": section_summary,
                "summary_model": self.summary_model_name,
                "title_model": self.title_model_name,
            }

            log.debug(
                "[SectionSummarizationTool] Summarized scorable %r (title=%r)",
                getattr(scorable, "id", None),
                section_title,
            )

        except Exception as exc:
            log.exception(
                "[SectionSummarizationTool] Failed to summarize scorable %r: %s",
                getattr(scorable, "id", None),
                exc,
            )

        return scorable
