# stephanie/components/information/agents/section_research.py
from __future__ import annotations

from typing import Dict, List, Any, Optional

import logging

from stephanie.agents.base_agent import BaseAgent


log = logging.getLogger(__name__)

class SectionResearchAgent(BaseAgent):
    """
    Post-ingest section researcher.

    Responsibilities:
      - For each ingested document (from InformationIngestAgent):
        * Load the CaseBook + MemCube.
        * Find section cases (case_kind == "section").
        * For each section:
            - Build an embedding query.
            - Retrieve evidence snippets via embeddings.
            - (Optionally) generate bullets via LLM.
            - (Optionally) generate a draft section via LLM.
            - Verify faithfulness of the draft against evidence.
            - Persist evidence/draft/metrics as:
                - DynamicScorable(s)
                - MemCube.extra_data["sections"]
      - Write a compact summary to context["section_research"].
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger) -> None:
        super().__init__(cfg, memory, container, logger)

        # --- Config wiring ---
        self.ingest_key: str = cfg.get("ingest_key", "information_ingest")
        self.output_key: str = cfg.get("output_key", "section_research")

        # Embedding / retrieval
        self.embedding_space: str = cfg.get("embedding_space", "papers")
        self.top_k: int = cfg.get("top_k_evidence", 12)
        self.include_ner: bool = cfg.get("include_ner", False)
        self.target_type: str = cfg.get(
            "target_type", "document"
        )  # or "document"
        self.min_cosine_for_evidence: float = cfg.get(
            "min_cosine_for_evidence", 0.55
        )

        # Bullets
        self.bullets_cfg: Dict[str, Any] = cfg.get("bullets", {})
        self.max_bullets: int = self.bullets_cfg.get("max_bullets", 7)

        # Draft
        self.draft_cfg: Dict[str, Any] = cfg.get("draft", {})
        self.draft_enabled: bool = self.draft_cfg.get("enabled", True)

        # Verification
        self.ver_cfg: Dict[str, Any] = cfg.get("verification", {})
        self.verify_enabled: bool = self.ver_cfg.get("enabled", True)
        self.faithfulness_cosine_threshold: float = self.ver_cfg.get(
            "faithfulness_cosine_threshold", 0.75
        )
        self.max_untrusted_fraction: float = self.ver_cfg.get(
            "max_untrusted_fraction", 0.4
        )

        # Persistence flags
        self.write_dynamic_scorables: bool = cfg.get(
            "write_dynamic_scorables", True
        )
        self.write_memcube_sections: bool = cfg.get(
            "write_memcube_sections", True
        )

        # --- Backing stores (through memory) ---
        # Adjust these attribute names if your Memory object differs.
        self.casebook_store = getattr(self.memory, "casebooks", None)
        self.dynamic_scorable_store = getattr(self.memory, "dynamic_scorables", None)

        self.VERIFIED_TAG = "[SELF-VERIFIED]"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects context[self.ingest_key] to be populated by InformationIngestAgent:

            context["information_ingest"] = {
                "status": "ok",
                "documents": [ {...}, ... ]
            }

        Writes summary to context[self.output_key].
        """
        ingest_res = context.get(self.ingest_key)
        if not ingest_res or ingest_res.get("status") != "ok":
            log.warning(
                f"SectionResearchAgent: skipping, no valid '{self.ingest_key}' in context"
            )
            context[self.output_key] = {
                "status": "skipped",
                "reason": "no_valid_ingest_output",
            }
            return context

        docs = ingest_res.get("documents") or []
        results: List[Dict[str, Any]] = []

        for doc in docs:
            try:
                processed = await self._process_document(doc)
                results.append(processed)
            except Exception as exc:  # noqa: BLE001
                doc_id = doc.get("document_id")
                log.exception(
                    f"SectionResearchAgent: error processing document {doc_id}: {exc}"
                )
                results.append(
                    {
                        "document_id": doc_id,
                        "error": str(exc),
                    }
                )

        context[self.output_key] = {
            "status": "ok",
            "documents": results,
        }
        return context

    # ------------------------------------------------------------------
    # Document-level processing
    # ------------------------------------------------------------------
    async def _process_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        doc_id = doc.get("document_id")
        memcube_id = doc.get("memcube_id")
        casebook_id = doc.get("casebook_id")

        if not memcube_id or not casebook_id:
            raise ValueError(
                f"Document {doc_id} missing memcube_id or casebook_id in ingest summary"
            )

        # 1. Load core objects
        memcube = self.memory.memcubes.get_by_id(memcube_id)
        casebook = self.casebook_store.get_casebook(casebook_id)

        # 2. Determine section cases
        section_cases = self._get_section_cases(casebook)
        log.info(
            f"SectionResearchAgent: document {doc_id} has {len(section_cases)} section cases"
        )

        sections_processed: List[Dict[str, Any]] = []

        for case in section_cases:
            try:
                evidence = await self._build_evidence_bundle(case, memcube)
                self._persist_evidence(case, evidence, memcube)
                sections_processed.append(
                    self._format_section_result(case, evidence)
                )
            except Exception as exc:  # noqa: BLE001
                log.exception(
                    f"SectionResearchAgent: section processing failed for "
                    f"case_id={getattr(case, 'id', '?')}: {exc}"
                )

        # 3. Optional: log self-improvement metrics
        self._log_self_improvement(doc_id, sections_processed)

        return {
            "document_id": doc_id,
            "memcube_id": memcube_id,
            "casebook_id": casebook_id,
            "sections_processed": len(sections_processed),
            "sections": sections_processed,
        }

    def _get_section_cases(self, casebook) -> List[Any]:
        """
        Fetch section cases using your existing tagging convention.

        Assumes Case objects are accessible via casebook_store.get_cases(casebook.id)
        and have a .meta mapping with "case_kind" and "section_index" keys.

        Adjust this method if your schema is different.
        """
        all_cases = self.casebook_store.get_cases_for_casebook(casebook.id)
        section_cases = []
        for case in all_cases:
            meta = getattr(case, "meta", {}) or {}
            if meta.get("case_kind") == "section":
                section_cases.append(case)
        return section_cases

    # ------------------------------------------------------------------
    # Evidence building
    # ------------------------------------------------------------------
    async def _build_evidence_bundle(self, case, memcube) -> Dict[str, Any]:
        """
        Build the evidence bundle (snippets, optional bullets + draft, metrics)
        for a single section case.
        """
        section_name = getattr(case, "name", "Unknown Section")
        meta = getattr(case, "meta", {}) or {}
        section_index = meta.get("section_index", 0)
        description = meta.get("description", "") or ""

        # A. Build query text
        query_text = (
            f"Section: {section_name}. "
            f"Goal: explain '{description or section_name}' for ML engineers. "
            f"Focus on intuition and how Stephanie uses this idea."
        )

        # B. Retrieve evidence via embeddings
        snippets = self._retrieve_evidence(query_text)

        # C. (Optional) bullets via LLM
        bullets: List[str] = []
        if self.bullets_cfg and hasattr(self, "llm"):
            bullets = await self._generate_bullets(section_name, snippets)

        # D. (Optional) draft via LLM
        draft_markdown: Optional[str] = None
        if self.draft_enabled and hasattr(self, "llm"):
            draft_markdown = await self._generate_draft(
                section_name, bullets, snippets
            )

        # E. Faithfulness / metrics
        metrics = self._compute_metrics(draft_markdown, snippets)

        return {
            "section_name": section_name,
            "section_index": section_index,
            "query_text": query_text,
            "snippets": snippets,
            "bullets": bullets,
            "draft_markdown": draft_markdown,
            "metrics": metrics,
        }

    def _retrieve_evidence(self, query_text: str) -> List[Dict[str, Any]]:
        """
        Use the embedding store to retrieve top-K evidence snippets.

        Assumes your embedding store exposes something like:

            vec = embedding_store.get_embedding(text)
            results = embedding_store.search_similar(
                vec,
                space=self.embedding_space,
                top_k=self.top_k_evidence,
            )

        and each result is a dict with "text" and "cosine" keys.
        Adjust to your real API if different.
        """

        docs = self.memory.embedding.search_related_scorables(
            query_text, top_k=self.top_k, include_ner=self.include_ner, target_type=self.target_type
        )

        snippets: List[Dict[str, Any]] = []
        for r in docs:
            cosine = r.get("cosine") or r.get("similarity") or 0.0
            if cosine < self.min_cosine_for_evidence:
                continue
            snippets.append(
                {
                    "id": r.get("id"),
                    "text": r.get("text", ""),
                    "source_type": r.get("source_type", "unknown"),
                    "source_id": r.get("source_id"),
                    "cosine": float(cosine),
                }
            )

        return snippets

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------
    async def _generate_bullets(
        self,
        section_name: str,
        snippets: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Call the LLM to extract factual bullets from evidence snippets.
        """
        if not snippets:
            return []

        # Take the top N snippet texts
        text_snippets = [s["text"] for s in snippets[:5]]
        joined_snippets = "\n\n".join(text_snippets)

        prompt = (
            "You are Stephanie's research assistant.\n"
            "Given the section topic and evidence snippets, extract ONLY factual bullets "
            "directly supported by the evidence.\n\n"
            "Constraints:\n"
            f"- Section name: {section_name}\n"
            "- Each bullet <= 12 words\n"
            "- NO speculation, NO new facts\n"
            "- Output format:\n"
            "  - bullet 1\n"
            "  - bullet 2\n"
            "  - ...\n\n"
            "Evidence snippets:\n"
            f"{joined_snippets}\n"
            "\nBullets:\n"
        )

        max_tokens = int(self.bullets_cfg.get("max_tokens", 256))

        response = await self.llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.1,
        )

        # Very simple parsing: lines starting with "-"
        lines = [ln.strip() for ln in response.splitlines()]
        bullets = [ln[1:].strip() for ln in lines if ln.startswith("-")]

        # Keep at most max_bullets
        return bullets[: self.max_bullets]

    async def _generate_draft(
        self,
        section_name: str,
        bullets: List[str],
        snippets: List[Dict[str, Any]],
    ) -> str:
        """
        Call the LLM to generate a draft markdown section based on bullets + evidence.
        """
        bullets_text = "\n".join(f"- {b}" for b in bullets) if bullets else "(no bullets)"
        top_snippets = "\n\n".join(s["text"] for s in snippets[:3])

        prompt = (
            "You are Stephanie, writing an AI Encyclopedia entry.\n\n"
            "Using ONLY the bullets and snippets below, write a short, intuitive "
            "section explaining the concept for an ML engineer.\n\n"
            "Rules:\n"
            "- Length: ~200 words\n"
            "- NO new factual claims beyond what's in bullets/snippets\n"
            "- Clear, concrete language\n"
            "- Use paragraphs, but no headings\n\n"
            f"Section: {section_name}\n\n"
            "Bullets:\n"
            f"{bullets_text}\n\n"
            "Evidence snippets:\n"
            f"{top_snippets}\n\n"
            "Draft:\n"
        )

        max_tokens = int(self.draft_cfg.get("max_tokens", 512))

        response = await self.llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return response.strip()

    # ------------------------------------------------------------------
    # Faithfulness / metrics
    # ------------------------------------------------------------------
    def _compute_metrics(
        self,
        draft_markdown: Optional[str],
        snippets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute basic metrics:
          - faithfulness (1 - hallucination_rate)
          - hallucination_rate via sentence-level nearest neighbor
          - compression_ratio (draft len / evidence len)
        """
        metrics: Dict[str, Any] = {
            "faithfulness": None,
            "hallucination_rate": None,
            "compression_ratio": None,
            "evidence_count": len(snippets),
        }

        if not snippets:
            return metrics

        total_evidence_chars = sum(len(s["text"]) for s in snippets)
        if draft_markdown:
            metrics["compression_ratio"] = len(draft_markdown) / (
                total_evidence_chars + 1
            )

        if not (self.verify_enabled and draft_markdown and self.embedding_store):
            return metrics

        # Sentence-level faithfulness
        sentences = [
            s.strip()
            for s in draft_markdown.replace("\n", " ").split(".")
            if s.strip()
        ]
        if not sentences:
            return metrics

        snippet_texts = [s["text"] for s in snippets]
        snippet_vecs = [
            self.embedding_store.get_embedding(
                txt, space=self.embedding_space
            )
            for txt in snippet_texts
        ]

        untrusted = 0
        for sent in sentences:
            sent_vec = self.embedding_store.get_embedding(
                sent, space=self.embedding_space
            )
            best_cos = self._max_cosine(sent_vec, snippet_vecs)
            if best_cos < self.faithfulness_cosine_threshold:
                untrusted += 1

        halluc_rate = untrusted / max(len(sentences), 1)
        metrics["hallucination_rate"] = halluc_rate
        metrics["faithfulness"] = 1.0 - halluc_rate

        return metrics

    @staticmethod
    def _max_cosine(v, vecs) -> float:
        """Compute max cosine similarity between v and each vector in vecs."""
        if not vecs:
            return 0.0
        # naive implementation; you probably have a better one in EmbeddingStore
        import numpy as np  # type: ignore

        v_arr = np.asarray(v, dtype=float)
        best = -1.0
        for w in vecs:
            w_arr = np.asarray(w, dtype=float)
            denom = (np.linalg.norm(v_arr) * np.linalg.norm(w_arr)) or 1.0
            cos = float(np.dot(v_arr, w_arr) / denom)
            if cos > best:
                best = cos
        return best

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _persist_evidence(self, case, evidence: Dict[str, Any], memcube) -> None:
        """
        Persist evidence as:
          - DynamicScorable per section (if enabled)
          - MemCube.extra_data["sections"] (if enabled)
        """
        # 1) DynamicScorable
        if self.write_dynamic_scorables and self.dynamic_scorable_store is not None:
            self.dynamic_scorable_store.create(
                case_id=getattr(case, "id"),
                scorable_type="section_evidence",
                data={
                    "section_name": evidence["section_name"],
                    "section_index": evidence["section_index"],
                    "bullets": evidence["bullets"],
                    "metrics": evidence["metrics"],
                    "snippets": evidence["snippets"][:3],  # store top few
                },
            )

        # 2) MemCube.extra_data["sections"]
        if self.write_memcube_sections and self.memory.memcubes is not None:
            extra = getattr(memcube, "extra_data", {}) or {}
            sections = extra.get("sections") or []

            section_data = {
                "case_id": getattr(case, "id"),
                "section_index": evidence["section_index"],
                "section_name": evidence["section_name"],
                "metrics": evidence["metrics"],
                "has_draft": bool(evidence["draft_markdown"]),
            }

            # Update if existing, else append
            idx = next(
                (
                    i
                    for i, s in enumerate(sections)
                    if s.get("case_id") == getattr(case, "id")
                ),
                None,
            )
            if idx is not None:
                sections[idx] = section_data
            else:
                sections.append(section_data)

            extra["sections"] = sections
            memcube.extra_data = extra
            self.memory.memcubes.upsert(memcube)

    def _log_self_improvement(
        self,
        doc_id: Any,
        sections: List[Dict[str, Any]],
    ) -> None:
        """
        Log per-document hallucination summary for future HRM / analytics.
        """
        if not sections:
            return

        vals = [
            s.get("metrics", {}).get("hallucination_rate")
            for s in sections
        ]
        vals = [v for v in vals if v is not None]
        if not vals:
            return

        avg_halluc = sum(vals) / len(vals)
        log.info(
            f"SectionResearchAgent: document {doc_id} average hallucination rate = {avg_halluc:.3f}"
        )

    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _format_section_result(
        case,
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "case_id": getattr(case, "id"),
            "section_index": evidence["section_index"],
            "section_name": evidence["section_name"],
            "metrics": evidence["metrics"],
            "has_draft": bool(evidence["draft_markdown"]),
        }
