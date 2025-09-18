# stephanie/agents/thought/paper_blog.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.utils.casebook_utils import generate_casebook_name


SENTS_MIN_DEFAULT = 4
SENTS_MAX_DEFAULT = 20


class SimplePaperBlogAgent(BaseAgent):
    """
    Inputs (context[self.input_key]): list of docs with at least:
      - id (int/str)
      - title (str)
      - summary (str)  # arXiv summary (author provided / arXiv auto)
    Will look up abstract from memory.document_sections for the doc id.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.min_sents = int(cfg.get("min_sents", SENTS_MIN_DEFAULT))
        self.max_sents = int(cfg.get("max_sents", SENTS_MAX_DEFAULT))
        self.training_min_overall = float(
            cfg.get("training_min_overall", 0.75)
        )
        self.training_max_halluc = float(cfg.get("training_max_halluc", 0.10))
        self.scoring = container.get("scoring")  # optional
        # sensible defaults for model keys
        self.model_key_ranker = cfg.get("model_key_ranker", "ranker.sicql.v1")
        self.model_key_retriever = cfg.get(
            "model_key_retriever", "retriever.mrq.v1"
        )
        self.casebook_action = cfg.get("casebook_action", "blog")

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        documents = context.get(self.input_key, [])
        self.report(
            {
                "event": "start",
                "step": "SimplePaperSummarizer",
                "details": f"Profiling {len(documents)} documents",
            }
        )

        out_map = {}
        for doc in documents:
            doc_id = doc.get("id")
            title = doc.get("title", "") or ""

            casebook_name = generate_casebook_name(self.casebook_action, title) 
            casebook = self.memory.casebooks.ensure_cb(self.casebook_action, casebook_name, tag=self.casebook_action)

            arxiv_summary = (
                doc.get("summary", "") or ""
            )  # treat as "author/arXiv summary"

            # --- fetch abstract from sections ---
            abstract = self._fetch_abstract(doc_id)

            merged_context = {
                "title": title,
                "summary": arxiv_summary,
                "min_sents": self.min_sents,
                "max_sents": self.max_sents,
                "abstract": abstract,
                **context,
            }

            prompt = self.prompt_loader.load_prompt(self.cfg, merged_context)
            model_out = self.call_llm(prompt, context=context)

            self.report(
                {
                    "event": "llm_output",
                    "step": "paper_summarizer",
                    "prompt": prompt[:12000],  # avoid huge logs
                    "response": (model_out or "")[:12000],
                }
            )

            parsed = self._parse_model_output(model_out or "")
            valid, msg, val_meta = self._validate_output(
                parsed.get("summary", ""), self.min_sents, self.max_sents
            )
            if not valid:
                self.logger.log(
                    "SummaryValidationFailed",
                    {"doc_id": doc_id, "reason": msg, **val_meta},
                )
                # still record raw, but mark invalid
                out_map[doc_id] = {
                    **parsed,
                    "valid": False,
                    "validation_reason": msg,
                }
                continue

            # --- compute metrics vs sources ---
            metrics = self._compute_metrics(
                parsed["summary"], abstract, arxiv_summary
            )

            # --- persist scorable w/ safe rollback on DB glitch ---
            paper = {
                "paper_id": doc.get("paper_id", doc_id),
                "title": title,
                "abstract": abstract,
                "arxiv_summary": arxiv_summary,
            }

            scorable_id = self._persist_scorable_document(
                paper=paper,
                summary_text=parsed["summary"],
                intro_text=parsed.get("intro", ""),
                metrics=metrics,
                context=context,
            )

            # --- training events (pointwise + optional pairwise vs arXiv summary) ---
            if scorable_id:
                try:
                    self._emit_training_events(
                        paper={
                            "paper_id": doc.get("paper_id", doc_id),
                            "title": title,
                            "abstract": abstract,
                            "arxiv_summary": arxiv_summary,
                        },
                        summary=parsed["summary"],
                        metrics=metrics,
                        context=context,
                    )
                except Exception as e:
                    self.memory.session.rollback()
                    self.logger.log(
                        "TrainingEventEmitError",
                        {"doc_id": doc_id, "error": str(e)},
                    )

            out_map[doc_id] = {
                **parsed,
                "valid": True,
                "metrics": metrics,
                "scorable_id": scorable_id,
            }

        context.setdefault(self.output_key, {})
        context[self.output_key]["summary_v0"] = out_map
        return context

    # ---------- persistence & events ----------

    def _persist_scorable_document(
        self,
        paper: Dict[str, Any],
        summary_text: str,
        intro_text: str,
        metrics: Dict[str, float],
        context: Dict[str, Any],
    ) -> Optional[str]:
        """
        Save a scorable/document and ensure an embedding exists. Rolls back on DB failure.
        """
        full_text = f"## Summary\n{summary_text}\n\n## Blog post introduction\n{intro_text}".strip()

        summary_scorable = self.memory.dynamic_scorables.add(
            pipeline_run_id=context.get("pipeline_run_id"),
            scorable_type=TargetType.DYNAMIC,
            source=self.name,
            text=full_text,
            source_scorable_id=paper.get("paper_id"),
            source_scorable_type="document",
            meta={
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "arxiv_summary": paper.get("arxiv_summary"), 
                "text": full_text,
                "summary": summary_text,
                "metrics": metrics,
            },
        )
        self.memory.embedding.get_or_create(full_text)
        return summary_scorable.id

    def _emit_training_events(
        self, paper: Dict[str, Any], summary: str, metrics: Dict[str, float], context: Dict[str, Any]
    ):
        """
        Emits:
          - Pointwise: positive example (summary) with weight = overall
          - Pairwise: baseline vs author/arXiv summary, if author summary present
        Uses thresholds to avoid training on low-quality samples.
        """
        if (
            metrics.get("overall", 0.0) < self.training_min_overall
            or metrics.get("hallucination_rate", 1.0)
            > self.training_max_halluc
        ):
            self.logger.log(
                "TrainingEventSkipped",
                {
                    "reason": "low_quality",
                    "overall": metrics.get("overall", 0.0),
                    "hallucination_rate": metrics.get(
                        "hallucination_rate", 1.0
                    ),
                },
            )
            return

        title = paper.get("title", "paper")
        # --- Pointwise (retriever) ---
        self.memory.training_events.add_pointwise(
            model_key=self.model_key_retriever,
            dimension="alignment",
            query_text=title,
            cand_text=summary,
            label=1,
            weight=float(metrics.get("overall", 0.7)),
            trust=float(metrics.get("overall", 0.7)),
            goal_id=None,
            pipeline_run_id=self._maybe_pipeline_run_id(),
            agent_name=self.name,
            source="track1_baseline",
            meta={
                "stage": "track1",
                "claim_coverage": metrics.get("claim_coverage"),
                "faithfulness": metrics.get("faithfulness"),
            },
        )

        # --- Pairwise (ranker) vs author/arXiv summary if available ---
        arxiv_summary = paper.get("summary") or ""
        if arxiv_summary.strip():
            author_metrics = self._compute_metrics(
                arxiv_summary, paper.get("abstract", ""), arxiv_summary
            )
            prefer_baseline = metrics["overall"] > author_metrics["overall"]

            pos_text = summary if prefer_baseline else arxiv_summary
            neg_text = arxiv_summary if prefer_baseline else summary

            self.memory.training_events.add_pairwise(
                model_key=self.model_key_ranker,
                dimension="alignment",
                query_text=title,
                pos_text=pos_text,
                neg_text=neg_text,
                weight=0.5,
                trust=0.3,
                goal_id=context.get("goal",{}).get("id"),
                pipeline_run_id=context.get("pipeline_run_id"),
                agent_name=self.name,
                source="track1_baseline",
                meta={
                    "stage": "track1",
                    "baseline_score": metrics["overall"],
                    "author_score": author_metrics["overall"],
                    "prefer_baseline": prefer_baseline,
                },
            )

    def _maybe_pipeline_run_id(self) -> Optional[str]:
        try:
            return getattr(self, "pipeline_run_id", None) or None
        except Exception:
            return None

    # ---------- parsing / validation ----------

    def _parse_model_output(self, text: str) -> Dict[str, str]:
        """
        Supports either:
          ## Summary
          ...
          ## Blog post introduction
          ...
        OR the older variant including Score/Rationale (will ignore for persistence).
        """

        def grab(section: str) -> str:
            m = re.search(
                rf"^##\s*{section}\s*\n(.+?)(?=^##|\Z)", text, re.S | re.M
            )
            return (m.group(1).strip() if m else "").strip()

        # Try new format
        summary = grab("Summary")
        intro = grab("Blog post introduction")

        # If missing, try legacy keys
        if not summary:
            summary = grab(
                "Blog post introduction for the paper"
            )  # fallback (rare)
        # Optional legacy score + rationale
        score_raw = grab("Score") or ""
        rationale = grab("Rational") or grab("Rationale") or ""

        return {
            "summary": summary,
            "intro": intro,
            "score_self": score_raw,
            "rationale": rationale,
        }

    def _validate_output(
        self, summary: str, min_sents: int, max_sents: int
    ) -> Tuple[bool, str, Dict[str, Any]]:
        if not summary:
            return False, "Missing 'Summary' section", {}

        sents = [
            s
            for s in re.split(r"(?<=[.!?])\s+", summary)
            if len(s.strip()) > 1
        ]
        if not (min_sents <= len(sents) <= max_sents):
            return (
                False,
                f"Summary must have {min_sents}-{max_sents} sentences (found {len(sents)})",
                {"sentence_count": len(sents)},
            )

        # lightweight hallucination markers
        hallucination_markers = [
            ("not specified", 0.4),
            ("we propose", 0.6),
            ("novel approach", 0.6),
        ]
        hscore = sum(
            w for t, w in hallucination_markers if t in summary.lower()
        )
        if hscore > 1.0:
            return (
                False,
                "Summary contains hallucination markers",
                {"hallucination_score": hscore},
            )

        return (
            True,
            "ok",
            {"sentence_count": len(sents), "hallucination_score": hscore},
        )

    # ---------- metrics (cheap + deterministic) ----------

    def _compute_metrics(
        self, summary: str, abstract: str, arxiv_summary: str
    ) -> Dict[str, float]:
        # 1) Claim coverage from abstract sentences (first 2–3 + numeric lines)
        claims = self._extract_key_claims(abstract)
        covered = sum(
            1 for claim in claims if self._contains_concept(summary, claim)
        )
        claim_coverage = covered / max(1, len(claims))

        # 2) Faithfulness via cosine on embeddings (summary vs sources)
        abstract_sim = (
            self._cosine_similarity(summary, abstract) if abstract else 0.0
        )
        author_sim = (
            self._cosine_similarity(summary, arxiv_summary)
            if arxiv_summary
            else 0.0
        )
        faithfulness = 0.7 * abstract_sim + 0.3 * author_sim

        # 3) Structure (problem→approach→results→implications heuristic)
        structure_score = self._evaluate_structure(summary)

        # 4) Hallucination: count sentences with verbs + not present in sources by fuzzy sim
        hallucination_issues = self._detect_hallucinations(
            summary, abstract, arxiv_summary
        )
        sent_count = max(
            1, len([s for s in re.split(r"(?<=[.!?])", summary) if s.strip()])
        )
        hallucination_rate = min(1.0, len(hallucination_issues) / sent_count)

        overall = (
            (claim_coverage * 0.4)
            + ((1 - hallucination_rate) * 0.4)
            + (structure_score * 0.2)
        )
        return {
            "claim_coverage": float(claim_coverage),
            "faithfulness": float(faithfulness),
            "structure": float(structure_score),
            "hallucination_rate": float(hallucination_rate),
            "sentence_count": sent_count,
            "tokens": len(summary.split()),
            "overall": float(overall),
        }

    # ---------- small helpers used by metrics ----------

    def _fetch_abstract(self, doc_id) -> str:
        try:
            sections = self.memory.document_sections.get_by_document(doc_id)
            for s in sections:
                sd = s.to_dict()
                if (
                    sd.get("section_name") or ""
                ).lower().strip() == "abstract":
                    return sd.get("section_text", "") or ""
        except Exception as e:
            self.logger.log(
                "AbstractFetchFailed", {"doc_id": doc_id, "error": str(e)}
            )
        return ""

    def _cosine_similarity(self, a: str, b: str) -> float:
        try:
            va = self.memory.embedding.get_or_create(a)
            vb = self.memory.embedding.get_or_create(b)
        except Exception as e:
            self.logger.log("EmbedSimFallback", {"error": str(e)})
            return 0.0
        # cosine
        dot = sum(x * y for x, y in zip(va, vb))
        na = max(1e-8, sum(x * x for x in va) ** 0.5)
        nb = max(1e-8, sum(y * y for y in vb) ** 0.5)
        return float(dot / (na * nb))

    def _extract_key_claims(self, abstract: str) -> List[str]:
        if not abstract:
            return []
        sents = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", abstract)
            if len(s.strip()) > 0
        ]
        # take first 2–3 + any sentence with numbers
        key = sents[:3]
        key += [s for s in sents[3:] if re.search(r"\d", s)]
        # unique & trimmed to ~3–5
        uniq = []
        for s in key:
            if s not in uniq:
                uniq.append(s)
        return uniq[:5]

    def _contains_concept(self, text: str, claim: str) -> bool:
        # quick semantic+lexical check
        if self._cosine_similarity(text, claim) >= 0.65:
            return True
        # fallback lexical overlap
        t = set(re.findall(r"[a-z0-9]+", text.lower()))
        c = set(re.findall(r"[a-z0-9]+", claim.lower()))
        if not c:
            return False
        overlap = len(t & c) / len(c)
        return overlap >= 0.25

    def _evaluate_structure(self, summary: str) -> float:
        # look for cues: problem/approach/results/implications
        s = summary.lower()
        cues = 0
        cues += 1 if re.search(r"(problem|challenge|gap|motivation)", s) else 0
        cues += (
            1
            if re.search(
                r"(we|the paper|the authors|method|approach|model|framework)",
                s,
            )
            else 0
        )
        cues += (
            1
            if re.search(
                r"(results|experiments|evaluation|improv(e|ement)|accuracy|performance)",
                s,
            )
            else 0
        )
        cues += (
            1
            if re.search(
                r"(implication|impact|application|future work|limitations)", s
            )
            else 0
        )
        return min(1.0, cues / 4.0)

    def _detect_hallucinations(
        self, summary: str, abstract: str, arxiv_summary: str
    ) -> List[str]:
        # any sentence far from both sources is suspicious
        issues = []
        for sent in [
            s.strip()
            for s in re.split(r"(?<=[.!?])", summary)
            if len(s.strip()) > 0
        ]:
            sim_a = (
                self._cosine_similarity(sent, abstract) if abstract else 0.0
            )
            sim_b = (
                self._cosine_similarity(sent, arxiv_summary)
                if arxiv_summary
                else 0.0
            )
            if max(sim_a, sim_b) < 0.45 and re.search(r"[A-Za-z]", sent):
                issues.append(sent)
        return issues
