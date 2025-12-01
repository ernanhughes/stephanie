# stephanie/components/nexus/agents/blossom_sharpen_executor.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.thought.paper_blog import SimplePaperBlogAgent

# Reuse your sharpening prompt/logic patterns
# If you want to import directly from SharpenedPaperSummarizerAgent, you can.
# Here we inline the bits we need to avoid circular imports.

MAX_BRANCHING = 3
DEPTH_LIMIT = 3
MIN_GAIN = 0.02
TARGET_CONF = 0.85
MIN_OVERALL = 0.75
MAX_ITERS = 3


def _cfg_get(cfg, key, default):
    try:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)
    except Exception:
        return default


class BlossomSharpenExecutorAgent(BaseAgent):
    """
    Blossom executor that uses a sharpen loop to expand each thought node (GoT/ToT-style).
    Works on paper-style inputs (title/abstract/summary) now, but the structure is generic.

    Input context:
      - documents: List[dict] with keys: id or paper_id, title, summary (optional)
      - goal (optional)
      - pipeline_run_id (optional)

    Effects:
      - Creates Blossom graph (models/blossom.py), nodes & edges via memory.blossoms
      - Logs SharpeningResult / Prediction via memory.sharpening_results / memory.sharpening_predictions
      - Emits training events for MRQ/SICQL, identical to your summarizer agent
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Search parameters
        self.max_branching = int(_cfg_get(cfg, "max_branching", MAX_BRANCHING))
        self.depth_limit = int(_cfg_get(cfg, "depth_limit", DEPTH_LIMIT))

        # Sharpen parameters
        self.max_iters = int(_cfg_get(cfg, "max_iters", MAX_ITERS))
        self.min_gain = float(_cfg_get(cfg, "min_gain", MIN_GAIN))
        self.target_conf = float(
            _cfg_get(cfg, "target_confidence", TARGET_CONF)
        )
        self.min_overall = float(_cfg_get(cfg, "min_overall", MIN_OVERALL))
        self.min_figure_score = float(_cfg_get(cfg, "min_figure_score", 0.70))
        self.min_sents = int(_cfg_get(cfg, "min_sents", 4))
        self.max_sents = int(_cfg_get(cfg, "max_sents", 20))

        # Models
        self.model_key_ranker = _cfg_get(
            cfg, "model_key_ranker", "ranker.sicql.v1"
        )
        self.model_key_retriever = _cfg_get(
            cfg, "model_key_retriever", "retriever.mrq.v1"
        )

        # Metrics helper (consistent with Track A/B)
        self.metrics = SimplePaperBlogAgent(cfg, memory, container, logger)

        # Optional scoring service
        self.scoring = container.get("scoring")

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.report(
            {
                "event": "start",
                "step": "BlossomExecutor",
                "details": "GoT/ToT blossom with sharpening",
            }
        )
        docs = context.get("documents") or []
        results = []

        for doc in docs:
            doc_id = doc.get("paper_id") or doc.get("id")
            if not doc_id:
                self.logger.log(
                    "BlossomSkipNoDocId", {"doc_keys": list(doc.keys())}
                )
                continue

            # 1) Seed blossom
            goal_id = (context.get("goal") or {}).get("id")
            run_id = context.get("pipeline_run_id")
            blossom = self.memory.blossoms.create_blossom(
                {
                    "goal_id": goal_id,
                    "pipeline_run_id": run_id,
                    "agent_name": self.name,
                    "strategy": "got",
                    "seed_type": "document",
                    "seed_id": str(doc_id),
                    "params": {
                        "max_branching": self.max_branching,
                        "depth_limit": self.depth_limit,
                        "sharpen": {
                            "max_iters": self.max_iters,
                            "min_gain": self.min_gain,
                            "target_conf": self.target_conf,
                            "min_overall": self.min_overall,
                            "min_figure_score": self.min_figure_score,
                            "min_sents": self.min_sents,
                            "max_sents": self.max_sents,
                        },
                    },
                    "status": "running",
                }
            )

            # 2) Root node from baseline (Track A)
            baseline_summary, baseline_obj = self._fetch_baseline_summary(
                doc_id, doc
            )
            if not baseline_summary:
                self.logger.log("BlossomBaselineMissing", {"doc_id": doc_id})
                self.memory.blossoms.update_status(blossom.id, "failed")
                continue

            base_metrics = self._score_summary(
                baseline_summary,
                self._fetch_abstract(doc.get("id") or doc_id),
                doc.get("summary", ""),
            )

            root = self.memory.blossoms.add_node(
                {
                    "blossom_id": blossom.id,
                    "parent_id": None,
                    "depth": 0,
                    "order_index": 0,
                    "state_text": baseline_summary,
                    "sharpened_text": baseline_summary,
                    "accepted": True,
                    "scores": {
                        "overall": base_metrics.get("overall", 0.0),
                        "metrics": base_metrics,
                    },
                    "features": None,
                    "tags": ["root", "baseline"],
                    "extra_data": {
                        "doc_id": str(doc_id),
                        "title": doc.get("title", ""),
                    },
                }
            )
            self.memory.blossoms.set_root(blossom.id, root.id)

            # 3) Expand with sharpened candidates breadth-first up to depth_limit
            frontier = [(root.id, 0)]
            created_nodes = 1
            created_edges = 0
            best_node = root
            best_overall = base_metrics.get("overall", 0.0)

            while frontier:
                parent_id, depth = frontier.pop(0)
                if depth >= self.depth_limit:
                    continue

                parent = self.memory.blossoms.get_node(parent_id)
                parent_overall = (parent.scores or {}).get("overall", 0.0)

                candidates = self._expand_and_sharpen(
                    doc=doc,
                    baseline_text=parent.sharpened_text
                    or parent.state_text
                    or "",
                    k=self.max_branching,
                )

                # Rank candidates by overall score; pick top-k (already limited by k)
                candidates.sort(
                    key=lambda c: c["metrics"]["overall"], reverse=True
                )

                for order_idx, cand in enumerate(candidates):
                    gain = cand["metrics"]["overall"] - parent_overall
                    accepted = (
                        cand["metrics"]["overall"] >= self.min_overall
                        and gain >= self.min_gain
                    )

                    child = self.memory.blossoms.add_node(
                        {
                            "blossom_id": blossom.id,
                            "parent_id": parent.id,
                            "depth": depth + 1,
                            "order_index": order_idx,
                            "state_text": cand["raw"],
                            "sharpened_text": cand["summary"],
                            "accepted": bool(accepted),
                            "scores": {
                                "overall": cand["metrics"]["overall"],
                                "metrics": cand["metrics"],
                            },
                            "features": None,
                            "tags": ["expand", "sharpened"]
                            + (["accepted"] if accepted else ["rejected"]),
                            "sharpen_passes": cand["iters"],
                            "sharpen_gain": gain,
                            "sharpen_meta": {
                                "iterations": cand["iterations"],
                                "hallucination_issues": cand[
                                    "hallucination_issues"
                                ],
                                "figure_results": cand["figure_results"],
                            },
                            "extra_data": {"prompt": cand["prompt"]},
                        }
                    )
                    self.memory.blossoms.add_edge(
                        {
                            "blossom_id": blossom.id,
                            "src_node_id": parent.id,
                            "dst_node_id": child.id,
                            "relation": "expand",
                            "score": cand["metrics"]["overall"],
                            "rationale": None,
                        }
                    )
                    created_nodes += 1
                    created_edges += 1

                    # Persist sharpening A/B record (baseline vs child) for analytics
                    try:
                        self._persist_sharpening_result(
                            goal_id=goal_id,
                            run_id=run_id,
                            doc=doc,
                            baseline_text=parent.sharpened_text
                            or parent.state_text
                            or "",
                            cand=cand,
                        )
                    except Exception as e:
                        self.logger.log(
                            "BlossomSharpenPersistWarn", {"error": str(e)}
                        )

                    if accepted:
                        frontier.append((child.id, depth + 1))
                        if cand["metrics"]["overall"] > best_overall:
                            best_overall = cand["metrics"]["overall"]
                            best_node = child

            # 4) Select best path (backtrack from best_node)
            best_path = self._backtrack_path(best_node.id)
            path_score = sum(
                (n["scores"] or {}).get("overall", 0.0) for n in best_path
            ) / max(1, len(best_path))

            # 5) Emit training events (aligns with your summarizer)
            try:
                title = doc.get("title", "")
                baseline_text = baseline_summary
                enhanced_text = (
                    best_node.sharpened_text
                    or best_node.state_text
                    or baseline_text
                )
                baseline_metrics = base_metrics
                enhanced_metrics = (best_node.scores or {}).get(
                    "metrics", {"overall": best_overall}
                )
                self._emit_training_events(
                    title=title,
                    baseline_summary=baseline_text,
                    enhanced_summary=enhanced_text,
                    baseline_metrics=baseline_metrics,
                    enhanced_metrics=enhanced_metrics,
                    goal_id=goal_id,
                    run_id=run_id,
                )
            except Exception as e:
                self.logger.log("BlossomTrainEventWarn", {"error": str(e)})

            # 6) Finalize
            self.memory.blossoms.finalize(
                blossom.id,
                {
                    "nodes": created_nodes,
                    "edges": created_edges,
                    "best_overall": best_overall,
                    "path_len": len(best_path),
                    "avg_path_score": path_score,
                },
            )

            results.append(
                {
                    "blossom_id": blossom.id,
                    "best_overall": best_overall,
                    "avg_path_score": path_score,
                    "path_node_ids": [n["id"] for n in best_path],
                }
            )

        context["blossom_results"] = results
        return context

    # ---------- Expansion via Sharpen loop ----------

    def _expand_and_sharpen(
        self, doc: Dict[str, Any], baseline_text: str, k: int
    ) -> List[Dict[str, Any]]:
        """
        Generate k candidates by running a short sharpen loop starting from baseline_text.
        """
        out: List[Dict[str, Any]] = []
        title = doc.get("title", "")
        abstract = self._fetch_abstract(doc.get("id") or doc.get("paper_id"))
        arxiv_summary = doc.get("summary", "")

        # Start from diverse rewrites: ask LLM for k different angles first, then sharpen each.
        diversity_prompt = self._build_diversity_prompt(
            title, abstract, baseline_text, k
        )
        diverse_seed = self.call_llm(diversity_prompt).strip()
        seeds = [
            s.strip("-• ").strip()
            for s in diverse_seed.split("\n")
            if s.strip()
        ]
        seeds = [s for s in seeds if len(s) > 10][:k] or [baseline_text]

        for seed in seeds[:k]:
            best_summary = seed
            best_metrics = self._score_summary(
                best_summary, abstract, arxiv_summary
            )
            iterations = []
            no_gain = 0
            last_prompt = ""

            for i in range(self.max_iters):
                prompt = self._build_super_sharpen_prompt(
                    title=title,
                    abstract=abstract,
                    summary=best_summary,
                    min_sents=self.min_sents,
                    max_sents=self.max_sents,
                )
                last_prompt = prompt  # <--- track last used prompt
                cand_text = self.call_llm(prompt).strip()

            ok_hall, hall_issues = self._verify_hallucinations(
                best_summary, abstract, arxiv_summary
            )
            fig_check = self._verify_figure_grounding(best_summary, doc)

            out.append(
                {
                    "raw": seed,
                    "summary": best_summary,
                    "metrics": best_metrics,
                    "iterations": iterations,
                    "iters": len(iterations),
                    "hallucination_issues": hall_issues,
                    "figure_results": fig_check,
                    "prompt": last_prompt,  # <--- use safe variable
                }
            )

        return out

    # ---------- Prompts & Metrics (reused patterns) ----------

    def _build_diversity_prompt(
        self, title: str, abstract: str, summary: str, k: int
    ) -> str:
        abstract_snip = (abstract or "")[:1000]
        return f"""
You are brainstorming {k} alternative perspectives for a paper summary, each emphasizing different aspects
(method, results, limitations, future work, applications, setup). Keep each to 1-2 sentences.

Paper: {title}

Abstract:
\"\"\"
{abstract_snip}
\"\"\"

Current summary:
\"\"\"{summary}\"\"\"

List {k} distinct alternative angles (each on its own line):
""".strip()

    def _build_super_sharpen_prompt(
        self,
        *,
        title: str,
        abstract: str,
        summary: str,
        min_sents: int,
        max_sents: int,
    ) -> str:
        abstract_snip = (abstract or "")[:1000]
        return f"""
You are an expert science editor. Improve the paper summary below using a combined **GROWS + CRITIC + REFLECT** loop.

Constraints:
- Output **one paragraph** of {min_sents}-{max_sents} sentences.
- Use ONLY facts present in the abstract; if a detail is missing, prefer generic phrasing over guessing.
- Avoid first person, questions, citations/links, and equations.

Paper Title: {title}

Abstract:
\"\"\"
{abstract_snip}
\"\"\"

Current summary:
\"\"\"{summary}\"\"\"

Rewrite now (one paragraph, {min_sents}-{max_sents} sentences):
""".strip()

    def _extract_summary(self, text: str) -> str:
        # If the model wrapped it in ## Summary, prefer that block; else return whole text
        import re

        m = re.search(r"(?mi)^##\s*Summary\s*\n(.+?)(?=^##|\Z)", text, re.S)
        return m.group(1).strip() if m else text.strip()

    def _score_summary(
        self, summary: str, abstract: str, arxiv_summary: str
    ) -> Dict[str, float]:
        return self.metrics._compute_metrics(summary, abstract, arxiv_summary)

    def _verify_hallucinations(
        self, summary: str, abstract: str, arxiv_summary: str
    ):
        issues = self.metrics._detect_hallucinations(
            summary, abstract, arxiv_summary
        )
        return len(issues) == 0, issues

    def _verify_figure_grounding(
        self, summary: str, paper_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Simple claim/citation heuristic (same as your agent)
        import re

        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()
        ]
        claims = []
        for sent in sentences:
            m = re.findall(
                r"(\b\d+\.?\d*\s*(%|percent|accuracy|precision|recall|f1|auc|rmse|mae|bleu)\b)",
                sent,
                flags=re.I,
            )
            if m:
                claims.append(
                    {
                        "claim": sent,
                        "value": m[0][0],
                        "metric": m[0][1],
                        "has_citation": any(
                            t in sent.lower()
                            for t in [
                                "figure",
                                "fig.",
                                "table",
                                "as shown",
                                "see",
                            ]
                        ),
                    }
                )
        cited = sum(1 for c in claims if c["has_citation"])
        rate = cited / max(1, len(claims))
        return {
            "total_claims": len(claims),
            "properly_cited": cited,
            "citation_rate": rate,
            "overall_figure_score": rate,
            "claims": claims,
        }

    def _fetch_baseline_summary(
        self, doc_id, doc: Dict[str, Any]
    ) -> Tuple[str, Optional[Any]]:
        """
        Try dynamic_scorables provenance first (Track A), else fallback to doc.summary if present.
        """
        baseline_obj = None
        try:
            sid = int(doc_id)
            baseline_obj = (
                self.memory.dynamic_scorables.get_latest_by_source_pointer(
                    source="paper_summarizer",
                    source_scorable_type="document",
                    source_scorable_id=sid,
                )
            )
        except Exception:
            baseline_obj = (
                self.memory.dynamic_scorables.get_latest_by_source_and_meta(
                    source="paper_summarizer",
                    meta_key="paper_id",
                    meta_value=str(doc_id),
                )
            )
        baseline_text = ""
        if baseline_obj:
            t = baseline_obj.text or ""
            baseline_text = self._extract_summary(t) or (
                baseline_obj.meta or {}
            ).get("summary", "")
        if not baseline_text:
            baseline_text = (doc.get("summary") or "").strip()
        return baseline_text, baseline_obj

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

    def _backtrack_path(self, node_id: int) -> List[Dict[str, Any]]:
        """
        Follow parent pointers to root; return path [root..node].
        """
        path = []
        seen = set()
        current = self.memory.blossoms.get_node(node_id)
        while current and current.id not in seen:
            path.append(current.to_dict())
            seen.add(current.id)
            current = current.parent
        path.reverse()
        return path

    def _emit_training_events(
        self,
        title: str,
        baseline_summary: str,
        enhanced_summary: str,
        baseline_metrics: Dict[str, float],
        enhanced_metrics: Dict[str, float],
        goal_id: Optional[int],
        run_id: Optional[int],
    ):
        if not enhanced_summary or not baseline_summary:
            return

        base_overall = float((baseline_metrics or {}).get("overall", 0.0))
        enh_overall = float((enhanced_metrics or {}).get("overall", 0.0))
        gain = enh_overall - base_overall
        if math.isnan(gain):
            gain = 0.0

        w = max(0.1, min(1.0, gain + 0.3))
        trust_pt = max(0.1, min(1.0, enh_overall))

        # pointwise (+)
        self.memory.training_events.insert_pointwise(
            {
                "model_key": self.model_key_retriever,
                "dimension": "alignment",
                "query_text": title,
                "cand_text": enhanced_summary,
                "label": 1,
                "weight": enh_overall,
                "trust": trust_pt,
                "goal_id": goal_id,
                "pipeline_run_id": run_id,
                "agent_name": self.name,
                "source": "blossom",
                "meta": {"stage": "blossom", "gain": gain},
            },
            dedup=True,
        )

        # pairwise (+) vs baseline
        self.memory.training_events.insert_pairwise(
            {
                "model_key": self.model_key_ranker,
                "dimension": "alignment",
                "query_text": title,
                "pos_text": enhanced_summary,
                "neg_text": baseline_summary,
                "weight": w,
                "trust": w * 0.6,
                "goal_id": goal_id,
                "pipeline_run_id": run_id,
                "agent_name": self.name,
                "source": "blossom",
                "meta": {
                    "stage": "blossom",
                    "enhanced_score": enh_overall,
                    "baseline_score": base_overall,
                    "gain": gain,
                },
            },
            dedup=True,
        )

    def _persist_sharpening_result(
        self, goal_id, run_id, doc, baseline_text: str, cand: Dict[str, Any]
    ):
        """
        Persist A/B comparison into SharpeningResult + SharpeningPrediction tables if available.
        """
        title = doc.get("title", "")
        self.memory.sharpening_results.insert(
            {
                "id": None,  # let DB/autogen handle if configured
                "goal": title,
                "prompt": (cand.get("prompt") or "")[:2000],
                "template": "super_sharpen_v1",
                "original_output": baseline_text,
                "sharpened_output": cand["summary"],
                "preferred_output": cand["summary"]
                if cand["metrics"]["overall"] >= self.min_overall
                else baseline_text,
                "winner": "b"
                if cand["metrics"]["overall"] >= self.min_overall
                else "a",
                "improved": cand["metrics"]["overall"] >= self.min_overall,
                "comparison": f"gain={cand['metrics']['overall']:.4f}",
                "score_a": 0.0,  # optional: plug in baseline 'overall'
                "score_b": cand["metrics"]["overall"],
                "score_diff": cand["metrics"][
                    "overall"
                ],  # baseline assumed 0 if unknown
                "best_score": cand["metrics"]["overall"],
                "prompt_template": "GROWS+CRITIC+REFLECT",
            }
        )
        self.memory.sharpening_predictions.insert(
            {
                "goal_id": goal_id or 0,
                "prompt_text": (cand.get("prompt") or "")[:512],
                "output_a": baseline_text[:512],
                "output_b": cand["summary"][:512],
                "preferred": "b",
                "predicted": "b",
                "value_a": 0.0,
                "value_b": cand["metrics"]["overall"],
            }
        )
