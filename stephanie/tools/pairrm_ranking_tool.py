# stephanie/tools/pairrm_ranking_tool.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from stephanie.scoring.scorable import Scorable
from stephanie.tools.base_tool import BaseTool

log = logging.getLogger(__name__)


@dataclass
class PairRMCandidateScore:
    index: int
    id: str
    score: float


class PairRMRankingTool(BaseTool):
    """
    Fast ranking of multiple candidates using a small reward model.

    Typical usage:
        - context: short source description (paper summary, seed blog summary, etc.)
        - candidates: 3–8 alternative blog posts or sections

    This tool:
        - scores each candidate with a PairRM-style reward model,
        - sorts candidates by score (descending),
        - optionally persists scores to memory.

    Integration patterns:

      1) Direct call from your agent:
            tool = PairRMRankingTool(cfg, memory, container, logger)
            result = tool.rank_candidates(context_text, candidate_texts)

      2) Via .apply() inside a pipeline:
            # scorable.text is used as context
            # context["candidates"] is a list of candidate texts
            scorable = await tool.apply(scorable, {"candidates": candidate_texts, "run_id": run_id})
    """

    name = "pairrm_ranker"

    def __init__(self, cfg: Dict[str, Any], memory, container, logger) -> None:
        super().__init__(cfg, memory, container, logger)

        # HF model configuration
        self.model_name: str = cfg.get(
            "model_name",
            "llm-blender/PairRM-hf",  # you can switch to "llm-blender/PairRM"
        )
        self.max_input_tokens: int = int(cfg.get("max_input_tokens", 2048))
        self.store_to_memory: bool = bool(cfg.get("store_to_memory", True))

        # Optional prompt template for context + candidate
        # You can override this via cfg if you want a more PairRM-style format.
        self.prompt_template: str = cfg.get(
            "prompt_template",
            (
                "You are a reward model scoring the quality of a candidate answer.\n\n"
                "Context:\n{context}\n\n"
                "Candidate:\n{candidate}\n"
            ),
        )

        # New: write-back keys + top-k selection
        self.candidates_in_key: str = cfg.get(
            "candidates_in_key", "blog_candidates"
        )  # preferred
        self.legacy_candidates_in_key: str = cfg.get(
            "legacy_candidates_in_key", "candidates"
        )

        self.ranked_out_key: str = cfg.get(
            "ranked_out_key", "blog_candidates_ranked"
        )
        self.winner_out_key: str = cfg.get(
            "winner_out_key", "blog_candidate_winner"
        )
        self.preference_pairs_out_key: str = cfg.get(
            "preference_pairs_out_key", "preference_pairs"
        )

        self.top_k: int = int(cfg.get("top_k", 1))
        self.preview_chars: int = int(
            cfg.get("preview_chars", 240)
        )  # avoid huge meta
        self.attach_full_text: bool = bool(
            cfg.get("attach_full_text", False)
        )  # opt-in

        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(
            "[PairRMRankingTool] Using device=%s, model=%s",
            self.device,
            self.model_name,
        )

        # Load model/tokenizer once and reuse
        self._tokenizer, self._model = self._load_model(self.model_name)

    # ------------------------------------------------------------------
    # HF helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_name: str):
        """
        Load a reward / classification model + tokenizer.

        We use half-precision on CUDA for memory/speed; full precision on CPU.
        """
        log.info("[PairRMRankingTool] Loading model %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.device == "cuda":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            ).to(self.device)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            ).to(self.device)

        model.eval()
        return tokenizer, model

    def _encode(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize + move to device, truncating long inputs.
        """
        return self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_input_tokens,
            truncation=True,
        ).to(self.device)

    def _build_input(self, context_text: str, candidate_text: str) -> str:
        """
        Build model input from context + candidate.

        If you later want closer adherence to the official PairRM prompt format,
        you can update this function or pass a different prompt_template in cfg.
        """
        return self.prompt_template.format(
            context=context_text.strip(),
            candidate=candidate_text.strip(),
        )

    def _score_candidate(
        self, context_text: str, candidate_text: str
    ) -> float:
        """
        Compute a scalar reward score for one candidate given the shared context.
        Higher is better.
        """
        if not candidate_text.strip():
            return float("-inf")

        model_input = self._build_input(context_text, candidate_text)
        inputs = self._encode(model_input)

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits

        # Heuristic: handle both single-logit and multi-logit heads.
        if logits.ndim == 2 and logits.size(0) == 1:
            if logits.size(1) == 1:
                score = logits[0, 0].item()
            else:
                # If there are multiple classes, take the mean as a simple scalar reward.
                score = logits[0].mean().item()
        else:
            score = logits.view(-1)[0].item()

        return float(score)

    # ------------------------------------------------------------------
    # Candidate normalization (NEW)
    # ------------------------------------------------------------------

    def _normalize_candidates(
        self,
        raw: List[Union[str, Dict[str, Any]]],
        candidate_ids: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """
        Returns:
          texts: List[str]
          ids:   List[str]
          payloads: List[dict]  # original metadata for write-back (id/meta/etc.)
        """
        if not raw:
            return [], [], []

        texts: List[str] = []
        ids: List[str] = []
        payloads: List[Dict[str, Any]] = []

        if candidate_ids is not None and len(candidate_ids) != len(raw):
            raise ValueError(
                f"candidate_ids length ({len(candidate_ids)}) must match candidates length ({len(raw)})"
            )

        for i, item in enumerate(raw):
            if isinstance(item, str):
                cid = candidate_ids[i] if candidate_ids else str(i)
                text = item
                payload = {"id": cid, "index": i, "kind": "text"}
            elif isinstance(item, dict):
                cid = str(
                    item.get("id")
                    or (candidate_ids[i] if candidate_ids else str(i))
                )
                # support common keys
                text = (
                    item.get("markdown")
                    or item.get("text")
                    or item.get("content")
                    or ""
                )
                payload = dict(item)
                payload.setdefault("id", cid)
                payload.setdefault("index", i)
                payload.setdefault("kind", "object")
            else:
                cid = candidate_ids[i] if candidate_ids else str(i)
                text = str(item)
                payload = {"id": cid, "index": i, "kind": "unknown"}

            ids.append(cid)
            texts.append(text)
            payloads.append(payload)

        return texts, ids, payloads

    def _make_preview(self, text: str) -> str:
        t = (text or "").strip().replace("\n", " ")
        if len(t) <= self.preview_chars:
            return t
        return t[: self.preview_chars] + "…"

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def rank_candidates(
        self,
        context_text: str,
        candidates: List[str],
        candidate_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Rank multiple candidate texts given a shared context.

        Args:
            context_text: Common source context (e.g. paper summary).
            candidates: List of candidate texts (blog posts or sections).
            candidate_ids: Optional stable IDs for each candidate; if omitted, uses '0', '1', ...

        Returns:
            {
                "context": <context_text>,
                "candidates": [
                    {"index": int, "id": str, "score": float},
                    ...
                ],
                "winner_index": int | None,
                "winner_id": str | None,
            }
        """
        if not candidates:
            return {
                "context": context_text,
                "candidates": [],
                "winner_index": None,
                "winner_id": None,
            }

        if candidate_ids is None:
            candidate_ids = [str(i) for i in range(len(candidates))]
        elif len(candidate_ids) != len(candidates):
            raise ValueError(
                f"candidate_ids length ({len(candidate_ids)}) must match candidates length ({len(candidates)})"
            )

        scored: List[PairRMCandidateScore] = []
        for idx, (cid, text) in enumerate(zip(candidate_ids, candidates)):
            score = self._score_candidate(context_text, text)
            scored.append(PairRMCandidateScore(index=idx, id=cid, score=score))

        scored_sorted = sorted(scored, key=lambda x: x.score, reverse=True)
        winner_index: Optional[int] = (
            scored_sorted[0].index if scored_sorted else None
        )
        winner_id: Optional[str] = (
            scored_sorted[0].id if scored_sorted else None
        )

        return {
            "context": context_text,
            "candidates": [
                {"index": s.index, "id": s.id, "score": s.score}
                for s in scored_sorted
            ],
            "winner_index": winner_index,
            "winner_id": winner_id,
        }

    # ------------------------------------------------------------------
    # Tool API (pipeline integration)
    # ------------------------------------------------------------------

    async def apply(
        self, scorable: Scorable, context: Dict[str, Any]
    ) -> Scorable:
        """
        Updated expected context (Paper→Blog loop):
          - context["blog_candidates"]: List[dict] preferred (id, markdown, meta...)
                OR context["candidates"]: List[str] legacy
          - context["candidate_ids"]: Optional[List[str]]
          - context["judge_context"]: Optional[str]  # overrides scorable.text
          - context["goal"]: Optional[str]           # prepended into judge_context
          - context["run_id"]: Optional[str/int]
        Writes back:
          - context[self.ranked_out_key]
          - context[self.winner_out_key]
          - context[self.preference_pairs_out_key]
        """
        raw_candidates = context.get(self.candidates_in_key)
        if raw_candidates is None:
            raw_candidates = context.get(self.legacy_candidates_in_key)

        raw_candidates = raw_candidates or []
        if not raw_candidates:
            log.warning(
                "[PairRMRankingTool] No candidates provided; skipping."
            )
            return scorable

        candidate_ids: Optional[List[str]] = context.get("candidate_ids")
        texts, ids, payloads = self._normalize_candidates(
            raw_candidates, candidate_ids=candidate_ids
        )

        judge_context: str = (
            context.get("judge_context") or scorable.text or ""
        )
        goal: str = context.get("goal") or ""
        if goal:
            judge_context = f"Goal:\n{goal.strip()}\n\n" + judge_context

        run_id: Optional[str] = context.get("run_id")

        result = self.rank_candidates(
            context_text=judge_context, candidates=texts, candidate_ids=ids
        )

        # Build ranked candidate objects for write-back
        ranked_items: List[Dict[str, Any]] = []
        for rank, item in enumerate(result["candidates"]):
            idx = item["index"]
            payload = payloads[idx]
            ranked = {
                "rank": rank,
                "pairrm_score": float(item["score"]),
                "id": payload.get("id", item["id"]),
                "index": idx,
                "meta": payload.get("meta", {}),
                "preview": self._make_preview(texts[idx]),
            }
            if self.attach_full_text:
                ranked["markdown"] = texts[idx]
            ranked_items.append(ranked)

        # Winner + top_k
        winner_id = result["winner_id"]
        winner = ranked_items[0] if ranked_items else None
        top_k = ranked_items[: max(1, self.top_k)]

        # Preference pairs (winner vs each loser)
        pref_pairs: List[Dict[str, Any]] = []
        if ranked_items:
            w = ranked_items[0]
            for loser in ranked_items[1:]:
                pref_pairs.append(
                    {
                        "winner_id": w["id"],
                        "loser_id": loser["id"],
                        "winner_score": w["pairrm_score"],
                        "loser_score": loser["pairrm_score"],
                        "delta": float(
                            w["pairrm_score"] - loser["pairrm_score"]
                        ),
                        "run_id": run_id,
                        "tool": self.name,
                    }
                )

        # Attach to scorable.meta (lightweight)
        meta: Dict[str, Any] = scorable.meta
        judgements: Dict[str, Any] = meta.setdefault("judgements", {})
        judgements[self.name] = {
            "context_char_len": len(judge_context),
            "n_candidates": len(texts),
            "winner_id": winner_id,
            "top_k": [
                {
                    "id": x["id"],
                    "rank": x["rank"],
                    "pairrm_score": x["pairrm_score"],
                }
                for x in top_k
            ],
            "ranked": [
                {
                    "id": x["id"],
                    "rank": x["rank"],
                    "pairrm_score": x["pairrm_score"],
                }
                for x in ranked_items
            ],
        }

        # Write-back to pipeline context (THIS is what you’ll use in the blog pipeline)
        context[self.ranked_out_key] = ranked_items
        context[self.winner_out_key] = winner
        context[self.preference_pairs_out_key] = pref_pairs

        # Optional: persist to memory (same as before, but now uses normalized IDs)
        if self.store_to_memory:
            store = getattr(self.memory, "pairrm_rankings", None)
            if store is None:
                log.debug(
                    "[PairRMRankingTool] memory.pairrm_rankings store not available; skipping persistence."
                )
            else:
                try:
                    for item in ranked_items:
                        idx = int(item["index"])
                        row = {
                            "scorable_type": scorable.target_type,
                            "scorable_id": scorable.id,
                            "tool_name": self.name,
                            "candidate_id": item["id"],
                            "candidate_index": idx,
                            "rank": int(item["rank"]),
                            "score": float(item["pairrm_score"]),
                            "run_id": run_id,
                            "context_char_len": len(judge_context),
                            "candidate_char_len": len(texts[idx]),
                        }
                        store.upsert(row)
                except Exception as exc:
                    log.exception(
                        "[PairRMRankingTool] Failed to persist rankings for scorable %r: %s",
                        getattr(scorable, "id", None),
                        exc,
                    )

        log.debug(
            "[PairRMRankingTool] Ranked %d candidates; winner=%r",
            len(texts),
            winner_id,
        )
        return scorable

    def rank_texts(
        self, context_text: str, candidates: List[str]
    ) -> Dict[str, Any]:
        return self.rank_candidates(
            context_text=context_text, candidates=candidates
        )
