# stephanie/tools/pairrm_ranking_tool.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from stephanie.scoring.scorable import Scorable
from stephanie.tools.base_tool import BaseTool  # same interface as EmbeddingTool

log = logging.getLogger(__name__)


@dataclass
class PairRMCandidateScore:
    index: int
    id: str
    score: float


class PairRMRankingTool(BaseTool):
    """
    Fast pairwise-style ranking of multiple candidates using a small reward model.

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

        # Device selection
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
            model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
                self.device
            )

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

    def _score_candidate(self, context_text: str, candidate_text: str) -> float:
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
    # Public helpers (for agents)
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
                f"candidate_ids length ({len(candidate_ids)}) "
                f"must match candidates length ({len(candidates)})"
            )

        scored: List[PairRMCandidateScore] = []
        for idx, (cid, text) in enumerate(zip(candidate_ids, candidates)):
            score = self._score_candidate(context_text, text)
            scored.append(PairRMCandidateScore(index=idx, id=cid, score=score))

        scored_sorted = sorted(scored, key=lambda x: x.score, reverse=True)

        winner_index: Optional[int] = scored_sorted[0].index if scored_sorted else None
        winner_id: Optional[str] = scored_sorted[0].id if scored_sorted else None

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

    async def apply(self, scorable: Scorable, context: Dict[str]) -> Scorable:
        """
        Main entry point for the tool pipeline.

        Expected context:
            context["candidates"]: List[str]           # candidate blog texts
            context["candidate_ids"]: Optional[List]   # ids for each candidate
            context["judge_context"]: Optional[str]    # override scorable.text as context
            context["run_id"]: Optional[str/int]       # for logging

        Behaviour:
            - Uses scorable.text (or judge_context) as shared context.
            - Ranks candidates with PairRM.
            - Writes result into scorable.meta["judgements"][self.name].
            - Optionally logs to self.memory.pairrm_rankings if available.
        """
        candidates: List[str] = context.get("candidates") or []
        if not candidates:
            log.warning(
                "[PairRMRankingTool] No candidates provided in context; skipping."
            )
            return scorable

        judge_context: str = context.get("judge_context") or scorable.text or ""
        candidate_ids: Optional[List[str]] = context.get("candidate_ids")
        run_id: Optional[str] = context.get("run_id")

        result = self.rank_candidates(
            context_text=judge_context,
            candidates=candidates,
            candidate_ids=candidate_ids,
        )

        # Attach to scorable.meta
        meta: Dict[str, Any] = scorable.meta
        judgements: Dict[str, Any] = meta.setdefault("judgements", {})
        judgements[self.name] = {
            "context_char_len": len(judge_context),
            "n_candidates": len(candidates),
            **result,
        }

        # Optional: persist to memory if the store is available
        if self.store_to_memory:
            store = getattr(self.memory, "pairrm_rankings", None)
            if store is None:
                log.debug(
                    "[PairRMRankingTool] memory.pairrm_rankings store not available; "
                    "skipping persistence."
                )
            else:
                try:
                    for rank_idx, item in enumerate(result["candidates"]):
                        cid = item["id"]
                        score = item["score"]
                        row = {
                            "scorable_type": scorable.target_type,
                            "scorable_id": scorable.id,
                            "tool_name": self.name,
                            "candidate_id": cid,
                            "candidate_index": item["index"],
                            "rank": rank_idx,
                            "score": score,
                            "run_id": run_id,
                            "context_char_len": len(judge_context),
                            "candidate_char_len": len(
                                candidates[item["index"]]
                            ),
                        }
                        store.upsert(row)
                except Exception as exc:
                    log.exception(
                        "[PairRMRankingTool] Failed to persist rankings for scorable %r: %s",
                        getattr(scorable, "id", None),
                        exc,
                    )

        log.debug(
            "[PairRMRankingTool] Ranked %d candidates for scorable %r; winner=%r",
            len(candidates),
            getattr(scorable, "id", None),
            result["winner_id"],
        )

        return scorable

    # Convenience API: rank arbitrary texts without a Scorable
    def rank_texts(
        self, context_text: str, candidates: List[str]
    ) -> Dict[str, Any]:
        return self.rank_candidates(context_text=context_text, candidates=candidates)
