from __future__ import annotations
import asyncio
import logging
import re
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Protocol,
    runtime_checkable,
)


# Contract from monitor.py
@runtime_checkable
class PairScorer(Protocol):
    async def score_text_pair(
        self,
        goal: str,
        reply: str,
        *,
        model_alias: str = "chat",
        monitor_alias: str = "selfcheck",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]: ...


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


@dataclass
class SelfCheckConfig:
    samples: int = 5  # K: resamples from the chat model
    temperature: float = 0.8
    max_tokens: int = 512
    nli_batch: int = 16  # batch size when scoring entailment/contradiction
    timeout_s: float = (
        6.0  # overall budget (keep low for “finalized reply only”)
    )
    min_sentence_len: int = 20  # skip trivial sentences
    lexical_floor: float = 0.25  # if no NLI, use lexical-agreement fallback


class SelfCheckScorer(PairScorer):
    """
    SelfCheckGPT-style monitor:
      1) Sample K additional responses (goal -> y_k)
      2) Compare original reply’s sentences against samples using NLI (contradiction) if available;
         fall back to lexical/embedding agreement heuristic.
      3) Risk = fraction of sentences that are contradicted or low-agreement.
      4) Confidence = agreement mass; Δ-gap ≈ risk proxy (mapped in aligner later if needed).

    Expects on container:
      - chat_sampler: async def sample(goal, temperature, max_tokens, n) -> List[str]
      - tiny_nli: async def score(premises: List[str], hypotheses: List[str]) -> List[Dict[str,float]]
        where each dict has probs for {"entail","neutral","contradict"}
      (Both are optional; we degrade gracefully.)
    """

    def __init__(
        self,
        container: Any,
        logger: Optional[logging.Logger] = None,
        cfg: Optional[SelfCheckConfig] = None,
    ):
        self.container = container
        self.cfg = cfg or SelfCheckConfig()
        self.logger = logger or logging.getLogger(__name__)

        self.chat_sampler = getattr(container, "chat_sampler", None)
        self.tiny_nli = getattr(container, "tiny_nli", None)
        self.embedder = getattr(
            container, "embedder", None
        )  # optional for cosine sim fallback

    # --------------- public -------------------------------------------------
    async def score_text_pair(
        self,
        goal: str,
        reply: str,
        *,
        model_alias: str = "chat",
        monitor_alias: str = "selfcheck",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        try:
            samples = await self._resample(goal)
            sentences = [
                s.strip()
                for s in _SENT_SPLIT.split(reply)
                if len(s.strip()) >= self.cfg.min_sentence_len
            ]
            if not sentences or not samples:
                # fallback neutral if we can’t run the protocol
                return dict(
                    confidence01=0.5,
                    faithfulness_risk01=0.5,
                    ood_hat01=0.5,
                    delta_gap01=0.5,
                )

            # Score each sentence by contradiction/consistency across samples
            risks, agrees = await self._score_sentences(sentences, samples)

            # Aggregate
            halluc_fraction = sum(risks) / max(
                1, len(risks)
            )  # 0..1, higher = worse
            agreement = sum(agrees) / max(
                1, len(agrees)
            )  # 0..1, higher = better

            return dict(
                confidence01=_clamp01(agreement),
                faithfulness_risk01=_clamp01(halluc_fraction),
                ood_hat01=float(
                    context.get("ood_hat01", 0.5) if context else 0.5
                ),  # leave to other monitors/aligner
                delta_gap01=_clamp01(
                    halluc_fraction
                ),  # reasonable proxy; aligner can recompute with baseline
                sentences=len(sentences),
                samples=len(samples),
            )
        except Exception:
            self.logger.exception(
                "SelfCheckScorer failed; returning neutral metrics"
            )
            return dict(
                confidence01=0.5,
                faithfulness_risk01=0.5,
                ood_hat01=0.5,
                delta_gap01=0.5,
            )

    # --------------- internals ---------------------------------------------
    async def _resample(self, goal: str) -> List[str]:
        if callable(self.chat_sampler):
            try:
                return await asyncio.wait_for(
                    self.chat_sampler(
                        goal,
                        temperature=self.cfg.temperature,
                        max_tokens=self.cfg.max_tokens,
                        n=self.cfg.samples,
                    ),
                    timeout=self.cfg.timeout_s,
                )
            except Exception:
                self.logger.warning(
                    "chat_sampler failed; falling back to empty"
                )
        return []

    async def _score_sentences(
        self, sentences: List[str], samples: List[str]
    ) -> Tuple[List[float], List[float]]:
        # For each sentence s in reply:
        #   compute contradiction rate vs samples (NLI), or lexical/embedding agreement.
        risks, agrees = [], []
        if callable(self.tiny_nli):
            premises, hypos = [], []
            for s in sentences:
                for y in samples:
                    premises.append(y)
                    hypos.append(s)
            # batched NLI
            out: List[Dict[str, float]] = []
            for i in range(0, len(premises), self.cfg.nli_batch):
                batch_p = premises[i : i + self.cfg.nli_batch]
                batch_h = hypos[i : i + self.cfg.nli_batch]
                try:
                    res = await self.tiny_nli(
                        batch_p, batch_h
                    )  # returns list of {entail,neutral,contradict}
                except Exception:
                    res = [
                        dict(entail=0.33, neutral=0.34, contradict=0.33)
                        for _ in batch_p
                    ]
                out.extend(res)
            # reduce per sentence
            idx = 0
            for s in sentences:
                n = len(samples)
                contrad = sum(
                    out[idx + j].get("contradict", 0.0) for j in range(n)
                ) / max(1, n)
                entail = sum(
                    out[idx + j].get("entail", 0.0) for j in range(n)
                ) / max(1, n)
                idx += n
                risks.append(_clamp01(contrad))
                agrees.append(_clamp01(entail))
        else:
            # lexical/embedding fallback
            sample_sets = [
                set(re.findall(r"[A-Za-z0-9]+", y.lower())) for y in samples
            ]
            for s in sentences:
                sset = set(re.findall(r"[A-Za-z0-9]+", s.lower()))
                jac = sum(_jaccard(sset, t) for t in sample_sets) / max(
                    1, len(sample_sets)
                )
                # if embedder exists, blend with cosine similarity
                if callable(self.embedder):
                    try:
                        vec_s = await self.embedder([s])
                        vec_t = await self.embedder(samples)
                        import numpy as np

                        cos = float(
                            np.mean(
                                [
                                    (
                                        np.dot(vec_s[0], vec_t[i])
                                        / (
                                            np.linalg.norm(vec_s[0])
                                            * np.linalg.norm(vec_t[i])
                                            + 1e-8
                                        )
                                    )
                                    for i in range(len(vec_t))
                                ]
                            )
                        )
                        jac = 0.5 * jac + 0.5 * (cos + 1) / 2.0
                    except Exception:
                        pass
                agree = max(jac, self.cfg.lexical_floor)
                agrees.append(_clamp01(agree))
                risks.append(_clamp01(1.0 - agree))
        return risks, agrees
