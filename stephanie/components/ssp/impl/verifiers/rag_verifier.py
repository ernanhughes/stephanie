# stephanie/components/ssp/impl/verifiers/rag_verifier.py
from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple

from stephanie.components.ssp.core.protocols import VerificationResult


class RAGVerifier:
    """
    Adversarial verifier (paper-style judge):
    Compares Proposer's SEED_ANSWER (A) vs Solver's predicted answer (B),
    judged by one or more LLMs using a strict 3-line output format.

    Primary API (paper-style):
        verify(question, seed_answer, predicted_answer, evidence, context)
          -> (solver_wins: bool, score_1_to_100: float, details: Dict)

    Optional adapter (protocol-friendly):
        verify_as_result(...) -> VerificationResult
    """

    _WIN_RE  = re.compile(r"^\s*winner\s*:\s*([ab])\s*$", re.IGNORECASE | re.MULTILINE)
    _CONF_RE = re.compile(r"^\s*confidence\s*:\s*([\d]+(?:\.\d+)?)\s*$", re.IGNORECASE | re.MULTILINE)
    _RAT_RE  = re.compile(r"^\s*rationale\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

    def __init__(self, container, logger, memory, cfg: Optional[Dict[str, Any]] = None):
        self.container = container
        self.logger = logger
        self.memory = memory
        self.cfg = cfg or {}
        self.prompt_service = container.get("prompt")

        vcfg = self.cfg.get("verifier", self.cfg) or {}
        # Models used to judge A vs B
        # Note need larger models with good instruction-following ability
        self.judge_models: List[Any] = list(vcfg.get("judge_models", ["ollama/qwen3"]))
        self.max_evidence: int = int(vcfg.get("max_evidence", 5))
        self.auto_win_on_exact_match: bool = bool(vcfg.get("auto_win_on_exact_match", True))

    # --------------------------- public API ---------------------------

    async def verify(
        self,
        question: str,
        seed_answer: str,
        predicted_answer: str,
        evidence: List[str],
        context: Dict[str, Any],
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Run an adversarial judgment between two answers.

        Returns:
            (solver_wins, confidence_1_to_100, details)
        """
        # Guard: no prediction → proposer wins by default with low confidence
        if not predicted_answer or not str(predicted_answer).strip():
            return False, 0.0, {"winner": "A", "reason": "empty_prediction"}

        # Optional fast-path: exact match
        if self.auto_win_on_exact_match and str(predicted_answer).strip() == str(seed_answer).strip():
            details = {
                "winner": "B",
                "votes": {"A": 0, "B": 1},
                "confidences": {"exact_match": 100.0},
                "avg_confidence": 100.0,
                "rationales": {"exact_match": "Predicted answer equals the ground truth seed."},
                "raw_outputs": {"exact_match": "rationale: identical\nwinner: B\nconfidence: 100"},
                "models": ["exact_match"],
                "evidence_used": min(len(evidence), self.max_evidence),
            }
            return True, 100.0, details

        prompt = self._build_prompt(
            question=question,
            seed_answer=seed_answer,
            predicted_answer=predicted_answer,
            evidence_subset=evidence[: self.max_evidence],
        )

        try:
            # Prefer multi-model competition so PromptService can log pairwise/pointwise training events
            if hasattr(self.prompt_service, "run_prompt_multi"):
                # Supply a judge callback so TrainingEventStore logs labels & pairs
                raw = await self.prompt_service.run_prompt_multi(
                    prompt_text=prompt,
                    models=self.judge_models,
                    judge=self._judge_callback,  # <— enables TrainingEventStore logging
                    context=context,
                    dimension="answer_quality",
                    goal_id=(context or {}).get("goal_id"),
                    pipeline_run_id=(context or {}).get("pipeline_run_id"),
                    agent_name="rag-verifier",
                )
                # run_prompt_multi returns {"outputs": {model: text}, "winner": key|None, "scores": {key: score}}
                outputs: Dict[str, str] = raw.get("outputs", {}) if isinstance(raw, dict) else {}
            else:
                # Fallback: ask each judge independently
                async def ask(model: Any) -> Tuple[str, str]:
                    out = await self.prompt_service.run_prompt(
                        prompt_text=prompt,
                        context={**(context or {}), "judge_model": (getattr(model, "name", None) or model)},
                        model=(model if isinstance(model, (str, dict)) else None),
                    )
                    key = getattr(model, "name", None) or str(model)
                    return key, out

                pairs = await asyncio.gather(*[ask(m) for m in self.judge_models])
                outputs = {m: out for (m, out) in pairs}

            # Parse each judge’s 3-line verdict
            winners: Dict[str, str] = {}
            confidences: Dict[str, float] = {}
            rationales: Dict[str, str] = {}
            for model_key, txt in outputs.items():
                w, c, r = self._parse_single_judgment(str(txt))
                if w is None:
                    w, c, r = "A", 50.0, r or "unparsed"
                winners[model_key] = w
                confidences[model_key] = c
                rationales[model_key] = r

            # Aggregate decision (majority vote; tie -> higher avg confidence)
            a_votes = sum(1 for w in winners.values() if w.upper() == "A")
            b_votes = sum(1 for w in winners.values() if w.upper() == "B")
            avg_conf = sum(confidences.values()) / max(len(confidences), 1)

            if b_votes > a_votes:
                winner = "B"
            elif a_votes > b_votes:
                winner = "A"
            else:
                avg_b = self._avg_conf_for(winners, confidences, "B")
                avg_a = self._avg_conf_for(winners, confidences, "A")
                winner = "B" if avg_b > avg_a else "A"

            solver_wins = (winner.upper() == "B")
            score_1_to_100 = float(max(0.0, min(100.0, avg_conf)))

            details = {
                "winner": winner,
                "votes": {"A": a_votes, "B": b_votes},
                "confidences": confidences,
                "avg_confidence": score_1_to_100,
                "rationales": rationales,
                "raw_outputs": outputs,
                "models": [getattr(m, "name", m) for m in self.judge_models],
                "evidence_used": min(len(evidence), self.max_evidence),
            }
            return solver_wins, score_1_to_100, details

        except Exception as e:
            self.logger.error("RAGVerifier exception", extra={"error": str(e)})
            return False, 0.0, {"error": str(e), "winner": "A", "reason": "exception"}

    # -------- Optional adapter to your VerificationResult dataclass --------

    async def verify_as_result(
        self,
        question: str,
        seed_answer: str,
        predicted_answer: str,
        evidence: List[str],
        context: Dict[str, Any],
    ):
        """
        Same as `verify`, but returns a VerificationResult (if available).
        Useful when the rest of the pipeline expects the protocol object.
        """
        ok, score, details = await self.verify(
            question=question,
            seed_answer=seed_answer,
            predicted_answer=predicted_answer,
            evidence=evidence,
            context=context,
        )
        return VerificationResult(
            is_valid=ok,
            score=score,
            reason=details.get("winner", "A"),
            filter_results={},
            verification_details=details,
        )

    # --------------------------- internals ---------------------------

    def _build_prompt(
        self,
        question: str,
        seed_answer: str,
        predicted_answer: str,
        evidence_subset: List[str],
    ) -> str:
        ev = "\n".join(f"- {e}" for e in evidence_subset)
        return (
            "You are a fair and rigorous judge. Two AIs answered the same question.\n"
            "Decide which answer is better based on factual accuracy, clarity, and support from the EVIDENCE.\n\n"
            f"Question:\n{question}\n\n"
            f"EVIDENCE:\n{ev}\n\n"
            "---\nAnswer A (Proposer's Seed Answer):\n"
            f"{seed_answer}\n\n"
            "---\nAnswer B (Solver's Predicted Answer):\n"
            f"{predicted_answer}\n\n"
            "OUTPUT FORMAT — EXACTLY THREE LINES (no extra text):\n"
            "rationale: <1–2 sentences explaining the decision>\n"
            "winner: A | B\n"
            "confidence: <1–100>\n"
        )

    def _parse_single_judgment(self, text: str) -> Tuple[Optional[str], float, str]:
        """Parse one model's output into (winner 'A'/'B'/None, confidence(1..100), rationale)."""
        winner: Optional[str] = None
        conf: float = 50.0
        rat: str = ""

        m_w = self._WIN_RE.search(text or "")
        if m_w:
            w = m_w.group(1).upper()
            if w in ("A", "B"):
                winner = w

        m_c = self._CONF_RE.search(text or "")
        if m_c:
            try:
                conf = float(m_c.group(1))
                conf = max(0.0, min(100.0, conf))
            except Exception:
                conf = 50.0

        m_r = self._RAT_RE.search(text or "")
        if m_r:
            rat = m_r.group(1).strip()

        return winner, conf, rat

    @staticmethod
    def _avg_conf_for(winners: Dict[str, str], confidences: Dict[str, float], label: str) -> float:
        vals = [confidences[m] for m, w in winners.items() if w.upper() == label.upper()]
        return sum(vals) / len(vals) if vals else 0.0

    # --------------------------- judge callback ---------------------------

    def _judge_callback(self, outputs: Dict[str, str]) -> Tuple[Optional[str], Dict[str, float]]:
        """
        Judge callback passed to PromptService.run_prompt_multi so it can:
          - pick a winner key
          - emit pointwise/pairwise training events using the returned scores
        Returns: (winner_key or None, {model_key: score_0_1})
        We convert 1..100 confidences to 0..1 for the TrainingEventStore's trust field.
        """
        winners: Dict[str, str] = {}
        confidences: Dict[str, float] = {}
        for key, txt in outputs.items():
            w, c, _ = self._parse_single_judgment(txt or "")
            if w is None:
                w, c = "A", 50.0
            winners[key] = w
            confidences[key] = c

        # Decide winner (majority; tie → higher avg confidence; if still tie → first key)
        a_votes = sum(1 for w in winners.values() if w.upper() == "A")
        b_votes = sum(1 for w in winners.values() if w.upper() == "B")
        if b_votes > a_votes:
            winner_key = next((k for k, w in winners.items() if w.upper() == "B"), None)
        elif a_votes > b_votes:
            winner_key = next((k for k, w in winners.items() if w.upper() == "A"), None)
        else:
            # tie → pick the side with higher mean confidence; if perfect tie, pick first model deterministically
            avg_b = self._avg_conf_for(winners, confidences, "B")
            avg_a = self._avg_conf_for(winners, confidences, "A")
            target = "B" if avg_b > avg_a else "A"
            winner_key = next((k for k, w in winners.items() if w.upper() == target), next(iter(outputs.keys()), None))

        # Convert confidences from 1..100 → 0..1 for trust
        scores_0_1 = {k: max(0.0, min(1.0, c / 100.0)) for k, c in confidences.items()}
        return winner_key, scores_0_1
