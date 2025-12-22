# stephanie/tools/rubric_judge_tool.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stephanie.scoring.scorable import Scorable
from stephanie.tools.base_tool import BaseTool

log = logging.getLogger(__name__)


@dataclass
class RubricCriterionResult:
    name: str
    score: float
    rationale: str


@dataclass
class RubricJudgeResult:
    candidate_id: Optional[str]
    overall_score: float
    criteria: Dict[str, RubricCriterionResult]
    raw_text: str


class RubricJudgeTool(BaseTool):
    """
    LLM-as-a-Judge using a simple rubric.

    - Evaluates a single candidate text against a shared context.
    - Runs 4 tiny judge calls (or however many criteria you configure):
        * faithfulness
        * coverage
        * clarity
        * usefulness
    - Each call returns a JSON { "score": float, "rationale": str }.
    - Combines them into a weighted overall score.

    Typical usage:
        tool = RubricJudgeTool(cfg, memory, container, logger)
        result = tool.evaluate(context_text, candidate_text, candidate_id="blog_v1")

    Or via .apply() on a Scorable:
        scorable = await tool.apply(scorable, {"judge_context": "...", "candidate_id": "blog_v1"})
    """

    name = "rubric_judge"

    def __init__(self, cfg: Dict[str, Any], memory, container, logger) -> None:
        super().__init__(cfg, memory, container, logger)

        # Model config (e.g., "prometheus-eval/prometheus-7b-v2.0" or Flow-Judge, etc.)
        self.model_name: str = cfg.get(
            "model_name",
            "prometheus-eval/prometheus-7b-v2.0",
        )
        self.max_input_tokens: int = int(cfg.get("max_input_tokens", 3072))
        self.max_new_tokens: int = int(cfg.get("max_new_tokens", 256))
        self.temperature: float = float(cfg.get("temperature", 0.1))
        self.top_p: float = float(cfg.get("top_p", 0.9))

        # Criteria & weights
        # You can override this entire block in Hydra if you want.
        default_criteria = [
            {
                "name": "faithfulness",
                "description": "How factually faithful is the blog to the source paper? No unsupported claims.",
                "weight": 0.45,
            },
            {
                "name": "coverage",
                "description": "Does the blog cover the key contributions, results, and limitations?",
                "weight": 0.25,
            },
            {
                "name": "clarity",
                "description": "Is the blog clear, well-structured, and easy to follow?",
                "weight": 0.20,
            },
            {
                "name": "usefulness",
                "description": "How useful is the blog for a technically literate reader interested in the paper?",
                "weight": 0.10,
            },
        ]
        self.criteria: List[Dict[str, Any]] = cfg.get(
            "criteria",
            default_criteria,
        )

        # Normalise weights to sum to 1
        total_weight = sum(float(c.get("weight", 0.0)) for c in self.criteria) or 1.0
        for c in self.criteria:
            c["weight"] = float(c.get("weight", 0.0)) / total_weight

        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(
            "[RubricJudgeTool] Using device=%s, model=%s",
            self.device,
            self.model_name,
        )

        self._tokenizer, self._model = self._load_model(self.model_name)

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_name: str):
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        if self.device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            ).to(self.device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        model.eval()
        return tok, model

    def _encode(self, text: str) -> Dict[str, torch.Tensor]:
        return self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)

    def _generate(self, prompt: str) -> str:
        inputs = self._encode(prompt)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text

    # ------------------------------------------------------------------
    # Rubric evaluation
    # ------------------------------------------------------------------

    def _build_prompt_for_criterion(
        self,
        criterion_name: str,
        criterion_desc: str,
        context_text: str,
        candidate_text: str,
    ) -> str:
        """
        Build a small, self-contained prompt for a single criterion.
        Returns model instructions + context + candidate, asking for JSON.
        """
        return (
            "You are an expert editor evaluating an AI-generated blog post "
            "about a research paper.\n\n"
            "Context (paper summary + neighbourhood):\n"
            f"{context_text.strip()}\n\n"
            "Blog Post:\n"
            f"{candidate_text.strip()}\n\n"
            f"Criterion: {criterion_name}\n"
            f"Definition: {criterion_desc}\n\n"
            "On a scale of 1 to 10, where 1 is very poor and 10 is excellent, "
            f"rate this blog on the '{criterion_name}' criterion.\n\n"
            "Think step by step, then respond ONLY with a JSON object in this exact form:\n"
            "{\n"
            '  \"score\": <number between 1 and 10>,\n'
            '  \"rationale\": \"<short explanation>\"\n'
            "}\n"
        )

    def _parse_criterion_response(
        self, raw_text: str, fallback_score: float = 5.0
    ) -> Tuple[float, str]:
        """
        Try to parse the model's output as JSON; be robust to extra chatter.
        """
        raw_text = raw_text.strip()

        # Attempt to find a JSON object in the text by locating the first '{'
        # and the last '}' and parsing that substring.
        try:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = raw_text[start : end + 1]
                data = json.loads(json_str)
            else:
                data = json.loads(raw_text)
        except Exception:
            log.warning(
                "[RubricJudgeTool] Failed to parse JSON from judge output; "
                "falling back to default score. raw=%r",
                raw_text[:2000],
            )
            return fallback_score, raw_text

        score = data.get("score", fallback_score)
        try:
            score = float(score)
        except Exception:
            score = fallback_score

        rationale = str(data.get("rationale", "")).strip() or raw_text
        return score, rationale

    def evaluate(
        self,
        context_text: str,
        candidate_text: str,
        candidate_id: Optional[str] = None,
    ) -> RubricJudgeResult:
        """
        Evaluate one candidate using the full rubric.

        Returns a RubricJudgeResult with:
            - per-criterion scores + rationale
            - weighted overall score
        """
        criteria_results: Dict[str, RubricCriterionResult] = {}
        total_score = 0.0

        for crit in self.criteria:
            name = crit["name"]
            desc = crit.get("description", name)
            weight = float(crit.get("weight", 0.0))

            prompt = self._build_prompt_for_criterion(
                criterion_name=name,
                criterion_desc=desc,
                context_text=context_text,
                candidate_text=candidate_text,
            )
            raw = self._generate(prompt)
            score, rationale = self._parse_criterion_response(raw)

            criteria_results[name] = RubricCriterionResult(
                name=name,
                score=score,
                rationale=rationale,
            )
            total_score += weight * score

        return RubricJudgeResult(
            candidate_id=candidate_id,
            overall_score=total_score,
            criteria=criteria_results,
            raw_text="",  # you can store raw if you like; omitted by default
        )

    # ------------------------------------------------------------------
    # Tool API: apply to a Scorable (single candidate)
    # ------------------------------------------------------------------

    async def apply(self, scorable: Scorable, context: Dict[str, Any]) -> Scorable:
        """
        Uses scorable.text as the blog text, and context["judge_context"]
        (or scorable.meta["judge_context"]) as the source context.

        Writes results into scorable.meta["judgements"][self.name].
        """
        blog_text: str = scorable.text or ""
        if not blog_text.strip():
            log.warning("[RubricJudgeTool] Empty scorable.text; skipping judgement.")
            return scorable

        ctx_text: str = (
            context.get("judge_context")
            or scorable.meta.get("judge_context", "")
            or ""
        )

        candidate_id: Optional[str] = context.get("candidate_id") or str(
            getattr(scorable, "id", None)
        )

        result = self.evaluate(
            context_text=ctx_text,
            candidate_text=blog_text,
            candidate_id=candidate_id,
        )

        meta: Dict[str, Any] = scorable.meta
        judgements: Dict[str, Any] = meta.setdefault("judgements", {})

        crit_scores = {
            name: {
                "score": r.score,
                "rationale": r.rationale,
            }
            for name, r in result.criteria.items()
        }
        judgements[self.name] = {
            "candidate_id": result.candidate_id,
            "overall_score": result.overall_score,
            "criteria": crit_scores,
        }

        return scorable

    # Convenience: judge arbitrary text without Scorable
    def judge_text(
        self, context_text: str, candidate_text: str, candidate_id: Optional[str] = None
    ) -> Dict[str, Any]:
        res = self.evaluate(context_text, candidate_text, candidate_id)
        return {
            "candidate_id": res.candidate_id,
            "overall_score": res.overall_score,
            "criteria": {
                name: {
                    "score": r.score,
                    "rationale": r.rationale,
                }
                for name, r in res.criteria.items()
            },
        }
