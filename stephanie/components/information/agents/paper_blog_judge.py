# stephanie/components/information/agents/paper_blog_judge.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Union

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.information.learning.judge_stack import JudgeStack
from stephanie.services.prompt_service import LLMRole

log = logging.getLogger(__name__)


# ----------------------------
# IO helpers
# ----------------------------


def _safe_read_text(path: str, max_chars: int) -> str:
    try:
        p = Path(path)
        if not p.exists():
            return ""
        txt = p.read_text(encoding="utf-8", errors="ignore")
        return txt[:max_chars]
    except Exception:
        return ""


_KV_LINE_RE = re.compile(
    r"^\s*([a-zA-Z_][a-zA-Z0-9_\- ]*)\s*[:\-]\s*(.*?)\s*$"
)


def _parse_kv_block(raw: str) -> Dict[str, str]:
    """
    Robust KV parser.
    Accepts:
      key: value
      key - value
    Ignores non-matching lines.
    """
    raw = (raw or "").strip()
    out: Dict[str, str] = {}
    if not raw:
        return out
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _KV_LINE_RE.match(line)
        if not m:
            continue
        k = m.group(1).strip().lower().replace(" ", "_")
        v = m.group(2).strip()
        if k:
            out[k] = v
    return out


def _parse_score(fields: Dict[str, str], key: str = "score") -> float:
    s = (fields.get(key) or "").strip()
    if not s:
        return 0.0
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return 0.0
    try:
        val = float(m.group(1))
    except Exception:
        val = 0.0
    return max(0.0, min(100.0, val))


def _parse_winner(fields: Dict[str, str]) -> str:
    w = (fields.get("winner") or "").strip().lower()
    if w in {"a", "b", "tie"}:
        return w
    # tolerate variants
    if "a" == w[:1]:
        return "a"
    if "b" == w[:1]:
        return "b"
    if "tie" in w:
        return "tie"
    return "tie"


def _pick_rationale(fields: Dict[str, str]) -> str:
    return (fields.get("rationale") or "").strip()


# ----------------------------
# Judge specs
# ----------------------------


@dataclass(frozen=True)
class JudgeDimension:
    name: str
    weight: float
    goal_text: str


@dataclass
class DimResult:
    score: float
    rationale: str
    per_model: Dict[
        str, Dict[str, Any]
    ]  # model_key -> {score, rationale, extras, raw}


# ----------------------------
# Agent
# ----------------------------


class PaperBlogJudgeAgent(BaseAgent):
    """
    Best-of-breed judge stack:
      1) Pairwise tournament (order-swap to reduce bias)
      2) Multi-dimension rubric scoring (faithfulness/coverage/clarity/usefulness)
      3) Faithfulness gate (threshold) to block hallucination-y drafts

    Output format from judge models is ALWAYS plain key/value lines:
      rationale: ...
      score: 0-100

    Pairwise judge returns:
      winner: A|B|tie
      rationale: ...
      score: 0-100   (optional, tolerated)
    """

    def __init__(self, cfg, memory, container, logger=None):
        super().__init__(cfg, memory, container, logger=logger)

        self.prompt = container.get("prompt")

        # --- judge model(s)
        # Accept either:
        #   cfg["judge_models"] = [...]
        # or:
        #   cfg["model"] / cfg["model_name"] for single
        jm = cfg.get("judge_models")
        if isinstance(jm, list) and jm:
            self.judge_models: List[Union[str, Dict[str, Any]]] = jm
        else:
            self.judge_models = [
                cfg.get("model") or cfg.get("model_name", "ollama/llama3.1:8b")
            ]

        # --- reading limits
        self.max_blog_chars = int(cfg.get("max_blog_chars", 18000))
        self.max_report_chars = int(cfg.get("max_report_chars", 6000))

        # --- generation controls for judge calls
        self.role = cfg.get("judge_role", LLMRole.CRITIC)
        self.max_tokens = int(cfg.get("max_tokens", 350))
        self.temperature = float(cfg.get("temperature", 0.2))
        self.top_p = float(cfg.get("top_p", 0.9))

        # --- tournament / selection
        self.enable_pairwise = bool(cfg.get("enable_pairwise", True))
        self.pairwise_top_k = int(cfg.get("pairwise_top_k", 3))
        self.order_swap = bool(cfg.get("pairwise_order_swap", True))

        # --- faithfulness gate
        self.gate_dimension = str(cfg.get("gate_dimension", "faithfulness"))
        self.gate_min_score = float(cfg.get("gate_min_score", 70.0))
        self.gate_mode = str(
            cfg.get("gate_mode", "disqualify")
        )  # "disqualify" | "penalize"
        self.gate_penalty = float(
            cfg.get("gate_penalty", 40.0)
        )  # used if gate_mode="penalize"

        # --- rubric dimensions (editable)
        # You can override by providing cfg["dimensions"] as list of dicts

        self.stack_cfg = cfg.get("judge_stack", {}) or {}
        self.judge_stack = JudgeStack(
            cfg=self.stack_cfg,
            tools={
                # You can wrap your existing methods here:
                # "pairrm": self._pairrm_pairwise_tournament,  # async wrapper
                # "rubric": self._rubric_score_one_candidate,
                # "factuality": self._factuality_gate_one_candidate,
            },
            logger=log,
        )

        dims_cfg = cfg.get("dimensions")
        if isinstance(dims_cfg, list) and dims_cfg:
            dims: List[JudgeDimension] = []
            for d in dims_cfg:
                if not isinstance(d, dict):
                    continue
                dims.append(
                    JudgeDimension(
                        name=str(d.get("name", "")),
                        weight=float(d.get("weight", 1.0)),
                        goal_text=str(d.get("goal_text", "")),
                    )
                )
            self.dimensions = [d for d in dims if d.name and d.goal_text]
        else:
            self.dimensions = [
                JudgeDimension(
                    name="faithfulness",
                    weight=0.45,
                    goal_text=(
                        "Evaluate faithfulness to the paper. Penalize ANY unsupported claim, invented result, "
                        "fabricated citation, or confident statement not grounded in the provided paper evidence."
                    ),
                ),
                JudgeDimension(
                    name="coverage",
                    weight=0.25,
                    goal_text=(
                        "Evaluate coverage of the paper's key contributions, method, results, and limitations. "
                        "Reward capturing the main ideas without missing crucial constraints."
                    ),
                ),
                JudgeDimension(
                    name="clarity",
                    weight=0.20,
                    goal_text=(
                        "Evaluate clarity & structure for a blog audience. Reward good narrative flow, headings, "
                        "and explanations that reduce jargon while staying accurate."
                    ),
                ),
                JudgeDimension(
                    name="usefulness",
                    weight=0.10,
                    goal_text=(
                        "Evaluate usefulness to the intended reader. Reward concrete takeaways, correct framing "
                        "of when/why the method matters, and what to do next."
                    ),
                ),
            ]

    # ----------------------------
    # Prompt builders (your format)
    # ----------------------------

    def _prompt_kv(
        self,
        *,
        goal_text: str,
        input_text: str,
        extra_return_fields: Optional[str] = None,
    ) -> str:
        base = f"""### Goal
{goal_text}

### Text
{input_text}

Does this text align with the goal's intent and the system's broader values (e.g. accuracy, fairness, constructive purpose)?

Return your review in the exact structured format below. Do not include headings, markdown, or additional commentary. Use only plain text fields as shown:

rationale: <brief explanation>
score: <0–100>""".strip()

        if extra_return_fields:
            base += "\n" + extra_return_fields.strip()
        return base

    def _prompt_pairwise(
        self, *, goal_text: str, paper_pack: str, a_text: str, b_text: str
    ) -> str:
        return f"""### Goal
{goal_text}

### Text
PAPER PACK:
{paper_pack}

CANDIDATE A:
{a_text}

CANDIDATE B:
{b_text}

Pick which candidate is better for the goal. If both are similarly good or similarly bad, choose tie.

Return your decision in the exact structured format below. Do not include headings, markdown, or additional commentary. Use only plain text fields as shown:

winner: <A|B|tie>
rationale: <brief explanation>
score: <0–100>""".strip()

    # ----------------------------
    # Core judge execution
    # ----------------------------

    async def _run_one_model(
        self, prompt_text: str, model: Union[str, Dict[str, Any]]
    ) -> str:
        return await self.prompt.run_prompt(
            prompt_text=prompt_text,
            context=None,
            model=model,
            role=self.role,
            sys_preamble=None,
            params={
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        )

    async def _run_ensemble(self, prompt_text: str) -> Dict[str, str]:
        # Uses your PromptService's parallel multi-call
        if len(self.judge_models) == 1:
            m = self.judge_models[0]
            out = await self._run_one_model(prompt_text, m)
            key = m["name"] if isinstance(m, dict) and "name" in m else str(m)
            return {key: out}

        res = await self.prompt.run_prompt_multi(
            prompt_text=prompt_text,
            models=self.judge_models,
            judge=None,
            role=self.role,
            sys_preamble=None,
            params={
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
            context=None,
            timeout=None,
            dimension="blog_judge",
            goal_id=None,
            pipeline_run_id=None,
            agent_name="paper_blog_judge",
        )
        outputs = (res or {}).get("outputs") or {}
        return {str(k): (v or "") for k, v in outputs.items()}

    async def _judge_dimension(
        self, *, dim: JudgeDimension, input_text: str
    ) -> DimResult:
        prompt_text = self._prompt_kv(
            goal_text=dim.goal_text, input_text=input_text
        )

        outputs = await self._run_ensemble(prompt_text)

        # parse each model output
        per_model: Dict[str, Dict[str, Any]] = {}
        scores: List[float] = []
        for mk, raw in outputs.items():
            fields = _parse_kv_block(raw)
            sc = _parse_score(fields, "score")
            rat = _pick_rationale(fields)
            extras = {
                k: v
                for k, v in fields.items()
                if k not in {"score", "rationale"}
            }
            per_model[mk] = {
                "score": sc,
                "rationale": rat,
                "extras": extras,
                "raw": raw,
            }
            scores.append(sc)

        agg = float(median(scores)) if scores else 0.0

        # pick rationale from the model whose score is closest to median
        best_key = None
        best_dist = 1e9
        for mk, info in per_model.items():
            dist = abs(float(info.get("score", 0.0)) - agg)
            if dist < best_dist:
                best_dist = dist
                best_key = mk
        rationale = (
            (per_model.get(best_key, {}).get("rationale") or "")
            if best_key
            else ""
        )

        return DimResult(score=agg, rationale=rationale, per_model=per_model)

    async def _pairwise_compare(
        self, *, goal_text: str, paper_pack: str, a_text: str, b_text: str
    ) -> Dict[str, Any]:
        prompt_text = self._prompt_pairwise(
            goal_text=goal_text,
            paper_pack=paper_pack,
            a_text=a_text,
            b_text=b_text,
        )
        outs = await self._run_ensemble(prompt_text)

        votes = {"a": 0, "b": 0, "tie": 0}
        per_model: Dict[str, Dict[str, Any]] = {}

        for mk, raw in outs.items():
            fields = _parse_kv_block(raw)
            w = _parse_winner(fields)
            votes[w] += 1
            per_model[mk] = {
                "winner": w,
                "rationale": _pick_rationale(fields),
                "score": _parse_score(fields, "score"),
                "raw": raw,
            }

        # majority
        winner = max(votes.items(), key=lambda kv: kv[1])[0]
        # if tied vote counts, declare tie
        top = sorted(votes.values(), reverse=True)
        if len(top) >= 2 and top[0] == top[1]:
            winner = "tie"

        return {"winner": winner, "votes": votes, "per_model": per_model}

    async def _pairwise_compare_with_order_swap(
        self, *, goal_text: str, paper_pack: str, a_text: str, b_text: str
    ) -> Dict[str, Any]:
        r1 = await self._pairwise_compare(
            goal_text=goal_text,
            paper_pack=paper_pack,
            a_text=a_text,
            b_text=b_text,
        )
        if not self.order_swap:
            return {"winner": r1["winner"], "rounds": [r1]}

        r2 = await self._pairwise_compare(
            goal_text=goal_text,
            paper_pack=paper_pack,
            a_text=b_text,
            b_text=a_text,
        )

        # invert r2 winner back into original frame
        inv = r2["winner"]
        if inv == "a":
            inv = "b"
        elif inv == "b":
            inv = "a"

        # reconcile
        w1 = r1["winner"]
        w2 = inv
        if w1 == w2:
            final = w1
        elif "tie" in (w1, w2):
            final = w1 if w2 == "tie" else w2
        else:
            final = "tie"

        return {
            "winner": final,
            "rounds": [r1, r2],
            "swap_inverted_winner": w2,
        }

    # ----------------------------
    # Evidence pack builder
    # ----------------------------

    def _build_paper_pack(self, context: Dict[str, Any]) -> str:
        arxiv_id = (
            context.get("arxiv_id") or context.get("paper_arxiv_id") or ""
        )
        title = context.get("paper_title") or context.get("title") or arxiv_id
        summary = context.get("paper_summary") or ""
        report_path = context.get("paper_pipeline_report_path") or context.get(
            "pipeline_report_path"
        )
        report_md = (
            _safe_read_text(report_path, self.max_report_chars)
            if report_path
            else ""
        )
        # If you have richer “paper pack” already prepared, you can pass it via context["paper_pack_md"]
        paper_pack_md = context.get("paper_pack_md") or ""

        parts = [
            f"arXiv: {arxiv_id}",
            f"Title: {title}",
            f"Summary: {summary}",
        ]
        if paper_pack_md:
            parts.append(
                "Paper pack (provided):\n"
                + str(paper_pack_md)[: self.max_report_chars]
            )
        if report_md:
            parts.append("Pipeline report (truncated):\n" + report_md)
        return "\n\n".join([p for p in parts if p.strip()])

    def _load_candidate_texts(
        self, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Candidate formats supported:
          - context["blog_candidates"] = [{"path": "...", "name": "x"}, {"text": "..."}]
          - fallback to single blog via paper_blog_path/blog_path or paper_blog_markdown
        """
        cands = context.get("blog_candidates")
        out: List[Dict[str, Any]] = []

        if isinstance(cands, list) and cands:
            for i, c in enumerate(cands):
                if not isinstance(c, dict):
                    continue
                path = c.get("path") or c.get("blog_path")
                text = c.get("text") or c.get("markdown")
                name = c.get("name") or f"cand_{i}"
                if path and not text:
                    text = _safe_read_text(str(path), self.max_blog_chars)
                if (text or "").strip():
                    out.append(
                        {
                            "name": name,
                            "path": path,
                            "text": text[: self.max_blog_chars],
                        }
                    )
            if out:
                return out

        # single fallback
        blog_path = context.get("paper_blog_path") or context.get("blog_path")
        blog_md = (
            _safe_read_text(blog_path, self.max_blog_chars)
            if blog_path
            else (context.get("paper_blog_markdown") or "")
        )
        blog_md = (blog_md or "")[: self.max_blog_chars]
        if (blog_md or "").strip():
            out.append({"name": "single", "path": blog_path, "text": blog_md})
        return out

    # ----------------------------
    # Main run
    # ----------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        candidates = self._load_candidate_texts(context)
        if not candidates:
            context["paper_blog_judge"] = {"error": "no_blog_content"}
            return context

        paper_pack = self._build_paper_pack(context)

        # ---- Layer A: Pairwise tournament to pick top-K
        idxs = list(range(len(candidates)))
        if self.enable_pairwise and len(candidates) > 1:
            goal_text = "Choose the better blog post overall, prioritizing faithfulness to the paper, then coverage, then clarity."
            wins = {i: 0.0 for i in idxs}
            pair_meta: List[Dict[str, Any]] = []

            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a = idxs[i]
                    b = idxs[j]
                    ra = candidates[a]["text"]
                    rb = candidates[b]["text"]
                    cmp_res = await self._pairwise_compare_with_order_swap(
                        goal_text=goal_text,
                        paper_pack=paper_pack,
                        a_text=ra,
                        b_text=rb,
                    )
                    w = cmp_res.get("winner", "tie")
                    if w == "a":
                        wins[a] += 1.0
                    elif w == "b":
                        wins[b] += 1.0
                    else:
                        wins[a] += 0.5
                        wins[b] += 0.5
                    pair_meta.append({"a": a, "b": b, "result": cmp_res})

            ranked = sorted(idxs, key=lambda i: wins[i], reverse=True)
            top_k = ranked[: max(1, min(self.pairwise_top_k, len(ranked)))]
        else:
            wins = {i: 0.0 for i in idxs}
            pair_meta = []
            top_k = idxs[:1]

        # ---- Layer B/C: Rubric scoring + gate on top-K
        cand_results: List[Dict[str, Any]] = []
        best_idx = None
        best_final = -1e9

        wsum = sum(max(0.0, d.weight) for d in self.dimensions) or 1.0

        for i in idxs:
            cand = candidates[i]
            full_input = f"""PAPER PACK:
{paper_pack}

BLOG CANDIDATE:
{cand["text"]}""".strip()

            # If not in top_k, you can skip heavy scoring and keep tournament score only
            if i not in top_k:
                cand_results.append(
                    {
                        "idx": i,
                        "name": cand.get("name"),
                        "path": cand.get("path"),
                        "wins": wins.get(i, 0.0),
                        "scored": False,
                    }
                )
                continue

            dim_out: Dict[str, Any] = {}
            weighted_total = 0.0

            for d in self.dimensions:
                r = await self._judge_dimension(dim=d, input_text=full_input)
                dim_out[d.name] = {
                    "score": r.score,
                    "rationale": r.rationale,
                    "per_model": r.per_model,
                    "weight": d.weight,
                }
                weighted_total += max(0.0, d.weight) * r.score

            weighted_total = weighted_total / wsum

            # Gate (faithfulness by default)
            gate_score = float(
                dim_out.get(self.gate_dimension, {}).get("score", 0.0)
            )
            failed_gate = gate_score < self.gate_min_score

            if failed_gate and self.gate_mode == "disqualify":
                final_score = -1.0
            elif failed_gate and self.gate_mode == "penalize":
                final_score = weighted_total - self.gate_penalty
            else:
                final_score = weighted_total

            cand_record = {
                "idx": i,
                "name": cand.get("name"),
                "path": cand.get("path"),
                "wins": wins.get(i, 0.0),
                "scored": True,
                "dimensions": dim_out,
                "weighted_total": weighted_total,
                "gate_dimension": self.gate_dimension,
                "gate_score": gate_score,
                "failed_gate": failed_gate,
                "final_score": final_score,
            }
            cand_results.append(cand_record)

            if final_score > best_final:
                best_final = final_score
                best_idx = i

        # If everything disqualified, fall back to highest weighted_total among scored
        if best_idx is None:
            scored = [c for c in cand_results if c.get("scored")]
            if scored:
                scored.sort(
                    key=lambda x: float(x.get("weighted_total", 0.0)),
                    reverse=True,
                )
                best_idx = int(scored[0]["idx"])
                best_final = float(scored[0].get("weighted_total", 0.0))
            else:
                best_idx = 0
                best_final = 0.0

        winner = candidates[best_idx]

        result = {
            "method": "judge_stack_v1",
            "winner_idx": best_idx,
            "winner_name": winner.get("name"),
            "winner_path": winner.get("path"),
            "winner_final_score": best_final,
            "pairwise": {
                "enabled": self.enable_pairwise and len(candidates) > 1,
                "wins": wins,
                "comparisons": pair_meta,
            },
            "candidates": cand_results,
        }

        context["paper_blog_judge"] = result
        context["ai_blog_score"] = float(best_final)
        # pick a readable rationale: faithfulness rationale if available, else first dimension rationale
        best_cand = next(
            (
                c
                for c in cand_results
                if c.get("idx") == best_idx and c.get("scored")
            ),
            None,
        )
        rationale = ""
        if isinstance(best_cand, dict):
            dims = best_cand.get("dimensions") or {}
            rationale = (dims.get(self.gate_dimension, {}) or {}).get(
                "rationale"
            ) or ""
            if not rationale and dims:
                first = next(iter(dims.values()))
                rationale = (first or {}).get("rationale") or ""
        context["ai_blog_rationale"] = rationale

        # Optional: preference pairs for DPO (winner vs everyone else)
        try:
            pairs = []
            win_txt = winner.get("text") or ""
            for i, c in enumerate(candidates):
                if i == best_idx:
                    continue
                lose_txt = c.get("text") or ""
                if win_txt.strip() and lose_txt.strip():
                    pairs.append(
                        {
                            "chosen": win_txt,
                            "rejected": lose_txt,
                            "meta": {"winner_idx": best_idx, "loser_idx": i},
                        }
                    )
            context["blog_preference_pairs"] = pairs
        except Exception:
            pass

        return context
