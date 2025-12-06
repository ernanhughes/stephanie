from __future__ import annotations

import logging
import os
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stephanie.agents.base_agent import BaseAgent
from stephanie.memory.codecheck_store import CodeCheckStore
from stephanie.data.base_config import BaseConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config model (lightweight view over the YAML)
# ---------------------------------------------------------------------------


@dataclass
class CodeCheckImproverConfig(BaseConfig):
    """
    Shallow view over the YAML config.

    For now we only promote the most important knobs to attributes; everything
    else stays accessible on the raw cfg dict (self.cfg).
    """

    run_id: Optional[str] = None
    max_files: int = 32
    max_suggestions_per_file: int = 5

    # selection weights (mirrors config/agents/codecheck_improver.yaml)
    recent_weight: float = 0.7
    dirty_weight: float = 0.3
    max_age_days: int = 7

    # which metrics matter for "dirtiness"
    dirty_metrics: Tuple[str, ...] = (
        "security.bandit_high",
        "readability.ruff_style",
        "semantic.risk",
        "vibe.instruction_compliance",
    )

    def update_from_raw(self, raw: Dict[str, Any]) -> None:
        """
        Populate fields from the raw cfg dict (Hydra/YAML).

        This is intentionally tolerant: missing keys are fine.
        """
        if not raw:
            return

        self.run_id = raw.get("run_id", self.run_id)
        self.max_files = int(raw.get("max_files", self.max_files))
        self.max_suggestions_per_file = int(
            raw.get("max_suggestions_per_file", self.max_suggestions_per_file)
        )

        sel = raw.get("selection", {})
        if sel:
            self.recent_weight = float(sel.get("recent_weight", self.recent_weight))
            self.dirty_weight = float(sel.get("dirty_weight", self.dirty_weight))
            self.max_age_days = int(sel.get("max_age_days", self.max_age_days))

            dirty_metrics = sel.get("dirty_metrics")
            if dirty_metrics:
                self.dirty_metrics = tuple(dirty_metrics)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CodeCheckImproverAgent(BaseAgent):
    """
    CodeCheckImproverAgent

    Phase 0 / 0.5 implementation:

      * takes a single CodeCheck run_id
      * selects a handful of "recent + dirty" files
      * generates heuristic suggestions for each
      * (optionally) asks a local model for extra suggestions
      * persists everything via CodeCheckStore.add_suggestions()

    Design constraints:

      - meta-scale friendly (never walk all file contents at once)
      - no open ORM sessions outside CodeCheckStore helpers
      - selection is cheap and purely numeric (recency + metrics)
    """

    def __init__(self, cfg: Dict[str, Any], memory: Any, container: Any, logger: Any):
        # BaseAgent will stash cfg/memory/container/logger on self
        super().__init__(cfg, memory, container, logger)

        # Make it explicit for type checkers / readers
        self.cfg: Dict[str, Any] = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        # Store handle (same pattern as MetricStore / NexusStore)
        self.store: CodeCheckStore = getattr(memory, "codecheck")

        self.repo_root = "./stephanie"

        # Light config view – keeps the common knobs in one place
        self.ccfg = CodeCheckImproverConfig()
        self.ccfg.update_from_raw(cfg)

    # ------------------------------------------------------------------ public API

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entrypoint.

        Typical entry from SIS:

            await agent.run(context={"codecheck_run_id": run_id})

        or from a pipeline:

            await agent.run(run_id=run_id)
        """
        run_id = context.get("pipeline_run_id")

        if not run_id:
            raise ValueError("CodeCheckImproverAgent requires a `run_id`.")

        # Keep ccfg in sync in case caller overrides run_id
        self.ccfg.run_id = run_id

        # Fetch run once so we can get repo_root without touching relationships
        run = self.store.get_run(run_id)
        if not run:
            raise ValueError(
                f"CodeCheckImproverAgent: no CodeCheck run found for id={run_id}"
            )

        repo_root = run.repo_root or self.repo_root

        # 1) Select priority files for this run
        files = self._select_priority_files(
            run_id=run_id,
            repo_root=repo_root,
            limit=self.ccfg.max_files,
        )
        log.info(
            "CodeCheckImprover: selected %d priority files for run %s",
            len(files),
            run_id,
        )

        total_suggestions = 0
        per_file_counts: Dict[int, int] = {}

        # 2) For each file, build suggestions and persist
        suggestions_payload: List[Dict[str, Any]] = []

        for f in files:
            file_id = f.id
            rel_path = f.path
            abs_path = os.path.join(repo_root, rel_path)

            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
            except OSError as e:  # file might have been deleted/renamed
                log.warning(
                    "CodeCheckImprover: unable to read %s for run %s: %s",
                    abs_path,
                    run_id,
                    e,
                )
                continue

            metrics = self._extract_metrics_dict(f)
            # If you don't have this yet, you can stub list_issues_for_file in CodeCheckStore
            issues = self.store.list_issues_for_file(file_id)

            file_suggestions = await self._build_suggestions_for_file(
                run_id=run_id,
                file_id=file_id,
                file_path=rel_path,
                content=content,
                metrics=metrics,
                issues=issues,
                max_suggestions=self.ccfg.max_suggestions_per_file,
            )

            if not file_suggestions:
                continue

            for s in file_suggestions:
                s.setdefault("status", "pending")
                s.setdefault("applied_ts", None)

            suggestions_payload.extend(file_suggestions)
            total_suggestions += len(file_suggestions)
            per_file_counts[file_id] = len(file_suggestions)

        # 3) Persist all suggestions in one go (single transactional scope)
        if suggestions_payload:
            self.store.add_suggestions(
                run_id=run_id,
                suggestions=suggestions_payload,
            )

        log.info(
            "CodeCheckImprover: created %d suggestions across %d files for run %s",
            total_suggestions,
            len(per_file_counts),
            run_id,
        )

        # Optional: telemetry hook – you can plug this into your datalog later
        if self.cfg.get("telemetry", {}).get("log_suggestions_summary", True):
            log.debug(
                "CodeCheckImprover summary for %s: %s",
                run_id,
                {
                    "files_considered": len(files),
                    "suggestions_total": total_suggestions,
                    "per_file_counts": per_file_counts,
                },
            )

        return {
            "run_id": run_id,
            "files_considered": len(files),
            "suggestions_total": total_suggestions,
            "per_file_counts": per_file_counts,
        }

    # ------------------------------------------------------------------ selection

    def _select_priority_files(
        self,
        run_id: str,
        repo_root: str,
        limit: int,
    ) -> List[Any]:
        """
        Select top-N files by a blend of:

          - recency (based on filesystem mtime)
          - dirtiness (based on a subset of metrics)

        We *only* touch what we need and keep all ORM access inside
        CodeCheckStore calls to avoid session leaks.
        """
        # 1) Load candidate files; store should eager-load metrics for us
        files = self.store.list_files_for_run(run_id, limit=100_000)

        if not files:
            return []

        now = time.time()
        max_age_seconds = max(self.ccfg.max_age_days, 1) * 86400

        dirty_metrics = set(self.ccfg.dirty_metrics)

        scored: List[Tuple[float, Any]] = []
        max_dirty_score = 0.0

        # First pass: compute dirty_score + recency_score per file
        tmp: List[Tuple[float, float, Any]] = []

        for f in files:
            rel_path = f.path
            abs_path = os.path.join(repo_root, rel_path)

            # Recency score ∈ [0,1]; newer files get higher scores
            try:
                mtime = os.path.getmtime(abs_path)
                age_seconds = max(now - mtime, 0.0)
                age_ratio = min(age_seconds / max_age_seconds, 1.0)
                recency_score = 1.0 - age_ratio
            except OSError:
                recency_score = 0.0

            metrics = self._extract_metrics_dict(f)

            # Dirtiness: simple sum over the dirty metrics
            dirty_score = 0.0
            for name in dirty_metrics:
                val = metrics.get(name)
                if val is None:
                    continue

                # Convention: higher is worse, except for vibe.instruction_compliance
                if name == "vibe.instruction_compliance":
                    # assume this is in [0,1] where 1 is good → invert
                    val = 1.0 - max(0.0, min(float(val), 1.0))
                else:
                    val = float(val)

                dirty_score += max(val, 0.0)

            max_dirty_score = max(max_dirty_score, dirty_score)
            tmp.append((recency_score, dirty_score, f))

        # Second pass: blend into a single priority score and sort
        recent_w = self.ccfg.recent_weight
        dirty_w = self.ccfg.dirty_weight

        norm = max_dirty_score or 1.0  # avoid division by zero

        for recency_score, dirty_score, f in tmp:
            severity_norm = dirty_score / norm
            priority = recent_w * recency_score + dirty_w * severity_norm
            scored.append((priority, f))

        scored.sort(key=lambda t: t[0], reverse=True)

        # Return top-N files
        return [f for _, f in scored[:limit]]

    # ------------------------------------------------------------------ helpers

    def _extract_metrics_dict(self, file_orm: Any) -> Dict[str, float]:
        """
        Extracts a flat metric dict from the CodeCheckFileMetricsORM attached
        to the given file object.

        We support both the vector JSON field and the (columns, values) pair.
        """
        m = getattr(file_orm, "metrics", None)
        if m is None:
            return {}

        # Preferred: the JSON mapping, if present
        vec = getattr(m, "vector", None)
        if isinstance(vec, dict):
            out: Dict[str, float] = {}
            for k, v in vec.items():
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    continue
            return out

        # Fallback: columns + values
        cols = getattr(m, "columns", None)
        vals = getattr(m, "values", None)
        if cols and vals and len(cols) == len(vals):
            out: Dict[str, float] = {}
            for k, v in zip(cols, vals):
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    continue
            return out

        return {}

    # ------------------------------------------------------------------ suggestion building

    async def _build_suggestions_for_file(
        self,
        run_id: str,
        file_id: int,
        file_path: str,
        content: str,
        metrics: Dict[str, float],
        issues: Sequence[Any],
        max_suggestions: int,
    ) -> List[Dict[str, Any]]:
        """
        Combine cheap heuristics + (eventually) model-based analysis.
        """
        suggestions: List[Dict[str, Any]] = []

        # 1) Heuristic suggestions – always safe and fast
        suggestions.extend(
            self._heuristic_suggestions(
                file_path=file_path,
                content=content,
                metrics=metrics,
                issues=issues,
            )
        )

        # 2) If we still have room, ask the local model (when wired up)
        remaining = max_suggestions - len(suggestions)
        if remaining > 0:
            ai_suggestions = await self._model_based_suggestions(
                run_id=run_id,
                file_id=file_id,
                file_path=file_path,
                content=content,
                metrics=metrics,
                issues=issues,
                remaining=remaining,
            )
            suggestions.extend(ai_suggestions)

        return suggestions[:max_suggestions]

    def _heuristic_suggestions(
        self,
        file_path: str,
        content: str,
        metrics: Dict[str, float],
        issues: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        """
        Very cheap, deterministic things we know are always good:

          - split mega-files
          - encourage docstrings for public functions
          - surface obvious TODO/FIXME debt

        These act as a seed set while the model-based critic is still coming online.
        """
        out: List[Dict[str, Any]] = []

        # Basic structural stats
        lines = content.splitlines()
        loc = len(lines)
        num_defs = content.count("def ")
        num_classes = content.count("class ")
        num_todos = sum(
            1 for ln in lines if "TODO" in ln or "FIXME" in ln
        )

        # 1) Mega-file refactor suggestion
        if loc > 800 or num_defs > 40:
            out.append(
                {
                    "kind": "refactor",
                    "title": "Split oversized module",
                    "summary": (
                        f"{file_path} is {loc} lines with {num_defs} functions; "
                        "consider splitting into smaller modules."
                    ),
                    "detail": (
                        "Identify cohesive groups of functions/classes and move them "
                        "into dedicated modules, keeping the public API surface stable. "
                        "Start by extracting clearly separable responsibilities."
                    ),
                    "patch": None,
                    "patch_type": None,
                    "meta": {
                        "loc": loc,
                        "num_defs": num_defs,
                        "num_classes": num_classes,
                    },
                }
            )

        # 2) Missing docstrings for public functions
        if num_defs >= 5:
            approx_docstrings = content.count('"""') + content.count("'''")
            if approx_docstrings < max(1, num_defs // 3):
                out.append(
                    {
                        "kind": "style",
                        "title": "Add docstrings to public functions",
                        "summary": (
                            f"{file_path} defines {num_defs} functions but has relatively "
                            "few docstrings; documenting intent will make future changes safer."
                        ),
                        "detail": (
                            "Add concise docstrings to public functions and key classes, "
                            "focusing on inputs, outputs, and side effects. This makes it "
                            "easier for both humans and tools to reason about the code."
                        ),
                        "patch": None,
                        "patch_type": None,
                        "meta": {
                            "loc": loc,
                            "num_defs": num_defs,
                            "docstring_like": approx_docstrings,
                        },
                    }
                )

        # 3) Surface explicit TODO/FIXME debt
        if num_todos > 0:
            out.append(
                {
                    "kind": "debt",
                    "title": "Address outstanding TODO/FIXME comments",
                    "summary": (
                        f"{file_path} contains {num_todos} TODO/FIXME markers; "
                        "consider resolving or clarifying the most critical ones."
                    ),
                    "detail": (
                        "Review the TODO/FIXME comments in this file, prioritize the ones "
                        "that affect correctness or developer ergonomics, and either resolve "
                        "them or convert them into tracked issues."
                    ),
                    "patch": None,
                    "patch_type": None,
                    "meta": {"num_todos": num_todos},
                }
            )

        return out


    async def _model_based_suggestions(
        self,
        run_id: str,
        file_id: int,
        file_path: str,
        content: str,
        metrics: Dict[str, float],
        issues: Sequence[Any],
        remaining: int,
    ) -> List[Dict[str, Any]]:
        """
        Model-based suggestions using a local HF model (e.g. VibeThinker).

        IMPORTANT:
        - We do NOT ask for JSON.
        - We treat the model's whole reply as a single free-text suggestion.
        - The UI will just show this as "here's what to change".
        """

        # If you haven't wired the model yet, just bail out cleanly.
        if not hasattr(self, "_ensure_model"):
            return []

        # Lazily load the HF model + tokenizer
        self._ensure_model()

        mcfg = self.cfg.get("model", {})
        system_prompt = mcfg.get("system_prompt", "").strip()
        max_tokens = int(mcfg.get("max_tokens", 768))

        # Keep context small: head of file only
        head_lines = content.splitlines()[:400]
        code_head = "\n".join(head_lines)

        metrics_summary = ", ".join(
            f"{k}={v:.2f}" for k, v in sorted(metrics.items())
        )

        issues_block = self._format_issues_for_prompt(issues)

        # ---- Prompt: plain text, no JSON, with explicit format hints ----
        user_prompt = f"""
        You are helping a human developer improve a Python file
        in a large, messy codebase.

        File path: {file_path}

        Metrics summary (higher numbers usually mean more problems):
        {metrics_summary or "(none)"}

        Known static issues:
        {issues_block}

        Code (first 400 lines):
        ```python
        {code_head}
        ```

        Task:

        - Propose a few SMALL, CONCRETE improvements that can be done in ~5 minutes each.
        - Focus on:
          * splitting overly large functions or files,
          * renaming confusing things,
          * adding or fixing docstrings,
          * removing obvious dead code,
          * making future changes safer and clearer.
        - Avoid huge rewrites or large architectural changes.

        OUTPUT FORMAT (VERY IMPORTANT):

        - Do NOT output JSON.
        - Do NOT output XML or any machine-readable format.
        - Just write normal text like you're talking to the developer.
        - Use short headings:

            Suggestion 1:
            - ...
            - ...

            Suggestion 2:
            - ...

        - If you show code, use fenced code blocks like:

            Before:
            ```python
            # old code
            ```

            After:
            ```python
            # improved code
            ```

        - Keep everything focused and short so it's obvious what to change.
        """

        full_prompt = system_prompt + "\n\n" + textwrap.dedent(user_prompt).strip()

        inputs = self._tok(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._model.config.max_position_embeddings,
        ).to(self._device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
            )

        text = self._tok.decode(out[0], skip_special_tokens=True)

        # Some models echo the prompt; strip it if so
        if full_prompt in text:
            text = text[len(full_prompt):].lstrip()

        # Build ONE suggestion object using the raw text
        # (we could later split Suggestion 1 / 2 / 3, but not needed now)
        summary = text.splitlines()[0].strip() if text.strip() else ""
        if len(summary) > 140:
            summary = summary[:137] + "..."

        suggestion = {
            "run_id": run_id,
            "file_id": file_id,
            "kind": "analysis",  # or "refactor" if you prefer
            "title": f"Model suggestions for {file_path}",
            "summary": summary or f"Improvement suggestions generated for {file_path}",
            "detail": text,      # full free-text response
            "patch": None,       # we are not auto-applying patches yet
            "patch_type": None,
            "meta": {
                "file_path": file_path,
                "source": "vibethinker_text",
            },
        }

        # We still honour `remaining`, but it's basically "at most 1" for now
        if remaining <= 0:
            return []
        return [suggestion]


    def _ensure_model(self):
        if getattr(self, "_model", None) is not None:
            return

        mcfg = self.cfg.get("model", {})
        model_name = mcfg.get("model_name", "WeiboAI/VibeThinker-1.5B")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            self._model.to(device)
        self._device = device

    def _format_issues_for_prompt(self, issues):
        pass
