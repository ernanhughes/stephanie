from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from stephanie.agents.base_agent import BaseAgent
from stephanie.memory.codecheck_store import CodeCheckStore
from stephanie.models.base import BaseConfig


@dataclass
class CodeCheckImproverConfig(BaseConfig):
    run_id: Optional[str] = None
    max_files: int = 10
    max_suggestions_per_file: int = 5

    # how to rank files (weights over existing metrics)
    file_priority_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "security.bandit_high": 2.0,
            "readability.ruff_style": 1.0,
            "vibe.instruction_compliance": -1.0,  # higher vibe = better, so negative weight
        }
    )

    critic_model_name: str = "llama-3.1-8b-code-instruct"  # placeholder, local model
    critic_temperature: float = 0.2


class CodeCheckImproverAgent(BaseAgent):
    """
    Takes a CodeCheck run, selects the worst files, and uses a code critic
    (local model or heuristic) to generate concrete, small improvement suggestions.
    """

    def __init__(self, cfg: CodeCheckImproverConfig, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.cfg: CodeCheckImproverConfig = cfg
        self.store: CodeCheckStore = getattr(self.memory, "codecheck")

    async def run(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Main entrypoint.

        Returns:
            dict with summary: number of files, number of suggestions, etc.
        """
        run_id = self.cfg.run_id or kwargs.get("run_id")
        if not run_id:
            raise ValueError("CodeCheckImproverAgent requires `run_id`.")

        # 1. Select top-N problematic files for this run
        files = self._select_priority_files(run_id, self.cfg.max_files)
        self.logger.info(
            "Improver: selected %d priority files for run %s", len(files), run_id
        )

        total_suggestions = 0
        per_file_counts: Dict[int, int] = {}

        # 2. For each file, call critic and store suggestions
        for f in files:
            file_id = f.id
            abs_path = os.path.join(run_id and f.run.repo_root or "", f.path)  # adjust if needed

            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
            except OSError as e:
                self.logger.warning("Improver: unable to read %s: %s", abs_path, e)
                continue

            metrics = (f.metrics.vector if f.metrics and f.metrics.vector else {})  # type: ignore[attr-defined]
            issues = self.store.list_issues_for_file(file_id)

            suggestions = await self._generate_suggestions_for_file(
                file_path=f.path,
                content=content,
                metrics=metrics,
                issues=issues,
                max_suggestions=self.cfg.max_suggestions_per_file,
            )
            if not suggestions:
                continue

            self.store.add_suggestions(
                run_id=run_id,
                file_id=file_id,
                suggestions=suggestions,
            )

            per_file_counts[file_id] = len(suggestions)
            total_suggestions += len(suggestions)

        return {
            "run_id": run_id,
            "files_considered": len(files),
            "suggestions_total": total_suggestions,
            "per_file_counts": per_file_counts,
        }

    # ------------------------------------------------------------------ internals

    def _select_priority_files(self, run_id: str, limit: int) -> List[Any]:
        """
        Select top-N files by a simple weighted priority over metrics.

        This still reads from CodeCheckFileORM, but you can make this more sophisticated
        later (e.g., direct SQL aggregation in the store).
        """
        files = self.store.list_files_for_run(run_id, limit=10000)
        scores: List[Tuple[float, Any]] = []

        for f in files:
            m = f.metrics.vector if f.metrics and f.metrics.vector else {}  # type: ignore[attr-defined]
            score = 0.0
            for name, weight in self.cfg.file_priority_weights.items():
                score += float(m.get(name, 0.0)) * weight
            scores.append((score, f))

        # sort descending: worst first
        scores.sort(key=lambda t: t[0], reverse=True)
        return [f for (_, f) in scores[:limit]]

    async def _generate_suggestions_for_file(
        self,
        file_path: str,
        content: str,
        metrics: Dict[str, float],
        issues: Sequence[Any],
        max_suggestions: int,
    ) -> List[Dict[str, Any]]:
        """
        Use heuristics + (optional) local model to generate small, concrete suggestions.
        """
        # 1) Start with a couple of deterministic heuristics (fast, no model)
        suggestions: List[Dict[str, Any]] = self._heuristic_suggestions(
            file_path=file_path,
            content=content,
            metrics=metrics,
            issues=issues,
        )

        # 2) Optionally augment with model-based suggestions
        if len(suggestions) < max_suggestions:
            ai_suggestions = await self._model_based_suggestions(
                file_path=file_path,
                content=content,
                metrics=metrics,
                issues=issues,
                remaining=max_suggestions - len(suggestions),
            )
            suggestions.extend(ai_suggestions)

        # Cap to max_suggestions
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
          - extract duplicated helpers
          - add docstrings to public functions
        """
        out: List[Dict[str, Any]] = []

        loc = len(content.splitlines())
        num_defs = content.count("def ")
        num_classes = content.count("class ")

        if loc > 800 or num_defs > 40:
            out.append(
                {
                    "kind": "refactor",
                    "title": "Split oversized module",
                    "summary": f"{file_path} is {loc} lines with {num_defs} functions; "
                               "consider splitting into smaller modules.",
                    "detail": "Identify cohesive groups of functions/classes and move them "
                              "into dedicated modules, keeping public API surface stable.",
                    "patch": None,
                    "patch_type": None,
                    "meta": {"loc": loc, "num_defs": num_defs, "num_classes": num_classes},
                }
            )

        # Example: docstring heuristic (very rough)
        if "def " in content and ('"""' not in content and "'''" not in content):
            out.append(
                {
                    "kind": "style",
                    "title": "Add docstrings to public functions",
                    "summary": "Public functions/classes are missing docstrings.",
                    "detail": "Add short, precise docstrings explaining inputs, outputs, and side effects.",
                    "patch": None,
                    "patch_type": None,
                    "meta": {},
                }
            )

        # You can also surface top 1–2 existing issues as explicit tasks
        for issue in list(issues)[:2]:
            out.append(
                {
                    "kind": "lint",
                    "title": f"Fix {issue.source} issue {issue.code or ''}".strip(),
                    "summary": issue.message,
                    "detail": f"At line {issue.line}, column {issue.col}",
                    "patch": None,
                    "patch_type": None,
                    "meta": {
                        "issue_id": issue.id,
                        "source": issue.source,
                        "code": issue.code,
                    },
                }
            )

        return out

    async def _model_based_suggestions(
        self,
        file_path: str,
        content: str,
        metrics: Dict[str, float],
        issues: Sequence[Any],
        remaining: int,
    ) -> List[Dict[str, Any]]:
        """
        Placeholder for local model integration.

        You'd route this through your local inference stack (Ollama, HF, etc.).
        For now it returns [] so nothing breaks.
        """
        # TODO: integrate with your local model runner via container or a service client.
        # The model prompt would include:
        #   - file path
        #   - snippet(s) of content (or whole file if small)
        #   - metrics summary
        #   - top few issues
        # And ask for `remaining` bullet-point suggestions with optional patch hints.
        return []
