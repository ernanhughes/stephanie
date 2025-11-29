from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os
import time

from stephanie.agents.base_agent import BaseAgent
from stephanie.memory.codecheck_store import CodeCheckStore
from stephanie.models.base_config import BaseConfig


# ---------------------------------------------------------------------------
# Config loading (brain-dead YAML → dataclass)
# ---------------------------------------------------------------------------

# repo_root / config / agents / codecheck_improver.yaml
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "agents" / "codecheck_improver.yaml"
)


@dataclass
class CodeCheckImproverConfig(BaseConfig):
    """
    Thin wrapper around the YAML config for the improver.

    We only materialise the pieces the agent actually needs. The full
    raw YAML node is kept in `raw` so you can inspect / log later.
    """

    run_id: Optional[str] = None

    # File selection
    max_files: int = 32
    max_suggestions_per_file: int = 5
    recency_weight: float = 0.7
    dirty_metrics: List[str] = field(default_factory=list)

    # Optional: keep the whole YAML subtree for later use
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(
        cls,
        run_id: str,
        path: Optional[os.PathLike[str] | str] = None,
    ) -> "CodeCheckImproverConfig":
        """
        Load `config/agents/codecheck_improver.yaml` and build a config.

        SIS can just do:

            cfg = CodeCheckImproverConfig.from_yaml(run_id)
            agent = CodeCheckImproverAgent(cfg, memory, container, logger)

        No Hydra required.
        """
        import yaml  # local import to avoid hard dependency at module import

        cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        node = data.get("codecheck_improver", data) or {}

        selection = node.get("selection") or {}
        dirty_metrics = list(selection.get("dirty_metrics") or [])
        recency_weight = float(selection.get("recent_weight", 0.7))

        return cls(
            run_id=run_id,
            max_files=int(node.get("max_files", 32)),
            max_suggestions_per_file=int(node.get("max_suggestions_per_file", 5)),
            recency_weight=recency_weight,
            dirty_metrics=dirty_metrics,
            raw=node,
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CodeCheckImproverAgent(BaseAgent):
    """
    Given a CodeCheck run, pick the worst recent files and propose fixes.

    This is deliberately simple for now:

      * file selection: “recent then dirty” using the metrics in the run
      * suggestions: cheap heuristic bullets (no local model yet)
      * storage: writes rows into CodeCheckSuggestionORM via CodeCheckStore

    Once you're happy with the plumbing you can replace
    `_heuristic_suggestions_for_file` with a call out to your local SLM.
    """

    def __init__(
        self,
        cfg: CodeCheckImproverConfig | Dict[str, Any],
        memory: Any,
        container: Any,
        logger: Any,
    ) -> None:
        # Normalise cfg into a CodeCheckImproverConfig instance
        if isinstance(cfg, CodeCheckImproverConfig):
            self.cfg = cfg
            base_cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else asdict(cfg)
        else:
            # Allow passing the raw YAML node; we only pull out what we need
            node = cfg.get("codecheck_improver", cfg) if isinstance(cfg, dict) else {}
            selection = (node or {}).get("selection") or {}
            self.cfg = CodeCheckImproverConfig(
                run_id=node.get("run_id"),
                max_files=int(node.get("max_files", 32)),
                max_suggestions_per_file=int(node.get("max_suggestions_per_file", 5)),
                recency_weight=float(selection.get("recent_weight", 0.7)),
                dirty_metrics=list(selection.get("dirty_metrics") or []),
                raw=node or {},
            )
            base_cfg = node

        super().__init__(base_cfg, memory, container, logger)
        self.memory = memory
        self.container = container
        self.logger = logger

        # CodeCheckStore hangs off memory in the usual way (self.memory.codecheck)
        self.store: CodeCheckStore = getattr(self.memory, "codecheck")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Main entrypoint used by SIS / Supervisor.

        Args:
            context: optional pipeline context (ignored for now)
            kwargs: can contain an explicit `run_id`

        Returns:
            Summary dict with counts etc. (also returned to SIS).
        """
        context = context or {}
        run_id = (
            kwargs.get("run_id")
            or context.get("codecheck_run_id")
            or self.cfg.run_id
        )
        if not run_id:
            raise ValueError("CodeCheckImproverAgent.run() requires a `run_id`.")

        self.cfg.run_id = run_id

        # 1. Select priority files for this run
        files = self._select_priority_files(run_id, self.cfg.max_files)
        self.logger.info(
            "CodeCheckImprover: selected %d priority files for run %s",
            len(files),
            run_id,
        )

        total_suggestions = 0
        per_file_counts: Dict[int, int] = {}

        # 2. For each file, generate and persist suggestions
        run_orm = self.store.get_run(run_id)
        repo_root = run_orm.repo_root if run_orm is not None else "."

        for f in files:
            file_id = f.id
            abs_path = os.path.join(repo_root, f.path)

            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
            except OSError as e:
                self.logger.warning("Improver: unable to read %s: %s", abs_path, e)
                continue

            suggestions = self._generate_suggestions_for_file(run_id, f, content)

            if not suggestions:
                continue

            # truncate per config
            suggestions = suggestions[: self.cfg.max_suggestions_per_file]

            created = self.store.add_suggestions(
                run_id=run_id,
                file_id=file_id,
                suggestions=suggestions,
            )
            count = len(created)
            total_suggestions += count
            per_file_counts[file_id] = count

        self.logger.info(
            "CodeCheckImprover: created %d suggestions across %d files",
            total_suggestions,
            len(per_file_counts),
        )

        return {
            "run_id": run_id,
            "files_considered": len(files),
            "suggestions_total": total_suggestions,
            "per_file_counts": per_file_counts,
        }

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select_priority_files(self, run_id: str, limit: int) -> List[Any]:
        """
        Select top-N files for a run using “recent then dirty”.

        "Recent" = filesystem mtime
        "Dirty"  = sum of metrics in cfg.dirty_metrics (default: empty ⇒ 0)

        The priority score is:

            score = alpha * recency + (1 - alpha) * severity

        where alpha = cfg.recency_weight in [0, 1].
        """
        files = self.store.list_files_for_run(run_id, limit=None) or []
        if not files:
            return []

        # 1) Pre-fetch metrics for all files
        metrics_by_file: Dict[int, Dict[str, float]] = {}
        for f in files:
            m = getattr(f, "metrics", None)
            if m is not None and getattr(m, "vector", None):
                try:
                    metrics_by_file[f.id] = {
                        k: float(v) for k, v in (m.vector or {}).items()
                    }
                except Exception:
                    metrics_by_file[f.id] = {}
            else:
                metrics_by_file[f.id] = {}

        # 2) Compute severity from dirty metrics
        dirty_names = self.cfg.dirty_metrics or [
            "security.bandit_high",
            "readability.ruff_style",
            "semantic.risk",
        ]

        severity_scores: Dict[int, float] = {}
        for f in files:
            vec = metrics_by_file.get(f.id, {})
            sev = 0.0
            for name in dirty_names:
                sev += float(vec.get(name, 0.0))
            severity_scores[f.id] = sev

        # 3) Compute recency from filesystem mtime
        run_orm = self.store.get_run(run_id)
        repo_root = run_orm.repo_root if run_orm is not None else "."

        mtimes: Dict[int, float] = {}
        now = time.time()
        for f in files:
            abs_path = os.path.join(repo_root, f.path)
            try:
                mt = os.path.getmtime(abs_path)
            except OSError:
                mt = now  # treat unreadable files as “now” to not penalise too much
            mtimes[f.id] = mt

        max_mtime = max(mtimes.values())
        min_mtime = min(mtimes.values())

        recency_scores: Dict[int, float] = {}
        if max_mtime == min_mtime:
            for file_id in mtimes:
                recency_scores[file_id] = 0.5
        else:
            span = max_mtime - min_mtime
            for file_id, mt in mtimes.items():
                recency_scores[file_id] = (mt - min_mtime) / span

        # 4) Combine into final priority
        alpha = max(0.0, min(1.0, float(self.cfg.recency_weight)))
        scored: List[Tuple[float, Any]] = []

        for f in files:
            sev = severity_scores.get(f.id, 0.0)
            rec = recency_scores.get(f.id, 0.5)
            score = alpha * rec + (1.0 - alpha) * sev
            scored.append((score, f))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [f for _, f in scored[:limit]]
        return top

    # ------------------------------------------------------------------
    # Suggestion generation
    # ------------------------------------------------------------------

    def _generate_suggestions_for_file(
        self,
        run_id: str,
        file_orm: Any,
        content: str,
    ) -> List[Dict[str, Any]]:
        """
        Return a list of suggestion dicts for a single file.

        Shape matches CodeCheckSuggestionORM fields (minus ids / timestamps).
        """
        # 1) Look up metrics + issues for context
        metrics = getattr(file_orm, "metrics", None)
        issues = self.store.list_issues_for_file(file_orm.id) or []

        metric_vector = metrics.vector if metrics is not None else {}

        # 2) Start with cheap heuristic suggestions
        suggestions: List[Dict[str, Any]] = self._heuristic_suggestions_for_file(
            file_orm=file_orm,
            content=content,
            metrics=metric_vector,
            issues=issues,
        )

        # (Later) model-based suggestions can be added here

        # Attach a tiny bit of debugging meta
        for s in suggestions:
            meta = s.setdefault("meta", {})
            meta.setdefault("metric_snapshot", metric_vector)
            meta.setdefault("issue_count", len(issues))

        return suggestions

    def _heuristic_suggestions_for_file(
        self,
        file_orm: Any,
        content: str,
        metrics: Dict[str, float],
        issues: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        """Rule-of-thumb suggestions that are always safe to compute."""
        suggestions: List[Dict[str, Any]] = []

        path = getattr(file_orm, "path", "<unknown>")

        loc = len(content.splitlines())
        if loc > 500:
            suggestions.append(
                {
                    "kind": "refactor",
                    "title": "File is very long – consider splitting",
                    "summary": f"{path} has {loc} lines; consider extracting helpers or submodules.",
                    "detail": (
                        "Extremely long files are hard to reason about. "
                        "Identify natural boundaries (independent classes, utility functions, "
                        "or unrelated concerns) and move them into separate modules."
                    ),
                    "patch_type": "unified_diff",
                    "patch_text": None,
                    "status": "pending",
                }
            )

        # Docstring / comments hint
        if "def " in content and '"""' not in content:
            suggestions.append(
                {
                    "kind": "documentation",
                    "title": "Functions without docstrings",
                    "summary": f"{path} defines functions but no obvious docstrings.",
                    "detail": (
                        "Add short docstrings to public functions and classes explaining "
                        "what they do, their main arguments, and key side-effects."
                    ),
                    "patch_type": "unified_diff",
                    "patch_text": None,
                    "status": "pending",
                }
            )

        # Metric-driven hints
        bandit_high = float(metrics.get("security.bandit_high", 0.0))
        if bandit_high > 0:
            suggestions.append(
                {
                    "kind": "security",
                    "title": "Security issues detected by Bandit",
                    "summary": f"Bandit reported {int(bandit_high)} high-severity findings in {path}.",
                    "detail": (
                        "Run Bandit locally on this file and address the reported issues. "
                        "Focus first on high and medium severity findings."
                    ),
                    "patch_type": "unified_diff",
                    "patch_text": None,
                    "status": "pending",
                }
            )

        ruff_style = float(metrics.get("readability.ruff_style", 0.0))
        if ruff_style > 0:
            suggestions.append(
                {
                    "kind": "style",
                    "title": "Style / lint issues",
                    "summary": f"Ruff reported {int(ruff_style)} style issues in {path}.",
                    "detail": (
                        "Run Ruff on this file and fix the reported issues. "
                        "This usually results in clearer, more consistent code."
                    ),
                    "patch_type": "unified_diff",
                    "patch_text": None,
                    "status": "pending",
                }
            )

        # Existing issues from the static tools
        for issue in issues:
            issue_summary = getattr(issue, "summary", None) or getattr(issue, "message", "")
            issue_source = getattr(issue, "source", "tool")
            issue_severity = getattr(issue, "severity", "info")

            suggestions.append(
                {
                    "kind": "issue",
                    "title": f"{issue_source} reported issue ({issue_severity})",
                    "summary": issue_summary,
                    "detail": getattr(issue, "detail", issue_summary),
                    "patch_type": "unified_diff",
                    "patch_text": None,
                    "status": "pending",
                }
            )

        return suggestions

    def _model_suggestions_for_file(
        self,
        file_orm: Any,
        content: str,
        metrics: Dict[str, float],
        issues: Sequence[Any],
        remaining: int,
    ) -> List[Dict[str]()]()
