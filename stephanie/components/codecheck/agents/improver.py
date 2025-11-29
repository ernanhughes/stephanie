from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os

from stephanie.agents.base_agent import BaseAgent
from stephanie.memory.codecheck_store import CodeCheckStore
from stephanie.models.base_config import BaseConfig

import logging
log = logging.getLogger(__name__)

@dataclass
class CodeCheckImproverConfig(BaseConfig):
    """
    Lightweight, typed config for the CodeCheckImproverAgent.

    This sits on top of your normal Hydra/agent config or a plain YAML dict.
    You can either:
      * Construct it directly in SIS (CodeCheckImproverConfig(run_id=...)), or
      * Build it from a Hydra/OmegaConf object / YAML dict via from_source().
    """

    run_id: Optional[str] = None
    max_files: int = 10
    max_suggestions_per_file: int = 5

    # how to rank files (weights over existing metrics)
    file_priority_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "security.bandit_high": 2.0,
            "readability.ruff_style": 1.0,
            "vibe.instruction_compliance": -1.0,
        }
    )

    # how much recency vs severity matters (0..1)
    recency_weight: float = 0.5  # "50% should be the data added to the system"

    critic_model_name: str = "llama-3.1-8b-code-instruct"
    critic_temperature: float = 0.2

    @classmethod
    def from_source(
        cls,
        src: Any,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "CodeCheckImproverConfig":
        """
        Build a typed config from:
          - Hydra/OmegaConf dict-like object
          - A plain dict (e.g. loaded from YAML)
          - Any object with attributes matching our field names

        Also understands nested blocks like:
          selection.recent_weight      → recency_weight
          selection.dirty_metrics      → file_priority_weights (uniform)
          model.model_name             → critic_model_name
          model.temperature            → critic_temperature
        """
        overrides = overrides or {}
        data: Dict[str, Any] = {}

        def get_nested(obj: Any, path: Sequence[str]) -> Any:
            cur = obj
            for key in path:
                if cur is None:
                    return None
                # Attribute-style (DictConfig, simple objects)
                if hasattr(cur, key):
                    cur = getattr(cur, key)
                    continue
                # Mapping-style (dict, OmegaConf mapping)
                if isinstance(cur, dict) and key in cur:
                    cur = cur[key]
                    continue
                return None
            return cur

        # First pass: top-level fields + overrides
        for f in fields(cls):
            name = f.name
            if name in overrides:
                data[name] = overrides[name]
                continue

            # Prefer attribute-style access
            if hasattr(src, name):
                data[name] = getattr(src, name)
            # Fallback to mapping-style access
            elif isinstance(src, dict) and name in src:
                data[name] = src[name]

        # Second pass: fill from nested 'selection' and 'model' if not already set
        if "recency_weight" not in data or data["recency_weight"] is None:
            val = get_nested(src, ("selection", "recent_weight"))
            if val is not None:
                data["recency_weight"] = float(val)

        if "file_priority_weights" not in data or data["file_priority_weights"] is None:
            dm = get_nested(src, ("selection", "dirty_metrics"))
            if dm:
                # Uniform weights over listed dirty metrics
                data["file_priority_weights"] = {str(name): 1.0 for name in dm}

        if "critic_model_name" not in data or data["critic_model_name"] is None:
            val = get_nested(src, ("model", "model_name"))
            if val is not None:
                data["critic_model_name"] = str(val)

        if "critic_temperature" not in data or data["critic_temperature"] is None:
            val = get_nested(src, ("model", "temperature"))
            if val is not None:
                data["critic_temperature"] = float(val)

        return cls(**data)


class CodeCheckImproverAgent(BaseAgent):
    """
    Takes a CodeCheck run, selects the worst files, and uses a code critic
    (local model or heuristic) to generate concrete, small improvement suggestions.

    The agent config comes from:
      * YAML / dict (e.g. config/agents/codecheck_improver.yaml), OR
      * Hydra (cfg.agents.codecheck_improver), OR
      * A direct CodeCheckImproverConfig instance.
    """

    def __init__(self, cfg: Any, memory, container, logger):
        # Preserve the original config (Hydra dict, DictConfig, dataclass, or YAML dict)
        self._raw_cfg = cfg

        # Build the typed overlay we use inside the agent
        if isinstance(cfg, CodeCheckImproverConfig):
            self.cfg = cfg
            cfg_dict: Dict[str, Any] = asdict(cfg)
        else:
            # Derive the typed view from whatever we were given
            self.cfg = CodeCheckImproverConfig.from_source(cfg)
            # For BaseAgent we still want a flat dict so generic features work
            if isinstance(cfg, dict):
                cfg_dict = dict(cfg)
            else:
                cfg_dict = asdict(self.cfg)

        super().__init__(cfg_dict, memory, container, logger)
        self.store: CodeCheckStore = getattr(self.memory, "codecheck")

    # ------------------------------------------------------------------ public API

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entrypoint (kept synchronous so SIS can call it from a threadpool).

        `run_id` resolution priority:
          1. Explicit cfg.run_id
          2. context["codecheck_run_id"] / context["run_id"] / context["pipeline_run_id"]
          3. kwargs["run_id"]
        """
        run_id = context.get("pipeline_run_id")

        run = self.store.get_run(run_id)
        repo_root = run.repo_root


        if not run_id:
            raise ValueError("CodeCheckImproverAgent requires `run_id` in cfg or context.")

        # 1. Select top-N problematic files for this run
        files = self._select_priority_files(run_id, self.cfg.get("max_files", 10))
        log.info(
            "Improver: selected %d priority files for run %s", len(files), run_id
        )

        total_suggestions = 0
        per_file_counts: Dict[int, int] = {}

        # 2. For each file, call critic and store suggestions
        for f in files:
            file_id = f.id
            abs_path = os.path.join(repo_root, f.path)

            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
            except OSError as e:
                log.warning("Improver: unable to read %s: %s", abs_path, e)
                continue

            metrics = (
                f.metrics.vector if getattr(f, "metrics", None) and f.metrics.vector else {}
            )  # type: ignore[attr-defined]
            issues = self.store.list_issues_for_file(file_id)

            suggestions = self._generate_suggestions_for_file(
                file_path=f.path,
                content=content,
                metrics=metrics,
                issues=issues,
                max_suggestions=self.cfg.get("max_suggestions_per_file", 5),
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
        Select top-N files by a blend of:
          - severity (from metrics)
          - recency (from filesystem mtime)

        priority = recency_weight * recency_score + (1 - recency_weight) * severity_score

        Where:
          - recency_score is in [0, 1], 1 = most recently modified.
          - severity_score is a weighted sum of metrics.
        """
        # 1) Get run so we know repo_root
        run = self.store.get_run(run_id)
        repo_root = run.repo_root

        # 2) Load files (with metrics eager-loaded in the store)
        files = self.store.list_files_for_run(run_id, limit=100000)

        # We'll compute severity first, collect mtimes, then normalize.
        mtimes: Dict[int, float] = {}
        severity_scores: Dict[int, float] = {}

        # 3) First pass: compute severity and mtime
        for f in files:
            file_id = f.id

            # --- severity score from metrics
            m = f.metrics.vector if getattr(f, "metrics", None) and f.metrics.vector else {}
            severity = 0.0
            for name, weight in self.cfg.get("file_priority_weights", {}).items():
                severity += float(m.get(name, 0.0)) * float(weight)
            severity_scores[file_id] = severity

            # --- recency from filesystem mtime
            abs_path = os.path.join(repo_root, f.path)
            try:
                mt = os.path.getmtime(abs_path)
            except OSError:
                # If the file is missing (deleted, moved), treat as very old
                mt = 0.0
            mtimes[file_id] = mt

        if not files:
            return []

        # 4) Normalize recency to [0, 1]  (1 = most recent)
        max_mtime = max(mtimes.values())
        min_mtime = min(mtimes.values())

        recency_scores: Dict[int, float] = {}
        if max_mtime == min_mtime:
            # all the same age; recency doesn't differentiate
            for file_id in mtimes:
                recency_scores[file_id] = 0.5
        else:
            span = max_mtime - min_mtime
            for file_id, mt in mtimes.items():
                recency_scores[file_id] = (mt - min_mtime) / span  # 0..1

        # 5) Combine into final priority
        alpha = float(self.cfg.get("recency_weight", 0.5))
        alpha = max(0.0, min(1.0, alpha))  # clamp just in case

        scored: List[Tuple[float, Any]] = []
        for f in files:
            file_id = f.id
            severity = severity_scores[file_id]
            recency = recency_scores[file_id]

            # NOTE: severity can be positive or negative depending on weights;
            # sorting still works because we just want "largest priority first".
            priority = alpha * recency + (1.0 - alpha) * severity
            scored.append((priority, f))

        # 6) Sort descending: highest priority (new + severe) first
        scored.sort(key=lambda t: t[0], reverse=True)

        return [f for (_, f) in scored[:limit]]

    def _generate_suggestions_for_file(
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
            ai_suggestions = self._model_based_suggestions(
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
          - add docstrings to public functions
          - surface top lint issues as todos
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

    def _model_based_suggestions(
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
