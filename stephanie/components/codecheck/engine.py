from __future__ import annotations

import logging
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from stephanie.memory.codecheck_store import CodeCheckStore
from stephanie.models.base_config import BaseConfig

# adjust this import to wherever your BaseAgent actually lives

log = logging.getLogger(__name__)

VIBE_INSTRUCTION_COMPLIANCE = "vibe.instruction_compliance"

@dataclass
class CodeCheckConfig(BaseConfig):
    # … same as before …
    repo_root: str = field(
        default=".",
        metadata={"help": "Absolute or relative path to the repository root to be scanned."},
    )
    file_extensions: List[str] = field(
        default_factory=lambda: [".py"],
        metadata={"help": "File extensions to include in the scan."},
    )
    exclude_paths: List[str] = field(
        default_factory=lambda: ["__pycache__", ".git", "venv", "dist", "build"],
        metadata={"help": "Directory or path fragments to exclude from the scan."},
    )
    language: str = field(
        default="python",
        metadata={"help": "Primary language of the repository."},
    )
    metric_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "security.bandit": 0.4,
            "readability.ruff": 0.3,
            "test_coverage": 0.2,
            VIBE_INSTRUCTION_COMPLIANCE: 0.1,
        },
        metadata={"help": "Weights for combining individual metrics into a composite score."},
    )
    critic_model_name: str = field(
        default="llama-7b-code-instruct",
        metadata={"help": "Local SLM / code model name for the critic policy (Phase 2+)."},
    )
    vram_budget_gb: int = field(
        default=10,
        metadata={"help": "Maximum VRAM budget for local model inference."},
    )


def _scan_repo_files(root_path: str, extensions: List[str], exclude_paths: List[str]) -> List[str]:
    root_path = os.path.abspath(root_path)
    all_files: List[str] = []

    exclude_dir_names = set(os.path.basename(os.path.normpath(p)) for p in exclude_paths)

    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in exclude_dir_names]

        rel_dir = os.path.relpath(dirpath, root_path)
        if rel_dir == ".":
            rel_dir = ""

        for filename in filenames:
            if not any(filename.endswith(ext) for ext in extensions):
                continue

            rel_path = os.path.join(rel_dir, filename) if rel_dir else filename
            norm_rel = os.path.normpath(rel_path)

            if any(ex in norm_rel.split(os.sep) for ex in exclude_paths):
                continue

            all_files.append(norm_rel)

    return all_files


def _detect_repo_context(repo_root: str) -> Tuple[Optional[str], Optional[str]]:
    repo_root = os.path.abspath(repo_root)

    def _run_git(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(  # noqa: S603, S607
                ["git", *args],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            )
            return out.decode("utf-8", "ignore").strip()
        except Exception:  # noqa: BLE001
            return None

    commit_hash = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])

    return branch, commit_hash


def _looks_like_test(path: str) -> bool:
    path = path.replace("\\", "/")
    name = os.path.basename(path)
    return (
        "/tests/" in path
        or path.startswith("tests/")
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


class CodeCheckEngine:
    """
    Main orchestration layer for CodeCheck (Freestyle Code Critic).

    Phase 0 responsibilities:
      - Create a run record.
      - Discover files to analyze.
      - Extract basic per-file stats.
      - Run a (currently stubbed) metric swarm.
      - Persist everything via CodeCheckStore.

    Later phases will add:
      - Graph-aware episodes (CodeGraph).
      - Real static tools (Bandit/Ruff/mypy/coverage).
      - HRM scoring and critic policy / patch generation.
    """

    def __init__(self, cfg: CodeCheckConfig, memory, container, logger):
        self.cfg: CodeCheckConfig = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        # repo root from cfg
        self.repo_root = os.path.abspath(self.cfg.repo_root)

        # CodeCheck store is exposed via memory
        # (if you call it something else, adjust this attribute name)
        self.store: CodeCheckStore = getattr(self.memory, "codecheck")

        # Placeholder for CodeGraph (Phase 1)
        self.code_graph: Optional[Any] = None

    #  public API

    async def run(self, *args, **kwargs) -> str:
        """
        Agent-style entry point that wraps run_analysis().

        Returns:
            run_id (str)
        """
        return self.run_analysis()


    def run_analysis(self) -> str:
        """
        Execute a full CodeCheck run over the configured repo.

        Returns:
            run_id (str): ID of the created run record.
        """
        run_id = str(uuid.uuid4())
        branch, commit_hash = _detect_repo_context(self.repo_root)

        log.info(
            "Starting CodeCheck run %s on %s (branch=%s, commit=%s)",
            run_id,
            self.repo_root,
            branch or "?",
            (commit_hash or "")[:8] if commit_hash else "?",
        )

        run_cfg_dict = (
            self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else dict(self.cfg.__dict__)
        )

        # 1. Create run record
        self.store.create_run(
            run_id=run_id,
            repo_root=self.repo_root,
            branch=branch,
            commit_hash=commit_hash,
            language=self.cfg.language,
            config=run_cfg_dict,
        )

        # Streaming summary accumulators (O(1) memory)
        files_total = 0
        files_with_metrics = 0
        vibe_sum = 0.0
        security_sum = 0.0
        readability_sum = 0.0

        try:
            # 2. Discover files
            file_paths = _scan_repo_files(
                self.repo_root,
                self.cfg.file_extensions,
                self.cfg.exclude_paths,
            )
            log.info(
                "CodeCheck run %s: found %d files to process", run_id, len(file_paths)
            )

            # 3. Process each file as an "episode"
            for rel_path in file_paths:
                abs_path = os.path.join(self.repo_root, rel_path)
                files_total += 1

                try:
                    metric_vector = self._process_file_episode(run_id, rel_path, abs_path)
                except Exception as file_err:  # noqa: BLE001
                    log.error(
                        "Error processing file %s in run %s: %s",
                        rel_path,
                        run_id,
                        file_err,
                        exc_info=True,
                    )
                    continue

                if not metric_vector:
                    continue

                files_with_metrics += 1
                vibe_sum += float(metric_vector.get(VIBE_INSTRUCTION_COMPLIANCE, 0.0))
                security_sum += float(metric_vector.get("security.bandit_high", 0.0))
                readability_sum += float(metric_vector.get("readability.ruff_style", 0.0))

            # 4. Build summary from accumulators (no extra DB reads)
            summary: Dict[str, float] = {
                "files.count": float(files_total),
            }
            if files_with_metrics:
                summary.update(
                    {
                        "vibe.mean": vibe_sum / files_with_metrics,
                        "security.total": security_sum,
                        "readability.total": readability_sum,
                    }
                )

            # 5. Finalize run
            self.store.update_run_status(
                run_id,
                status="success",
                status_message=f"Successfully analyzed {files_total} files.",
                summary_metrics=summary,
                finished=True,
            )

            log.info("CodeCheck run %s finished successfully", run_id)
            return run_id

        except Exception as e:  # noqa: BLE001
            log.error("CodeCheck run %s failed: %s", run_id, e, exc_info=True)
            self.store.update_run_status(
                run_id,
                status="failed",
                status_message=f"Analysis failed: {str(e)[:512]}",
                finished=True,
            )
            raise

    # ------------------------------------------------------------------ internals

    def _process_file_episode(
        self,
        run_id: str,
        file_path_rel: str,
        file_path_abs: str,
    ) -> Dict[str, float] | None:
        """
        Process a single file and persist its metrics.

        Returns:
            metric_vector dict, or None if we couldn't read the file.
        """
        try:
            with open(file_path_abs, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except OSError as e:
            log.warning("Unable to read file %s: %s", file_path_abs, e)
            return None

        loc = len(content.splitlines()) if content else 0
        content_hash = hex(hash(content)) if content else None

        file_orm = self.store.upsert_file(
            run_id=run_id,
            path=file_path_rel,
            language=self.cfg.language,
            loc=loc,
            content_hash=content_hash,
            is_test=_looks_like_test(file_path_rel),
        )

        metric_vector = self._run_metric_swarm(file_path_abs, content)

        self.store.upsert_file_metrics(
            file_id=file_orm.id,
            columns=list(metric_vector.keys()),
            values=[float(v) for v in metric_vector.values()],
            vector=metric_vector,
        )

        # TODO later: derive issues from real tools and bulk insert

        return metric_vector


    def _run_metric_swarm(self, file_path_abs: str, file_content: str) -> Dict[str, float]:
        has_def = "def " in file_content
        has_triple_quotes = '"""' in file_content or "'''" in file_content

        if has_def and has_triple_quotes:
            vibe_score = 1.0
        elif has_def:
            vibe_score = 0.5
        else:
            vibe_score = 0.0

        security_issues = float(file_content.count("subprocess") + file_content.count("os.system"))
        readability_issues = float(file_content.count("lambda") + file_content.count("TODO"))
        semantic_risk = 0.0

        return {
            VIBE_INSTRUCTION_COMPLIANCE: vibe_score,
            "security.bandit_high": security_issues,
            "readability.ruff_style": readability_issues,
            "semantic.risk": semantic_risk,
        }

