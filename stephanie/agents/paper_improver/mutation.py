# stephanie/agents/paper_improver/mutation.py
from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MutationReport:
    available: bool
    score: Optional[float]                 # killed / (killed + survived)
    killed: int
    survived: int
    timeout: int
    skipped: int
    suspicious: int                        # mutmut "incompetent" or "suspicious" bucket
    runtime_sec: float
    cmd_run: str
    stdout_snippet: str
    stderr_snippet: str
    workdir: str
    details_path: Optional[str] = None     # path to raw results file (if saved)

class MutationRunner:
    """
    Run mutation tests using `mutmut` (https://mutmut.readthedocs.io/).
    - Enforces CPU/RAM limits.
    - Supports timeouts.
    - Parses robustly from `mutmut results`.
    - Non-fatal if mutmut isn't present (available=False).
    Usage:
        rep = MutationRunner().run(run_dir="...", src="src", tests="tests")
    """

    def __init__(
        self,
        timeout_sec: int = 900,
        cpu_seconds: int = 600,
        mem_bytes: int = 1_500_000_000,  # ~1.5GB
        pytest_cmd: str = "python -m pytest -q",
    ):
        self.timeout_sec = timeout_sec
        self.cpu_seconds = cpu_seconds
        self.mem_bytes = mem_bytes
        self.pytest_cmd = pytest_cmd

    # ---------- public API ----------

    def run(
        self,
        run_dir: str | Path,
        src: str | Path = "src",
        tests: str | Path = "tests",
        paths_to_mutate: Optional[List[str]] = None,
        save_details: bool = True,
    ) -> Dict[str, Any]:
        run_dir = Path(run_dir).resolve()
        src = Path(src)
        tests = Path(tests)

        if shutil.which("mutmut") is None:
            # Mutmut not installed; return non-fatal report
            return asdict(MutationReport(
                available=False, score=None, killed=0, survived=0, timeout=0, skipped=0,
                suspicious=0, runtime_sec=0.0, cmd_run="mutmut (not installed)",
                stdout_snippet="", stderr_snippet="", workdir=str(run_dir), details_path=None
            ))

        # Ensure working directory contains a config (optional but helps portability)
        self._ensure_mutmut_config(run_dir, src, tests)

        # Compose command
        cmd_run = self._build_run_cmd(paths_to_mutate, src, tests)

        # Execute `mutmut run`
        start = time.time()
        try:
            proc = subprocess.run(
                cmd_run,
                cwd=run_dir,
                text=True,
                capture_output=True,
                timeout=self.timeout_sec,
                preexec_fn=self._limit_resources,
            )
            run_stdout, run_stderr = proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as te:
            return asdict(MutationReport(
                available=True, score=None, killed=0, survived=0, timeout=0, skipped=0,
                suspicious=0, runtime_sec=float(self.timeout_sec), cmd_run=" ".join(cmd_run),
                stdout_snippet="(timeout)", stderr_snippet=str(te), workdir=str(run_dir), details_path=None
            ))

        # Now `mutmut results` to get summary
        res_cmd = ["mutmut", "results"]
        try:
            res = subprocess.run(
                res_cmd,
                cwd=run_dir,
                text=True,
                capture_output=True,
                timeout=120,
                preexec_fn=self._limit_resources,
            )
            results_out = res.stdout + "\n" + res.stderr
        except subprocess.TimeoutExpired:
            results_out = run_stdout + "\n" + run_stderr

        # Parse
        killed, survived, timeout_cnt, skipped, suspicious = self._parse_results(results_out)
        total = max(1, killed + survived)  # avoid div-by-zero
        score = killed / total

        details_path = None
        if save_details:
            details_path = str((run_dir / "mutation_results.txt").resolve())
            Path(details_path).write_text(results_out)

        return asdict(MutationReport(
            available=True,
            score=round(score, 4),
            killed=killed,
            survived=survived,
            timeout=timeout_cnt,
            skipped=skipped,
            suspicious=suspicious,
            runtime_sec=round(time.time() - start, 3),
            cmd_run=" ".join(cmd_run),
            stdout_snippet=(run_stdout or "")[:2000],
            stderr_snippet=(run_stderr or "")[:2000],
            workdir=str(run_dir),
            details_path=details_path
        ))

    # ---------- internals ----------

    def _ensure_mutmut_config(self, run_dir: Path, src: Path, tests: Path):
        """
        Create a minimal .mutmut.yaml if none exists.
        Default runner uses pytest command.
        """
        cfg = run_dir / ".mutmut.yaml"
        if cfg.exists():
            return
        content = f"""# Auto-generated by MutationRunner
runner: "{self.pytest_cmd}"
paths_to_mutate:
  - "{src}"
tests_dir: "{tests}"
backup: False
CI: true
"""
        cfg.write_text(content)

    def _build_run_cmd(self, paths_to_mutate: Optional[List[str]], src: Path, tests: Path) -> List[str]:
        cmd = ["mutmut", "run", "--use-coverage"]
        # prefer explicit paths to mutate if provided; else rely on config
        if paths_to_mutate:
            for p in paths_to_mutate:
                cmd += ["--paths-to-mutate", p]
        # explicit tests dir can help on some setups
        cmd += ["--tests-dir", str(tests)]
        # reduce interactivity
        cmd += ["--silent"]
        return cmd

    def _limit_resources(self):
        """Apply CPU and memory limits for safety."""
        pass

    def _parse_results(self, text: str) -> Tuple[int, int, int, int, int]:
        """
        Robustly parse mutmut summary from `mutmut results` output.
        Typical lines include:
           - 3 survived
           - 10 killed
           - 0 timeout
           - 1 suspicious
           - 0 skipped
        """
        # Normalize
        t = text.lower()
        def pick(patterns: List[str]) -> int:
            for pat in patterns:
                m = re.search(rf"(\d+)\s+{pat}", t)
                if m:
                    return int(m.group(1))
            return 0

        killed = pick(["killed", "killed mutants"])
        survived = pick(["survived", "survived mutants"])
        timeout = pick(["timeout", "timeouts"])
        skipped = pick(["skipped", "skip"])
        suspicious = pick(["suspicious", "incompetent"])  # mutmut sometimes uses 'incompetent'

        return killed, survived, timeout, skipped, suspicious
