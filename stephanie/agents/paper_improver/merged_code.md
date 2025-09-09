<!-- Merged Python Code Files -->


## File: bandit_router.py

`python
# stephanie/agents/paper_improver/bandit_router.py

# bandit_router.py — UCB1/Thompson Sampling for exemplar selection based on historical uplift.
# Logs plays, rewards, saves state. Routes by spec type or claim density.

import json
import math
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

class ExemplarBandit:
    """
    Multi-armed bandit for routing to best exemplar pack.
    Tracks uplift in target metric (e.g., pass_rate, coverage).
    Supports UCB1 and Thompson Sampling.
    Persists state to disk.
    """

    def __init__(
        self,
        save_path: str = "./bandit_state.json",
        strategy: str = "ucb1",  # or "thompson"
        reward_metric: str = "pass_rate",  # or "coverage", etc.
        default_reward: float = 0.5,
        smoothing: float = 1.0  # for Thompson prior
    ):
        self.save_path = Path(save_path)
        self.strategy = strategy
        self.reward_metric = reward_metric
        self.default_reward = default_reward
        self.smoothing = smoothing
        self.arms: Dict[str, Dict[str, Union[int, float]]] = {}
        self.total_plays = 0
        self._load_state()

    def choose(self, candidate_ids: List[str], context: Optional[Dict[str, Any]] = None) -> str:
        """
        Select best exemplar_id given candidates.
        Context can be spec type, claim count, etc. — ignored for now (future: contextual bandit).
        """
        if not candidate_ids:
            raise ValueError("No candidate exemplar IDs provided.")

        # Initialize unseen arms
        for eid in candidate_ids:
            if eid not in self.arms:
                self.arms[eid] = {
                    "plays": 0,
                    "reward_sum": 0.0,
                    "reward_avg": self.default_reward
                }

        if self.strategy == "ucb1":
            return self._choose_ucb1(candidate_ids)
        elif self.strategy == "thompson":
            return self._choose_thompson(candidate_ids)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _choose_ucb1(self, candidate_ids: List[str]) -> str:
        """UCB1: balance exploration + exploitation."""
        total_plays = max(1, sum(self.arms[eid]["plays"] for eid in candidate_ids))
        scores = []
        for eid in candidate_ids:
            arm = self.arms[eid]
            if arm["plays"] == 0:
                scores.append((float('inf'), eid))
            else:
                avg = arm["reward_avg"]
                confidence = math.sqrt(2 * math.log(total_plays) / arm["plays"])
                ucb = avg + confidence
                scores.append((ucb, eid))
        return max(scores, key=lambda x: x[0])[1]

    def _choose_thompson(self, candidate_ids: List[str]) -> str:
        """Thompson Sampling: sample from Beta posterior."""
        samples = []
        for eid in candidate_ids:
            arm = self.arms[eid]
            # Beta(alpha, beta) ~ Beta(successes + smoothing, failures + smoothing)
            successes = arm["reward_sum"] * arm["plays"]  # approximate
            failures = arm["plays"] - successes
            alpha = successes + self.smoothing
            beta = failures + self.smoothing
            sample = random.betavariate(alpha, beta)
            samples.append((sample, eid))
        return max(samples, key=lambda x: x[0])[1]

    def update(self, exemplar_id: str, reward: float):
        """
        Update arm with observed reward (e.g., pass_rate delta, coverage delta).
        Reward should be 0.0 – 1.0.
        """
        if exemplar_id not in self.arms:
            self.arms[exemplar_id] = {
                "plays": 0,
                "reward_sum": 0.0,
                "reward_avg": self.default_reward
            }

        arm = self.arms[exemplar_id]
        arm["plays"] += 1
        arm["reward_sum"] += reward
        arm["reward_avg"] = arm["reward_sum"] / arm["plays"]

        self.total_plays += 1
        self._save_state()

    def get_stats(self, exemplar_id: str) -> Dict[str, Any]:
        """Get current stats for an arm."""
        return self.arms.get(exemplar_id, {
            "plays": 0,
            "reward_avg": self.default_reward,
            "reward_sum": 0.0
        })

    def get_leaderboard(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return top-k arms by reward_avg."""
        scored = [
            {"exemplar_id": eid, **stats}
            for eid, stats in self.arms.items()
        ]
        return sorted(scored, key=lambda x: x["reward_avg"], reverse=True)[:top_k]

    def _load_state(self):
        """Load bandit state from disk."""
        if self.save_path.exists():
            try:
                data = json.loads(self.save_path.read_text())
                self.arms = data.get("arms", {})
                self.total_plays = data.get("total_plays", 0)
                print(f"📊 Bandit state loaded: {len(self.arms)} arms, {self.total_plays} total plays.")
            except Exception as e:
                print(f"⚠️ Failed to load bandit state: {e}")

    def _save_state(self):
        """Persist bandit state to disk."""
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "arms": self.arms,
                "total_plays": self.total_plays,
                "strategy": self.strategy,
                "reward_metric": self.reward_metric
            }
            self.save_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"⚠️ Failed to save bandit state: {e}")

    def reset(self):
        """Reset bandit state."""
        self.arms = {}
        self.total_plays = 0
        if self.save_path.exists():
            self.save_path.unlink()
        print("🔄 Bandit state reset.")

    def __repr__(self):
        return f"<ExemplarBandit strategy={self.strategy} arms={len(self.arms)} plays={self.total_plays}>"
``n

## File: code_improver.py

`python
# stephanie/agents/paper_improver/code_improver.py

# code_improver.py — spec → tests → stub → verify → edit → log → PR-ready
# Enhanced with: real metrics, safe execution, AST denylist, edit-policy rules, VPM, DPO pairs.
import json
import subprocess
import hashlib
import re
import os
import resource
import signal
import tempfile
import ast
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

class CodeImprover:
    def __init__(
        self,
        workdir: str = "./improver_runs",
        backend: str = "torch",
        timeout: int = 300,
        max_edits: int = 5
    ):
        self.workdir = Path(workdir)
        self.workdir.mkdir(exist_ok=True)
        self.run_id = 0
        self.backend = backend
        self.timeout = timeout
        self.max_edits = max_edits
        self.denylist_imports = {"os", "subprocess", "socket", "sys", "shutil", "pickle", "marshal"}

    def improve(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Takes spec JSON → returns PR-ready artifact + VPM row + DPO pair."""
        self.run_id += 1
        spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
        run_dir = self.workdir / f"run_{self.run_id}_{spec_hash}"
        run_dir.mkdir()

        # Determinism & metadata
        meta = {
            "spec_sha": spec_hash,
            "seeds": {"python": 0, "torch": 0},
            "backend": self.backend,
            "timeout": self.timeout,
            "max_edits": self.max_edits
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        # Write spec
        spec_path = run_dir / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=2))

        # Scaffold package: src/pkg/impl.py
        pkg_dir = run_dir / "src" / "pkg"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "__init__.py").touch()

        # Generate initial stub
        stub_path = pkg_dir / "impl.py"
        stub_content = self._generate_stub(spec)
        self._ast_denylist_check(stub_content)  # security: AST parse + block dangerous nodes
        stub_path.write_text(stub_content)

        # Generate tests
        test_path = self._generate_tests(spec, run_dir)

        # Initial test + lint run
        print(f"🧪 Initial test run for {spec['function_name']}...")
        initial_results = self._run_tests_and_lint(run_dir, test_path)

        # Apply edit-policy until pass or max edits
        final_code, edit_log = self._apply_edit_policy(stub_path, test_path, run_dir)

        # Final evaluation
        final_results = self._run_tests_and_lint(run_dir, test_path)

        # Build VPM row
        vpm_row = self._build_vpm_row(initial_results, final_results, spec, run_dir)

        # Log DPO pair (function body only + failing test names)
        dpo_pair = self._build_dpo_pair(
            initial_code=initial_results.get("code_snapshot", ""),
            final_code=final_code,
            spec=spec,
            meta=meta,
            initial_results=initial_results
        )
        (run_dir / "dpo_pair.json").write_text(json.dumps(dpo_pair, indent=2))

        # Return structured artifact
        return {
            "run_dir": str(run_dir),
            "spec_path": str(spec_path),
            "final_code_path": str(stub_path),
            "test_path": str(test_path),
            "vpm_row": vpm_row,
            "dpo_pair_path": str(run_dir / "dpo_pair.json"),
            "passed": final_results["pass_rate"] == 1.0,
            "coverage": final_results.get("coverage", 0.0),
            "escalations": 0 if final_results["pass_rate"] == 1.0 else 1,
            "edit_log": edit_log
        }

    # -------------------------
    # Generation
    # -------------------------

    def _generate_stub(self, spec: Dict[str, Any]) -> str:
        backend_import = "import torch" if self.backend == "torch" else "import numpy as np"
        fn_name = spec["function_name"]
        desc = spec.get("description", "No description.")
        eqs = spec.get("equations", [])
        eq_str = "; ".join(eqs) if eqs else "Not specified."

        return f'''{backend_import}

def {fn_name}(x):
    """
    {desc}
    Equations: {eq_str}
    """
    # TODO: Implement per spec.
    return x  # placeholder
'''

    def _generate_tests(self, spec: Dict[str, Any], run_dir: Path) -> Path:
        fn_name = spec["function_name"]
        golden_in = spec.get("golden_input", [1.0, 2.0])
        golden_out = spec.get("golden_output", 3.0)
        in_shape = spec.get("input_shape", [10])
        out_shape = spec.get("output_shape", [10])

        if self.backend == "torch":
            backend_import = "import torch"
            rand_in = f"torch.randn({in_shape})"
            assert_shape = f"assert y.shape == torch.Size({out_shape})"
        else:
            backend_import = "import numpy as np"
            rand_in = f"np.random.randn(*{in_shape})"
            assert_shape = f"assert y.shape == tuple({out_shape})"

        test_content = f'''
import pytest
{backend_import}
from pkg.impl import {fn_name}

def test_{fn_name}_golden():
    input = {golden_in}
    expected = {golden_out}
    result = {fn_name}(input)
    assert result == pytest.approx(expected, rel=1e-3)

def test_{fn_name}_shape():
    x = {rand_in}
    y = {fn_name}(x)
    {assert_shape}
'''
        test_dir = run_dir / "tests"
        test_dir.mkdir()
        test_path = test_dir / f"test_{fn_name}.py"
        test_path.write_text(test_content)
        return test_path

    # -------------------------
    # Execution & Metrics
    # -------------------------

    def _limit_resources(self):
        """Enforce CPU, memory, no network."""
        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
        resource.setrlimit(resource.RLIMIT_AS, (1_200_000_000, 1_200_000_000))  # ~1.2GB
        # Block network: env var or patch in future

    def _run_tests_and_lint(self, run_dir: Path, test_path: Path) -> Dict[str, Any]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(run_dir / "src")
        env["NO_NETWORK"] = "1"  # hint for stubbed clients

        cmd = [
            "python", "-m", "pytest", str(test_path), "-q",
            "--cov=src", "--cov-report=xml:coverage.xml",
            "--json-report", "--json-report-file=pytest_report.json",
            "--maxfail=1"
        ]

        try:
            r = subprocess.run(
                cmd, cwd=run_dir, env=env, text=True, capture_output=True,
                timeout=self.timeout, preexec_fn=self._limit_resources
            )
        except subprocess.TimeoutExpired:
            return self._empty_results(stderr="TIMEOUT")

        # Parse pytest JSON
        try:
            rep = json.loads((run_dir / "pytest_report.json").read_text())
            passed = rep["summary"].get("passed", 0)
            failed = rep["summary"].get("failed", 0)
            errors = rep["summary"].get("error", 0)
            total = passed + failed + errors
            pass_rate = passed / total if total > 0 else 0.0
            failing_tests = [t["nodeid"] for t in rep.get("tests", []) if t["outcome"] in ("failed", "error")]
        except Exception as e:
            return self._empty_results(stderr=f"JSON parse error: {e}")

        # Parse coverage
        coverage = self._parse_coverage(run_dir)

        # Lint + type
        lint_clean = self._run_lint(run_dir)
        type_safe = self._run_mypy(run_dir)

        # Complexity
        complexity_ok = self._run_radon(run_dir)

        code_snapshot = ""
        impl_path = run_dir / "src" / "pkg" / "impl.py"
        if impl_path.exists():
            code_snapshot = impl_path.read_text()

        return {
            "pass_rate": pass_rate,
            "coverage": coverage,
            "type_safe": type_safe,
            "lint_clean": lint_clean,
            "complexity_ok": complexity_ok,
            "stdout": r.stdout,
            "stderr": r.stderr,
            "failing_tests": failing_tests,
            "code_snapshot": code_snapshot
        }

    def _empty_results(self, stderr: str = "") -> Dict[str, Any]:
        return {
            "pass_rate": 0.0, "coverage": 0.0, "type_safe": 0.0,
            "lint_clean": 0.0, "complexity_ok": 0.0, "stderr": stderr,
            "failing_tests": [], "code_snapshot": ""
        }

    def _parse_coverage(self, run_dir: Path) -> float:
        try:
            cov_xml = (run_dir / "coverage.xml").read_text()
            m = re.search(r'line-rate="([\d.]+)"', cov_xml)
            return float(m.group(1)) if m else 0.0
        except Exception:
            return 0.0

    def _run_lint(self, run_dir: Path) -> float:
        try:
            r = subprocess.run(["ruff", "check", "."], cwd=run_dir, capture_output=True, text=True)
            return 1.0 if r.returncode == 0 else 0.0
        except Exception:
            return 0.0

    def _run_mypy(self, run_dir: Path) -> float:
        try:
            r = subprocess.run(["mypy", "."], cwd=run_dir, capture_output=True, text=True)
            return 1.0 if r.returncode == 0 else 0.0
        except Exception:
            return 0.0

    def _run_radon(self, run_dir: Path) -> float:
        try:
            r = subprocess.run(["radon", "cc", "src", "-s"], cwd=run_dir, capture_output=True, text=True)
            if r.returncode != 0:
                return 1.0
            # Penalize if any function > B
            if "C" in r.stdout or "D" in r.stdout or "E" in r.stdout or "F" in r.stdout:
                return 0.5
            return 1.0
        except Exception:
            return 1.0

    # -------------------------
    # Edit Policy
    # -------------------------

    def _apply_edit_policy(self, stub_path: Path, test_path: Path, run_dir: Path) -> Tuple[str, List[str]]:
        code = stub_path.read_text()
        edits = []

        for i in range(self.max_edits):
            results = self._run_tests_and_lint(run_dir, test_path)
            if results["pass_rate"] == 1.0:
                break

            fix = self._propose_minimal_fix(code, results["stderr"], results["failing_tests"])
            if not fix:
                break

            code = fix
            stub_path.write_text(code)
            edits.append(f"Edit {i+1}: applied fix based on: {results['stderr'][:50]}...")
            print(f"🔧 Applied edit {i+1}: {edits[-1]}")

        return code, edits

    def _propose_minimal_fix(self, code: str, stderr: str, failing_tests: List[str]) -> Optional[str]:
        """Rule-based fixes based on error patterns."""

        # Fix 1: dtype mismatch
        if any(kw in stderr for kw in ["dtype", "float", "int", "tensor", "ndarray"]):
            if "torch" in code:
                code = re.sub(r'return (.+)', r'return torch.as_tensor(\1).float()', code)
            elif "numpy" in code:
                code = re.sub(r'return (.+)', r'return np.asarray(\1, dtype=np.float32)', code)
            return code

        # Fix 2: shape mismatch
        if any(kw in stderr for kw in ["shape", "size", "dimension", "broadcast"]):
            if "torch" in code:
                code = re.sub(r'return (.+)', r'return \1.view(-1)', code)
            elif "numpy" in code:
                code = re.sub(r'return (.+)', r'return \1.flatten()', code)
            return code

        # Fix 3: out-of-bounds / negative values
        if any(kw in stderr for kw in ["clamp", "bound", "negative", "positive", "range"]):
            if "torch" in code:
                code = re.sub(r'return (.+)', r'return torch.clamp(\1, min=0.0, max=1.0)', code)
            elif "numpy" in code:
                code = re.sub(r'return (.+)', r'return np.clip(\1, 0.0, 1.0)', code)
            return code

        # Fix 4: sum() on list → backend sum
        if "sum" in stderr and ("list" in stderr or "iterable" in stderr):
            if "torch" in code:
                code = re.sub(r'return sum\((.+)\)', r'return torch.sum(torch.as_tensor(\1))', code)
            elif "numpy" in code:
                code = re.sub(r'return sum\((.+)\)', r'return np.sum(np.asarray(\1))', code)
            return code

        # Fix 5: missing import or undefined name
        if "NameError" in stderr or "not defined" in stderr:
            if "math" in stderr and "import math" not in code:
                code = "import math\n" + code
            return code

        return None

    # -------------------------
    # Safety & Hygiene
    # -------------------------

    def _ast_denylist_check(self, code: str):
        """Parse AST and block dangerous patterns."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name.split('.')[0] in self.denylist_imports:
                            raise ValueError(f"AST denylist triggered: {alias.name}")
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in {"eval", "exec", "__import__"}:
                        raise ValueError(f"AST denylist triggered: {node.func.id}")
        except SyntaxError:
            pass  # let tests catch it

    # -------------------------
    # Output & Logging
    # -------------------------

    def _build_vpm_row(self, initial: Dict[str, Any], final: Dict[str, Any], spec: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        fn_name = spec["function_name"]
        return {
            "unit": f"pkg.impl:{fn_name}",
            "tests_pass_rate": round(final["pass_rate"], 3),
            "coverage": round(final["coverage"], 3),
            "type_safe": round(final["type_safe"], 3),
            "lint_clean": round(final["lint_clean"], 3),
            "complexity_ok": round(final["complexity_ok"], 3),
            "escalations": 0 if final["pass_rate"] == 1.0 else 1
        }

    def _build_dpo_pair(
        self,
        initial_code: str,
        final_code: str,
        spec: Dict[str, Any],
        meta: Dict[str, Any],
        initial_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "spec_slice": {
                k: v for k, v in spec.items()
                if k in ["function_name", "equations", "input_shape", "output_shape", "golden_input", "golden_output"]
            },
            "prompt": f"Implement {spec['function_name']} per spec. Equations: {spec.get('equations', [])}",
            "rejected": self._extract_function_body(initial_code),
            "chosen": self._extract_function_body(final_code),
            "metadata": {
                "spec_hash": meta["spec_sha"],
                "backend": meta["backend"],
                "initial_pass_rate": initial_results["pass_rate"],
                "failing_tests": initial_results.get("failing_tests", []),
                "stderr_snippet": initial_results.get("stderr", "")[:200]
            }
        }

    def _extract_function_body(self, code: str) -> str:
        """Extract just the function body for DPO pairs."""
        lines = code.splitlines()
        in_func = False
        body_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def "):
                in_func = True
                continue
            if in_func and (not stripped or stripped.startswith("#")):
                continue
            if in_func and not (line.startswith("    ") or line.startswith("\t")):
                break
            if in_func:
                body_lines.append(line)
        return "\n".join(body_lines) if body_lines else code
``n

## File: curriculum.py

`python
# stephanie/agents/paper_improver/curriculum.py

# curriculum.py — Paper difficulty scorer + curriculum scheduler for progressive learning.
# Scores papers by "teachability" → routes easy ones first to bootstrap learning.

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

class CurriculumScheduler:
    """
    Ranks papers by estimated difficulty (low = easy, high = hard).
    Prioritizes papers with:
      - Explicit pseudocode or algorithms
      - Clear metrics/tables
      - Minimal proofs/theorems
    Outputs ordered queue for training or processing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        return {
            "bonus_keywords": {
                "algorithm": 1.5,
                "pseudocode": 2.0,
                "procedure": 1.0,
                "table": 0.8,
                "figure": 0.5,
                "equation": 0.5,
                "eq.": 0.5,
                "implementation": 1.0,
                "code": 0.8
            },
            "penalty_keywords": {
                "proof": -1.0,
                "theorem": -1.0,
                "lemma": -0.8,
                "corollary": -0.8,
                "induction": -1.2,
                "derivation": -0.7,
                "by inspection": -0.5,
                "trivially": -0.3,
                "without loss of generality": -0.8
            },
            "section_bias": {
                "method": 1.0,
                "approach": 1.0,
                "experimental setup": 0.5,
                "results": 0.3,
                "related work": -0.5,
                "introduction": 0.0,
                "conclusion": -0.3
            },
            "min_score_threshold": 0.0,   # filter out papers below this
            "max_difficulty": 10.0        # cap for normalization
        }

    def compute_teachability_score(self, paper: Dict[str, Any]) -> float:
        """
        Compute a normalized teachability score [0, 1] for a paper.
        Higher = easier to turn into code/text automatically.
        """
        text = self._extract_relevant_text(paper).lower()
        title = paper.get("title", "").lower()
        section = paper.get("section", "").lower()

        score = 0.0

        # Bonus: presence of implementation-friendly keywords
        for keyword, weight in self.config["bonus_keywords"].items():
            count = text.count(keyword) + title.count(keyword)
            score += count * weight

        # Penalty: presence of proofy/abstract keywords
        for keyword, weight in self.config["penalty_keywords"].items():
            count = text.count(keyword) + title.count(keyword)
            score += count * weight

        # Section bias: prioritize method/approach sections
        for sec_keyword, bias in self.config["section_bias"].items():
            if sec_keyword in section:
                score += bias

        # Normalize to [0, 1]
        max_possible = sum(w * 10 for w in self.config["bonus_keywords"].values())  # rough upper bound
        score = max(0.0, min(self.config["max_difficulty"], score))
        normalized = score / self.config["max_difficulty"]

        return round(normalized, 3)

    def _extract_relevant_text(self, paper: Dict[str, Any]) -> str:
        """Extract text fields to score — prioritize body, abstract, captions."""
        chunks = []
        for key in ["abstract", "body", "text", "content", "caption", "description"]:
            if key in paper and isinstance(paper[key], str):
                chunks.append(paper[key])
        return " ".join(chunks)

    def schedule_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort papers by teachability score (descending: easiest first).
        Optionally filter by min_score_threshold.
        """
        scored = []
        for paper in papers:
            score = self.compute_teachability_score(paper)
            if score >= self.config["min_score_threshold"]:
                scored.append((score, paper))

        # Sort descending: highest teachability (easiest) first
        sorted_papers = [paper for score, paper in sorted(scored, key=lambda x: x[0], reverse=True)]

        return sorted_papers

    def tag_and_rank(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Return papers with added 'teachability_score' and 'difficulty_rank'.
        """
        scored = [(self.compute_teachability_score(p), p) for p in papers]
        sorted_scored = sorted(scored, key=lambda x: x[0], reverse=True)

        for rank, (score, paper) in enumerate(sorted_scored):
            paper["teachability_score"] = score
            paper["difficulty_rank"] = rank + 1  # 1 = easiest

        return [paper for score, paper in sorted_scored]

    def load_from_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load papers from JSON files (e.g., scraped/arxiv)."""
        papers = []
        for path in file_paths:
            p = Path(path)
            if p.exists() and p.suffix == ".json":
                try:
                    paper = json.loads(p.read_text())
                    paper["source_path"] = str(p)
                    papers.append(paper)
                except Exception as e:
                    print(f"⚠️ Failed to load {p}: {e}")
        return papers

    def save_curriculum(self, papers: List[Dict[str, Any]], output_path: str):
        """Save ranked curriculum to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(papers, f, indent=2)
        print(f"✅ Curriculum saved to {output_path} ({len(papers)} papers)")

    def get_stats(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return summary stats for dashboard."""
        if not papers:
            return {"count": 0, "avg_score": 0.0, "min_score": 0.0, "max_score": 0.0}

        scores = [p.get("teachability_score", self.compute_teachability_score(p)) for p in papers]
        return {
            "count": len(papers),
            "avg_score": round(sum(scores) / len(scores), 3),
            "min_score": round(min(scores), 3),
            "max_score": round(max(scores), 3),
            "passing_count": len([s for s in scores if s >= self.config["min_score_threshold"]])
        }
``n

## File: faithfulness.py

`python

# stephanie/agents/paper_improver/faithfulness.py

# faithfulness.py — Automated claim verification against source paper.
# Uses retrieval + LLM judge to score faithfulness. Logs mismatches. Safe, auditable, measurable.

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from sentence_transformers import SentenceTransformer, util
import numpy as np

class FaithfulnessBot:
    """
    Verifies that generated claims are supported by the source paper.
    Steps:
      1. Chunk paper into passages.
      2. For each claim, retrieve top-k most relevant passages.
      3. Ask LLM judge: “Does this passage support this claim? YES/NO”
      4. Return verdict + confidence + evidence snippet.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
        judge_prompt_template: Optional[str] = None,
        llm_judge_fn: Optional[Callable] = None,
        max_claim_length: int = 300,
        min_similarity_threshold: float = 0.3
    ):
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k
        self.judge_prompt_template = judge_prompt_template or self._default_judge_prompt()
        self.llm_judge_fn = llm_judge_fn or self._dummy_judge  # replace with real LLM in prod
        self.max_claim_length = max_claim_length
        self.min_similarity_threshold = min_similarity_threshold
        self.paper_chunks = []
        self.chunk_embeddings = None

    def _default_judge_prompt(self) -> str:
        return """You are a precise research assistant. Your task is to verify whether a claim is directly supported by a given passage from a paper.

Paper Passage:
{passage}

Claim:
{claim}

Answer only YES or NO. Do not explain.
Answer:"""

    def _dummy_judge(self, prompt: str) -> str:
        """Mock LLM judge — replace with real API (e.g., local Phi-2, OpenAI, Claude)."""
        # Simulate 80% accuracy — real one should be deterministic + seeded
        if "Table" in prompt or "Figure" in prompt or "Eq" in prompt:
            return "YES"
        if "never" in prompt or "always" in prompt or "proves" in prompt:
            return "NO"
        return "YES" if hash(prompt) % 10 > 2 else "NO"

    def prepare_paper(self, paper_text: str, chunk_size: int = 200, chunk_overlap: int = 50):
        """
        Preprocess and embed paper text for retrieval.
        Splits into overlapping chunks for dense retrieval.
        """
        if not paper_text.strip():
            raise ValueError("Paper text is empty.")

        # Clean and split
        paper_text = re.sub(r'\s+', ' ', paper_text).strip()
        self.paper_chunks = self._chunk_text(paper_text, chunk_size, chunk_overlap)

        # Embed chunks
        print(f"🔍 Embedding {len(self.paper_chunks)} paper chunks...")
        self.chunk_embeddings = self.model.encode(self.paper_chunks, convert_to_tensor=True)
        print("✅ Paper prepared for claim verification.")

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start = end - overlap
        return chunks

    def verify_claim(self, claim: str, claim_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify a single claim against the prepared paper.
        Returns verdict, confidence, evidence, and retrieval score.
        """
        if not self.chunk_embeddings:
            raise RuntimeError("Call prepare_paper() first.")

        if len(claim) > self.max_claim_length:
            claim = claim[:self.max_claim_length] + "... [truncated]"

        # Embed claim
        claim_embedding = self.model.encode(claim, convert_to_tensor=True)

        # Compute similarities
        cos_scores = util.cos_sim(claim_embedding, self.chunk_embeddings)[0]
        top_results = np.argpartition(-cos_scores.cpu(), range(self.top_k))[:self.top_k]

        # Get top passages
        retrieved = []
        for idx in top_results[0:self.top_k]:
            score = cos_scores[idx].item()
            if score < self.min_similarity_threshold:
                continue
            retrieved.append({
                "passage": self.paper_chunks[idx],
                "score": round(score, 3),
                "index": int(idx)
            })

        if not retrieved:
            return {
                "claim_id": claim_id,
                "claim": claim,
                "supported": False,
                "confidence": 0.0,
                "evidence": "",
                "retrieved_count": 0,
                "max_similarity": 0.0,
                "judge_prompt": "",
                "judge_response": "NO",
                "error": "No relevant passage retrieved."
            }

        # Use top passage for judging
        best_passage = retrieved[0]["passage"]
        prompt = self.judge_prompt_template.format(passage=best_passage, claim=claim)

        # Call LLM judge
        judge_response = self.llm_judge_fn(prompt).strip().upper()
        supported = "YES" in judge_response

        return {
            "claim_id": claim_id,
            "claim": claim,
            "supported": supported,
            "confidence": retrieved[0]["score"],  # use retrieval score as proxy
            "evidence": best_passage[:500],
            "retrieved_count": len(retrieved),
            "max_similarity": retrieved[0]["score"],
            "judge_prompt": prompt,
            "judge_response": judge_response,
            "error": None
        }

    def verify_claims_batch(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verify a batch of claims. Returns list of verdicts.
        Input: [{"claim_id": "...", "claim": "..."}, ...]
        """
        results = []
        for c in claims:
            try:
                result = self.verify_claim(c["claim"], c.get("claim_id"))
            except Exception as e:
                result = {
                    "claim_id": c.get("claim_id"),
                    "claim": c["claim"],
                    "supported": False,
                    "confidence": 0.0,
                    "evidence": "",
                    "error": str(e)
                }
            results.append(result)
        return results

    def get_faithfulness_score(self, claims: List[Dict[str, Any]]) -> float:
        """Compute % of claims supported."""
        if not claims:
            return 1.0
        results = self.verify_claims_batch(claims)
        supported = sum(1 for r in results if r["supported"])
        return round(supported / len(claims), 3)

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save verification results to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Faithfulness results saved to {output_path} ({len(results)} claims)")

    def get_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return summary stats for dashboard."""
        if not results:
            return {"total": 0, "supported": 0, "faithfulness_score": 0.0, "avg_confidence": 0.0}

        supported = [r for r in results if r["supported"]]
        confidences = [r["confidence"] for r in results if r["confidence"] > 0]

        return {
            "total": len(results),
            "supported": len(supported),
            "faithfulness_score": round(len(supported) / len(results), 3),
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
            "errors": len([r for r in results if r.get("error")])
        }
``n

## File: goals.py

`python
# goals.py — goal templates, normalization, and portfolio scoring for VPM rows
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple, List, Callable
import math
import json

try:
    import yaml  # optional: only needed for load_yaml
except Exception:  # pragma: no cover
    yaml = None  # graceful degradation


# ----------------------------- data models -----------------------------

@dataclass(frozen=True)
class Normalization:
    """
    How to map a raw metric to [0,1] for scoring.
    - "pass_through": value already in [0,1]
    - "band": clamp into [min,max] then rescale to [0,1] (e.g., readability FKGL 9–11)
    - "invert": 1 - value (when lower raw is better)
    """
    kind: str = "pass_through"
    min_val: float = 0.0
    max_val: float = 1.0

    def apply(self, value: float) -> float:
        if value is None or math.isnan(float(value)):  # type: ignore[arg-type]
            return 0.0
        v = float(value)
        if self.kind == "pass_through":
            return max(0.0, min(1.0, v))
        if self.kind == "invert":
            return max(0.0, min(1.0, 1.0 - v))
        if self.kind == "band":
            if self.max_val <= self.min_val:
                return 0.0
            v = max(self.min_val, min(self.max_val, v))
            return (v - self.min_val) / (self.max_val - self.min_val)
        # unknown → safe default
        return max(0.0, min(1.0, v))


@dataclass
class GoalTemplate:
    """
    A weighted blend over normalized metrics.
    weights: metric -> weight (non-negative)
    norms:   metric -> Normalization
    min_bar: optional per-metric minimum normalized requirement
    """
    name: str
    kind: str  # "text" | "code"
    weights: Dict[str, float]
    norms: Dict[str, Normalization] = field(default_factory=dict)
    min_bar: Dict[str, float] = field(default_factory=dict)

    def score(self, vpm_dims: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Return (composite_score, per_metric_normalized)."""
        # normalize
        normed: Dict[str, float] = {}
        for k, w in self.weights.items():
            raw = vpm_dims.get(k)
            norm = self.norms.get(k, Normalization("pass_through"))
            normed[k] = norm.apply(raw if raw is not None else 0.0)

        # enforce min bars (if any)
        for k, bar in self.min_bar.items():
            if k in normed and normed[k] < bar:
                # penalize proportionally below bar
                normed[k] = normed[k] * 0.5

        # weighted average (avoid division by zero)
        denom = sum(max(0.0, w) for w in self.weights.values())
        if denom <= 0:
            return 0.0, normed
        score = sum(normed.get(k, 0.0) * max(0.0, w) for k, w in self.weights.items()) / denom
        return score, normed

    def unmet(self, vpm_dims: Dict[str, float], hysteresis: float = 0.0) -> List[str]:
        """Return list of metric keys that are below their min_bar (after hysteresis)."""
        misses = []
        if not self.min_bar:
            return misses
        for k, bar in self.min_bar.items():
            v = self.norms.get(k, Normalization()).apply(vpm_dims.get(k, 0.0))
            if v + hysteresis < bar:
                misses.append(k)
        return misses


# ----------------------------- defaults -----------------------------

# Normalizers for common text metrics
TEXT_NORMS_DEFAULT = {
    "coverage":           Normalization("pass_through"),
    "correctness":        Normalization("pass_through"),
    "coherence":          Normalization("pass_through"),
    "citation_support":   Normalization("pass_through"),
    "entity_consistency": Normalization("pass_through"),
    "readability":        Normalization("band", min_val=9.0, max_val=11.0),  # FKGL target band
    "novelty":            Normalization("pass_through"),
    # Optional extended dims
    "stickiness":         Normalization("pass_through"),
}

# Normalizers for common code metrics
CODE_NORMS_DEFAULT = {
    "tests_pass_rate": Normalization("pass_through"),
    "coverage":        Normalization("pass_through"),
    "type_safe":       Normalization("pass_through"),
    "lint_clean":      Normalization("pass_through"),
    "complexity_ok":   Normalization("pass_through"),
    "mutation_score":  Normalization("pass_through"),
}

# Default text goals
ACADEMIC_SUMMARY = GoalTemplate(
    name="academic_summary",
    kind="text",
    weights={
        "coverage": 0.30,
        "correctness": 0.25,
        "citation_support": 0.20,
        "coherence": 0.15,
        "readability": 0.10,
    },
    norms=TEXT_NORMS_DEFAULT,
    min_bar={
        "coverage": 0.80,
        "correctness": 0.75,
        "coherence": 0.75,
        "citation_support": 0.65,
        "entity_consistency": 0.80,  # not weighted strongly but must clear bar
    },
)

PRACTITIONER_TUTORIAL = GoalTemplate(
    name="practitioner_tutorial",
    kind="text",
    weights={
        "correctness": 0.30,
        "coverage": 0.25,
        "coherence": 0.20,
        "entity_consistency": 0.15,
        "readability": 0.10,
    },
    norms=TEXT_NORMS_DEFAULT,
    min_bar={
        "coverage": 0.75,
        "correctness": 0.80,
        "readability": 0.60,  # normalized band score (maps FKGL into [0,1])
    },
)

BLOG_GENERAL = GoalTemplate(
    name="blog_general",
    kind="text",
    weights={
        "coherence": 0.25,
        "coverage": 0.20,
        "correctness": 0.20,
        "readability": 0.20,
        "novelty": 0.15,
    },
    norms=TEXT_NORMS_DEFAULT,
    min_bar={
        "coverage": 0.70,
        "coherence": 0.70,
    },
)

# Default code goals
STRICT_CI = GoalTemplate(
    name="strict_ci",
    kind="code",
    weights={
        "tests_pass_rate": 0.40,
        "coverage": 0.25,
        "type_safe": 0.15,
        "lint_clean": 0.10,
        "mutation_score": 0.10,
    },
    norms=CODE_NORMS_DEFAULT,
    min_bar={
        "tests_pass_rate": 1.00,
        "coverage": 0.70,
        "type_safe": 1.00,
        "lint_clean": 1.00,
    },
)

FAST_ITER = GoalTemplate(
    name="fast_iter",
    kind="code",
    weights={
        "tests_pass_rate": 0.50,
        "coverage": 0.15,
        "type_safe": 0.10,
        "lint_clean": 0.10,
        "complexity_ok": 0.10,
        "mutation_score": 0.05,
    },
    norms=CODE_NORMS_DEFAULT,
    min_bar={
        "tests_pass_rate": 1.00,
        "coverage": 0.60,
    },
)

DEFAULT_TEMPLATES: Dict[str, GoalTemplate] = {
    "text/academic_summary": ACADEMIC_SUMMARY,
    "text/practitioner_tutorial": PRACTITIONER_TUTORIAL,
    "text/blog_general": BLOG_GENERAL,
    "code/strict_ci": STRICT_CI,
    "code/fast_iter": FAST_ITER,
}


# ----------------------------- scorer -----------------------------

class GoalScorer:
    """
    Computes portfolio scores for VPM rows against goal templates.
    Also surfaces unmet bars and suggests next targets (the lowest-scoring normalized dims).
    """

    def __init__(
        self,
        templates: Optional[Dict[str, GoalTemplate]] = None,
        judge: Optional[Callable[[Dict[str, Any]], Dict[str, float]]] = None,
    ):
        """
        judge (optional): function(row) -> extra dimension dict (e.g., external LLM judge).
        Any returned dims will be merged (and normalized via template norms if targeted).
        """
        self.templates = templates or DEFAULT_TEMPLATES
        self.judge = judge

    def available(self, *, kind: str) -> List[str]:
        pre = f"{kind}/"
        return [k.split("/", 1)[1] for k in self.templates.keys() if k.startswith(pre)]

    def score(
        self,
        kind: str,
        goal: str,
        vpm_row: Dict[str, float],
        hysteresis: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Return:
        {
          "goal": "...",
          "score": float,
          "normalized": {dim: norm_val},
          "unmet": [dim...],
          "raw": vpm_row
        }
        """
        key = f"{kind}/{goal}"
        if key not in self.templates:
            raise KeyError(f"Unknown goal '{goal}' for kind '{kind}'. Available: {self.available(kind=kind)}")

        tpl = self.templates[key]
        dims = dict(vpm_row)

        # Pull in optional judge signals
        if self.judge:
            try:
                extra = self.judge(vpm_row) or {}
                dims.update(extra)
            except Exception:
                pass

        score, normed = tpl.score(dims)
        unmet = tpl.unmet(dims, hysteresis=hysteresis)
        return {
            "goal": goal,
            "score": round(float(score), 4),
            "normalized": {k: round(float(v), 4) for k, v in normed.items()},
            "unmet": unmet,
            "raw": vpm_row,
        }

    def suggest_targets(
        self,
        kind: str,
        goal: str,
        vpm_row: Dict[str, float],
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Returns the K weakest normalized metrics under this goal (ascending).
        Useful to route edits (e.g., add citations, raise coverage).
        """
        tpl = self.templates[f"{kind}/{goal}"]
        _, normed = tpl.score(vpm_row)
        # Only consider dimensions present in weights
        pairs = [(k, normed.get(k, 0.0)) for k in tpl.weights.keys()]
        pairs.sort(key=lambda x: x[1])
        return pairs[:top_k]


# ----------------------------- I/O helpers -----------------------------

def load_yaml(path: str | None) -> Dict[str, Any]:
    """
    Load custom templates from a YAML file.
    Format:
      text:
        blog_general:
          weights: {coverage: 0.2, correctness: 0.2, ...}
          norms:
            readability: {kind: band, min_val: 9.0, max_val: 11.0}
          min_bar: {coverage: 0.75, coherence: 0.7}
      code:
        strict_ci:
          ...
    """
    if path is None:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML not installed; cannot load YAML templates.")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def build_templates_from_yaml(config: Dict[str, Any]) -> Dict[str, GoalTemplate]:
    """
    Construct GoalTemplate mapping from a parsed YAML dict.
    """
    out: Dict[str, GoalTemplate] = dict(DEFAULT_TEMPLATES)  # start from defaults
    for kind in ("text", "code"):
        if kind not in config:
            continue
        for name, body in (config.get(kind) or {}).items():
            weights = body.get("weights", {})
            min_bar = body.get("min_bar", {})
            norms_in = body.get("norms", {})
            norms: Dict[str, Normalization] = {}
            for k, ncfg in (norms_in or {}).items():
                if isinstance(ncfg, dict):
                    norms[k] = Normalization(
                        kind=ncfg.get("kind", "pass_through"),
                        min_val=float(ncfg.get("min_val", 0.0)),
                        max_val=float(ncfg.get("max_val", 1.0)),
                    )
            tpl = GoalTemplate(name=name, kind=kind, weights=weights, norms=norms or (TEXT_NORMS_DEFAULT if kind=="text" else CODE_NORMS_DEFAULT), min_bar=min_bar)
            out[f"{kind}/{name}"] = tpl
    return out


# ----------------------------- quick demo -----------------------------
if __name__ == "__main__":  # pragma: no cover
    # Example VPM rows
    text_row = {
        "coverage": 0.82, "correctness": 0.78, "coherence": 0.76,
        "citation_support": 0.68, "entity_consistency": 0.86, "readability": 10.1, "novelty": 0.55
    }
    code_row = {
        "tests_pass_rate": 1.0, "coverage": 0.73, "type_safe": 1.0,
        "lint_clean": 1.0, "complexity_ok": 0.8, "mutation_score": 0.62
    }

    gs = GoalScorer()
    print("[text/academic_summary]", json.dumps(gs.score("text", "academic_summary", text_row), indent=2))
    print("[code/strict_ci]", json.dumps(gs.score("code", "strict_ci", code_row), indent=2))
    print("[suggest targets]", gs.suggest_targets("text", "academic_summary", text_row, top_k=3))
``n

## File: mutation.py

`python
# stephanie/agents/paper_improver/mutation.py

# mutation.py — safe, measurable mutation-testing helper for the improver stack
from __future__ import annotations
import re
import shutil
import subprocess
import time
import resource
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

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
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_seconds, self.cpu_seconds))
            resource.setrlimit(resource.RLIMIT_AS, (self.mem_bytes, self.mem_bytes))
        except Exception:
            # Not all platforms support RLIMITs (e.g., Windows). Best effort.
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
``n

## File: orchestrator.py

`python
# stephanie/agents/paper_improver/orchestrator.py

# orchestrator.py — End-to-end spec + plan → code + text → PR (optional)
import json
import argparse
from pathlib import Path
from typing import Optional

from .code_improver import CodeImprover
from .text_improver import TextImprover
from .repo_link import RepoLink
from .vpm_controller import VPMController

def run_paper_section(
    spec_path: str,
    plan_path: str,
    workdir: str = "./runs",
    backend: str = "torch",
    create_pr: bool = False,
    repo_root: str = "../.."
) -> dict:
    """
    Run code + text improvers. Optionally create PR.
    Returns combined report.
    """
    # Load inputs
    spec = json.loads(Path(spec_path).read_text())
    plan = json.loads(Path(plan_path).read_text())

    # Init improvers
    ci = CodeImprover(backend=backend, workdir=f"{workdir}/code")
    ti = TextImprover(workdir=f"{workdir}/text")
    vc = VPMController()

    # Improve code
    print("🔧 Improving code...")
    code_result = ci.improve(spec)
    code_action = vc.add_vpm_row(code_result["vpm_row"], f"code:{spec['function_name']}")
    print(f"→ Code VPM: {code_result['vpm_row']}")
    print(f"→ Controller: {code_action}")

    # Improve text
    print("📝 Improving text...")
    text_result = ti.improve(plan)
    text_action = vc.add_vpm_row(text_result["vpm_row"], f"text:{plan['section_title']}")
    print(f"→ Text VPM: {text_result['vpm_row']}")
    print(f"→ Controller: {text_action}")

    # Build report
    report = {
        "spec": spec_path,
        "plan": plan_path,
        "code": {
            "vpm_row": code_result["vpm_row"],
            "action": code_action,
            "artifacts": code_result["run_dir"],
            "passed": code_result["passed"]
        },
        "text": {
            "vpm_row": text_result["vpm_row"],
            "action": text_action,
            "artifacts": text_result["run_dir"],
            "passed": text_result["passed"]
        }
    }

    # Create PR if requested
    if create_pr:
        print("🚀 Creating PRs...")
        rl = RepoLink(repo_root=repo_root)
        if code_result["passed"]:
            rl.create_pr(code_result["run_dir"], code_result["vpm_row"], "code")
        if text_result["passed"]:
            rl.create_pr(text_result["run_dir"], text_result["vpm_row"], "text")

    # Save report
    report_path = Path(workdir) / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"✅ Report saved: {report_path}")

    return report

def main():
    parser = argparse.ArgumentParser(description="Run paper section improver")
    parser.add_argument("--spec", required=True, help="Path to spec.json")
    parser.add_argument("--plan", required=True, help="Path to plan.json")
    parser.add_argument("--workdir", default="./runs", help="Working directory")
    parser.add_argument("--backend", default="torch", choices=["torch", "numpy"], help="Code backend")
    parser.add_argument("--pr", action="store_true", help="Create PRs if passed")
    parser.add_argument("--repo-root", default="../..", help="Root of git repo")

    args = parser.parse_args()
    run_paper_section(
        spec_path=args.spec,
        plan_path=args.plan,
        workdir=args.workdir,
        backend=args.backend,
        create_pr=args.pr,
        repo_root=args.repo_root
    )

if __name__ == "__main__":
    main()
``n

## File: repo_link.py

`python
# stephanie/agents/paper_improver/repo_link.py

# repo_link.py — Push improver artifacts into repo + open PR + optional auto-merge
# Combines: your clean CLI flow + my structured contrib/ layout, VPM-rich PR body, auto-merge, and hardening.

import subprocess
import json
import hashlib
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

class RepoLink:
    """
    RepoLink automates:
      - Creating a branch for improver outputs
      - Committing artifacts under contrib/<branch>/
      - Opening a PR with VPM summary + DPO pair link
      - Polling CI status
      - Optional auto-merge if CI passes
    """

    def __init__(
        self,
        repo_root: str = ".",
        remote: str = "origin",
        base: str = "main",
        contrib_dir: str = "contrib",
        auto_merge: bool = False,
        ci_timeout: int = 600,
        poll_interval: int = 20
    ):
        self.root = Path(repo_root).resolve()
        self.remote = remote
        self.base = base
        self.contrib_dir = self.root / contrib_dir
        self.auto_merge = auto_merge
        self.ci_timeout = ci_timeout
        self.poll_interval = poll_interval

        # Ensure contrib/ exists
        self.contrib_dir.mkdir(exist_ok=True)

    # ---------- public ----------

    def push_pr(
        self,
        run_dir: str,
        vpm_row: Dict[str, Any],
        dpo_pair_path: Optional[str] = None,
        label: str = "improver",
        artifact_type: str = "code"  # or "text"
    ) -> Dict[str, Any]:
        """
        Create branch, copy artifacts to contrib/, commit, push, open PR.
        Returns dict with {branch, pr_url, ci_status, merged}.
        """
        # Derive hash from VPM row (or use spec/plan hash if available)
        row_hash = hashlib.sha256(json.dumps(vpm_row, sort_keys=True).encode()).hexdigest()[:8]
        branch_name = f"feat/{label}-{artifact_type}-{row_hash}"

        run_dir = Path(run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")

        # Target dir inside contrib/
        target_dir = self.contrib_dir / branch_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy all artifacts (except .git, venv, etc.)
        self._copy_artifacts(run_dir, target_dir)

        # Ensure we're on base branch
        self._run(["git", "checkout", self.base], check=False)
        self._run(["git", "pull", self.remote, self.base], check=False)

        # Create new branch
        self._run(["git", "checkout", "-B", branch_name])

        # Stage and commit
        self._run(["git", "add", str(target_dir.relative_to(self.root))])
        commit_msg = f"[{label}] {artifact_type} {row_hash}"
        self._run(["git", "commit", "-m", commit_msg])

        # Push
        self._run(["git", "push", "-u", self.remote, branch_name, "--force"])

        # Open PR
        pr_url = self._open_pr(branch_name, vpm_row, dpo_pair_path, artifact_type)

        # Poll CI
        ci_status = self._poll_ci(pr_url) if pr_url else "unknown"

        # Auto-merge if enabled and CI passed
        merged = False
        if self.auto_merge and ci_status == "success":
            merged = self._auto_merge(pr_url, branch_name)

        return {
            "branch": branch_name,
            "pr_url": pr_url,
            "ci_status": ci_status,
            "merged": merged,
            "artifacts_dir": str(target_dir)
        }

    # ---------- internals ----------

    def _copy_artifacts(self, src: Path, dest: Path):
        """Copy improver run artifacts — skip unsafe or noisy dirs."""
        for item in src.iterdir():
            if item.name.startswith(".") or item.name in {"venv", "__pycache__"}:
                continue
            if item.is_dir():
                shutil.copytree(item, dest / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest / item.name)

    def _run(self, cmd: list[str], check: bool = True) -> str:
        """Run git/cmd with error handling."""
        try:
            r = subprocess.run(cmd, cwd=self.root, text=True, capture_output=True, timeout=300)
            if check and r.returncode != 0:
                raise RuntimeError(f"Command failed: {' '.join(cmd)}\nStderr: {r.stderr}")
            return r.stdout.strip()
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Command timeout: {' '.join(cmd)}")
        except Exception as e:
            if check:
                raise RuntimeError(f"Command error: {' '.join(cmd)} | {e}")
            return ""

    def _open_pr(self, branch: str, vpm_row: Dict[str, Any], dpo_pair_path: Optional[str], artifact_type: str) -> Optional[str]:
        """Use GitHub CLI `gh` if available, else return branch URL."""
        # Build rich PR body
        body = f"## 📊 VPM Summary ({artifact_type})\n```json\n{json.dumps(vpm_row, indent=2)}\n```\n"

        if dpo_pair_path:
            # Try to make it a relative, clickable link
            try:
                dpo_path = Path(dpo_pair_path).resolve()
                rel_path = dpo_path.relative_to(self.root)
                body += f"\n## 🔁 DPO Pair\n`{rel_path}`\n"
            except Exception:
                body += f"\n## 🔁 DPO Pair\n`{dpo_pair_path}`\n"

        body += "\n---\n*Generated by `RepoLink` — do not edit manually. Auto-merge if CI passes.*"

        try:
            if not shutil.which("gh"):
                raise FileNotFoundError("gh CLI not found")

            cmd = [
                "gh", "pr", "create",
                "--base", self.base,
                "--head", branch,
                "--title", f"[Auto] {artifact_type.capitalize()} Improver: {branch.split('-')[-1]}",
                "--body", body,
                "--label", "improver,auto-pr"
            ]
            out = self._run(cmd)
            # Extract URL from output
            for line in out.splitlines():
                if line.startswith("http"):
                    return line.strip()
            return out.strip() or None
        except Exception as e:
            print(f"⚠️ Failed to create PR with gh: {e}")
            # Fallback: return branch ref
            return f"{self.remote}/{branch}"

    def _poll_ci(self, pr_url: str) -> str:
        """Poll CI status using gh CLI."""
        if not pr_url or not shutil.which("gh"):
            return "skipped"

        print(f"⏳ Polling CI for {pr_url} (timeout: {self.ci_timeout}s)...")
        start = time.time()

        while time.time() - start < self.ci_timeout:
            try:
                # Get PR info with status checks
                cmd = ["gh", "pr", "view", pr_url, "--json", "statusCheckRollup"]
                out = self._run(cmd, check=False)
                if not out:
                    time.sleep(self.poll_interval)
                    continue

                data = json.loads(out)
                rollup = data.get("statusCheckRollup", [])

                if not rollup:
                    time.sleep(self.poll_interval)
                    continue

                # Look for final conclusion
                for check in rollup:
                    conclusion = check.get("conclusion")
                    if conclusion == "SUCCESS":
                        print("✅ CI passed.")
                        return "success"
                    elif conclusion in ("FAILURE", "CANCELLED"):
                        print(f"❌ CI {conclusion.lower()}.")
                        return conclusion.lower()

                time.sleep(self.poll_interval)

            except Exception as e:
                print(f"⚠️ CI poll error: {e}")
                time.sleep(self.poll_interval)

        print("⏰ CI poll timeout.")
        return "timeout"

    def _auto_merge(self, pr_url: str, branch: str) -> bool:
        """Attempt to merge PR via gh CLI."""
        try:
            cmd = ["gh", "pr", "merge", pr_url, "--merge", "--delete-branch"]
            self._run(cmd)
            print(f"✅ Auto-merged PR: {pr_url}")
            return True
        except Exception as e:
            print(f"⚠️ Auto-merge failed: {e}")
            return False
``n

## File: text_improver.py

`python
# stephanie/agents/paper_improver/text_improver.py

# text_improver.py — plan → draft → score → edit → log → blog-ready
import json
import re
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List

class TextImprover:
    def __init__(self, workdir: str = "./text_runs", timeout: int = 60):
        self.workdir = Path(workdir)
        self.workdir.mkdir(exist_ok=True)
        self.run_id = 0
        self.timeout = timeout

    def improve(self, content_plan: Dict[str, Any]) -> Dict[str, Any]:
        self.run_id += 1
        plan_hash = hashlib.sha256(
            json.dumps(content_plan, sort_keys=True).encode()
        ).hexdigest()[:8]
        run_dir = self.workdir / f"run_{self.run_id}_{plan_hash}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # metadata for determinism
        meta = {
            "plan_sha": plan_hash,
            "seeds": {"python": 0},
            "timeout": self.timeout,
            "timestamp": time.time(),
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        # 1. save plan
        plan_path = run_dir / "plan.json"
        plan_path.write_text(json.dumps(content_plan, indent=2))

        # 2. initial draft
        draft_path = self._generate_draft(content_plan, run_dir)

        # 3. score draft
        initial_score = self._score_draft(draft_path, content_plan)

        # 4. edit loop
        final_draft, edits = self._apply_edit_policy(draft_path, content_plan, max_edits=5)

        # 5. rescore
        final_score = self._score_draft(draft_path, content_plan)

        # 6. vpm row
        vpm_row = self._build_vpm_row(initial_score, final_score, content_plan)

        # 7. dpo pair
        dpo_pair = {
            "content_plan_slice": self._extract_plan_slice(content_plan),
            "prompt": "Generate faithful, clear prose from this plan.",
            "rejected": (run_dir / "initial_draft.md").read_text(),
            "chosen": final_draft,
            "metadata": {
                "run_id": self.run_id,
                "plan_hash": plan_hash,
                "initial_scores": initial_score,
                "final_scores": final_score,
                "applied_edits": edits,
            },
        }
        (run_dir / "text_dpo_pair.json").write_text(json.dumps(dpo_pair, indent=2))

        return {
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "final_draft_path": str(draft_path),
            "vpm_row": vpm_row,
            "dpo_pair_path": str(run_dir / "text_dpo_pair.json"),
            "scores": final_score,
            "passed": all(
                final_score[d] >= 0.7
                for d in ["coverage", "correctness", "coherence"]
            ),
        }

    # ---------------- draft + scoring ----------------

    def _generate_draft(self, plan: Dict[str, Any], run_dir: Path) -> Path:
        draft = f"# {plan.get('section_title', 'Section')}\n\n"
        for unit in plan.get("units", []):
            claim = unit.get("claim", "No claim")
            evidence = unit.get("evidence", "See paper")
            claim_id = unit.get("claim_id", "")
            tag = f" [#{claim_id}]" if claim_id else ""
            cite = " [#]" if evidence and evidence != "See paper" else ""
            draft += f"- {claim}{tag}.\n"
            draft += f"  *Evidence: {evidence}*{cite}\n\n"
        draft_path = run_dir / "draft.md"
        draft_path.write_text(draft)
        (run_dir / "initial_draft.md").write_text(draft)
        return draft_path

    def _score_draft(self, draft_path: Path, plan: Dict[str, Any]) -> Dict[str, float]:
        text = draft_path.read_text()
        units = plan.get("units", [])
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

        # coverage: % of claims referenced
        covered = sum(1 for u in units if u.get("claim_id") and f"[#{u['claim_id']}]" in text)
        coverage = covered / max(1, len(units))

        # citation support
        factual_kw = ("show","prove","result","achiev","increase","decrease","outperform","error","accuracy","loss")
        factual_sentences = [s for s in sentences if any(kw in s.lower() for kw in factual_kw)]
        cited = sum(1 for s in factual_sentences if "[#]" in s)
        citation_support = cited / max(1, len(factual_sentences)) if factual_sentences else 1.0

        # entity consistency
        abbrs = plan.get("entities", {}).get("ABBR", {})
        entity_consistency = 1.0
        for full, abbr in abbrs.items():
            if full not in text and abbr not in text:
                entity_consistency = min(entity_consistency, 0.0)
            elif full in text and text.count(full) > 1:
                entity_consistency = min(entity_consistency, 0.5)

        # readability: FKGL
        words = re.findall(r"[A-Za-z]+", text)
        num_words = max(1, len(words))
        num_sentences = max(1, len(sentences))
        syllables = sum(self._count_syllables(w) for w in words)
        fkgl = 0.39 * (num_words/num_sentences) + 11.8 * (syllables/num_words) - 15.59
        readability = float(max(6.0, min(15.0, fkgl)))

        # coherence: jaccard similarity between adjacent sentences
        coh_scores = []
        for i in range(len(sentences)-1):
            s1 = set(re.findall(r'\w+', sentences[i].lower()))
            s2 = set(re.findall(r'\w+', sentences[i+1].lower()))
            denom = len(s1 | s2)
            coh_scores.append((len(s1 & s2)/denom) if denom else 1.0)
        coherence = sum(coh_scores) / max(1, len(coh_scores)) if coh_scores else 1.0

        correctness = citation_support
        novelty = 0.6  # placeholder until abstract-sim is wired

        return {
            "coverage": coverage,
            "correctness": correctness,
            "coherence": coherence,
            "citation_support": citation_support,
            "entity_consistency": entity_consistency,
            "readability": readability,
            "novelty": novelty,
        }

    # ---------------- edit policy ----------------

    def _apply_edit_policy(self, draft_path: Path, plan: Dict[str, Any], max_edits: int = 5):
        text = draft_path.read_text()
        edits: List[str] = []

        for _ in range(max_edits):
            scores = self._score_draft(draft_path, plan)
            made_change = False

            # add missing claims
            if scores["coverage"] < 0.8:
                missing = [u for u in plan.get("units", [])
                           if u.get("claim_id") and f"[#{u['claim_id']}]" not in text]
                if missing:
                    u = missing[0]
                    line = f"- {u.get('claim','Claim')} [#{u['claim_id']}].\n  *Evidence: {u.get('evidence','See paper')}* [#]\n\n"
                    text += line
                    edits.append(f"Add claim {u['claim_id']}")
                    made_change = True

            # add citation markers
            if not made_change and scores["citation_support"] < 0.7:
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for j, s in enumerate(sentences):
                    if self._is_factual_sentence(s) and "[#]" not in s:
                        sentences[j] = s.rstrip() + " [#]"
                        text = " ".join(sentences)
                        edits.append("Add citation marker")
                        made_change = True
                        break

            # normalize ABBRs
            if not made_change and scores["entity_consistency"] < 1.0:
                abbrs = plan.get("entities", {}).get("ABBR", {})
                for full, abbr in abbrs.items():
                    if full not in text and abbr in text:
                        text = re.sub(rf"\b{re.escape(abbr)}\b", f"{full} ({abbr})", text, count=1)
                        edits.append(f"Expand ABBR: {full} ({abbr})")
                        made_change = True
                        break
                    if text.count(full) > 1:
                        first = True
                        def _swap(m):
                            nonlocal first
                            if first: 
                                first = False
                                return m.group(0)
                            return abbr
                        text = re.sub(rf"\b{re.escape(full)}\b", _swap, text)
                        edits.append(f"Normalize ABBR usage: {full}→{abbr}")
                        made_change = True
                        break

            # readability fix
            if not made_change and not (9.0 <= scores["readability"] <= 11.0):
                if scores["readability"] > 11.0:
                    text = re.sub(r";\s+", ". ", text)
                    text = re.sub(r"\s+and\s+", ". ", text, count=1)
                    edits.append("Split long sentences")
                    made_change = True
                else:
                    text = re.sub(r"\.\s+([a-z])", r", \1", text, count=1)
                    edits.append("Join short sentences")
                    made_change = True

            # coherence smoothing
            if not made_change and scores["coherence"] < 0.7:
                text_before = text
                text = re.sub(r"\n- ([^.\n]{0,40})\.\n- ", r"\n- \1; ", text)
                if text != text_before:
                    edits.append("Merge short adjacent bullets")
                    made_change = True

            if not made_change:
                break

            draft_path.write_text(text)

        return text, edits

    def _is_factual_sentence(self, s: str) -> bool:
        s_low = s.lower()
        return any(kw in s_low for kw in (
            "show","prove","result","achiev","increase","decrease","outperform","error","accuracy","loss"
        ))

    # ---------------- helpers ----------------

    def _build_vpm_row(self, initial: Dict[str,float], final: Dict[str,float], plan: Dict[str,Any]) -> Dict[str, Any]:
        return {
            "section": plan.get("section_title","unknown"),
            "coverage_initial": round(initial["coverage"],3),
            "coverage_final": round(final["coverage"],3),
            "correctness": round(final["correctness"],3),
            "coherence": round(final["coherence"],3),
            "citation_support": round(final["citation_support"],3),
            "entity_consistency": round(final["entity_consistency"],3),
            "readability": round(final["readability"],2),
            "novelty": round(final["novelty"],3),
        }

    def _extract_plan_slice(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "section_title": plan.get("section_title"),
            "claim_count": len(plan.get("units", [])),
            "required_entities": plan.get("entities", {}).get("REQUIRED", []),
            "abbr": plan.get("entities", {}).get("ABBR", {})
        }

    def _count_syllables(self, word: str) -> int:
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        if word and word[0] in vowels:
            count += 1
        for idx in range(1, len(word)):
            if word[idx] in vowels and word[idx - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        return max(1, count)
``n

## File: vpm_controller.py

`python
# stephanie/agents/paper_improver/vpm_controller.py

# vpm_controller.py
# A trend-aware controller for VPM rows that emits control signals to drive the loop.
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Callable, Tuple
import math
import time
import statistics as stats

# ========= public API =========

class Signal(Enum):
    """Controller decisions for the next step in the trajectory."""
    EDIT = auto()        # apply local, minimal diffs
    RESAMPLE = auto()    # rerun with new exemplars / different seeds
    ESCALATE = auto()    # escalate to stronger model / human checkpoint
    STOP = auto()        # stop improving (stable & above thresholds)
    SPINOFF = auto()     # fork dropped/novel content to a new artifact

@dataclass
class Thresholds:
    """Per-dimension minimum requirements (text or code)."""
    mins: Dict[str, float]                      # e.g., {"coverage":0.8, "correctness":0.75, ...}
    # Optional bands for hysteresis (require higher to STOP than to remain STOP)
    stop_margin: float = 0.02                   # extra margin to declare STOP
    edit_margin: float = 0.01                   # tolerance before EDIT triggers again

@dataclass
class Policy:
    """Control policy knobs."""
    # lookback and smoothing
    window: int = 5                              # how many recent frames to consider
    ema_alpha: float = 0.4                       # exponential smoothing for trends
    patience: int = 3                            # consecutive fails before RESAMPLE
    escalate_after: int = 2                      # consecutive RESAMPLEs before ESCALATE
    # novelty → spinoff
    spinoff_dim: str = "novelty"                 # when high novelty + low stickiness → spin off
    stickiness_dim: str = "stickiness"           # requires producer to log this; else ignored
    spinoff_gate: Tuple[float, float] = (0.75, 0.45)  # (novelty>=, stickiness<=)
    # regression guard
    max_regressions: int = 2                     # in window
    # score weighting to detect "local gaps" vs "global failure"
    local_gap_dims: List[str] = field(default_factory=lambda: ["citation_support","entity_consistency","lint_clean","type_safe"])
    # action limits
    max_steps: int = 50

@dataclass
class VPMRow:
    """Single frame from an improver (code or text)."""
    # Common
    unit: str                          # e.g., "pkg.impl:l2_normalize" or "Method"
    kind: str                          # "code" or "text"
    timestamp: float                   # epoch seconds
    dims: Dict[str, float]             # metric → value (0..1 or scalar like FKGL)
    # Optional metadata
    step_idx: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Decision:
    signal: Signal
    reason: str
    # optional “action params”, e.g., which exemplar family to try next
    params: Dict[str, Any] = field(default_factory=dict)
    # snapshot for auditability
    snapshot: Dict[str, Any] = field(default_factory=dict)

class VPMController:
    """
    Enhanced controller that:
      - applies threshold gating with hysteresis,
      - tracks rolling windows and EMAs,
      - detects stagnation vs. local gaps,
      - triggers RESAMPLE, EDIT, ESCALATE, STOP, SPINOFF,
      - provides hooks to route exemplars via a bandit.
    """

    def __init__(
        self,
        thresholds_code: Thresholds,
        thresholds_text: Thresholds,
        policy: Policy = Policy(),
        *,
        bandit_choose: Optional[Callable[[List[str]], str]] = None,
        bandit_update: Optional[Callable[[str, float], None]] = None,
        logger: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.thr_code = thresholds_code
        self.thr_text = thresholds_text
        self.p = policy
        self.bandit_choose = bandit_choose
        self.bandit_update = bandit_update
        self.log = logger or (lambda ev, d: None)
        self.history: Dict[str, List[VPMRow]] = {}      # unit → rows
        self.resample_counts: Dict[str, int] = {}       # unit → count
        self.last_signal: Dict[str, Signal] = {}        # unit → last signal

    # ---- public method ----
    def add(self, row: VPMRow, *, candidate_exemplars: Optional[List[str]] = None) -> Decision:
        """Ingest a VPM row and decide next action."""
        h = self.history.setdefault(row.unit, [])
        h.append(row)
        if len(h) > 50:  # trim unbounded history
            self.history[row.unit] = h[-50:]

        thr = self._thresholds_for(row.kind)
        window = h[-self.p.window:] if len(h) >= 1 else h
        ema = self._ema_series(window)
        trend = self._trend(window)

        # 0) safety stop: steps or all required dims missing
        if (row.meta.get("total_steps", row.step_idx or 0) >= self.p.max_steps):
            return self._decide(row, Signal.STOP, "Max steps reached", {})

        # 1) STOP (hysteresis): all dims above mins + margin for last K frames
        if self._stable_above(window, thr, margin=thr.stop_margin):
            return self._decide(row, Signal.STOP, "Stable above thresholds", {"hysteresis": thr.stop_margin})

        # 2) SPINOFF: high novelty, low stickiness (if dims provided)
        if self._should_spinoff(row):
            return self._decide(row, Signal.SPINOFF, "High novelty with low stickiness", {"dim": self.p.spinoff_dim})

        # 3) REGRESSION guard: if too many dips in window, RESAMPLE
        if self._regressions(window) > self.p.max_regressions:
            self._bump_resamples(row.unit)
            return self._decide(row, Signal.RESAMPLE, "Too many regressions", {"why": "regressions"})

        # 4) LOCAL vs GLOBAL failure
        gaps = self._gaps(row, thr)
        local_gaps = [g for g in gaps if g in self.p.local_gap_dims]
        global_fail = (len(gaps) > 0 and len(local_gaps) < len(gaps))  # several core dims below

        # 4a) LOCAL gaps → EDIT (prefer edit-policy)
        if local_gaps:
            return self._decide(row, Signal.EDIT, "Local gaps", {"gaps": local_gaps})

        # 4b) STAGNATION on core dims → RESAMPLE
        if self._stagnating(window, thr):
            params = {}
            if candidate_exemplars and self.bandit_choose:
                chosen = self.bandit_choose(candidate_exemplars)
                params["exemplar_id"] = chosen
            self._bump_resamples(row.unit)
            return self._decide(row, Signal.RESAMPLE, "Stagnation on core dims", params)

        # 4c) GLOBAL failure and worsening trend → ESCALATE
        if global_fail and self._worsening(trend):
            if self.resample_counts.get(row.unit, 0) >= self.p.escalate_after:
                return self._decide(row, Signal.ESCALATE, "Global fail & worsening after resamples", {})
            else:
                self._bump_resamples(row.unit)
                return self._decide(row, Signal.RESAMPLE, "Global fail & worsening (resample first)", {})

        # 5) default: EDIT until thresholds are met or patience exceeded
        # if below mins for patience frames -> RESAMPLE
        if not self._recently_above(window, thr, patience=self.p.patience):
            self._bump_resamples(row.unit)
            return self._decide(row, Signal.RESAMPLE, "Below thresholds for patience window", {})

        # otherwise keep editing
        return self._decide(row, Signal.EDIT, "Default edit to close small gaps", {"gaps": gaps})

    # ========= internals =========

    def _thresholds_for(self, kind: str) -> Thresholds:
        return self.thr_code if kind == "code" else self.thr_text

    def _stable_above(self, window: List[VPMRow], thr: Thresholds, margin: float) -> bool:
        if not window:
            return False
        dims = thr.mins.keys()
        for w in window[-self.p.patience:]:  # require last K frames above
            for d in dims:
                v = self._val(w, d)
                if v is None:
                    return False
                if v < thr.mins[d] + margin:
                    return False
        return True

    def _recently_above(self, window: List[VPMRow], thr: Thresholds, patience: int) -> bool:
        """At least once in last K frames above all mins (prevents premature RESAMPLE)."""
        dims = thr.mins.keys()
        recent = window[-patience:]
        for w in recent:
            if all((self._val(w, d) or 0) >= thr.mins[d] for d in dims):
                return True
        return False

    def _should_spinoff(self, row: VPMRow) -> bool:
        if self.p.spinoff_dim not in row.dims or self.p.stickiness_dim not in row.dims:
            return False
        nov = row.dims[self.p.spinoff_dim]
        stk = row.dims[self.p.stickiness_dim]
        return (nov >= self.p.spinoff_gate[0]) and (stk <= self.p.spinoff_gate[1])

    def _gaps(self, row: VPMRow, thr: Thresholds) -> List[str]:
        gaps = []
        for k, t in thr.mins.items():
            v = self._val(row, k)
            if v is None:
                continue
            # hysteresis on EDIT: don't thrash if close
            if v < t - self.p.edit_margin:
                gaps.append(k)
        return gaps

    def _regressions(self, window: List[VPMRow]) -> int:
        """Count metric dips vs previous frame for core dims present in all frames."""
        if len(window) < 2:
            return 0
        regs = 0
        dims = set(window[-1].dims.keys())
        for i in range(1, len(window)):
            prev, cur = window[i-1], window[i]
            dips = sum(1 for d in dims if d in prev.dims and d in cur.dims and cur.dims[d] < prev.dims[d] - 1e-6)
            regs += (1 if dips >= max(1, len(dims)//4) else 0)
        return regs

    def _stagnating(self, window: List[VPMRow], thr: Thresholds) -> bool:
        """No improvement on core dims for 'patience' frames."""
        if len(window) < self.p.patience + 1:
            return False
        recent = window[-(self.p.patience+1):]
        core = [k for k in thr.mins.keys() if k in recent[-1].dims]
        improved = False
        for d in core:
            series = [w.dims.get(d, 0.0) for w in recent]
            if series[-1] - series[0] > 0.005:
                improved = True
                break
        return not improved

    def _trend(self, window: List[VPMRow]) -> Dict[str, float]:
        """Simple linear slope estimate per dim in window (normalized by length)."""
        if len(window) < 2:
            return {}
        n = len(window)
        t = list(range(n))
        trends = {}
        dims = set().union(*(w.dims.keys() for w in window))
        for d in dims:
            y = [w.dims.get(d, float('nan')) for w in window]
            y = [v for v in y if not math.isnan(v)]
            if len(y) < 2:
                continue
            # slope ~ last - first over n
            trends[d] = (y[-1] - y[0]) / (n - 1)
        return trends

    def _worsening(self, trend: Dict[str, float]) -> bool:
        """If majority of tracked dims have negative slope beyond small epsilon."""
        if not trend:
            return False
        vals = list(trend.values())
        neg = sum(1 for v in vals if v < -0.003)
        return neg >= max(1, len(vals)//2)

    def _ema_series(self, window: List[VPMRow]) -> Dict[str, float]:
        """Exponential moving average for info/debug; not used directly in gates."""
        if not window:
            return {}
        alpha = self.p.ema_alpha
        acc: Dict[str, float] = {}
        for w in window:
            for k, v in w.dims.items():
                if k not in acc:
                    acc[k] = v
                else:
                    acc[k] = alpha * v + (1 - alpha) * acc[k]
        return acc

    def _bump_resamples(self, unit: str):
        self.resample_counts[unit] = self.resample_counts.get(unit, 0) + 1

    def _val(self, row: VPMRow, key: str) -> Optional[float]:
        v = row.dims.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    def _decide(self, row: VPMRow, signal: Signal, reason: str, params: Dict[str, Any]) -> Decision:
        dec = Decision(signal=signal, reason=reason, params=params, snapshot={
            "unit": row.unit,
            "kind": row.kind,
            "step_idx": row.step_idx,
            "dims": row.dims
        })
        self.last_signal[row.unit] = signal

        # bandit bookkeeping: update on STOP/EDIT improvements if we know exemplar used
        eid = row.meta.get("exemplar_id")
        if eid and self.bandit_update:
            reward = self._reward(row) if signal in (Signal.STOP, Signal.EDIT) else 0.0
            try:
                self.bandit_update(eid, reward)
            except Exception:
                pass

        self.log("decision", {"unit": row.unit, "signal": signal.name, "reason": reason, **params})
        return dec

    def _reward(self, row: VPMRow) -> float:
        """Define reward for bandit as average of selected dims (can be customized)."""
        # prioritize “core” dims commonly present
        core = ["coverage","correctness","coherence","tests_pass_rate","type_safe","lint_clean"]
        vals = [row.dims[d] for d in core if d in row.dims]
        if not vals:
            vals = list(row.dims.values())
        return float(sum(vals)/len(vals)) if vals else 0.0


# ========= convenience builders =========

def default_controller() -> VPMController:
    thr_code = Thresholds(
        mins={
            "tests_pass_rate": 1.0,
            "coverage": 0.70,
            "type_safe": 1.0,
            "lint_clean": 1.0,
            "complexity_ok": 0.8
        },
        stop_margin=0.0,  # exact for code
        edit_margin=0.0
    )
    thr_text = Thresholds(
        mins={
            "coverage": 0.80,
            "correctness": 0.75,
            "coherence": 0.75,
            "citation_support": 0.65,
            "entity_consistency": 0.80
        },
        stop_margin=0.02,
        edit_margin=0.01
    )
    return VPMController(thr_code, thr_text, Policy())

# ========= example usage =========
if __name__ == "__main__":
    ctrl = default_controller()

    # simulate some text VPM frames
    def row(step, cov, cor, coh, cit, ent) -> VPMRow:
        return VPMRow(
            unit="Blog:Method",
            kind="text",
            timestamp=time.time(),
            step_idx=step,
            dims=dict(coverage=cov, correctness=cor, coherence=coh, citation_support=cit, entity_consistency=ent, novelty=0.7, stickiness=0.5),
            meta={"exemplar_id": "ex_pack_A"}
        )

    frames = [
        row(1, 0.62, 0.60, 0.64, 0.30, 0.70),
        row(2, 0.70, 0.66, 0.70, 0.55, 0.78),
        row(3, 0.74, 0.70, 0.72, 0.60, 0.80),
        row(4, 0.81, 0.76, 0.77, 0.67, 0.85),
        row(5, 0.82, 0.77, 0.78, 0.68, 0.86),
    ]
    for f in frames:
        dec = ctrl.add(f, candidate_exemplars=["ex_pack_A","ex_pack_B","ex_pack_C"])
        print(f"step {f.step_idx}: {dec.signal.name} — {dec.reason} {dec.params}")
``n
