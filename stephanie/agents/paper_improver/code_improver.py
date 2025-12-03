# stephanie/agents/paper_improver/code_improver.py
from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stephanie.memory.casebook_store import CaseBookStore
from stephanie.scoring.scorable import ScorableType
from stephanie.services.bus.knowledge_bus import KnowledgeBus
from stephanie.utils.json_sanitize import safe_json

from .bandit_router import ExemplarBandit

# Optional mutation testing (graceful if absent)
try:
    from .mutation import \
        MutationRunner  # our module; returns dict with "score"

    HAS_MUTATION = True
except Exception:
    HAS_MUTATION = False


class CodeImprover:
    def __init__(
        self,
        workdir: str = "./improver_runs",
        backend: str = "torch",
        timeout: int = 300,
        max_edits: int = 5,
        kb: Any = None,
        casebooks: CaseBookStore | None = None,
    ):
        self.workdir = Path(workdir)
        self.workdir.mkdir(exist_ok=True)
        self.run_id = 0
        self.backend = backend
        self.timeout = timeout
        self.max_edits = max_edits
        # keep the truly dangerous imports blocked; allow sys/shutil
        self.denylist_imports = {
            "os",
            "subprocess",
            "socket",
            "pickle",
            "marshal",
        }
        self.bandit = ExemplarBandit(save_path="./bandit_state.json")
        self.kb = kb or KnowledgeBus()
        self.casebook_store = CaseBookStore(root="./knowledge/casebooks")

    def improve(self, spec: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Takes spec JSON → returns PR-ready artifact + VPM row + DPO pair."""
        self.run_id += 1
        spec_hash = hashlib.sha256(
            json.dumps(spec, sort_keys=True).encode()
        ).hexdigest()[:8]
        run_dir = self.workdir / f"run_{self.run_id}_{spec_hash}"
        run_dir.mkdir(parents=True, exist_ok=True)

        self._set_seeds()

        candidates = [
            cb.name
            for cb in self.casebooks.list_casebooks_by_tag("exemplar_code")
        ]
        chosen_exemplar = (
            self.bandit.choose(candidates) if candidates else None
        )
        # optional: fetch champion code for ICL / pattern mining later:
        if chosen_exemplar:
            chosen_payload = self.casebooks.load(chosen_exemplar)
            # you can mine prior “final” code scorables or edit_snippets here

        meta = {
            "spec_sha": spec_hash,
            "seeds": {"python": 0, "torch": 0},
            "backend": self.backend,
            "timeout": self.timeout,
            "max_edits": self.max_edits,
            "exemplar_id": chosen_exemplar,
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        # Write spec
        spec_path = run_dir / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=2))

        # Scaffold package: src/pkg/impl.py
        pkg_dir = run_dir / "src" / "pkg"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "__init__.py").touch()

        # Generate initial stub (now aware of exemplar choice)
        stub_path = pkg_dir / "impl.py"
        stub_content = self._generate_stub(spec, exemplar_id=chosen_exemplar)
        self._ast_denylist_check(
            stub_content
        )  # security: AST parse + block dangerous nodes
        stub_path.write_text(stub_content)

        # Generate tests
        test_path = self._generate_tests(spec, run_dir)

        # Initial test + lint run
        print(f"🧪 Initial test run for {spec['function_name']}...")
        initial_results = self._run_tests_and_lint(run_dir, test_path)

        # Apply edit-policy until pass or max edits
        final_code, edit_log = self._apply_edit_policy(
            stub_path, test_path, run_dir
        )

        # Final evaluation
        final_results = self._run_tests_and_lint(run_dir, test_path)

        # Optional: mutation testing (only if tools available and tests not failing)
        mutation_score = 0.0
        if HAS_MUTATION and final_results.get("pass_rate", 0.0) == 1.0:
            try:
                mut_rep = MutationRunner(timeout_sec=600).run(
                    run_dir=run_dir, src="src", tests="tests"
                )
                mutation_score = float(mut_rep.get("score") or 0.0)
            except Exception:
                mutation_score = 0.0

        # Build VPM row
        vpm_row = self._build_vpm_row(
            initial_results,
            final_results,
            spec,
            run_dir,
            mutation_score,
            exemplar_id=chosen_exemplar,
        )

        # Update bandit reward based on improvement (Δ pass_rate, fallback Δ coverage)
        reward = self._compute_reward(initial_results, final_results)
        try:
            self.bandit.update(
                exemplar_id=chosen_exemplar,
                reward=reward,
                context={"kind": "code", "backend": self.backend},
            )
        except Exception:
            pass

        # after computing final vpm_row:
        reward = (
            vpm_row.get("tests_pass_rate") or vpm_row.get("coverage") or 0.0
        )
        if chosen_exemplar:
            self.bandit.update(chosen_exemplar, reward)

        # Log DPO pair (function body only + failing test names)
        dpo_pair = self._build_dpo_pair(
            initial_code=initial_results.get("code_snapshot", ""),
            final_code=final_code,
            spec=spec,
            meta=meta,
            initial_results=initial_results,
        )
        (run_dir / "dpo_pair.json").write_text(json.dumps(dpo_pair, indent=2))

        casebook_name = f"code_{spec['function_name']}"
        pipeline_run_id = context.get("pipeline_run_id")
        cb = self.casebooks.ensure_casebook(
            name=casebook_name,
            pipeline_run_id=pipeline_run_id,
            tags=["code_improver", "exemplar_code"],
            meta={"spec_sha": spec_hash, "backend": self.backend},
        )
        case = self.casebooks.add_case(
            casebook_name,
            json.dumps(spec),
            agent_name="code_improver",
            meta={"run_dir": str(run_dir), "chosen_exemplar": chosen_exemplar},
        )
        self.kb.publish(
            "case.added", {"casebook": cb.name, "case_id": case.id}
        )

        self.casebooks.add_scorable(
            casebook_name,
            case.id,
            role="vpm",
            text=safe_json(initial_results),
            meta={"stage": "initial"},
            scorable_type=ScorableType.DYNAMIC,
        )

        self.casebooks.add_scorable(
            casebook_name,
            case.id,
            role="code",
            text=(pkg_dir / "impl.py").read_text(),
            meta={"stage": "final"},
            scorable_type=ScorableType.DYNAMIC,
        )
        self.casebooks.add_scorable(
            casebook_name,
            case.id,
            role="text",
            text=(run_dir / "draft.md").read_text(),
            meta={"stage": "final"},
            scorable_type=ScorableType.DYNAMIC,
        )
        # DPO pair
        self.casebooks.add_scorable(
            casebook_name,
            case.id,
            role="dpo_pair",
            text=safe_json(dpo_pair),
            meta=dpo_pair["metadata"],
        )

        self.kb.publish(
            "trajectory.step",
            {"casebook": cb.name, "case_id": case.id, "vpm": vpm_row},
        )

        # Return structured artifact
        return {
            "run_dir": str(run_dir),
            "spec_path": str(spec_path),
            "final_code_path": str(stub_path),
            "test_path": str(test_path),
            "vpm_row": vpm_row,
            "dpo_pair_path": str(run_dir / "dpo_pair.json"),
            "passed": abs(final_results["pass_rate"] - 1.0) < 1e-9,
            "coverage": final_results.get("coverage", 0.0),
            "escalations": 0
            if abs(final_results["pass_rate"] - 1.0) < 1e-9
            else 1,
            "edit_log": edit_log,
        }

    # -------------------------
    # Generation
    # -------------------------

    def _generate_stub(
        self, spec: Dict[str, Any], exemplar_id: Optional[str] = None
    ) -> str:
        backend_import = (
            "import torch" if self.backend == "torch" else "import numpy as np"
        )
        fn_name = spec["function_name"]
        desc = spec.get("description", "No description.")
        eqs = spec.get("equations", [])
        eq_str = "; ".join(eqs) if eqs else "Not specified."

        # Slightly different scaffolds per exemplar (keeps bandit meaningful)
        body = "return x  # placeholder"
        if exemplar_id == "torch_norm_v1":
            body = (
                "x = torch.as_tensor(x).float()\n"
                "den = torch.linalg.vector_norm(x) + 1e-12\n"
                "return x / den\n"
                if self.backend == "torch"
                else "x = np.asarray(x, dtype=np.float32)\n"
                "den = (np.linalg.norm(x) + 1e-12)\n"
                "return x / den\n"
            )
        elif exemplar_id == "numpy_math_v2":
            body = (
                "x = torch.as_tensor(x).float()\nreturn torch.sum(x)\n"
                if self.backend == "torch"
                else "x = np.asarray(x, dtype=np.float32)\n"
                "return float(np.sum(x))\n"
            )

        return f'''{backend_import}

def {fn_name}(x):
    """
    {desc}
    Equations: {eq_str}
    """
    # Initial scaffold ({exemplar_id})
    {body}
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

        test_content = f"""
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
"""
        test_dir = run_dir / "tests"
        test_dir.mkdir(parents=True, exist_ok=True)
        test_path = test_dir / f"test_{fn_name}.py"
        test_path.write_text(test_content)
        return test_path

    # -------------------------
    # Execution & Metrics
    # -------------------------

    def _limit_resources(self):
        """Enforce CPU, memory (best effort); network blocking can be layered later."""
        pass

    def _run_tests_and_lint(
        self, run_dir: Path, test_path: Path
    ) -> Dict[str, Any]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(run_dir / "src")
        env["NO_NETWORK"] = "1"  # hint for stubbed clients

        cmd = [
            "python",
            "-m",
            "pytest",
            str(test_path),
            "-q",
            "--cov=src",
            "--cov-report=xml:coverage.xml",
            "--json-report",
            "--json-report-file=pytest_report.json",
            "--maxfail=1",
        ]

        try:
            r = subprocess.run(
                cmd,
                cwd=run_dir,
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                preexec_fn=self._limit_resources,
            )
        except subprocess.TimeoutExpired:
            return self._empty_results(stderr="TIMEOUT")

        # Parse pytest JSON
        try:
            rep = json.loads((run_dir / "pytest_report.json").read_text())
            passed = int(rep["summary"].get("passed", 0))
            failed = int(rep["summary"].get("failed", 0))
            errors = int(rep["summary"].get("error", 0))
            total = passed + failed + errors
            pass_rate = passed / total if total > 0 else 0.0
            failing_tests = [
                t["nodeid"]
                for t in rep.get("tests", [])
                if t["outcome"] in ("failed", "error")
            ]
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
            "code_snapshot": code_snapshot,
        }

    def _empty_results(self, stderr: str = "") -> Dict[str, Any]:
        return {
            "pass_rate": 0.0,
            "coverage": 0.0,
            "type_safe": 0.0,
            "lint_clean": 0.0,
            "complexity_ok": 0.0,
            "stderr": stderr,
            "failing_tests": [],
            "code_snapshot": "",
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
            r = subprocess.run(
                ["ruff", "check", "."],
                cwd=run_dir,
                capture_output=True,
                text=True,
            )
            return 1.0 if r.returncode == 0 else 0.0
        except Exception:
            return 0.0

    def _run_mypy(self, run_dir: Path) -> float:
        try:
            r = subprocess.run(
                ["mypy", "."], cwd=run_dir, capture_output=True, text=True
            )
            return 1.0 if r.returncode == 0 else 0.0
        except Exception:
            return 0.0

    def _run_radon(self, run_dir: Path) -> float:
        try:
            r = subprocess.run(
                ["radon", "cc", "src", "-s"],
                cwd=run_dir,
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                return 1.0
            # Penalize if any function > B
            if any(x in r.stdout for x in (" C ", " D ", " E ", " F ")):
                return 0.5
            return 1.0
        except Exception:
            return 1.0

    # -------------------------
    # Edit Policy
    # -------------------------

    def _apply_edit_policy(
        self, stub_path: Path, test_path: Path, run_dir: Path
    ) -> Tuple[str, List[str]]:
        code = stub_path.read_text()
        edits: List[str] = []

        for i in range(self.max_edits):
            results = self._run_tests_and_lint(run_dir, test_path)
            if results["pass_rate"] == 1.0:
                break

            fix = self._propose_minimal_fix(
                code, results["stderr"], results["failing_tests"]
            )
            if not fix or fix == code:
                break

            # safety: re-check denylist on each new code version
            self._ast_denylist_check(fix)

            code = fix
            stub_path.write_text(code)
            edits.append(
                f"Edit {i + 1}: applied fix based on: {results['stderr'][:80]}..."
            )
            print(f"🔧 Applied edit {i + 1}: {edits[-1]}")

        return code, edits

    def _propose_minimal_fix(
        self, code: str, stderr: str, failing_tests: List[str]
    ) -> Optional[str]:
        """Rule-based fixes based on error patterns. One minimal, targeted change per call."""
        new_code = code

        # Fix 1: dtype mismatch
        if any(
            kw in stderr
            for kw in ("dtype", "tensor(", "ndarray", "float(", "int(")
        ):
            if "import torch" in code:
                new_code = re.sub(
                    r"^\s*return\s+(.+)$",
                    r"    return torch.as_tensor(\1).float()",
                    new_code,
                    count=1,
                    flags=re.M,
                )
            elif "import numpy" in code or "import numpy as np" in code:
                new_code = re.sub(
                    r"^\s*return\s+(.+)$",
                    r"    return np.asarray(\1, dtype=np.float32)",
                    new_code,
                    count=1,
                    flags=re.M,
                )
            if new_code != code:
                return new_code

        # Fix 2: shape mismatch
        if any(
            kw in stderr for kw in ("shape", "size", "dimension", "broadcast")
        ):
            if "import torch" in code:
                new_code = re.sub(
                    r"^\s*return\s+(.+)$",
                    r"    return \1.view(-1)",
                    new_code,
                    count=1,
                    flags=re.M,
                )
            elif "import numpy" in code or "import numpy as np" in code:
                new_code = re.sub(
                    r"^\s*return\s+(.+)$",
                    r"    return \1.flatten()",
                    new_code,
                    count=1,
                    flags=re.M,
                )
            if new_code != code:
                return new_code

        # Fix 3: out-of-bounds / negative values
        if any(
            kw in stderr
            for kw in ("clamp", "bound", "negative", "positive", "range")
        ):
            if "import torch" in code:
                new_code = re.sub(
                    r"^\s*return\s+(.+)$",
                    r"    return torch.clamp(\1, min=0.0, max=1.0)",
                    new_code,
                    count=1,
                    flags=re.M,
                )
            elif "import numpy" in code or "import numpy as np" in code:
                new_code = re.sub(
                    r"^\s*return\s+(.+)$",
                    r"    return np.clip(\1, 0.0, 1.0)",
                    new_code,
                    count=1,
                    flags=re.M,
                )
            if new_code != code:
                return new_code

        # Fix 4: sum() on list → backend sum
        if "sum(" in stderr and ("list" in stderr or "iterable" in stderr):
            if "import torch" in code:
                new_code = re.sub(
                    r"return\s+sum\((.+)\)",
                    r"    return torch.sum(torch.as_tensor(\1))",
                    new_code,
                    count=1,
                )
            elif "import numpy" in code or "import numpy as np" in code:
                new_code = re.sub(
                    r"return\s+sum\((.+)\)",
                    r"    return np.sum(np.asarray(\1))",
                    new_code,
                    count=1,
                )
            if new_code != code:
                return new_code

        # Fix 5: missing import or undefined name
        if "NameError" in stderr or "not defined" in stderr:
            if (
                "math" in stderr or "math." in stderr
            ) and "import math" not in code:
                return "import math\n" + code

        return None

    # -------------------------
    # Safety & Hygiene
    # -------------------------

    def _ast_denylist_check(self, code: str):
        """Parse AST and block dangerous patterns."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        if alias.name.split(".")[0] in self.denylist_imports:
                            raise ValueError(
                                f"AST denylist triggered: {alias.name}"
                            )
                elif isinstance(node, ast.Call):
                    # block eval/exec/__import__
                    if isinstance(node.func, ast.Name) and node.func.id in {
                        "eval",
                        "exec",
                        "__import__",
                    }:
                        raise ValueError(
                            f"AST denylist triggered: {node.func.id}"
                        )
                    # block os.system / subprocess.Popen, etc.
                    if isinstance(node.func, ast.Attribute):
                        root = getattr(node.func.value, "id", None)
                        if root in {"os", "subprocess"}:
                            raise ValueError(
                                f"AST denylist triggered: {root}.{node.func.attr}"
                            )
        except SyntaxError:
            # Let pytest/mypy report syntax; we only guard dangerous ops
            pass

    # -------------------------
    # Output & Logging
    # -------------------------

    def _build_vpm_row(
        self,
        initial: Dict[str, Any],
        final: Dict[str, Any],
        spec: Dict[str, Any],
        run_dir: Path,
        mutation_score: float,
        exemplar_id: str,
    ) -> Dict[str, Any]:
        fn_name = spec["function_name"]
        return {
            "unit": f"pkg.impl:{fn_name}",
            "tests_pass_rate": round(final.get("pass_rate", 0.0), 3),
            "coverage": round(final.get("coverage", 0.0), 3),
            "type_safe": round(final.get("type_safe", 0.0), 3),
            "lint_clean": round(final.get("lint_clean", 0.0), 3),
            "complexity_ok": round(final.get("complexity_ok", 0.0), 3),
            "mutation_score": round(float(mutation_score), 3),
            "exemplar_id": exemplar_id,
            "escalations": 0 if final.get("pass_rate", 0.0) == 1.0 else 1,
        }

    def _build_dpo_pair(
        self,
        initial_code: str,
        final_code: str,
        spec: Dict[str, Any],
        meta: Dict[str, Any],
        initial_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "spec_slice": {
                k: v
                for k, v in spec.items()
                if k
                in [
                    "function_name",
                    "equations",
                    "input_shape",
                    "output_shape",
                    "golden_input",
                    "golden_output",
                ]
            },
            "prompt": f"Implement {spec['function_name']} per spec. Equations: {spec.get('equations', [])}",
            "rejected": self._extract_function_body(initial_code),
            "chosen": self._extract_function_body(final_code),
            "metadata": {
                "spec_hash": meta["spec_sha"],
                "backend": meta["backend"],
                "initial_pass_rate": initial_results.get("pass_rate", 0.0),
                "failing_tests": initial_results.get("failing_tests", []),
                "stderr_snippet": (initial_results.get("stderr", "") or "")[
                    :200
                ],
                "exemplar_id": meta.get("exemplar_id"),
            },
        }

    def _extract_function_body(self, code: str) -> str:
        """Extract just the function body for DPO pairs."""
        lines = code.splitlines()
        in_func = False
        body_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def "):
                in_func = True
                continue
            if in_func and (not stripped or stripped.startswith("#")):
                continue
            if in_func and not (
                line.startswith("    ") or line.startswith("\t")
            ):
                break
            if in_func:
                body_lines.append(line)
        return "\n".join(body_lines) if body_lines else code

    # -------------------------
    # Internal helpers
    # -------------------------

    def _set_seeds(self):
        random.seed(0)
        try:
            import numpy as np

            np.random.seed(0)
        except Exception:
            pass
        if self.backend == "torch":
            try:
                import torch

                torch.manual_seed(0)
            except Exception:
                pass

    def _compute_reward(
        self, initial: Dict[str, Any], final: Dict[str, Any]
    ) -> float:
        """Map improvement to [0,1] reward: Δpass_rate primary, Δcoverage fallback."""

        def clamp01(x: float) -> float:
            return max(0.0, min(1.0, x))

        delta = final.get("pass_rate", 0.0) - initial.get("pass_rate", 0.0)
        if abs(delta) < 1e-9:
            delta = final.get("coverage", 0.0) - initial.get("coverage", 0.0)
        delta = max(-1.0, min(1.0, float(delta)))
        return clamp01(0.5 + 0.5 * delta)
