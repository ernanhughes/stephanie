# stephanie/agents/paper_improver/repo_link.py
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional


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

        # After auto-merge:
        if self.auto_merge and merged:
            # Optional: delete local branch
            try:
                self._run(["git", "branch", "-d", branch_name], check=False)
            except Exception:
                pass

        return {
            "branch": branch_name,
            "pr_url": pr_url,
            "ci_status": ci_status,
            "merged": merged,
            "artifacts_dir": str(target_dir)
        }

    def create_pr(self, run_dir, vpm_row, artifact_type="code", dpo_pair_path=None, label="improver"):
        return self.push_pr(run_dir=run_dir, vpm_row=vpm_row,
                            dpo_pair_path=dpo_pair_path, label=label,
                            artifact_type=artifact_type)

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