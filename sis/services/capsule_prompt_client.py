"""Deterministic Capsule prompt preparation; deliberately does not pass credentials."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


class CapsulePromptError(RuntimeError):
    pass


class CapsulePromptClient:
    def __init__(self, root: str | Path | None = None, timeout_seconds: int = 120):
        self.root = Path(root or os.environ.get("STEPHANIE_CAPSULE_ROOT", r"C:\Projects\capsule")).resolve()
        self.timeout_seconds = max(10, min(180, int(timeout_seconds)))

    def _environment(self) -> dict[str, str]:
        ai_root = Path(os.environ.get("STEPHANIE_AI_ROOT", r"C:\Projects\ai")).resolve()
        paths = [str(self.root / "src")]
        if (ai_root / "src").is_dir():
            paths.append(str(ai_root / "src"))
        # Deliberately exclude CAPSULE_DSN, provider credentials and all other secrets.
        env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE", "LOCALAPPDATA", "HOME", "INSPECT_PROJECTS", "CAPSULE_HANDOFF_DIR") if os.environ.get(key)}
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def subjects(self) -> list[dict[str, str]]:
        if not self.root.is_dir() or not (self.root / "src" / "capsule_app").is_dir():
            raise CapsulePromptError(f"Capsule source is unavailable at {self.root}")
        try:
            completed = subprocess.run([sys.executable, "-m", "capsule_app", "subjects"], cwd=self.root,
                env=self._environment(), capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=15, check=False, shell=False)
        except subprocess.TimeoutExpired as exc:
            raise CapsulePromptError("Capsule subject discovery timed out") from exc
        if completed.returncode:
            raise CapsulePromptError(f"Capsule subject discovery failed: {completed.stderr[-2000:]}")
        try:
            rows = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise CapsulePromptError("Capsule returned invalid subject JSON") from exc
        if not isinstance(rows, list):
            raise CapsulePromptError("Capsule returned an invalid subject list")
        return rows

    def processes(self) -> list[dict[str, Any]]:
        """Read the declarative catalogue; design-only entries are filtered by callers."""
        try:
            completed = subprocess.run([sys.executable, "-m", "capsule_app", "processes"], cwd=self.root,
                env=self._environment(), capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=15, check=False, shell=False)
        except subprocess.TimeoutExpired as exc:
            raise CapsulePromptError("Capsule process discovery timed out") from exc
        if completed.returncode:
            raise CapsulePromptError(f"Capsule process discovery failed: {completed.stderr[-2000:]}")
        try:
            rows = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise CapsulePromptError("Capsule returned invalid process JSON") from exc
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise CapsulePromptError("Capsule returned an invalid process catalogue")
        return rows

    def prepare(self, *, subject: str, process_id: str = "repository.review", revision: str = "latest", instructions: str = "",
                destination: str = "codex", size_profile: str = "standard",
                token_budget: int | None = None) -> dict[str, Any]:
        if not self.root.is_dir() or not (self.root / "src" / "capsule_app").is_dir():
            raise CapsulePromptError(f"Capsule source is unavailable at {self.root}")
        if len(instructions) > 8000:
            raise CapsulePromptError("Instructions exceed the 8,000-character limit")
        command = [sys.executable, "-m", "capsule_app", "--format", "json", "--subject", subject,
                   "--revision", revision, "--destination", destination, "--size-profile", size_profile,
                   "--instructions", instructions]
        if token_budget is not None:
            command.extend(("--token-budget", str(token_budget)))
        if not process_id.startswith(("repository.", "project.")):
            raise CapsulePromptError("Unsupported repository process")
        command.extend(("process", "prepare-prompt", process_id))
        try:
            completed = subprocess.run(command, cwd=self.root, env=self._environment(), capture_output=True, text=True,
                                       encoding="utf-8", errors="replace", timeout=self.timeout_seconds,
                                       check=False, shell=False)
        except subprocess.TimeoutExpired as exc:
            raise CapsulePromptError("Capsule prompt preparation exceeded its deterministic execution timeout") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-3000:]
            raise CapsulePromptError(f"Capsule returned {completed.returncode}: {detail}")
        try:
            package = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise CapsulePromptError("Capsule returned invalid JSON") from exc
        dispatch = package.get("dispatch", {}) if isinstance(package, dict) else {}
        if not isinstance(package, dict) or not (
            dispatch.get("dispatched") is False or dispatch.get("status") == "not_dispatched"
        ):
            raise CapsulePromptError("Capsule output did not prove that dispatch was skipped")
        return package

    def prepare_document(self, *, document_context: dict[str, Any], process_id: str,
                         instructions: str = "", destination: str = "codex",
                         size_profile: str = "standard", token_budget: int | None = None) -> dict[str, Any]:
        """Pass document text through a bounded temporary file, never process arguments."""
        if process_id not in {"document.review","document.improve","chapter.review","chapter.improve_readability","book.remove_artifacts","book.technical.create"}:
            raise CapsulePromptError("Unsupported document process")
        serialized=json.dumps(document_context,ensure_ascii=False,separators=(",",":"))
        if len(serialized)>1_200_000 or len(instructions)>8_000:
            raise CapsulePromptError("Document context exceeds the 1,200,000-character handoff bound")
        if not self.root.is_dir() or not (self.root / "src" / "capsule_app").is_dir():
            raise CapsulePromptError(f"Capsule source is unavailable at {self.root}")
        temp_path=None
        try:
            with tempfile.NamedTemporaryFile("w",encoding="utf-8",suffix=".json",delete=False) as handle:
                handle.write(serialized); temp_path=Path(handle.name)
            subject=str(document_context.get("document_id",""))
            command=[sys.executable,"-m","capsule_app","--format","json","--subject",subject,
                "--revision","latest","--destination",destination,"--size-profile",size_profile,
                "--instructions",instructions,"--document-context-file",str(temp_path),
                "process","prepare-prompt",process_id]
            if token_budget is not None: command.extend(("--token-budget",str(token_budget)))
            completed=subprocess.run(command,cwd=self.root,env=self._environment(),capture_output=True,text=True,
                encoding="utf-8",errors="replace",timeout=self.timeout_seconds,check=False,shell=False)
            if completed.returncode:
                raise CapsulePromptError(f"Capsule returned {completed.returncode}: {(completed.stderr or completed.stdout)[-3000:]}")
            package=json.loads(completed.stdout)
            dispatch=package.get("dispatch",{}) if isinstance(package,dict) else {}
            if not isinstance(package,dict) or not (dispatch.get("dispatched") is False or dispatch.get("status")=="not_dispatched"):
                raise CapsulePromptError("Capsule output did not prove that dispatch was skipped")
            return package
        except subprocess.TimeoutExpired as exc:
            raise CapsulePromptError("Capsule document prompt preparation timed out") from exc
        except json.JSONDecodeError as exc:
            raise CapsulePromptError("Capsule returned invalid document prompt JSON") from exc
        finally:
            if temp_path is not None:
                try: temp_path.unlink(missing_ok=True)
                except OSError: pass

