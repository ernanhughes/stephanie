# stephanie/tools/code_analyzer_tool.py
from __future__ import annotations

import ast
import hashlib
import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stephanie.scoring.scorable import Scorable
from stephanie.tools.base_tool import BaseTool

log = logging.getLogger(__name__)


# -----------------------------
# Small data helpers
# -----------------------------

@dataclass(frozen=True)
class FileFingerprint:
    rel_path: str
    abs_path: str
    sha256: str
    mtime: float
    size_bytes: int


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _safe_slug(s: str) -> str:
    # stable filename-ish slug for cache files
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]


def _read_text(path: Path, max_bytes: int) -> str:
    # Read up to max_bytes safely; skip if binary-ish
    raw = path.read_bytes()[:max_bytes]
    if b"\x00" in raw:
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw.decode("latin-1")
        except Exception:
            return ""


# -----------------------------
# The tool
# -----------------------------

class CodeAnalyzerTool(BaseTool):
    """
    Repo code analyzer designed for Stephanie:
    - Deterministic findings (AST + optional ruff) => grounded signal
    - Optional LLM advisor on top-K risky files => actionable guidance
    - Fingerprint + disk cache => repeat runs are fast
    - Writes results to scorable.meta["code_analysis"][self.name]
    """

    name = "code_analyzer"

    # Optional process-wide model cache (avoid re-loading in same process)
    _MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}

    def __init__(self, cfg: Dict[str, Any], memory, container, logger) -> None:
        super().__init__(cfg, memory, container, logger)

        self.base_path: str = cfg.get("base_path", "stephanie/")
        self.file_extensions: List[str] = cfg.get("file_extensions", [".py"])

        # Excludes (fast default; you can tune)
        self.exclude_dirs: List[str] = cfg.get(
            "exclude_dirs",
            [".venv", "venv", "__pycache__", ".git", ".mypy_cache", ".ruff_cache", "build", "dist"],
        )
        self.exclude_globs: List[str] = cfg.get(
            "exclude_globs",
            ["**/migrations/**", "**/site-packages/**", "**/.tox/**"],
        )

        self.max_file_size_bytes: int = int(cfg.get("max_file_size_bytes", 200_000))  # 200KB
        self.max_read_bytes: int = int(cfg.get("max_read_bytes", 500_000))  # cap read
        self.force: bool = bool(cfg.get("force", False))
        self.store_to_memory: bool = bool(cfg.get("store_to_memory", True))

        # Deterministic analyzers
        self.enable_ast_checks: bool = bool(cfg.get("enable_ast_checks", True))
        self.enable_ruff: bool = bool(cfg.get("enable_ruff", True))
        self.ruff_bin: str = cfg.get("ruff_bin", "ruff")
        self.ruff_timeout_s: float = float(cfg.get("ruff_timeout_s", 10.0))

        # LLM advisor (optional)
        self.enable_llm: bool = bool(cfg.get("enable_llm", False))
        self.llm_model_name: str = cfg.get("llm_model_name", "Qwen/Qwen2.5-Coder-0.5B-Instruct")
        self.llm_device: str = cfg.get("llm_device", "cpu")
        self.llm_max_input_chars: int = int(cfg.get("llm_max_input_chars", 12_000))
        self.llm_max_new_tokens: int = int(cfg.get("llm_max_new_tokens", 400))
        self.llm_top_k: int = int(cfg.get("llm_top_k", 25))

        # Cache (disk) - makes repeat runs fast without DB changes
        self.cache_dir: Path = Path(cfg.get("cache_dir", ".stephanie_cache/code_analysis"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "[CodeAnalyzerTool] base_path=%s exts=%s force=%s ruff=%s llm=%s cache_dir=%s",
            self.base_path,
            self.file_extensions,
            self.force,
            self.enable_ruff,
            self.enable_llm,
            str(self.cache_dir),
        )

        # LLM is lazy-loaded so that deterministic-only runs are cheap
        self._tokenizer = None
        self._model = None

    # -----------------------------
    # File discovery + fingerprinting
    # -----------------------------

    def _is_excluded(self, p: Path) -> bool:
        parts = set(p.parts)
        if any(d in parts for d in self.exclude_dirs):
            return True
        sp = str(p).replace("\\", "/")
        for g in self.exclude_globs:
            # cheap glob check
            if Path(sp).match(g):
                return True
        return False

    def _walk_files(self) -> List[Path]:
        root = Path(self.base_path)
        if not root.exists():
            log.warning("[CodeAnalyzerTool] base_path not found: %s", root)
            return []
        out: List[Path] = []
        for ext in self.file_extensions:
            for p in root.rglob(f"*{ext}"):
                if not p.is_file():
                    continue
                if self._is_excluded(p):
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                if st.st_size > self.max_file_size_bytes:
                    continue
                out.append(p)
        return out

    def _fingerprint(self, path: Path) -> FileFingerprint:
        st = path.stat()
        raw = path.read_bytes()
        sha = _sha256_bytes(raw)
        rel = str(path.relative_to(Path(self.base_path)).as_posix())
        return FileFingerprint(
            rel_path=rel,
            abs_path=str(path.resolve()),
            sha256=sha,
            mtime=float(st.st_mtime),
            size_bytes=int(st.st_size),
        )

    # -----------------------------
    # Cache IO
    # -----------------------------

    def _cache_key(self, fp: FileFingerprint) -> str:
        return f"{fp.rel_path}:{fp.sha256}"

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{_safe_slug(key)}.json"

    def _cache_load(self, fp: FileFingerprint) -> Optional[Dict[str, Any]]:
        key = self._cache_key(fp)
        cpath = self._cache_path(key)
        if not cpath.exists():
            return None
        try:
            return json.loads(cpath.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _cache_save(self, fp: FileFingerprint, payload: Dict[str, Any]) -> None:
        key = self._cache_key(fp)
        cpath = self._cache_path(key)
        try:
            cpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            log.debug("[CodeAnalyzerTool] cache save failed: %s", e)

    # -----------------------------
    # Deterministic analyzers
    # -----------------------------

    def _ast_findings(self, text: str) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            return [{
                "kind": "syntax_error",
                "severity": "high",
                "message": f"SyntaxError: {e.msg}",
                "line": getattr(e, "lineno", None),
                "col": getattr(e, "offset", None),
            }]

        # helpers
        def add(kind: str, severity: str, message: str, node: ast.AST):
            findings.append({
                "kind": kind,
                "severity": severity,
                "message": message,
                "line": getattr(node, "lineno", None),
                "col": getattr(node, "col_offset", None),
            })

        for node in ast.walk(tree):
            # broad except
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    add("broad_except", "medium", "Bare except: catches everything; narrow it.", node)
                elif isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException"):
                    add("broad_except", "medium", f"Broad except {node.type.id}; consider narrowing.", node)

            # eval / exec
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec"):
                    add("dangerous_call", "high", f"Use of {node.func.id} is dangerous; avoid if possible.", node)

            # mutable defaults
            if isinstance(node, ast.FunctionDef):
                for d in node.args.defaults:
                    if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                        add("mutable_default", "high", "Mutable default argument; use None + initialize inside.", node)

            # subprocess shell=True (very rough)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("Popen", "run", "call", "check_call", "check_output"):
                    for kw in node.keywords or []:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            add("shell_true", "high", "subprocess(..., shell=True) can be dangerous; avoid if possible.", node)

        return findings

    def _complexity_metrics(self, text: str) -> Dict[str, Any]:
        # lightweight “complexity-ish” signal: count branchy nodes
        try:
            tree = ast.parse(text)
        except Exception:
            return {"branch_nodes": None, "func_defs": None, "class_defs": None}

        branch = 0
        func_defs = 0
        class_defs = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match, ast.BoolOp)):
                branch += 1
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_defs += 1
            if isinstance(node, ast.ClassDef):
                class_defs += 1

        return {"branch_nodes": branch, "func_defs": func_defs, "class_defs": class_defs}

    def _ruff_check(self, file_path: Path) -> List[Dict[str, Any]]:
        if not self.enable_ruff:
            return []
        if shutil.which(self.ruff_bin) is None:
            return [{
                "kind": "tool_missing",
                "severity": "low",
                "message": f"ruff not found ({self.ruff_bin}); skipping.",
            }]

        cmd = [self.ruff_bin, "check", "--output-format", "json", str(file_path)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=self.ruff_timeout_s)
        except subprocess.TimeoutExpired:
            return [{"kind": "ruff_timeout", "severity": "low", "message": "ruff timed out"}]
        except Exception as e:
            return [{"kind": "ruff_error", "severity": "low", "message": f"ruff failed: {e}"}]

        if p.returncode not in (0, 1):  # ruff returns 1 on lint findings
            return [{"kind": "ruff_error", "severity": "low", "message": p.stderr.strip()[:500]}]

        try:
            data = json.loads(p.stdout) if p.stdout.strip() else []
            # normalize shape
            issues: List[Dict[str, Any]] = []
            for it in data:
                issues.append({
                    "code": it.get("code"),
                    "message": it.get("message"),
                    "filename": it.get("filename"),
                    "location": it.get("location"),
                    "end_location": it.get("end_location"),
                    "fix": it.get("fix"),
                })
            return issues
        except Exception:
            return [{"kind": "ruff_parse_error", "severity": "low", "message": "Could not parse ruff JSON output"}]

    # -----------------------------
    # LLM advisor (optional)
    # -----------------------------

    def _ensure_llm_loaded(self) -> None:
        if not self.enable_llm:
            return
        if self._model is not None and self._tokenizer is not None:
            return

        # process-wide cache
        if self.llm_model_name in self._MODEL_CACHE:
            self._tokenizer, self._model = self._MODEL_CACHE[self.llm_model_name]
            return

        log.info("[CodeAnalyzerTool] Loading LLM advisor model=%s device=%s", self.llm_model_name, self.llm_device)

        # lazy imports
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.llm_model_name)

        # keep memory sane on CPU: prefer float32 correctness but allow bfloat16 if you want
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        model.to(self.llm_device)
        model.eval()

        self._tokenizer, self._model = tok, model
        self._MODEL_CACHE[self.llm_model_name] = (tok, model)

    def _truncate_for_llm(self, text: str) -> str:
        if len(text) <= self.llm_max_input_chars:
            return text
        # keep head + tail
        head = text[: int(self.llm_max_input_chars * 0.65)]
        tail = text[-int(self.llm_max_input_chars * 0.35):]
        return head + "\n\n# --- truncated ---\n\n" + tail

    def _llm_advise(
        self,
        rel_path: str,
        code_text: str,
        findings: List[Dict[str, Any]],
        ruff_issues: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ) -> str:
        if not self.enable_llm:
            return ""

        self._ensure_llm_loaded()
        if self._model is None or self._tokenizer is None:
            return ""

        findings_txt = json.dumps(findings[:25], indent=2, ensure_ascii=False)
        ruff_txt = json.dumps(ruff_issues[:25], indent=2, ensure_ascii=False)

        code_clip = self._truncate_for_llm(code_text)

        prompt = f"""You are a senior Python reviewer inside an automated code-quality system.
            Your job: produce grounded, actionable improvements for this file.

            FILE: {rel_path}
            METRICS: {json.dumps(metrics)}
            AST_FINDINGS (top): {findings_txt}
            RUFF_ISSUES (top): {ruff_txt}

            CODE (truncated):
            ```python
            {code_clip}
            ```

            Return EXACTLY this structure:

            ## Summary

            (2-4 sentences, grounded in findings)

            ##Fix Plan (bullets)

            (bullet: concrete change + why)

            ## Risks / Bugs

            (bullet: specific risk, reference finding/ruff when possible)

            Refactor Opportunities

            (bullet: specific refactor, scope estimate)
            """
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.llm_device) for k, v in inputs.items()}
        import torch
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.llm_max_new_tokens,
                do_sample=True,
                temperature=0.2,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        text_out = self._tokenizer.decode(out[0], skip_special_tokens=True)
        # strip prompt echo if present
        if text_out.startswith(prompt):
            text_out = text_out[len(prompt):].strip()

        return text_out.strip()


    def _risk_score(
        self,
        findings: List[Dict[str, Any]],
        ruff_issues: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ) -> float:
        sev_w = {"low": 0.5, "medium": 1.5, "high": 3.0}
        s = 0.0
        for f in findings:
            s += sev_w.get(f.get("severity", "low"), 0.5)
        # ruff issues: count
        if ruff_issues and isinstance(ruff_issues, list):
            # tool_missing etc are dicts too; weight only real "code"
            for it in ruff_issues:
                if it.get("code"):
                    s += 0.3
        # complexity
        bn = metrics.get("branch_nodes")
        if isinstance(bn, int):
            s += min(10.0, bn / 8.0)
        return s

# -----------------------------
# Tool API
# -----------------------------

    async def apply(self, scorable: Scorable, context: Dict[str, Any]) -> Scorable:
        meta: Dict[str, Any] = scorable.meta
        bucket: Dict[str, Any] = meta.setdefault("code_analysis", {})

        if not self.force and self.name in bucket:
            log.debug("[CodeAnalyzerTool] Analysis already exists for scorable %r", getattr(scorable, "id", None))
            return scorable

        t0 = time.time()
        files = self._walk_files()
        log.info("[CodeAnalyzerTool] Found %d candidate files", len(files))

        per_file: List[Dict[str, Any]] = []
        triage: List[Tuple[float, int]] = []  # (risk_score, index into per_file)

        for p in files:
            fp = self._fingerprint(p)

            # cache reuse
            cached = None if self.force else self._cache_load(fp)
            if cached is not None:
                per_file.append(cached)
                triage.append((cached.get("risk_score", 0.0), len(per_file) - 1))
                continue

            text = _read_text(p, self.max_read_bytes)
            if not text.strip():
                continue

            findings = self._ast_findings(text) if self.enable_ast_checks else []
            metrics = self._complexity_metrics(text)
            ruff_issues = self._ruff_check(p)

            risk = self._risk_score(findings, ruff_issues, metrics)

            row = {
                "rel_path": fp.rel_path,
                "abs_path": fp.abs_path,
                "sha256": fp.sha256,
                "mtime": fp.mtime,
                "size_bytes": fp.size_bytes,
                "metrics": metrics,
                "findings": findings,
                "ruff": ruff_issues,
                "risk_score": risk,
                "llm_advice": "",
                "tool": self.name,
                "run_id": context.get("run_id"),
                "ts": time.time(),
            }

            self._cache_save(fp, row)
            per_file.append(row)
            triage.append((risk, len(per_file) - 1))

        # LLM advisor on top-K risky files
        if self.enable_llm and per_file:
            triage.sort(key=lambda x: x[0], reverse=True)
            top = triage[: max(0, min(self.llm_top_k, len(triage)))]

            # Only advise those without advice (cache may already contain)
            for _, idx in top:
                r = per_file[idx]
                if r.get("llm_advice"):
                    continue
                try:
                    path = Path(self.base_path) / r["rel_path"]
                    text = _read_text(path, self.max_read_bytes)
                    advice = self._llm_advise(
                        r["rel_path"],
                        text,
                        r.get("findings", []),
                        r.get("ruff", []),
                        r.get("metrics", {}),
                    )
                    r["llm_advice"] = advice

                    # refresh cache with advice included
                    fp = FileFingerprint(
                        rel_path=r["rel_path"],
                        abs_path=r["abs_path"],
                        sha256=r["sha256"],
                        mtime=r["mtime"],
                        size_bytes=r["size_bytes"],
                    )
                    self._cache_save(fp, r)

                except Exception as e:
                    log.warning("[CodeAnalyzerTool] LLM advice failed for %s: %s", r.get("rel_path"), e)

        # repo rollup
        per_file.sort(key=lambda r: float(r.get("risk_score", 0.0)), reverse=True)
        top_hotspots = per_file[:20]

        totals = {
            "files_analyzed": len(per_file),
            "total_findings": sum(len(r.get("findings") or []) for r in per_file),
            "total_ruff_items": sum(
                len([x for x in (r.get("ruff") or []) if isinstance(x, dict) and x.get("code")])
                for r in per_file
            ),
        }

        rollup = {
            "tool": self.name,
            "base_path": self.base_path,
            "run_id": context.get("run_id"),
            "elapsed_s": round(time.time() - t0, 3),
            "totals": totals,
            "top_hotspots": [{
                "rel_path": r["rel_path"],
                "risk_score": r.get("risk_score"),
                "branch_nodes": (r.get("metrics") or {}).get("branch_nodes"),
                "findings_n": len(r.get("findings") or []),
                "ruff_n": len([x for x in (r.get("ruff") or []) if isinstance(x, dict) and x.get("code")]),
                "has_llm": bool(r.get("llm_advice")),
            } for r in top_hotspots],
            "files": per_file,
        }

        bucket[self.name] = rollup

        # Optional: store to DB if you have a store; keep this safe/non-fatal.
        if self.store_to_memory:
            try:
                if hasattr(self.memory, "code_repo_reports"):
                    self.memory.code_repo_reports.upsert({
                        "scorable_type": scorable.target_type,
                        "scorable_id": scorable.id,
                        "tool_name": self.name,
                        "run_id": context.get("run_id"),
                        "base_path": self.base_path,
                        "payload": rollup,
                        "elapsed_s": rollup["elapsed_s"],
                        "files_analyzed": totals["files_analyzed"],
                    })
            except Exception as e:
                log.debug("[CodeAnalyzerTool] memory persist skipped/failed: %s", e)

        return scorable
