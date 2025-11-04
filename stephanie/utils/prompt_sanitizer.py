from __future__ import annotations
import re
from typing import Dict, Tuple

_CODE_PUNCT = set("{}[]()<>;:=+-/*\\|&%!#@,.$^~`")
_FENCE_RE = re.compile(r"```(\w+)?\s*\n(.*?)\n```", re.DOTALL)       # ```py\n...\n```
_TILDE_RE = re.compile(r"~~~(\w+)?\s*\n(.*?)\n~~~", re.DOTALL)       # ~~~js\n...\n~~~
_INLINE_RE = re.compile(r"`([^`]+)`")                               # `inline`
_STACK_START_RE = re.compile(r"(?i)^traceback \(most recent call last\):")
_DIFF_HDR_RE = re.compile(r"^(@@|[-+]{3}|\+\+\+)\s")
_JSON_LIKE_RE = re.compile(r"^\s*[{[].*[}\]]\s*$")                   # one-line JSON-ish
_WHITESPACE_RE = re.compile(r"\n{3,}")

def _count_lines(s: str) -> int:
    return 0 if not s else s.count("\n") + 1

def _punct_ratio(line: str) -> float:
    if not line.strip():
        return 0.0
    p = sum(1 for ch in line if ch in _CODE_PUNCT)
    return p / max(1, len(line))

def _looks_code_line(line: str) -> bool:
    t = line.lstrip()
    if not t:
        return False
    if t.startswith(("def ", "class ", "import ", "from ", "#!", "//", "#", "/*")):
        return True
    if t.endswith(("{", "}", ";")):
        return True
    if _punct_ratio(t) >= 0.20:
        return True
    return False

def _collapse(text: str, label: str, placeholder_tpl: str) -> str:
    lines = _count_lines(text)
    return placeholder_tpl.format(lang=label or "code", lines=lines, chars=len(text))

def _summarize_code_head(code: str, keep_first: int) -> Tuple[str, int]:
    lines = code.splitlines()
    head = "\n".join(lines[:max(0, keep_first)]).strip()
    return (head, len(lines))

def estimate_tokens(s: str) -> int:
    # rough heuristic: ~4 chars/token
    return (len(s) // 4) + 1

def clean_text_for_prompt(
    text: str,
    cfg: Dict,
    *,
    is_code_goal: bool = False
) -> str:
    """
    Remove/collapse code from a message for prompt-building.
    - Fenced code blocks (``` / ~~~)
    - Indented code blocks (>= 4 spaces) with >= min_code_lines
    - Stack traces
    - Diffs
    - Big JSON one-liners
    - Long inline `code` spans

    Modes:
      keep      -> leave content intact
      remove    -> replace blocks with placeholders only
      collapse  -> keep 0..N head lines + placeholder

    Returns: cleaned text
    """
    if not text or not cfg.get("enabled", True) or cfg.get("mode", "collapse") == "keep":
        return text

    mode = cfg.get("mode", "collapse")
    keep_first = int(cfg.get("keep_first_code_lines", 2))
    min_indent_lines = int(cfg.get("min_code_lines", 3))
    max_inline_len = int(cfg.get("max_inline_code", 80))
    ph = cfg.get("placeholders", {}) or {}
    ph_code = ph.get("code", "[code:{lang}:{lines} lines]")
    ph_stack = ph.get("stack", "[stack-trace:{lines} lines]")
    ph_json = ph.get("json", "[json:{chars} chars]")
    ph_diff = ph.get("diff", "[diff:{lines} lines]")

    # Gate: if this is a programming/code goal, optionally skip cleaning
    if is_code_goal and cfg.get("skip_if_code_goal", True):
        return text

    s = text

    # 1) Collapse fenced code blocks: ```lang\n...\n```
    def _fence_sub(m):
        lang = (m.group(1) or "").strip().lower()
        body = m.group(2) or ""
        if mode == "remove":
            return _collapse(body, lang, ph_code)
        head, total = _summarize_code_head(body, keep_first)
        if head:
            head = head + "\n"
        return f"{head}{_collapse(body, lang, ph_code)}"
    s = _FENCE_RE.sub(_fence_sub, s)
    s = _TILDE_RE.sub(_fence_sub, s)

    # 2) Stack traces: collapse contiguous blocks
    lines = s.splitlines()
    out = []
    i = 0
    while i < len(lines):
        if _STACK_START_RE.match(lines[i]):
            j = i + 1
            while j < len(lines) and (lines[j].startswith("  File ") or lines[j].startswith("    ") or lines[j].strip()):
                # stop at empty separator after a traceback body
                if j > i and not lines[j].strip():
                    break
                j += 1
            block = "\n".join(lines[i:j])
            out.append(ph_stack.format(lines=_count_lines(block)))
            i = j
        else:
            out.append(lines[i])
            i += 1
    s = "\n".join(out)

    # 3) Diffs (---, +++, @@ headers) → collapse contiguous diff chunks
    lines = s.splitlines()
    out = []
    i = 0
    while i < len(lines):
        if _diff_hdr := _DIFF_HDR_RE.match(lines[i]):
            j = i + 1
            while j < len(lines) and (lines[j].startswith(("+", "-", " ")) or _DIFF_HDR_RE.match(lines[j])):
                j += 1
            block = "\n".join(lines[i:j])
            out.append(ph_diff.format(lines=_count_lines(block)))
            i = j
        else:
            out.append(lines[i])
            i += 1
    s = "\n".join(out)

    # 4) Indented code blocks (>=4 spaces) – collapse if at least min_indent_lines
    lines = s.splitlines()
    out = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("    "):
            j = i
            # grow while it's indented or empty but adjacent to indented region
            while j < len(lines) and (lines[j].startswith("    ") or (lines[j] == "" and j + 1 < len(lines) and lines[j+1].startswith("    "))):
                j += 1
            block = "\n".join(lines[i:j])
            total = _count_lines(block)
            if total >= min_indent_lines:
                if mode == "remove":
                    out.append(ph_code.format(lang="indent", lines=total))
                else:
                    head, _ = _summarize_code_head(block, keep_first)
                    out.append((head + "\n") if head else "")
                    out.append(ph_code.format(lang="indent", lines=total))
            else:
                out.append(block)
            i = j
        else:
            out.append(lines[i])
            i += 1
    s = "\n".join(out)

    # 5) One-line big JSON-like → collapse
    def _json_sub(m):
        body = m.group(0)
        # if tiny, keep it
        if len(body) <= 200:
            return body
        return ph_json.format(chars=len(body))
    s = _JSON_LIKE_RE.sub(_json_sub, s)

    # 6) Inline code: collapse long spans
    def _inline_sub(m):
        body = m.group(1)
        if len(body) <= max_inline_len:
            return f"`{body}`"
        return f"`…{len(body)} chars code…`"
    s = _INLINE_RE.sub(_inline_sub, s)

    # 7) Code-density heuristic: if too codey, aggressively prune codey lines
    density_gate = float(cfg.get("code_density_gate", 0.25))
    lines = s.splitlines()
    if lines:
        codey = sum(1 for ln in lines if _looks_code_line(ln))
        if codey / max(1, len(lines)) >= density_gate:
            # remove most code-like lines, keep human sentences
            kept = []
            for ln in lines:
                if _looks_code_line(ln) and len(ln) > 40:
                    # drop long code lines entirely
                    continue
                kept.append(ln)
            s = "\n".join(kept)

    # 8) Normalize whitespace
    s = _WHITESPACE_RE.sub("\n\n", s).strip()
    return s

def is_likely_code_goal(goal_text: str) -> bool:
    if not goal_text:
        return False
    gt = goal_text.lower()
    tokens = ("code", "python", "class ", "function", "bug", "traceback", "stack trace", "compile", "build error")
    return any(t in gt for t in tokens)
