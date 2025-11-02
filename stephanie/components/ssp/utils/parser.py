"""
Parser Utilities for SSP Components (text-only, line-oriented)

Robust to:
- Empty values after a key (e.g., "rationale:" + following lines)
- Multi-line blocks for rationale/question
- Extra code fences (``` / ~~~), quotes, commas
- Key synonyms (e.g., score/confidence/grade; verifiability/verification)
- Duplicate keys (keeps the last occurrence)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Pattern, Tuple

# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def _strip_code_fences(s: str) -> str:
    # Remove common code-fence wrappers
    s = s.strip()
    s = re.sub(r"^```.*?\n|\n```$|^~~~.*?\n|\n~~~$", "", s, flags=re.S)
    # Remove single-line fences too
    s = re.sub(r"^```.*?```$|^~~~.*?~~~$", "", s, flags=re.S)
    return s.strip()

def _clean_text(val: str) -> str:
    s = (val or "").strip()
    if len(s) >= 2:
        for lq, rq in [('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")]:
            if s.startswith(lq) and s.endswith(rq):
                s = s[1:-1].strip()
                break
    return s

def _int_in(val: str) -> Optional[int]:
    if not val:
        return None
    m = re.search(r"(-?\d+)", val)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _clamp01_int100(n: Optional[int], default: int = 0) -> int:
    if n is None:
        return default
    return max(0, min(100, int(n)))

def _bool_in(val: str) -> Optional[bool]:
    v = (val or "").strip().lower()
    if v in ("true", "yes", "1", "on", "valid", "correct"):
        return True
    if v in ("false", "no", "0", "off", "invalid", "incorrect"):
        return False
    return None

# Matches key: value  (value may be empty)
_LINE_RE: Pattern = re.compile(
    r'^\s*"?(?P<key>[A-Za-z_][A-Za-z0-9_\- ]*)"?\s*[:=]\s*(?P<value>.*?)(?:\s*,\s*)?$'
)

def _scan_keyed_lines(text: str) -> List[Tuple[str, str]]:
    """
    Returns list of (key_lower, value_raw) in order of appearance.
    Keeps empty values; does not merge duplicates.
    """
    pairs: List[Tuple[str, str]] = []
    for line in (text.splitlines() if text else []):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        k = m.group("key").strip().lower()
        v = m.group("value").strip()
        pairs.append((k, v))
    return pairs

def _capture_block(text: str, start_key: str, stop_keys: Tuple[str, ...]) -> Optional[str]:
    """
    Capture a multi-line block starting at 'start_key:' (which may have empty value)
    and continuing until the next stop key. Returns cleaned text or None.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    start_idx = None
    key_prefix = f"{start_key.lower()}:"
    stop_prefixes = tuple(f"{k.lower()}:" for k in stop_keys)

    for i, ln in enumerate(lines):
        low = ln.strip().lower()
        if low.startswith(key_prefix):
            start_idx = i
            break
    if start_idx is None:
        return None

    # First line: take the tail after 'key:'
    first = lines[start_idx]
    tail = first[first.lower().find(":")+1:].strip()
    chunks = [_clean_text(tail)] if tail else []

    # Subsequent lines until a stop key
    for j in range(start_idx + 1, len(lines)):
        lowj = lines[j].strip().lower()
        if any(lowj.startswith(p) for p in stop_prefixes):
            break
        chunks.append(lines[j].strip())

    text_out = " ".join([c for c in chunks if c]).strip()
    return text_out or None

# ---------------------------------------------------------------------
# Proposer
# ---------------------------------------------------------------------

def parse_proposer_lines(text: str) -> Dict[str, Any]:
    """
    Expected keys (any case): rationale, difficulty, verifiability, question
    Returns dict with fields + ok flag.
    """
    raw = _strip_code_fences((text or "").strip())
    out: Dict[str, Any] = {
        "rationale": "",
        "difficulty": 0,
        "verifiability": 0,
        "question": "",
        "raw": raw,
        "ok": False,
    }
    if not raw:
        return out

    pairs = _scan_keyed_lines(raw)
    # Prefer the LAST occurrence of each key
    seen: Dict[str, str] = {}
    for k, v in pairs:
        seen[k] = v  # overwrite earlier entries

    # Accept common synonyms
    diff_val = seen.get("difficulty")
    verif_val = seen.get("verifiability") or seen.get("verification")
    rationale_val = seen.get("rationale")
    question_val = seen.get("question")

    # Parse numbers
    out["difficulty"] = _clamp01_int100(_int_in(diff_val), default=0)
    out["verifiability"] = _clamp01_int100(_int_in(verif_val), default=0)

    # Clean single-line values (may be empty)
    out["rationale"] = _clean_text(rationale_val or "")
    out["question"] = _clean_text(question_val or "")

    # Multi-line recovery (rationale & question)
    if not out["rationale"]:
        blk = _capture_block(raw, "rationale", ("difficulty", "verifiability", "verification", "question"))
        if blk:
            out["rationale"] = blk
    if not out["question"]:
        blk = _capture_block(raw, "question", ("rationale", "difficulty", "verifiability", "verification"))
        if blk:
            out["question"] = blk

    # ok if we got a question; difficulty/verifiability default safely
    out["ok"] = bool(out["question"].strip())
    return out

# ---------------------------------------------------------------------
# Solver / Verifier (text, line-oriented)
# ---------------------------------------------------------------------

def parse_solver_lines(text: str) -> Dict[str, Any]:
    """
    Solver format:
      rationale: <text>
      score: <0-100>   (aka confidence/grade)
      result: <final answer paragraph>
    """
    raw = _strip_code_fences((text or "").strip())
    out = {"rationale": "", "score": 0, "result": "", "raw": raw, "ok": False}
    if not raw:
        return out

    pairs = _scan_keyed_lines(raw)
    seen: Dict[str, str] = {}
    for k, v in pairs:
        seen[k] = v

    # handle synonyms for score
    score_val = seen.get("score") or seen.get("confidence") or seen.get("grade")
    out["score"] = _clamp01_int100(_int_in(score_val), default=0)
    out["rationale"] = _clean_text(seen.get("rationale", ""))
    out["result"] = _clean_text(seen.get("result", ""))

    # Multi-line recovery for rationale/result
    if not out["rationale"]:
        blk = _capture_block(raw, "rationale", ("score", "confidence", "grade", "result"))
        if blk:
            out["rationale"] = blk
    if not out["result"]:
        blk = _capture_block(raw, "result", ("rationale", "score", "confidence", "grade"))
        if blk:
            out["result"] = blk

    out["ok"] = bool(out["result"].strip())
    return out

def parse_verifier_lines(text: str) -> Dict[str, Any]:
    """
    Verifier format (strict grader):
      rationale: <text>
      score: <0-100>   (85+ strong; 60–84 partial; <60 weak)
      result: <valid|invalid>
    """
    raw = _strip_code_fences((text or "").strip())
    out = {"rationale": "", "score": 0, "result": "", "raw": raw, "ok": False, "is_valid": False}
    if not raw:
        return out

    pairs = _scan_keyed_lines(raw)
    seen: Dict[str, str] = {}
    for k, v in pairs:
        seen[k] = v

    out["rationale"] = _clean_text(seen.get("rationale", ""))
    out["score"] = _clamp01_int100(_int_in(seen.get("score") or seen.get("grade")), default=0)

    res = (seen.get("result") or "").strip().lower()
    if not res:
        # backward compat: some prompts may emit 'answer:'
        res = (seen.get("answer") or "").strip().lower()
    out["result"] = res
    bv = _bool_in(res)
    out["is_valid"] = bool(bv) if bv is not None else (res.startswith("valid"))

    # Multi-line recovery for rationale (rare but safe)
    if not out["rationale"]:
        blk = _capture_block(raw, "rationale", ("score", "grade", "result", "answer"))
        if blk:
            out["rationale"] = blk

    out["ok"] = bool(out["rationale"].strip()) and (out["result"] != "")
    return out

# ---------------------------------------------------------------------
# Evidence parsing (search snippets)
# ---------------------------------------------------------------------

def parse_solution_search_lines(text: str, top_k: int = 3) -> List[str]:
    raw = _strip_code_fences((text or "").strip())
    if not raw:
        return []

    snips: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("snippet:"):
            snip = _clean_text(line.split(":", 1)[1])
            if snip:
                snips.append(snip)

    if not snips:
        # numbered
        snips = [ _clean_text(m) for m in re.findall(r"^\s*\d+\.\s+(.+)$", raw, re.M) ][:top_k]

    if not snips:
        # bullets
        snips = [ _clean_text(m) for m in re.findall(r"^\s*[-*•]\s+(.+)$", raw, re.M) ][:top_k]

    if not snips and len(raw.split()) > 10 and "." in raw:
        # sentences
        snips = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()][:top_k]

    # de-dup while preserving order
    seen = set()
    deduped: List[str] = []
    for s in snips:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped[:top_k]



_TRILINE: re.Pattern = re.compile(
    r'^\s*(rationale|score|result)\s*[:=]\s*(.+?)\s*$',
    re.IGNORECASE
)

def _clean_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2:
        pairs = [('"','"'), ("'","'"), ('“','”'), ('‘','’')]
        for l, r in pairs:
            if s.startswith(l) and s.endswith(r):
                return s[1:-1].strip()
    return s

def _float_in(s: str) -> Optional[float]:
    m = re.search(r'(-?\d+(?:\.\d+)?)', s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def parse_three_lines(text: str, *, default_score: float = 50.0) -> Dict[str, Any]:
    raw = (text or "").strip()
    out = {
        "rationale": "",
        "score": float(default_score),
        "result": "",
        "raw": raw,
        "ok": False,
    }
    if not raw:
        return out

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for ln in lines:
        m = _TRILINE.match(ln)
        if not m:
            continue
        key = m.group(1).lower()
        val = _clean_quotes(m.group(2))
        if key == "rationale":
            out["rationale"] = val
        elif key == "score":
            f = _float_in(val)
            if f is not None:
                out["score"] = max(0.0, min(100.0, float(f)))
        elif key == "result":
            out["result"] = val

    # Tolerate single-line raw responses (treat as result)
    if not out["result"] and len(lines) == 1 and ":" not in lines[0]:
        out["result"] = lines[0].strip()

    out["ok"] = bool(out["result"])
    return out
