# stephanie/expository/heuristics.py
import math
import re
from typing import Dict

SECTION_ALLOW = {"introduction","background","related work","preliminaries","method","method overview","problem setup"}
RHEX = {"standard","commonly used","previous work","we build on","classically","has been widely used","the usual","typical"}
NOVELTY = {"novel","we propose","we present","first","new method","state-of-the-art"}

CITE_PATTERNS = [r"\[\d+(?:\s*,\s*\d+)*\]", r"\([A-Z][A-Za-z\-]+ et al\., \d{4}\)"]
CROSS_REF = [r"\bsee (sec|section|fig|table)\b", r"\b(§|Fig\.|Tab\.)\s*\d+"]
MATH_LIKE = [r"[A-Za-z]\s*=", r"\b\d+\s*[\+\-\*/]\s*\d+", r"\bO\([nN]\)"]

def _count(rx_list, text: str) -> int:
    return sum(len(re.findall(rx, text, flags=re.IGNORECASE)) for rx in rx_list)

def citation_density(text: str, sentences: int) -> float:
    s = max(1, sentences)
    return _count(CITE_PATTERNS, text) / s

def crossref_density(text: str, sentences: int) -> float:
    s = max(1, sentences)
    return _count(CROSS_REF, text) / s

def symbol_density(text: str, tokens: int) -> float:
    t = max(1, tokens)
    return _count(MATH_LIKE, text) / t

def lex_hit_ratio(text: str, lex: set) -> float:
    tl = text.lower()
    return sum(1 for k in lex if k in tl) / max(1, len(lex))

def flesch_reading_ease(text: str) -> float:
    # light estimate; replace with your existing util if you have one
    words = re.findall(r"[A-Za-z]+", text)
    sents = max(1, len(re.findall(r"[.!?]+", text)))
    syll = sum(max(1, len(re.findall(r"[aeiouyAEIOUY]+", w))) for w in words)
    W = max(1, len(words))
    return 206.835 - 1.015 * (W / sents) - 84.6 * (syll / W)

def mid_band(x, lo, hi):
    # high when x in [lo, hi], falls to 0 outside
    if x < lo or x > hi: return 0.0
    if hi == lo: return 1.0
    # smoother triangular
    mid = (lo + hi) / 2
    return 1.0 - abs(x - mid) / (hi - lo)

def compute_features(text: str, section: str) -> Dict:
    tokens = len(re.findall(r"\S+", text))
    sents = max(1, len(re.findall(r"[.!?]+", text)))
    f = {
        "section_cue": 1.0 if section.lower() in SECTION_ALLOW else 0.0,
        "rhet_cues": lex_hit_ratio(text, RHEX),
        "cite_density": citation_density(text, sents),
        "novelty_lex": lex_hit_ratio(text, NOVELTY),
        "crossref_density": crossref_density(text, sents),
        "symbol_density": symbol_density(text, tokens),
        "len_tokens": tokens,
        "readability": flesch_reading_ease(text),
        "sentences": sents,
    }
    return f

def expository_score(f: Dict) -> float:
    return (
        f["section_cue"] +
        f["rhet_cues"] +
        mid_band(f["cite_density"], 0.2, 0.9) +
        (1 - f["novelty_lex"]) +
        (1 - f["crossref_density"]) +
        (1 - f["symbol_density"])
    ) / 6.0

def bloggability_score(text: str, f: Dict, cfg) -> float:
    ok_read = 1.0 if f["readability"] >= cfg.min_readability else 0.0
    ok_len  = 1.0 if (cfg.min_len_tokens <= f["len_tokens"] <= cfg.max_len_tokens) else 0.0
    ok_xref = 1.0 if f["crossref_density"] <= cfg.max_crossref_density else 0.0
    ok_sym  = 1.0 if f["symbol_density"] <= cfg.max_symbol_density else 0.0
    return (ok_read + ok_len + ok_xref + ok_sym) / 4.0

