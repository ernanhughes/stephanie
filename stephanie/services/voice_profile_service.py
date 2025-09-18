# stephanie/services/voice_profile_service.py
from __future__ import annotations
import math
import re
import time
import traceback
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

def _nz(x, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    Safe numeric clamp:
    - casts x to float
    - if NaN/inf/invalid -> returns midpoint of [lo, hi]
    - otherwise clamps into [lo, hi]
    """
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return (lo + hi) / 2.0
        return min(hi, max(lo, v))
    except Exception:
        return (lo + hi) / 2.0

def _safe_list(x):
    return x if isinstance(x, list) else []


class VoiceProfileService:
    """
    Builds a 'voice profile' from chat history and scores new text against it.
    - centroid: embedding centroid of your chat turns
    - stylometry: μ/Σ for simple style features
    - style_card: light-weight rhetorical "moves" + key-phrases for guidance

    Features:
    - Graceful fallback for missing embeddings
    - Incremental profile updates
    - Better key phrase extraction
    - Error handling and metrics tracking
    """

    FUNC_WORDS = set(
        """
        the of and to a in that it is was for on are as with be by this i you
        not or have from at we they an which but all can has their more if do
        will just so than into about then out up over like also one
    """.split()
    )

    def __init__(
        self, memory, logger, config: Optional[Dict[str, Any]] = None
    ):
        self.memory = memory
        self.logger = logger
        self.config = config or {}
        self._profile: Dict[str, Any] = {}
        self._last_updated = 0
        self._profile_version = 0
        self._load_profile()

    def _load_profile(self):
        """Load profile from persistent storage if available"""
        try:
            # Check if there's a stored profile
            profile_data = self.memory.meta.get("voice_profile")
            if profile_data:
                self._profile = profile_data
                self._last_updated = profile_data.get("last_updated", 0)
                self._profile_version = profile_data.get("version", 0)
                self.logger.log(
                    "VoiceProfileLoaded",
                    {
                        "version": self._profile_version,
                        "last_updated": self._last_updated,
                        "phrases": len(self._profile.get("key_phrases", [])),
                        "moves": self._profile.get("style_card", {}).get(
                            "moves", []
                        ),
                    },
                )
                return
        except Exception as e:
            self.logger.log(
                "VoiceProfileLoadError",
                {"error": str(e), "traceback": traceback.format_exc()},
            )

        # Initialize empty profile
        self._profile = {
            "centroid": [],
            "stylometry_mu": {},
            "stylometry_sigma": {},
            "style_card": {
                "tone": "conversational",
                "moves": [
                    "analogy",
                    "contrast",
                    "example",
                    "steps",
                    "audience_check",
                ],
                "key_phrases": [],
            },
            "last_updated": time.time(),
            "version": 0,
        }

    def _save_profile(self):
        """Save profile to persistent storage"""
        try:
            self._profile["last_updated"] = time.time()
            self._profile["version"] = self._profile_version
            self.memory.meta["voice_profile"] = self._profile
            self.logger.log(
                "VoiceProfileSaved",
                {
                    "version": self._profile_version,
                    "last_updated": self._profile["last_updated"],
                    "phrases": len(self._profile.get("key_phrases", [])),
                    "moves": self._profile.get("style_card", {}).get(
                        "moves", []
                    ),
                },
            )
        except Exception as e:
            self.logger.log(
                "VoiceProfileSaveError",
                {"error": str(e), "traceback": traceback.format_exc()},
            )

    def build_from_chats(
        self, chats: List[str], top_k_phrases: int = 40
    ) -> Dict[str, Any]:
        if not chats:
            self.logger.log(
                "VoiceProfileBuildSkipped", {"reason": "no_chats_provided"}
            )
            return self._profile
        try:
            embs = []
            for t in chats:
                if not t or not t.strip():
                    continue
                try:
                    embs.append(self.memory.embedding.get_or_create(t))
                except Exception as e:
                    self.logger.log(
                        "EmbeddingFailed", {"text": t[:100], "error": str(e)}
                    )

            if not embs:
                self.logger.log(
                    "VoiceProfileBuildFailed", {"reason": "no_embeddings"}
                )
                return self._profile

            dim = len(embs[0])
            centroid = [
                sum(e[i] for e in embs) / len(embs) for i in range(dim)
            ]

            feats = [self._features(t) for t in chats if t and t.strip()]
            keys = feats[0].keys()
            mu = {k: sum(f[k] for f in feats) / len(feats) for k in keys}
            # add epsilon to sigma to avoid 0
            sigma = {
                k: max(
                    1e-6,
                    math.sqrt(
                        sum((f[k] - mu[k]) ** 2 for f in feats)
                        / max(1, len(feats) - 1)
                    ),
                )
                for k in keys
            }

            key_phrases = self._key_phrases(chats, top_k=top_k_phrases)
            moves = [
                "analogy",
                "contrast",
                "example",
                "steps",
                "audience_check",
            ]

            self._profile = {
                "centroid": centroid,
                "stylometry_mu": mu,
                "stylometry_sigma": sigma,
                "style_card": {
                    "tone": "conversational",
                    "moves": moves,
                    "key_phrases": key_phrases,
                },
                "last_updated": time.time(),
                "version": self._profile_version + 1,
            }
            self._profile_version += 1
            self._save_profile()
            self.logger.log(
                "VoiceProfileBuilt",
                {
                    "phrases": len(key_phrases),
                    "version": self._profile_version,
                    "chats_processed": len(chats),
                },
            )
            return self._profile
        except Exception as e:
            self.logger.log(
                "VoiceProfileBuildError",
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            return self._profile

    def update_with_chats(
        self,
        new_chats: List[str],
        top_k_phrases: int = 40,
        decay: float = 0.95,
    ):
        """Online update: exponential decay of centroid and μ/σ; append phrases."""
        if not new_chats:
            return
        try:
            # centroid
            for t in new_chats:
                if not t or not t.strip():
                    continue
                try:
                    v = self.memory.embedding.get_or_create(t)
                except Exception:
                    continue
                c = _safe_list(self._profile.get("centroid")) or v
                if len(c) != len(v):  # reset if dimension mismatch
                    c = v
                self._profile["centroid"] = [
                    decay * c[i] + (1.0 - decay) * v[i] for i in range(len(v))
                ]

            # stylometry μ/σ
            feats = [self._features(t) for t in new_chats if t and t.strip()]
            if feats:
                mu = dict(self._profile.get("stylometry_mu", {}))
                sg = dict(self._profile.get("stylometry_sigma", {}))
                keys = feats[0].keys()
                for k in keys:
                    x_bar = sum(f[k] for f in feats) / len(feats)
                    mu[k] = decay * mu.get(k, x_bar) + (1.0 - decay) * x_bar
                    # rough sigma update: decay old, blend new dispersion
                    var_new = sum((f[k] - mu[k]) ** 2 for f in feats) / max(
                        1, len(feats) - 1
                    )
                    sg[k] = math.sqrt(
                        max(
                            1e-6,
                            decay * (sg.get(k, math.sqrt(var_new)) ** 2)
                            + (1.0 - decay) * var_new,
                        )
                    )
                self._profile["stylometry_mu"] = mu
                self._profile["stylometry_sigma"] = sg

            # phrases: merge & cap
            phrases = self._key_phrases(new_chats, top_k=top_k_phrases)
            base = set(
                _safe_list(
                    self._profile.get("style_card", {}).get("key_phrases", [])
                )
            )
            merged = list((base | set(phrases)))[
                : max(len(base), top_k_phrases)
            ]
            self._profile.setdefault("style_card", {}).setdefault(
                "key_phrases", []
            )
            self._profile["style_card"]["key_phrases"] = merged

            self._profile["last_updated"] = time.time()
            self._profile_version += 1
            self._save_profile()
            self.logger.log(
                "VoiceProfileUpdated",
                {
                    "added_chats": len(new_chats),
                    "phrases_now": len(merged),
                    "version": self._profile_version,
                },
            )
        except Exception as e:
            self.logger.log(
                "VoiceProfileUpdateError",
                {"error": str(e), "traceback": traceback.format_exc()},
            )

    def score_text(self, text: str) -> Dict[str, float]:
        if (
            not text
            or not text.strip()
            or not self._profile
            or not self._profile.get("centroid")
        ):
            return {
                "VS": 0.5,
                "VS1_embed": 0.5,
                "VS2_style": 0.5,
                "VS3_moves": 0.5,
                "text_sample": (text or "")[:200],
            }
        try:
            # VS1
            try:
                emb = self.memory.embedding.get_or_create(text)
                sim = self._cosine(emb, self._profile["centroid"])
                VS1 = (sim + 1.0) / 2.0
            except Exception:
                VS1 = 0.5

            # VS2
            x = self._features(text)
            mu = self._profile.get("stylometry_mu", {})
            sg = self._profile.get("stylometry_sigma", {})
            z2, k = 0.0, 0
            for key, xv in x.items():
                if key in mu and key in sg and sg[key] > 0:
                    z2 += ((xv - mu[key]) / sg[key]) ** 2
                    k += 1
            VS2 = math.exp(-0.5 * z2 / max(1, k)) if k > 0 else 0.5

            # VS3
            moves = self._profile.get("style_card", {}).get(
                "moves",
                ["analogy", "contrast", "example", "steps", "audience_check"],
            )
            VS3 = self._moves_score(text, moves)

            VS = 0.5 * VS1 + 0.3 * VS2 + 0.2 * VS3
            return {
                "VS": _nz(VS),
                "VS1_embed": _nz(VS1),
                "VS2_style": _nz(VS2),
                "VS3_moves": _nz(VS3),
                "text_sample": text[:200],
            }
        except Exception as e:
            self.logger.log(
                "VoiceScoreError",
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            return {
                "VS": 0.5,
                "VS1_embed": 0.5,
                "VS2_style": 0.5,
                "VS3_moves": 0.5,
                "text_sample": text[:200],
            }

    def style_card(self) -> Dict[str, Any]:
        """Get the current style card"""
        if not self._profile or not self._profile.get("style_card"):
            return {
                "tone": "conversational",
                "moves": [
                    "analogy",
                    "contrast",
                    "example",
                    "steps",
                    "audience_check",
                ],
                "key_phrases": [],
            }
        return self._profile["style_card"]

    def _cosine(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1e-8
        nb = math.sqrt(sum(y * y for y in b)) or 1e-8
        return dot / (na * nb)

    def _features(self, text: str) -> Dict[str, float]:
        """Calculate stylometric features for text"""
        sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        tokens = re.findall(r"[A-Za-z']+", text)
        words = [w.lower() for w in tokens]
        n_words = max(1, len(words))
        n_sents = max(1, len(sents))
        func_count = sum(1 for w in words if w in self.FUNC_WORDS)
        puncts = re.findall(r"[,:;\-—()]", text)
        caps = sum(1 for w in re.findall(r"\b[A-Z][a-z]+\b", text))

        # Calculate readability metrics
        avg_word_length = (
            sum(len(w) for w in words) / n_words if n_words > 0 else 0
        )
        avg_sent_length = n_words / n_sents if n_sents > 0 else 0

        return {
            "avg_sent_len": avg_sent_length,
            "func_word_ratio": func_count / n_words if n_words > 0 else 0,
            "punct_per_word": len(puncts) / n_words if n_words > 0 else 0,
            "caps_per_sent": caps / n_sents if n_sents > 0 else 0,
            "avg_word_len": avg_word_length,
            "sentences": n_sents,
            "words": n_words,
        }

    def _key_phrases(self, texts: List[str], top_k: int = 40) -> List[str]:
        """Extract key phrases from text corpus"""
        counts = Counter()
        for t in texts:
            # Clean text
            t = re.sub(r"[^a-z0-9\s']", " ", t.lower())
            words = t.split()

            # Extract n-grams (up to 3 words)
            for n in range(1, 4):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i : i + n])
                    # Skip single words that are too short
                    if n == 1 and len(phrase) < 3:
                        continue
                    counts[phrase] += 1

        # Filter out common stop words and very short phrases
        filtered = []
        for phrase, count in counts.most_common():
            # Skip stop words and very short phrases
            if (
                len(phrase) < 3
                or phrase
                in [
                    "the",
                    "of",
                    "and",
                    "to",
                    "a",
                    "in",
                    "that",
                    "it",
                    "is",
                    "was",
                    "for",
                    "on",
                    "are",
                    "as",
                    "with",
                    "be",
                    "by",
                    "this",
                    "i",
                    "you",
                    "not",
                    "or",
                    "have",
                    "from",
                    "at",
                    "we",
                    "they",
                    "an",
                    "which",
                    "but",
                    "all",
                    "can",
                    "has",
                    "their",
                    "more",
                    "if",
                    "do",
                    "will",
                    "just",
                    "so",
                    "than",
                    "into",
                    "about",
                    "then",
                    "out",
                    "up",
                    "over",
                    "like",
                    "also",
                    "one",
                ]
                or any(
                    w
                    in [
                        "the",
                        "of",
                        "and",
                        "to",
                        "a",
                        "in",
                        "that",
                        "it",
                        "is",
                        "was",
                    ]
                    for w in phrase.split()
                )
            ):
                continue
            filtered.append(phrase)

        return filtered[:top_k]

    def _moves_score(self, text: str, moves: List[str]) -> float:
        """Calculate score for rhetorical moves presence"""
        t = text.lower()
        score = 0.0
        checks = {
            "analogy": lambda: any(
                x in t
                for x in [
                    "like ",
                    "imagine ",
                    "it's as if",
                    "similar to",
                    "resembles",
                ]
            ),
            "contrast": lambda: any(
                x in t
                for x in [
                    "however",
                    "on the other hand",
                    "but",
                    "yet",
                    "nevertheless",
                    "in contrast",
                ]
            ),
            "example": lambda: "for example" in t
            or "e.g." in t
            or "such as" in t
            or "like" in t
            or "such as" in t,
            "steps": lambda: bool(
                re.search(
                    r"\b(step|first|next|then|finally|firstly|secondly|lastly)\b",
                    t,
                )
            ),
            "audience_check": lambda: any(
                x in t
                for x in [
                    "you can think",
                    "if you're",
                    "let's",
                    "you might wonder",
                    "imagine that",
                    "picture this",
                ]
            ),
        }
        per = 1.0 / max(1, len(moves))
        for m in moves:
            try:
                if checks.get(m, lambda: False)():
                    score += per
            except Exception:
                pass
        return min(1.0, score)
