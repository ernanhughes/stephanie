# stephanie/services/style_card_service.py
from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List, Optional

from stephanie.services.voice_profile_service import VoiceProfileService


def _norm_section(s: str) -> str:
    s = (s or "").strip().lower().replace("-", "_")
    aliases = {
        "intro": "introduction", "background": "related_work",
        "method": "methods", "results_and_discussion": "discussion"
    }
    return aliases.get(s, s)

class StyleCardService:
    """
    Retrieves section-type-specific style cards using the VoiceProfileService.
    Persists style cards into CaseBook for reuse/inspection and learns from champion selections.
    
    Features:
    - Section-type specific specialization
    - Persistent storage of style cards
    - Automatic learning from champion selections
    - Fallback to base style card when no specific style is available
    """
    def __init__(self, memory, logger, voice_profile: VoiceProfileService, config: Optional[Dict[str, Any]] = None):
        self.memory = memory
        self.logger = logger
        self.voice = voice_profile
        self.config = config or {}
        self.section_types = {"introduction","methods","results","discussion","abstract","conclusion","related_work","future_work"}
        self._style_cards: Dict[str, Dict[str, Any]] = {}
        self._moves_stats: Dict[str, Dict[str, float]] = {}  # EMA counts per section_type
        self._load_style_cards()

    def _load_style_cards(self):
        try:
            blob = self.memory.meta.get("style_cards_v1") or {}
            self._style_cards = blob.get("cards", {})
            self._moves_stats = blob.get("moves_stats", {})
            if self._style_cards:
                self.logger.log("StyleCardsLoaded", {"count": len(self._style_cards), "version": blob.get("version", 1)})
                return
        except Exception as e:
            self.logger.log("StyleCardsLoadError", {"error": str(e), "traceback": traceback.format_exc()})

        base = self.voice.style_card()
        now = time.time()
        for st in self.section_types:
            self._style_cards[st] = {**base, "section_type": st, "updated_at": now}
            self._moves_stats[st] = {m: 0.0 for m in base.get("moves", [])}
        self._save_style_cards()

    def _save_style_cards(self):
        try:
            self.memory.meta["style_cards_v1"] = {
                "version": 1,
                "saved_at": time.time(),
                "cards": self._style_cards,
                "moves_stats": self._moves_stats
            }
            self.logger.log("StyleCardsSaved", {"count": len(self._style_cards)})
        except Exception as e:
            self.logger.log("StyleCardsSaveError", {"error": str(e), "traceback": traceback.format_exc()})

    def get(self, section_type: str) -> Dict[str, Any]:
        st = _norm_section(section_type)
        if st not in self._style_cards:
            base = self.voice.style_card()
            self._style_cards[st] = {**base, "section_type": st, "updated_at": time.time()}
            self._moves_stats[st] = {m: 0.0 for m in base.get("moves", [])}
            self._save_style_cards()
            self.logger.log("StyleCardCreated", {"section_type": st, "from": "base"})
        return self._style_cards[st]

    def update_with_champion(self, section_type: str, champion_text: str, agent: str, alpha: float = 0.2):
        try:
            st = _norm_section(section_type)
            card = self.get(st)
            moves = card.get("moves", [])
            counts = self._count_moves(champion_text, moves)

            # EMA update of move stats
            stat = self._moves_stats.get(st, {m: 0.0 for m in moves})
            for m in moves:
                stat[m] = (1.0 - alpha)*stat.get(m, 0.0) + alpha*counts.get(m, 0.0)
            self._moves_stats[st] = stat

            # re-order moves by learned preference
            new_moves = sorted(moves, key=lambda m: stat[m], reverse=True)[:min(4, len(moves))]
            if new_moves != moves:
                card["moves"] = new_moves
                card["updated_at"] = time.time()
                self._style_cards[st] = card
                self._save_style_cards()

            self.logger.log("StyleCardUpdated", {"section_type": st, "agent": agent, "moves": card["moves"], "stats": stat})
        except Exception as e:
            self.logger.log("StyleCardUpdateError", {"error": str(e), "traceback": traceback.format_exc()})

    def _count_moves(self, text: str, moves: List[str]) -> Dict[str, int]:
        t = (text or "").lower()
        counts = {m: 0 for m in moves}
        def has_any(xs): return sum(1 for x in xs if x in t)
        for m in moves:
            if m == "analogy": counts[m] = has_any([" like ", "imagine ", "it's as if", "similar to", "resembles"])
            elif m == "contrast": counts[m] = has_any(["however", "on the other hand", " but ", " yet ", " nevertheless ", "in contrast"])
            elif m == "example": counts[m] = has_any(["for example", "e.g.", "such as", " like "])
            elif m == "steps": counts[m] = has_any([" step ", " first", " next", " then", " finally", " firstly", " secondly", " lastly"])
            elif m == "audience_check": counts[m] = has_any(["you can think", "if you're", "let's", "you might wonder", "imagine that", "picture this"])
        return counts

    def _load_style_cards(self):
        """Load style cards from persistent storage if available"""
        try:
            style_cards = self.memory.meta.get("style_cards", {})
            if style_cards:
                self._style_cards = style_cards
                self.logger.log("StyleCardsLoaded", {
                    "count": len(self._style_cards),
                    "section_types": list(self._style_cards.keys())
                })
                return
        except Exception as e:
            self.logger.log("StyleCardsLoadError", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        
        # Initialize with default style cards
        base_card = self.voice.style_card()
        self._style_cards = {
            section_type: {
                **base_card,
                "section_type": section_type,
                "moves": self._specialize_moves(section_type, base_card["moves"])
            }
            for section_type in self.section_types
        }
        self._save_style_cards()

    def _save_style_cards(self):
        """Save style cards to persistent storage"""
        try:
            self.memory.meta["style_cards"] = self._style_cards
            self.logger.log("StyleCardsSaved", {
                "count": len(self._style_cards),
                "section_types": list(self._style_cards.keys())
            })
        except Exception as e:
            self.logger.log("StyleCardsSaveError", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })

    def _specialize_moves(self, section_type: str, base_moves: List[str]) -> List[str]:
        """Specialize moves for a specific section type"""
        # Default specialization
        specialization = {
            "introduction": ["audience_check", "analogy", "example", "steps"],
            "methods": ["steps", "contrast", "example"],
            "results": ["contrast", "example", "steps"],
            "discussion": ["analogy", "audience_check", "contrast", "example"],
            "abstract": ["steps", "example", "audience_check"],
            "conclusion": ["audience_check", "analogy", "example"],
            "related_work": ["contrast", "example", "steps"],
            "future_work": ["audience_check", "analogy", "example"]
        }
        
        # Apply specialization if available
        if section_type in specialization:
            return specialization[section_type]
        
        # Fall back to base moves
        return base_moves

    def get(self, section_type: str) -> Dict[str, Any]:
        """Get style card for a section type, with fallbacks"""
        if section_type not in self._style_cards:
            # Try to find a close match
            for known_type in self.section_types:
                if known_type in section_type.lower():
                    return self._style_cards[known_type]
            # Fall back to default
            self.logger.log("StyleCardFallback", {
                "section_type": section_type,
                "fallback": "base"
            })
            return self._style_cards.get("introduction", self.voice.style_card())
        
        return self._style_cards[section_type]

    def update_with_champion(self, section_type: str, champion_text: str, agent: str):
        """Update style card based on champion selection"""
        try:
            # Get current style card
            current_card = self.get(section_type)
            
            # Analyze champion text for style improvements
            new_moves = self._analyze_moves(champion_text, current_card["moves"])
            
            # Update moves if there are improvements
            if new_moves != current_card["moves"]:
                self._style_cards[section_type]["moves"] = new_moves
                self._save_style_cards()
                self.logger.log("StyleCardUpdated", {
                    "section_type": section_type,
                    "old_moves": current_card["moves"],
                    "new_moves": new_moves,
                    "agent": agent
                })
        except Exception as e:
            self.logger.log("StyleCardUpdateError", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })

    def _analyze_moves(self, text: str, current_moves: List[str]) -> List[str]:
        """Analyze text to determine which moves are most effective"""
        # Count occurrences of each move type in text
        move_counts = {m: 0 for m in current_moves}
        t = text.lower()
        
        for move in current_moves:
            if move == "analogy":
                move_counts[move] = sum(1 for x in ["like ", "imagine ", "it's as if", "similar to", "resembles"] if x in t)
            elif move == "contrast":
                move_counts[move] = sum(1 for x in ["however", "on the other hand", "but", "yet", "nevertheless", "in contrast"] if x in t)
            elif move == "example":
                move_counts[move] = sum(1 for x in ["for example", "e.g.", "such as", "like", "such as"] if x in t)
            elif move == "steps":
                move_counts[move] = sum(1 for x in ["step", "first", "next", "then", "finally", "firstly", "secondly", "lastly"] if x in t)
            elif move == "audience_check":
                move_counts[move] = sum(1 for x in ["you can think", "if you're", "let's", "you might wonder", "imagine that", "picture this"] if x in t)
        
        # Sort moves by count
        sorted_moves = sorted(current_moves, key=lambda m: move_counts[m], reverse=True)
        
        # Return top 3-4 moves (or all if fewer)
        return sorted_moves[:min(4, len(sorted_moves))]