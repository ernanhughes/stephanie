# stephanie/arena/plugins/registry.py
from __future__ import annotations

from typing import Dict, Type

from stephanie.arena.plugins.interfaces import Play, Scorer

_PLAYS: Dict[str, Play] = {}
_SCORERS: Dict[str, Scorer] = {}

def register_play(play: Play):
    _PLAYS[play.name] = play
    return play

def get_play(name: str) -> Play:
    return _PLAYS[name]

def list_plays() -> Dict[str, Play]:
    return dict(_PLAYS)

def register_scorer(scorer: Scorer):
    _SCORERS[scorer.name] = scorer
    return scorer

def get_scorer(name: str) -> Scorer:
    return _SCORERS[name]

def list_scorers() -> Dict[str, Scorer]:
    return dict(_SCORERS)
