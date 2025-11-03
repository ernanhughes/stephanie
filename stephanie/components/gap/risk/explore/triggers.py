from __future__ import annotations

import random
import textwrap
from typing import List


def t_counterfactual(goal:str)->str:
    return f"{goal}\n\nAssume one key premise is false. Under that counterfactual, reason to a plausible answer."

def t_boundary(goal:str)->str:
    return f"{goal}\n\nAnswer by challenging common assumptions. List 3 edge cases that break the usual recipe."

def t_transfer(goal:str)->str:
    return f"{goal}\n\nReframe using an analogy from a different field (e.g., ecology ↔ systems). Derive 2 concrete implications."

def t_reverse(goal:str)->str:
    return f"{goal}\n\nGive the answer that would be *least expected* under standard doctrine, then justify it."

def t_noise(goal:str)->str:
    return f"[Speculative Mode]\n{goal}\n\nIf unknown, hypothesize boldly. Tag each speculative claim with [SPEC] and provide a falsifiable check."

def t_timewarp(goal:str)->str:
    return f"{goal}\n\nPretend it is 2030. What changed that makes the prior best answer obsolete?"

def t_contradict(goal:str)->str:
    return f"{goal}\n\nFirst state the usual answer in 1 line. Then argue the opposite for 5 lines with evidence or plausible mechanisms."

TRIGGERS = [t_counterfactual, t_boundary, t_transfer, t_reverse, t_noise, t_timewarp, t_contradict]

def apply_triggers(goal: str, k: int = 4) -> List[str]:
    funcs = random.sample(TRIGGERS, k=min(k, len(TRIGGERS)))
    return [fn(goal) for fn in funcs]
