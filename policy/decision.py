# policy/decision.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict


class Decision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    REVIEW = "review"
    FREEZE = "freeze"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class PolicyDecision:
    decision: Decision
    reason: str
    signals: Dict[str, float]
    metadata: Optional[Dict] = None
