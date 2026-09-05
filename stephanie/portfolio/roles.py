# stephanie/portfolio/roles.py
"""Portfolio roles are first-class (§3). Enables model x task x role learning."""
from __future__ import annotations

from enum import Enum


class PortfolioRole(str, Enum):
    PRIMARY = "primary"
    INDEPENDENT_REVIEWER = "independent_reviewer"
    CRITIC = "critic"
    BREADTH = "breadth"
    VERIFIER = "verifier"
    SYNTHESIZER = "synthesizer"


# Roles that must NEVER receive another candidate's answer (§8).
INDEPENDENT_ROLES = frozenset({PortfolioRole.PRIMARY, PortfolioRole.INDEPENDENT_REVIEWER, PortfolioRole.BREADTH})

# Roles that legitimately receive candidate answers for anchored critique.
ANCHORED_ROLES = frozenset({PortfolioRole.CRITIC, PortfolioRole.VERIFIER, PortfolioRole.SYNTHESIZER})
