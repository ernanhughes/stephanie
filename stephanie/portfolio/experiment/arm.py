# stephanie/portfolio/experiment/arm.py
"""Experimental arms A–E (§Experiment 001). A is the identical primary leg of B–E."""
from __future__ import annotations

from enum import Enum


class ExperimentArm(str, Enum):
    A_PRIMARY_ONLY = "A"
    B_SAME_FAMILY_CRITIC = "B"
    C_FRONTIER_REVIEWER = "C"
    D_BREADTH = "D"
    E_FULL = "E"


ARM_PURPOSE = {
    ExperimentArm.A_PRIMARY_ONLY: "baseline",
    ExperimentArm.B_SAME_FAMILY_CRITIC: "critique value with likely correlated errors",
    ExperimentArm.C_FRONTIER_REVIEWER: "high-quality independence",
    ExperimentArm.D_BREADTH: "marginal information per cost",
    ExperimentArm.E_FULL: "whether combining all roles is worth it",
}
