
def accept_margin_ratio(energy: float, tau: float) -> float:
    """
    Normalized margin relative to acceptance threshold.
    """
    return max(0.0, (tau - energy) / max(tau, 1e-6))
