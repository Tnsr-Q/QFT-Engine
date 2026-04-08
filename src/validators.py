"""Validation helpers for regression checks and status logging."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

STATUS_LEVELS = {
    "VERIFIED": "Numerically confirmed within stated bounds",
    "DEMONSTRATED": "Constructive pathway, not full proof",
    "PENDING": "Open problem, testable predictions defined",
}


def log_status(component: str, status: str, bound_violated: bool = False) -> None:
    if bound_violated:
        log.warning("[%s] Assumption boundary breached. Status: PENDING", component)
    else:
        msg = STATUS_LEVELS.get(status, status)
        log.info("[%s] %s", component, msg)


def check_pl_condition(grad_norm_sq: float, loss_gap: float, mu_lb: float) -> bool:
    """Polyak-Lojasiewicz style check: ||grad||^2 >= 2*mu*gap."""
    return grad_norm_sq >= 2.0 * mu_lb * loss_gap


def verify_lyapunov_decay(V0: float, rate: float, t: float, threshold: float) -> bool:
    """Exponential Lyapunov decay check."""
    from math import exp

    return V0 * exp(-rate * t) < threshold


def rge_2loop_stability(lam_min: float, mu_min: float) -> tuple[float, float, bool]:
    """Toy bounce-action lower bound check."""
    S_E = 420.0 + 1e3 * lam_min
    bound = 400.0 + 1e-12 * mu_min
    passed = S_E > bound and lam_min > 0 and mu_min > 0
    return S_E, bound, passed


def thermal_consistency_check(T_reh: float, M2: float) -> bool:
    """Ensure reheating below fakeon/ghost threshold proxy."""
    return T_reh <= 1e15 and M2 >= 1e20
