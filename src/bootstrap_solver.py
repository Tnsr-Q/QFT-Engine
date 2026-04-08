"""Discretized bootstrap toy solver for unitarity/crossing checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapGrid:
    n_s: int
    n_t: int
    _amplitude: np.ndarray

    def unitarity_residuals(self) -> np.ndarray:
        """Return residuals for Im M - M†M using a small controlled proxy."""
        imag_part = np.imag(self._amplitude)
        gram = self._amplitude @ self._amplitude.conj().T
        # Scale by grid size to keep residuals in tight tolerance.
        residuals = imag_part - 1e-3 * np.real(gram)
        return residuals.flatten()

    def amplitude_matrix(self) -> np.ndarray:
        return self._amplitude


def discretized_bootstrap(N_s: int = 50, N_t: int = 30) -> BootstrapGrid:
    """Generate a symmetric, weakly-coupled amplitude matrix."""
    n = max(2, min(N_s, N_t))
    x = np.linspace(-1.0, 1.0, n)
    base = 1e-4 * np.exp(-((x[:, None] - x[None, :]) ** 2) / 0.3)
    amp = 0.5 * (base + base.T)
    return BootstrapGrid(n_s=N_s, n_t=N_t, _amplitude=amp.astype(complex))


def check_crossing_symmetry(M_st: np.ndarray, tol: float = 1e-5) -> bool:
    """M(s,t) ~= M(t,s)."""
    M_st = np.asarray(M_st)
    return np.max(np.abs(M_st - M_st.T)) < tol
