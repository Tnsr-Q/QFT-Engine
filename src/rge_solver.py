from dataclasses import dataclass
from typing import Dict, Optional

import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
from scipy.integrate import solve_ivp

jax.config.update("jax_enable_x64", True)

from src.tolerance.dynamic_ledger import DynamicToleranceLedger
from src.tolerance.regime_detector import RegimeDetector

# Constants
PI = np.pi
T16 = 16.0 * PI**2


@dataclass
class JAXODESolution:
    """Minimal solve_ivp-like container for JAX-integrated trajectories."""

    t: np.ndarray
    y: np.ndarray
    success: bool
    nfev: int


class SIQGRGESolver:
    """
    2-Loop RGE system for Scale-Invariant Quadratic Gravity + SM.
    Matches Extended Lemma 2 & S.3 β-function closure.
    """

    def __init__(
        self,
        mu_start: float = 173.1,
        mu_end: float = 2.4e23,
        ledger: Optional[DynamicToleranceLedger] = None,
        detector: Optional[RegimeDetector] = None,
    ):
        self.mu_start = mu_start  # GeV (top mass scale)
        self.mu_end = mu_end  # GeV (fakeon threshold M2)
        self.t_span = (np.log(mu_start), np.log(mu_end))
        self.ledger = ledger
        self.detector = detector or RegimeDetector(M2_GeV=mu_end)
        self.state: Dict[str, float] = {"energy_scale": mu_start}

    @staticmethod
    def _beta_f2(f2: float, lam_HS: float, xi_H: float) -> float:
        """Return raw loop coefficients for β_{f2}; rhs() applies the overall /T16 normalization."""
        b1 = -(133.0 / 20.0) * f2**3
        b2_grav = (5196.0 / 5.0) / T16 * f2**5
        b2_sm = -12.0 * lam_HS * xi_H**2 / T16 * f2**3
        return b1 + b2_grav + b2_sm

    def rhs(self, t: float, g: np.ndarray) -> np.ndarray:
        """RGE vector field dg/dt = β(g)"""
        _ = t
        lam_H, lam_S, lam_HS, y_t, g1, g2, g3, f2, xi_H = g

        b_lam_H = (
            24 * lam_H**2
            + 0.5 * lam_HS**2
            - 6 * y_t**4
            + 0.375 * (2 * g2**4 + (g1**2 + g2**2) ** 2)
            + (1.5 * g2**2 + 0.5 * g1**2 - 6 * y_t**2) * lam_H
        )
        b_lam_S = 18 * lam_S**2 + 2 * lam_HS**2
        b_lam_HS = (
            4 * lam_HS**2
            + 12 * lam_H * lam_HS
            + 6 * lam_S * lam_HS
            - 6 * y_t**2 * lam_HS
            + (1.5 * g2**2 + 0.5 * g1**2) * lam_HS
        )

        b_yt = y_t * (4.5 * y_t**2 - 4 * g3**2 - 2.25 * g2**2 - 1.4166667 * g1**2)
        b_g1 = (41.0 / 6.0) * g1**3
        b_g2 = (-19.0 / 6.0) * g2**3
        b_g3 = -7.0 * g3**3

        b_f2 = self._beta_f2(f2, lam_HS, xi_H)
        b_xi_H = 0.0

        return np.array([b_lam_H, b_lam_S, b_lam_HS, b_yt, b_g1, b_g2, b_g3, b_f2, b_xi_H]) / T16

    @staticmethod
    @jax.jit
    def _rhs_jax(g: jnp.ndarray) -> jnp.ndarray:
        lam_H, lam_S, lam_HS, y_t, g1, g2, g3, f2, xi_H = g

        b_lam_H = (
            24.0 * lam_H**2
            + 0.5 * lam_HS**2
            - 6.0 * y_t**4
            + 0.375 * (2.0 * g2**4 + (g1**2 + g2**2) ** 2)
            + (1.5 * g2**2 + 0.5 * g1**2 - 6.0 * y_t**2) * lam_H
        )
        b_lam_S = 18.0 * lam_S**2 + 2.0 * lam_HS**2
        b_lam_HS = (
            4.0 * lam_HS**2
            + 12.0 * lam_H * lam_HS
            + 6.0 * lam_S * lam_HS
            - 6.0 * y_t**2 * lam_HS
            + (1.5 * g2**2 + 0.5 * g1**2) * lam_HS
        )

        b_yt = y_t * (4.5 * y_t**2 - 4.0 * g3**2 - 2.25 * g2**2 - 1.4166667 * g1**2)
        b_g1 = (41.0 / 6.0) * g1**3
        b_g2 = (-19.0 / 6.0) * g2**3
        b_g3 = -7.0 * g3**3

        b1 = -(133.0 / 20.0) * f2**3
        b2_grav = (5196.0 / 5.0) / T16 * f2**5
        b2_sm = -12.0 * lam_HS * xi_H**2 / T16 * f2**3
        b_f2 = b1 + b2_grav + b2_sm
        b_xi_H = 0.0

        return jnp.array([b_lam_H, b_lam_S, b_lam_HS, b_yt, b_g1, b_g2, b_g3, b_f2, b_xi_H]) / T16

    @staticmethod
    @partial(jax.jit, static_argnums=(3,))
    def _rk4_scan(g0: jnp.ndarray, t0: float, t1: float, n_steps: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        dt = (t1 - t0) / n_steps

        def step(g_prev, i):
            t = t0 + dt * i
            del t
            k1 = SIQGRGESolver._rhs_jax(g_prev)
            k2 = SIQGRGESolver._rhs_jax(g_prev + 0.5 * dt * k1)
            k3 = SIQGRGESolver._rhs_jax(g_prev + 0.5 * dt * k2)
            k4 = SIQGRGESolver._rhs_jax(g_prev + dt * k3)
            g_next = g_prev + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return g_next, g_next

        grid_idx = jnp.arange(n_steps)
        g_final, g_hist = jax.lax.scan(step, g0, grid_idx)
        t_hist = jnp.linspace(t0 + dt, t1, n_steps)
        return t_hist, jnp.vstack([g0, g_hist]), g_final

    def _solve_jax_rk4(self, g0: np.ndarray, n_steps: int) -> JAXODESolution:
        t0, t1 = self.t_span
        t_hist, y_hist, _ = self._rk4_scan(jnp.asarray(g0, dtype=jnp.float64), t0, t1, int(n_steps))
        full_t = np.concatenate([[t0], np.asarray(t_hist)])
        full_y = np.asarray(y_hist).T
        return JAXODESolution(t=full_t, y=full_y, success=True, nfev=4 * int(n_steps))

    def solve(
        self,
        g0: np.ndarray,
        rtol: float = 1e-8,
        atol: Optional[float] = None,
        backend: str = "jax",
        n_steps: int = 2048,
    ) -> Dict:
        """Integrate RGE from m_t to M2. Returns solution dict + stability flags."""
        if atol is None:
            if self.ledger is not None:
                regime = self.detector.classify(self.state, {})
                atol = self.ledger.get_tolerance("rge_atol", regime=regime)
            else:
                atol = 1e-10

        if backend == "jax":
            sol = self._solve_jax_rk4(g0, n_steps=n_steps)
        elif backend == "scipy":
            sol = solve_ivp(self.rhs, self.t_span, g0, method="RK45", rtol=rtol, atol=atol, dense_output=True)
        else:
            raise ValueError("backend must be either 'jax' or 'scipy'")

        g_uv = sol.y[:, -1]
        g_ir = sol.y[:, 0]

        step_min = float(np.diff(sol.t).min()) if len(sol.t) > 1 else 1.0
        regime = self.detector.classify(
            {
                "energy_scale": self.mu_end,
                "f2": float(g_uv[7]),
                "step_size": step_min,
            },
            {"RGE_residual": 0.0 if sol.success else 1.0},
        )

        if self.ledger is not None:
            residual = abs(float(g_uv[-1] - g0[-1]))
            self.ledger.update_from_residual("rge_atol", residual, "rge_solver")

        return {
            "sol": sol,
            "g_uv": g_uv,
            "g_ir": g_ir,
            "success": bool(sol.success),
            "nfev": int(sol.nfev),
            "tol_used": atol,
            "regime": regime.value,
            "backend": backend,
        }
