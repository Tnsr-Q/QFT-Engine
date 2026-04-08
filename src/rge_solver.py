import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict

# Constants
PI = np.pi
T16 = 16.0 * PI**2
T16_SQ = T16**2


class SIQGRGESolver:
    """
    2-Loop RGE system for Scale-Invariant Quadratic Gravity + SM.
    Matches Extended Lemma 2 & S.3 β-function closure.
    """

    def __init__(self, mu_start: float = 173.1, mu_end: float = 2.4e23):
        self.mu_start = mu_start  # GeV (top mass scale)
        self.mu_end = mu_end  # GeV (fakeon threshold M2)
        self.t_span = (np.log(mu_start), np.log(mu_end))

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

        # 1-loop scalar sector (QUFT- RGE-Thermal.txt §1)
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

        # 1-loop SM gauge & Yukawa (standard MS-bar)
        b_yt = y_t * (4.5 * y_t**2 - 4 * g3**2 - 2.25 * g2**2 - 1.4166667 * g1**2)
        b_g1 = (41.0 / 6.0) * g1**3
        b_g2 = (-19.0 / 6.0) * g2**3
        b_g3 = -7.0 * g3**3

        # f2 flow with 2-loop closure
        b_f2 = self._beta_f2(f2, lam_HS, xi_H)
        b_xi_H = 0.0  # Negligible running in perturbative regime per docs

        return np.array([b_lam_H, b_lam_S, b_lam_HS, b_yt, b_g1, b_g2, b_g3, b_f2, b_xi_H]) / T16

    def solve(self, g0: np.ndarray, rtol: float = 1e-8, atol: float = 1e-10) -> Dict:
        """Integrate RGE from m_t to M2. Returns solution dict + stability flags."""
        sol = solve_ivp(self.rhs, self.t_span, g0, method="RK45", rtol=rtol, atol=atol, dense_output=True)

        # Extract endpoints
        g_uv = sol.y[:, -1]
        g_ir = sol.y[:, 0]

        return {
            "sol": sol,
            "g_uv": g_uv,
            "g_ir": g_ir,
            "success": sol.success,
            "nfev": sol.nfev,
        }
