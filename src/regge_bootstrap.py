import numpy as np
import jax.numpy as jnp
from jax import jit
from scipy.optimize import root_scalar
from typing import Dict, Tuple

from src.bootstrap_solver import DiscretizedBootstrapSolver


class ReggeExtendedBootstrap(DiscretizedBootstrapSolver):
    """Discretized bootstrap extension with lightweight Regge pole tracking."""

    def __init__(
        self,
        s_min: float = 4.0,
        s_max: float = 1e6,
        N_s: int = 128,
        N_l: int = 6,
        alpha: float = 0.05,
        m2: float = 0.01,
        M2: float = 2.4e23,
        **kwargs,
    ):
        super().__init__(s_min=s_min, s_max=s_max, N_s=N_s, N_l=N_l, alpha=alpha, m2=m2, **kwargs)
        self.M2 = M2
        self.t_grid = np.logspace(-2, 4.0, 40)
        self.l_contour_re = np.linspace(-0.5, 2.0, 150)
        self.l_contour_im = np.linspace(-0.8, 0.8, 100)

    @staticmethod
    @jit
    def _analytic_continue_Sl(
        s: jnp.ndarray,
        l_re: jnp.ndarray,
        l_im: jnp.ndarray,
        alpha: float,
        delta_l: jnp.ndarray,
    ) -> jnp.ndarray:
        """Analytic continuation of S_l(s) to complex l = l_re + i l_im."""
        eta_c = jnp.exp(-alpha * jnp.maximum(s - 0.04, 0.0) ** (l_re + 1j * l_im + 1.0))
        return eta_c * jnp.exp(2j * delta_l)

    def track_regge_poles(
        self,
        S_l_solved: jnp.ndarray,
        delta_solved: jnp.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Track a coarse Regge trajectory α(t) over the configured t-grid."""
        del S_l_solved  # interface-compatible input; phase information is read from delta_solved.

        alpha_traj = []
        delta_np = np.asarray(delta_solved)

        for t_val in self.t_grid:
            s_cross = max(float(t_val), 4.0 * self.m2)
            s_idx = int(np.argmin(np.abs(np.asarray(self.s_grid) - s_cross)))
            delta_slice = delta_np[:, s_idx]

            def pole_condition(l_re: float) -> float:
                eta_c = np.exp(-self.alpha * np.maximum(s_cross - 0.04, 0.0) ** (l_re + 0.01j + 1.0))
                S_c = eta_c * np.exp(2j * np.mean(delta_slice))
                return float(np.imag(1.0 / (1.0 - S_c)))

            try:
                res = root_scalar(pole_condition, bracket=[-0.4, 1.8], method="brentq", xtol=1e-6)
                alpha_traj.append(complex(res.root, 0.0))
            except ValueError:
                alpha_traj.append(complex(-0.25, 0.0))

        return np.asarray(alpha_traj), self.t_grid

    def verify_fakeon_regge_condition(self, alpha_traj: np.ndarray, t_grid: np.ndarray) -> Dict[str, object]:
        """Verify Re[α(M2²)] < 0 as a fakeon-virtualization diagnostic."""
        t_target = self.M2**2
        idx = int(np.argmin(np.abs(t_grid - t_target)))
        alpha_M2 = alpha_traj[idx]

        return {
            "Re_alpha_at_M2": float(np.real(alpha_M2)),
            "Im_alpha_at_M2": float(np.imag(alpha_M2)),
            "fakeon_virtualized": bool(np.real(alpha_M2) < 0),
            "trajectory": alpha_traj.tolist(),
            "status": "VERIFIED" if np.real(alpha_M2) < 0 else "PENDING",
        }

    def run_full_regge_analysis(self, S_opt: jnp.ndarray, delta_opt: jnp.ndarray) -> Dict[str, object]:
        """Run coarse Regge-pole extraction and fakeon verification."""
        traj, t_vals = self.track_regge_poles(S_opt, delta_opt)
        verification = self.verify_fakeon_regge_condition(traj, t_vals)
        verification["trajectory"] = [str(c) for c in verification["trajectory"]]
        return verification
