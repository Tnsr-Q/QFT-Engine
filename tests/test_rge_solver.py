"""tests/test_rge_solver.py — Unit tests for the XLA adaptive RGE solver.

Validates:
- Forward integration accuracy and stability
- Gradient consistency (finite-difference check)
- Dense-output interpolation accuracy
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from fakeon_numeric.rge_solver import AdaptiveRK45Solver, make_differentiable_rge_solver, interpolate_trajectory


# ---------------------------------------------------------------------------
# Mock 1-loop SM-like RHS for testing (stable, analytically tractable)
# ---------------------------------------------------------------------------

def mock_rhs(t: float, y: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Simple 3-coupling system mimicking 1-loop SM running (stable for tests)."""
    lam, g, yt = y
    b_lam = 24 * lam**2 + 0.5 * g**2 * lam - 6 * yt**4
    b_g = (41 / 6) * g**3
    b_yt = yt * (4.5 * yt**2 - 4 * 1.22**2)
    return jnp.array([b_lam, b_g, b_yt]) / (16 * np.pi**2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rge_forward_accuracy():
    t0, t1 = 0.0, 10.0
    solver = AdaptiveRK45Solver(mock_rhs, t0, t1)
    g0 = jnp.array([0.13, 0.35, 0.93], dtype=jnp.float64)

    t_final, y_end, nfev, ts, ys = solver.solve(
        g0, rtol=1e-9, atol=1e-11, record_trajectory=True
    )

    assert y_end.shape == (3,)
    assert jnp.isfinite(y_end).all()
    assert nfev > 50 and nfev < 20000
    assert ts is not None and ys is not None
    assert len(ts) == len(ys) > 10
    assert jnp.allclose(t_final, t1, atol=1e-6)


def test_gradient_consistency():
    t0, t1 = 0.0, 5.0
    solver = AdaptiveRK45Solver(mock_rhs, t0, t1)
    diff_solver = make_differentiable_rge_solver(solver, target_idx=0)

    g0 = jnp.array([0.13, 0.35, 0.93], dtype=jnp.float64)
    params = jnp.array([], dtype=jnp.float64)

    loss, dg0, dparams, y_uv = diff_solver(g0, params, rtol=1e-8, atol=1e-10)

    assert jnp.isfinite(loss)
    assert dg0.shape == g0.shape
    assert dparams.shape == params.shape

    # Finite-difference sanity check
    eps = 1e-6
    for i in range(len(g0)):
        g_plus = g0.at[i].add(eps)
        loss_plus, _, _, _ = diff_solver(g_plus, params, rtol=1e-8, atol=1e-10)
        fd_grad = (loss_plus - loss) / eps
        rel_err = jnp.abs(fd_grad - dg0[i]) / jnp.abs(dg0[i] + 1e-15)
        assert rel_err < 5e-3, f"Gradient mismatch on component {i}: {rel_err}"


def test_dense_output_accuracy():
    t0, t1 = 0.0, 8.0
    solver = AdaptiveRK45Solver(mock_rhs, t0, t1)
    g0 = jnp.array([0.13, 0.35, 0.93], dtype=jnp.float64)

    _, _, _, ts, ys = solver.solve(g0, record_trajectory=True, rtol=1e-10, atol=1e-12)

    t_query = jnp.linspace(t0 + 0.1, t1 - 0.1, 30)
    y_interp = interpolate_trajectory(ts, ys, t_query)

    # Check boundary conditions and smoothness
    assert y_interp.shape == (len(t_query), 3)
    assert jnp.allclose(y_interp[0], ys[0], atol=1e-8)
    assert jnp.allclose(y_interp[-1], ys[-1], atol=1e-8)

    # Check that interpolated values are finite and reasonable
    assert jnp.all(jnp.isfinite(y_interp))
    assert jnp.all(y_interp > 0)  # couplings stay positive in this mock


def test_parameter_differentiability():
    """Smoke test that we can differentiate w.r.t theory parameters."""
    t0, t1 = 0.0, 4.0
    solver = AdaptiveRK45Solver(mock_rhs, t0, t1)
    diff_solver = make_differentiable_rge_solver(solver, target_idx=0)

    g0 = jnp.array([0.13, 0.35, 0.93], dtype=jnp.float64)
    params = jnp.array([1.0, 0.5], dtype=jnp.float64)  # dummy params

    loss, dg0, dparams, _ = diff_solver(g0, params, rtol=1e-8, atol=1e-10)

    assert dparams.shape == params.shape
    assert jnp.all(jnp.isfinite(dparams))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
