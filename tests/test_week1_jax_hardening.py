import numpy as np
import pytest


def test_sharded_regge_returns_convergence_mask():
    jnp = pytest.importorskip("jax.numpy")

    from src.regge_shard_map import ShardedReggeSolver

    solver = ShardedReggeSolver(N_t=16, t_min=1e-2, t_max=1e2, max_iter=32)
    delta_mock = jnp.zeros(16) + 0.05

    traj, converged = solver.scan_regge_trajectory_sharded(delta_mock, return_convergence=True)

    assert traj.shape == (16,)
    assert converged.shape == (16,)
    assert converged.dtype == jnp.bool_


def test_rge_solver_jax_backend_runs():
    from src.rge_solver import SIQGRGESolver

    solver = SIQGRGESolver()
    g0 = np.array([0.13, 0.01, 0.02, 0.93, 0.35, 0.64, 1.16, 1e-8, 0.1], dtype=np.float64)

    out = solver.solve(g0, backend="jax", n_steps=128)

    assert out["success"]
    assert out["g_uv"].shape == (9,)
    assert np.isfinite(out["g_uv"]).all()
