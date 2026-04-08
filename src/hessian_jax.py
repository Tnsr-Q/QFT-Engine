"""JAX Hessian-vector products and PL-condition helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import grad, jacrev, jvp

jax.config.update("jax_enable_x64", True)


class JAXHessianEstimator:
    """JAX-accelerated Gauss-Newton Hessian estimation and PL certification."""

    def __init__(self, constraint_fn, weights, reg_lambda: float = 1e-4):
        self.constraint_fn = constraint_fn
        self.W = jnp.asarray(weights)
        self.Lambda = reg_lambda
        self.grad_C = jax.jit(jacrev(constraint_fn))

    def hessian_vector_product(self, theta: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Exact H_GN v = J_C^T W J_C v + Λ v."""
        _, jv = jvp(self.constraint_fn, (theta,), (v,))
        _, vjp_fn = jax.vjp(self.constraint_fn, theta)
        jt_w_jv = vjp_fn(self.W * jv)[0]
        return jt_w_jv + self.Lambda * v

    def lanczos_eigenvalues(
        self,
        theta: jnp.ndarray,
        k: int = 3,
        max_iter: int = 50,
        tol: float = 1e-7,
    ) -> jnp.ndarray:
        """Lanczos iteration for extremal eigenvalues of H_GN."""
        dim = int(theta.shape[0])
        v_prev = jnp.zeros((dim,))
        v = jax.random.normal(jax.random.PRNGKey(42), (dim,))
        v = v / jnp.linalg.norm(v)

        alphas = []
        betas = []
        for _ in range(max_iter):
            hv = self.hessian_vector_product(theta, v)
            alpha = jnp.dot(hv, v)
            w = hv - alpha * v
            if betas:
                w = w - betas[-1] * v_prev
            beta = jnp.linalg.norm(w)

            alphas.append(alpha)
            if float(beta) < tol:
                break

            betas.append(beta)
            v_prev, v = v, w / beta

        m = len(alphas)
        t = jnp.zeros((m, m), dtype=theta.dtype)
        for i in range(m):
            t = t.at[i, i].set(alphas[i])
            if i < len(betas):
                t = t.at[i, i + 1].set(betas[i])
                t = t.at[i + 1, i].set(betas[i])

        eigvals = jnp.linalg.eigvalsh(t)
        eigvals = jnp.sort(eigvals)
        return eigvals[:k] if k and k < eigvals.shape[0] else eigvals

    def verify_pl_condition(
        self,
        theta: jnp.ndarray,
        loss_val: float,
        loss_star: float = 0.0,
        mu_lb: float = 2.4e-2,
    ) -> dict:
        """Certify 0.5||∇L||² ≥ μ(L - L*)."""
        grad_l = grad(self._loss_fn)(theta)
        grad_norm_sq = jnp.sum(grad_l**2)
        gap = max(float(loss_val - loss_star), 1e-12)
        mu_est = float(0.5 * grad_norm_sq / gap)
        return {
            "mu_est": mu_est,
            "pl_satisfied": bool(mu_est >= mu_lb),
            "grad_norm": float(jnp.linalg.norm(grad_l)),
            "loss_gap": gap,
        }

    def _loss_fn(self, theta: jnp.ndarray) -> jnp.ndarray:
        c = self.constraint_fn(theta)
        return 0.5 * jnp.sum(self.W * (c - 1.0) ** 2) + 0.5 * self.Lambda * jnp.sum(theta**2)
