"""Example of fitting a coupled Ornstein--Uhlenbeck process with Numpyro
using the :class:`~kaljax.model.ParametricSSM` interface.

The latent state is four-dimensional: polarity ``(p_x,p_y)`` driving the
observed position ``(x,y)``.  We simulate synthetic data and infer
dynamics/variance parameters with MCMC.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from kaljax.model import ParametricSSM


def _core_F(a: float, b: float, dt: float, eps: float = 1e-8) -> jnp.ndarray:
    """2x2 transition block for the polarity-position system."""
    phi = jnp.exp(a * dt)
    beta = jnp.where(jnp.abs(a) > eps, b * (phi - 1.0) / a, b * dt)
    return jnp.array([[phi, 0.0], [beta, 1.0]])


def _core_Q(a: float, b: float, sd0: float, sd1: float, dt: float, eps: float = 1e-8) -> jnp.ndarray:
    """2x2 process-noise covariance block."""
    expadt = jnp.exp(a * dt)
    exp2adt = expadt**2
    q00 = sd0**2 * (exp2adt - 1.0) / (2.0 * (a + eps))
    q01 = b * sd0**2 * (expadt - 1.0) ** 2 / (2.0 * (a + eps) ** 2)
    q11 = (
        sd1**2 * dt
        + (b**2 * sd0**2) / (a + eps) ** 2
        * (dt - 2.0 * (expadt - 1.0) / (a + eps) + (exp2adt - 1.0) / (2.0 * (a + eps)))
    )
    return jnp.array([[q00, q01], [q01, q11]])


def build_matrices(theta: jnp.ndarray, dt: float):
    """Construct state-space matrices from parameters."""
    a, b, sd0, sd1, sigma_obs = theta
    F_block = _core_F(a, b, dt)
    Q_block = _core_Q(a, b, sd0, sd1, dt)
    F = jnp.kron(F_block, jnp.eye(2))
    Q = jnp.kron(Q_block, jnp.eye(2))
    H = jnp.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    R = sigma_obs**2 * jnp.eye(2)
    return F, H, Q, R


def numpyro_model(y: jnp.ndarray, dt: float):
    """Numpyro model placing priors on OU parameters."""
    a = numpyro.sample("a", dist.Normal(-1.0, 0.5))
    b = numpyro.sample("b", dist.Normal(0.0, 0.5))
    sd0 = numpyro.sample("sd0", dist.LogNormal(-1.0, 0.2))
    sd1 = numpyro.sample("sd1", dist.LogNormal(-1.0, 0.2))
    sigma_obs = numpyro.sample("sigma_obs", dist.LogNormal(-3.0, 0.2))

    theta = jnp.array([a, b, sd0, sd1, sigma_obs])
    ssm = ParametricSSM(theta=theta, build_matrices=lambda th: build_matrices(th, dt))

    m0 = jnp.zeros(4)
    P0 = jnp.eye(4) * 100.0
    numpyro.factor("kalman_ll", ssm.loglik(y, m0, P0))


def simulate(key: jax.random.KeyArray, T: int, dt: float, params: jnp.ndarray) -> jnp.ndarray:
    """Simulate observations from the coupled OU process."""
    F, H, Q, R = build_matrices(params, dt)
    state = jnp.zeros(4)
    ys = []
    for _ in range(T):
        key, k1, k2 = jax.random.split(key, 3)
        noise = jax.random.multivariate_normal(k1, jnp.zeros(4), Q)
        state = F @ state + noise
        obs = H @ state + jax.random.multivariate_normal(k2, jnp.zeros(2), R)
        ys.append(obs)
    return jnp.stack(ys)


def main():  # pragma: no cover - illustrative example
    dt = 1 / 3
    true_params = jnp.array([-1.0, 0.3, 0.1, 0.1, 0.05])
    key = jax.random.PRNGKey(0)
    y = simulate(key, 100, dt, true_params)

    nuts = NUTS(numpyro_model)
    mcmc = MCMC(nuts, num_warmup=200, num_samples=200)
    mcmc.run(key, y, dt)
    print(mcmc.get_samples())


if __name__ == "__main__":  # pragma: no cover - example script
    main()
