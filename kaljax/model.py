"""Higher level parametric interface built with Equinox.

This module provides a small `eqx.Module` that stores a parameter PyTree and a
callable ``build_matrices``.  The callable maps the parameter PyTree to the
matrices required by the Kalman filter.  The module exposes ``__call__`` to run
the filter and ``loglik`` which returns the summed log-likelihood.  Both methods
are compatible with JAX transformations such as ``jax.grad`` and
``jax.vmap``.

Example
-------

```python
import jax.numpy as jnp
import equinox as eqx

from kaljax.model import ParametricSSM


def build(theta):
    F = jnp.array([[theta]])
    H = jnp.array([[1.0]])
    Q = jnp.eye(1)
    R = jnp.eye(1)
    return F, H, Q, R


model = ParametricSSM(theta=jnp.array(0.9), build_matrices=build)
```
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from .filter import forward


class ParametricSSM(eqx.Module):
    """State space model parametrised by a PyTree.

    Parameters
    ----------
    theta:
        PyTree of parameters used to construct the matrices.
    build_matrices:
        Callable mapping ``theta`` to ``(F, H, Q, R)`` matrices.  It is marked as
        static so that it is not treated as a differentiable parameter.
    """

    theta: jnp.ndarray
    build_matrices: Callable[[jnp.ndarray], tuple] = eqx.field(static=True)

    def __call__(self, y, m0, P0, *, G=None, u=None, mask=None):
        """Run the Kalman filter and return the full result."""

        F, H, Q, R = self.build_matrices(self.theta)
        return forward(y, F, H, Q, R, m0, P0, G=G, u=u, mask=mask)

    def loglik(self, y, m0, P0, *, G=None, u=None, mask=None):
        """Return the summed log-likelihood for observations ``y``."""

        res = self(y, m0, P0, G=G, u=u, mask=mask)
        return jnp.sum(res.ll)


def vmapped_loglik(model: ParametricSSM, thetas, y, m0, P0, *, G=None, u=None, mask=None):
    """Vectorised log-likelihood over a batch of parameter values.

    This is a small helper around ``jax.vmap`` that replaces ``model.theta``
    with each entry from ``thetas`` and evaluates ``model.loglik``.  It avoids
    users writing explicit Python loops when scanning over many parameter
    variants.
    """

    def _loglik(theta):
        inst = eqx.tree_at(lambda m: m.theta, model, theta)
        return inst.loglik(y, m0, P0, G=G, u=u, mask=mask)

    return jax.vmap(_loglik)(thetas)

