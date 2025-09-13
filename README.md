# kaljax

JAX-powered Kalman filtering and smoothing with a small parametric
interface built on top of [Equinox](https://github.com/patrick-kidger/equinox).

## Installation

The project is published as a Poetry package.  You can install it via pip:

```bash
pip install kaljax
```

Alternatively, clone the repository and run `poetry install` to get the
development environment.

## Basic filtering

The core `forward` function runs a standard Kalman filter.  It accepts the
system matrices ``F, H, Q, R`` along with initial mean and covariance:

```python
import jax.numpy as jnp
from kaljax.filter import forward

F = jnp.array([[0.9]])
H = jnp.array([[1.0]])
Q = jnp.eye(1)
R = jnp.eye(1)

T = 5
y = jnp.zeros((T, 1))
m0 = jnp.zeros(1)
P0 = jnp.eye(1)

res = forward(y, F, H, Q, R, m0, P0)
print(res.m)  # filtered means
```

The corresponding Rauch--Tung--Striebel smoother is available via
``kaljax.smooth.rts``:

```python
from kaljax.smooth import rts

smooth_res = rts(res.m, res.P, F, Q)
print(smooth_res.m)  # smoothed means
```

## Parametric state-space models

For models where the system matrices depend on parameters, use
``ParametricSSM``.  Supply a ``build_matrices`` callable that maps a parameter
PyTree to the ``(F, H, Q, R)`` matrices:

```python
import jax
import jax.numpy as jnp
import equinox as eqx
from kaljax.model import ParametricSSM, vmapped_loglik

def build(theta):
    F = jnp.array([[theta]])
    H = jnp.array([[1.0]])
    Q = jnp.eye(1)
    R = jnp.eye(1)
    return F, H, Q, R

model = ParametricSSM(theta=jnp.array(0.9), build_matrices=build)
ll = model.loglik(y, m0, P0)  # summed log-likelihood

# gradients flow through the parameters
grad_fn = jax.grad(lambda th: ParametricSSM(th, build).loglik(y, m0, P0))
print(grad_fn(jnp.array(0.9)))

# evaluate many parameter values in parallel
thetas = jnp.linspace(0.5, 0.99, 5)
ll_batch = vmapped_loglik(model, thetas, y, m0, P0)
print(ll_batch)
```

The Equinox module integrates seamlessly with ``jax.grad`` for optimisation
and ``jax.vmap`` for efficient parameter sweeps.

