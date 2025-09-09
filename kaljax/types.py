from typing import NamedTuple
import jax.numpy as jnp

class FilterResult(NamedTuple):
    m: jnp.ndarray
    P: jnp.ndarray
    ll: jnp.ndarray

class SmoothResult(NamedTuple):
    m: jnp.ndarray
    P: jnp.ndarray
