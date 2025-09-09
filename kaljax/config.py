import jax
import jax.numpy as jnp

JITTER = 1e-6

def eye(n, dtype=jnp.float64):
    return jnp.eye(n, dtype=dtype)

def enable_x64(flag: bool = True):
    jax.config.update("jax_enable_x64", flag)
