import jax
import jax.numpy as jnp

from .config import eye


def rollout(key, T, F, H, Q, R, m0, *, G=None, u=None):
    state_dim = m0.shape[0]
    obs_dim = H.shape[-2] if H.ndim == 2 else H.shape[1]
    F_seq = F if F.ndim == 3 else jnp.broadcast_to(F, (T,) + F.shape)
    H_seq = H if H.ndim == 3 else jnp.broadcast_to(H, (T,) + H.shape)
    Q_seq = Q if Q.ndim == 3 else jnp.broadcast_to(Q, (T,) + Q.shape)
    R_seq = R if R.ndim == 3 else jnp.broadcast_to(R, (T,) + R.shape)

    if G is not None:
        if u is None:
            raise ValueError("u must be provided when G is not None")
        if G.ndim != 2:
            raise ValueError(
                "G must be fixed in time with shape (state_dim, control_dim)"
            )
        u_seq = u if u.ndim == 2 else jnp.broadcast_to(u, (T,) + u.shape)

    def step(carry, inputs):
        x_prev = carry
        if G is None:
            F_t, H_t, Q_t, R_t, key_t = inputs
            u_t = None
        else:
            F_t, H_t, Q_t, R_t, u_t, key_t = inputs

        k1, k2 = jax.random.split(key_t, 2)
        Lq = jnp.linalg.cholesky(Q_t)
        Lr = jnp.linalg.cholesky(R_t)
        w = Lq @ jax.random.normal(k1, (state_dim,))
        v = Lr @ jax.random.normal(k2, (obs_dim,))

        x = F_t @ x_prev
        if G is not None:
            x = x + G @ u_t
        x = x + w

        y = H_t @ x + v
        return x, (x, y)

    keys = jax.random.split(key, T)

    if G is None:
        inputs = (F_seq, H_seq, Q_seq, R_seq, keys)
    else:
        inputs = (F_seq, H_seq, Q_seq, R_seq, u_seq, keys)

    (_, (xs, ys)) = jax.lax.scan(step, m0, inputs)
    return xs, ys
