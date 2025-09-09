from functools import partial
import jax
from jax import lax, jit
import jax.numpy as jnp
import jax.scipy.linalg as jsp
from .types import SmoothResult
from .config import JITTER, eye

def _as_seq(A, T):
    return A if A.ndim >= 3 else jnp.broadcast_to(A, (T,) + A.shape)

@jit
def rts(filtered_means, filtered_covs, F, Q, *, G=None, u=None):
    T = filtered_means.shape[0]
    state_dim = filtered_means.shape[1]
    I = eye(state_dim, dtype=filtered_means.dtype)
    F_seq = _as_seq(F, T)
    Q_seq = _as_seq(Q, T)

    m_last = filtered_means[-1]
    P_last = filtered_covs[-1]
    m_in = jnp.flip(filtered_means[:-1], axis=0)
    P_in = jnp.flip(filtered_covs[:-1], axis=0)
    F_in = jnp.flip(F_seq[:-1], axis=0)
    Q_in = jnp.flip(Q_seq[:-1], axis=0)

    def step(carry, inputs):
        m_next, P_next = carry
        m_t, P_t, F_t, Q_t = inputs
        P_pred_next = F_t @ P_t @ F_t.T + Q_t + JITTER * I
        L = jsp.cholesky(P_pred_next, lower=True)
        A = F_t @ P_t
        J = jsp.cho_solve((L, True), A.T).T
        m_pred_next = F_t @ m_t
        m_s = m_t + J @ (m_next - m_pred_next)
        P_s = P_t + J @ (P_next - P_pred_next) @ J.T
        return (m_s, P_s), (m_s, P_s)

    (_, _), (m_rev, P_rev) = lax.scan(step, (m_last, P_last), (m_in, P_in, F_in, Q_in))
    m_s = jnp.concatenate([jnp.flip(m_rev, axis=0), m_last[None, :]], axis=0)
    P_s = jnp.concatenate([jnp.flip(P_rev, axis=0), P_last[None, :, :]], axis=0)
    return SmoothResult(m=m_s, P=P_s)

@partial(jit, static_argnames=("unroll",))
def rts_batched(m_filt, P_filt, F, Q, *, G=None, u=None, unroll=1):
    T, B, S = m_filt.shape
    I = jnp.eye(S, dtype=m_filt.dtype)
    JIT = 1e-6

    def to_bt(A, tail_rank):
        if A is None:
            return None
        if A.ndim == tail_rank:            # (...tail) → (B,T,...tail)
            A = jnp.broadcast_to(A, (B, T) + A.shape)
        elif A.ndim == 1 + tail_rank:      # (B,...tail) → (B,T,...tail)
            A = jnp.broadcast_to(A[:, None, ...], (B, T) + A.shape[1:])
        elif A.ndim == 2 + tail_rank:      # (B,T,...tail)
            pass
        else:
            raise ValueError("bad rank")
        return A

    F_bt = to_bt(F, 2)   # (B,T,S,S)
    Q_bt = to_bt(Q, 2)   # (B,T,S,S)
    G_bt = to_bt(G, 2) if (G is not None) else None   # (B,T,S,C)
    u_bt = to_bt(u, 1) if (u is not None) else None   # (B,T,C)

    m_last = m_filt[-1]         # (B,S)
    P_last = P_filt[-1]         # (B,S,S)

    m_in = jnp.flip(m_filt[:-1], axis=0)             # (T-1,B,S)
    P_in = jnp.flip(P_filt[:-1], axis=0)             # (T-1,B,S,S)
    F_in = jnp.flip(F_bt[:, :-1], axis=1).swapaxes(0, 1)  # (T-1,B,S,S)
    Q_in = jnp.flip(Q_bt[:, :-1], axis=1).swapaxes(0, 1)  # (T-1,B,S,S)
    if G_bt is not None and u_bt is not None:
        G_in = jnp.flip(G_bt[:, :-1], axis=1).swapaxes(0, 1)  # (T-1,B,S,C)
        u_in = jnp.flip(u_bt[:, :-1], axis=1).swapaxes(0, 1)  # (T-1,B,C)
    else:
        G_in, u_in = None, None

    def step(carry, inputs):
        m_next, P_next = carry
        m_t, P_t, F_t, Q_t = inputs[:4]
        if G_in is not None and u_in is not None:
            G_t, u_t = inputs[4], inputs[5]

        P_pred = F_t @ P_t @ jnp.swapaxes(F_t, -1, -2) + Q_t + JIT * I
        L = jnp.linalg.cholesky(P_pred)
        A = F_t @ P_t
        Y = jax.lax.linalg.triangular_solve(L, A, left_side=True, lower=True)
        X = jax.lax.linalg.triangular_solve(L, Y, left_side=True, lower=True, transpose_a=True)
        Jgain = jnp.swapaxes(X, -1, -2)

        m_pred = (F_t @ m_t[..., None])[..., 0]
        if G_in is not None and u_in is not None:
            m_pred = m_pred + (G_t @ u_t[..., None])[..., 0]

        delta = m_next - m_pred
        m_s = m_t + (Jgain @ delta[..., None])[..., 0]
        P_s = P_t + Jgain @ (P_next - P_pred) @ jnp.swapaxes(Jgain, -1, -2)
        return (m_s, P_s), (m_s, P_s)

    if G_in is not None and u_in is not None:
        scan_inputs = (m_in, P_in, F_in, Q_in, G_in, u_in)
    else:
        scan_inputs = (m_in, P_in, F_in, Q_in)

    (_, _), (m_rev, P_rev) = lax.scan(step, (m_last, P_last), scan_inputs, unroll=unroll)
    m_s = jnp.concatenate([jnp.flip(m_rev, axis=0), m_last[None, ...]], axis=0)
    P_s = jnp.concatenate([jnp.flip(P_rev, axis=0), P_last[None, ...]], axis=0)
    return m_s, P_s

vsmooth = jax.vmap(rts, in_axes=(0, 0, 0, 0), out_axes=0)  # means,covs,F,Q per batch
