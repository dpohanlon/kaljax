from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp
from jax import jit, lax, random

from .config import JITTER, eye
from .types import SmoothResult


def _as_seq(A, T):
    return A if A.ndim >= 3 else jnp.broadcast_to(A, (T,) + A.shape)


def _sample_gaussian(key, mean, cov):
    cov = 0.5 * (cov + cov.T) + JITTER * eye(cov.shape[0], dtype=cov.dtype)
    L = jsp.cholesky(cov, lower=True)
    z = random.normal(key, mean.shape, dtype=mean.dtype)
    return mean + L @ z


@jit
def rts(filtered_means, filtered_covs, F, Q, *, G=None, u=None):
    T = filtered_means.shape[0]
    state_dim = filtered_means.shape[1]
    I = eye(state_dim, dtype=filtered_means.dtype)
    F_seq = _as_seq(F, T)
    Q_seq = _as_seq(Q, T)

    if G is not None:
        if u is None:
            raise ValueError("u must be provided when G is not None")
        if G.ndim != 2:
            raise ValueError(
                "G must be fixed in time with shape (state_dim, control_dim)"
            )
        u_seq = _as_seq(u, T)

    m_last = filtered_means[-1]
    P_last = filtered_covs[-1]
    m_in = jnp.flip(filtered_means[:-1], axis=0)
    P_in = jnp.flip(filtered_covs[:-1], axis=0)
    F_in = jnp.flip(F_seq[:-1], axis=0)
    Q_in = jnp.flip(Q_seq[:-1], axis=0)

    if G is not None:
        u_in = jnp.flip(u_seq[:-1], axis=0)

    def step(carry, inputs):
        m_next, P_next = carry

        if G is None:
            m_t, P_t, F_t, Q_t = inputs
            u_t = None
        else:
            m_t, P_t, F_t, Q_t, u_t = inputs

        P_t = 0.5 * (P_t + P_t.T) + JITTER * I
        Q_t = 0.5 * (Q_t + Q_t.T)

        P_pred_next = F_t @ P_t @ F_t.T + Q_t
        P_pred_next = 0.5 * (P_pred_next + P_pred_next.T) + JITTER * I

        L = jsp.cholesky(P_pred_next, lower=True)
        A = P_t @ F_t.T
        J = jsp.cho_solve((L, True), A.T).T

        m_pred_next = F_t @ m_t
        if G is not None:
            m_pred_next = m_pred_next + G @ u_t

        m_s = m_t + J @ (m_next - m_pred_next)
        P_s = P_t + J @ (P_next - P_pred_next) @ J.T
        P_s = 0.5 * (P_s + P_s.T) + JITTER * I
        return (m_s, P_s), (m_s, P_s)

    if G is None:
        scan_inputs = (m_in, P_in, F_in, Q_in)
    else:
        scan_inputs = (m_in, P_in, F_in, Q_in, u_in)

    (_, _), (m_rev, P_rev) = lax.scan(step, (m_last, P_last), scan_inputs)
    m_s = jnp.concatenate([jnp.flip(m_rev, axis=0), m_last[None, :]], axis=0)
    P_s = jnp.concatenate([jnp.flip(P_rev, axis=0), P_last[None, :, :]], axis=0)
    return SmoothResult(m=m_s, P=P_s)


@partial(jit, static_argnames=("unroll",))
def rts_batched(m_filt, P_filt, F, Q, *, G=None, u=None, unroll=1):
    B, T, S = m_filt.shape
    I = jnp.eye(S, dtype=m_filt.dtype)
    JIT = 1e-6

    def to_bt(A, tail_rank):
        if A is None:
            return None
        if A.ndim == tail_rank:
            A = jnp.broadcast_to(A, (B, T) + A.shape)
        elif A.ndim == 1 + tail_rank:
            A = jnp.broadcast_to(A[:, None, ...], (B, T) + A.shape[1:])
        elif A.ndim == 2 + tail_rank:
            pass
        else:
            raise ValueError("bad rank")
        return A

    def to_bt_u(u):
        if u is None:
            return None
        if u.ndim == 1:
            u = jnp.broadcast_to(u, (B, T) + u.shape)
        elif u.ndim == 2:
            if u.shape[0] == B:
                u = jnp.broadcast_to(u[:, None, :], (B, T, u.shape[-1]))
            elif u.shape[0] == T:
                u = jnp.broadcast_to(u[None, :, :], (B, T, u.shape[-1]))
            else:
                raise ValueError("bad u shape")
        elif u.ndim == 3:
            pass
        else:
            raise ValueError("bad u rank")
        return u

    F_bt = to_bt(F, 2)
    Q_bt = to_bt(Q, 2)

    if G is not None:
        if u is None:
            raise ValueError("u must be provided when G is not None")
        if G.ndim != 2:
            raise ValueError(
                "G must be fixed in time with shape (state_dim, control_dim)"
            )
        G_b = jnp.broadcast_to(G, (B,) + G.shape)
        u_bt = to_bt_u(u)
    else:
        G_b = None
        u_bt = None

    m_last = m_filt[:, -1]
    P_last = P_filt[:, -1]

    m_in = jnp.flip(m_filt[:, :-1], axis=1).swapaxes(0, 1)
    P_in = jnp.flip(P_filt[:, :-1], axis=1).swapaxes(0, 1)
    F_in = jnp.flip(F_bt[:, :-1], axis=1).swapaxes(0, 1)
    Q_in = jnp.flip(Q_bt[:, :-1], axis=1).swapaxes(0, 1)

    if G_b is not None:
        u_in = jnp.flip(u_bt[:, :-1], axis=1).swapaxes(0, 1)

    def step(carry, inputs):
        m_next, P_next = carry

        if G_b is None:
            m_t, P_t, F_t, Q_t = inputs
            u_t = None
        else:
            m_t, P_t, F_t, Q_t, u_t = inputs

        P_pred = F_t @ P_t @ jnp.swapaxes(F_t, -1, -2) + Q_t + JIT * I
        L = jnp.linalg.cholesky(P_pred)

        P_t = 0.5 * (P_t + jnp.swapaxes(P_t, -1, -2)) + JIT * I
        Q_t = 0.5 * (Q_t + jnp.swapaxes(Q_t, -1, -2))

        P_pred = F_t @ P_t @ jnp.swapaxes(F_t, -1, -2) + Q_t
        P_pred = 0.5 * (P_pred + jnp.swapaxes(P_pred, -1, -2)) + JIT * I
        L = jnp.linalg.cholesky(P_pred)

        A = P_t @ jnp.swapaxes(F_t, -1, -2)
        B = jnp.swapaxes(A, -1, -2)
        Y = jax.lax.linalg.triangular_solve(L, B, left_side=True, lower=True)
        X = jax.lax.linalg.triangular_solve(
            L, Y, left_side=True, lower=True, transpose_a=True
        )
        Jgain = jnp.swapaxes(X, -1, -2)

        m_pred = (F_t @ m_t[..., None])[..., 0]
        if G_b is not None:
            m_pred = m_pred + (G_b @ u_t[..., None])[..., 0]

        delta = m_next - m_pred
        m_s = m_t + (Jgain @ delta[..., None])[..., 0]
        P_s = P_t + Jgain @ (P_next - P_pred) @ jnp.swapaxes(Jgain, -1, -2)
        return (m_s, P_s), (m_s, P_s)

    if G_b is None:
        scan_inputs = (m_in, P_in, F_in, Q_in)
    else:
        scan_inputs = (m_in, P_in, F_in, Q_in, u_in)

    (_, _), (m_rev, P_rev) = lax.scan(
        step, (m_last, P_last), scan_inputs, unroll=unroll
    )

    m_s = jnp.concatenate(
        [jnp.flip(m_rev, axis=0).swapaxes(0, 1), m_last[:, None, :]], axis=1
    )
    P_s = jnp.concatenate(
        [jnp.flip(P_rev, axis=0).swapaxes(0, 1), P_last[:, None, :, :]], axis=1
    )
    return m_s, P_s


@jit
def backward_sample(filtered_means, filtered_covs, F, Q, key, *, G=None, u=None):
    T = filtered_means.shape[0]
    state_dim = filtered_means.shape[1]
    I = eye(state_dim, dtype=filtered_means.dtype)
    F_seq = _as_seq(F, T)
    Q_seq = _as_seq(Q, T)

    if G is not None:
        if u is None:
            raise ValueError("u must be provided when G is not None")
        if G.ndim != 2:
            raise ValueError(
                "G must be fixed in time with shape (state_dim, control_dim)"
            )
        u_seq = _as_seq(u, T)

    keys = random.split(key, T)

    P_last = 0.5 * (filtered_covs[-1] + filtered_covs[-1].T) + JITTER * I
    x_last = _sample_gaussian(keys[-1], filtered_means[-1], P_last)

    m_in = jnp.flip(filtered_means[:-1], axis=0)
    P_in = jnp.flip(filtered_covs[:-1], axis=0)
    F_in = jnp.flip(F_seq[:-1], axis=0)
    Q_in = jnp.flip(Q_seq[:-1], axis=0)
    k_in = jnp.flip(keys[:-1], axis=0)

    if G is not None:
        u_in = jnp.flip(u_seq[:-1], axis=0)

    def step(x_next, inputs):
        if G is None:
            m_t, P_t, F_t, Q_t, key_t = inputs
            u_t = None
        else:
            m_t, P_t, F_t, Q_t, u_t, key_t = inputs

        P_t = 0.5 * (P_t + P_t.T) + JITTER * I
        Q_t = 0.5 * (Q_t + Q_t.T)

        P_pred = F_t @ P_t @ F_t.T + Q_t
        P_pred = 0.5 * (P_pred + P_pred.T) + JITTER * I

        L = jsp.cholesky(P_pred, lower=True)
        A = P_t @ F_t.T
        J = jsp.cho_solve((L, True), A.T).T

        m_pred = F_t @ m_t
        if G is not None:
            m_pred = m_pred + G @ u_t

        mean_t = m_t + J @ (x_next - m_pred)
        cov_t = P_t - J @ P_pred @ J.T
        cov_t = 0.5 * (cov_t + cov_t.T) + JITTER * I
        x_t = _sample_gaussian(key_t, mean_t, cov_t)
        return x_t, x_t

    if G is None:
        scan_inputs = (m_in, P_in, F_in, Q_in, k_in)
    else:
        scan_inputs = (m_in, P_in, F_in, Q_in, u_in, k_in)

    _, x_rev = lax.scan(step, x_last, scan_inputs)
    x = jnp.concatenate([jnp.flip(x_rev, axis=0), x_last[None, :]], axis=0)
    return x


@partial(jit, static_argnames=("unroll",))
def backward_sample_batched(m_filt, P_filt, F, Q, key, *, G=None, u=None, unroll=1):
    B, T, S = m_filt.shape
    I = eye(S, dtype=m_filt.dtype)

    def to_bt(A, tail_rank):
        if A is None:
            return None
        if A.ndim == tail_rank:
            A = jnp.broadcast_to(A, (B, T) + A.shape)
        elif A.ndim == 1 + tail_rank:
            A = jnp.broadcast_to(A[:, None, ...], (B, T) + A.shape[1:])
        elif A.ndim == 2 + tail_rank:
            pass
        else:
            raise ValueError("bad rank")
        return A

    def to_bt_u(u):
        if u is None:
            return None
        if u.ndim == 1:
            u = jnp.broadcast_to(u, (B, T) + u.shape)
        elif u.ndim == 2:
            if u.shape[0] == B:
                u = jnp.broadcast_to(u[:, None, :], (B, T, u.shape[-1]))
            elif u.shape[0] == T:
                u = jnp.broadcast_to(u[None, :, :], (B, T, u.shape[-1]))
            else:
                raise ValueError("bad u shape")
        elif u.ndim == 3:
            pass
        else:
            raise ValueError("bad u rank")
        return u

    F_bt = to_bt(F, 2)
    Q_bt = to_bt(Q, 2)

    if G is not None:
        if u is None:
            raise ValueError("u must be provided when G is not None")
        if G.ndim != 2:
            raise ValueError(
                "G must be fixed in time with shape (state_dim, control_dim)"
            )
        G_b = jnp.broadcast_to(G, (B,) + G.shape)
        u_bt = to_bt_u(u)
    else:
        G_b = None
        u_bt = None

    keys = random.split(key, T * B).reshape(T, B, 2)

    def sample_last(k, m, P):
        P = 0.5 * (P + jnp.swapaxes(P, -1, -2)) + JITTER * I
        L = jnp.linalg.cholesky(P)
        z = random.normal(k, (S,), dtype=m.dtype)
        return m + L @ z

    x_last = jax.vmap(sample_last)(keys[-1], m_filt[:, -1], P_filt[:, -1])

    m_in = jnp.flip(m_filt[:, :-1], axis=1).swapaxes(0, 1)
    P_in = jnp.flip(P_filt[:, :-1], axis=1).swapaxes(0, 1)
    F_in = jnp.flip(F_bt[:, :-1], axis=1).swapaxes(0, 1)
    Q_in = jnp.flip(Q_bt[:, :-1], axis=1).swapaxes(0, 1)
    k_in = jnp.flip(keys[:-1], axis=0)

    if G_b is not None:
        u_in = jnp.flip(u_bt[:, :-1], axis=1).swapaxes(0, 1)

    def step(x_next, inputs):
        if G_b is None:
            m_t, P_t, F_t, Q_t, key_t = inputs
            u_t = None
        else:
            m_t, P_t, F_t, Q_t, u_t, key_t = inputs

        P_pred = F_t @ P_t @ jnp.swapaxes(F_t, -1, -2) + Q_t + JITTER * I
        L = jnp.linalg.cholesky(P_pred)

        P_t = 0.5 * (P_t + jnp.swapaxes(P_t, -1, -2)) + JIT * I
        Q_t = 0.5 * (Q_t + jnp.swapaxes(Q_t, -1, -2))

        P_pred = F_t @ P_t @ jnp.swapaxes(F_t, -1, -2) + Q_t
        P_pred = 0.5 * (P_pred + jnp.swapaxes(P_pred, -1, -2)) + JIT * I
        L = jnp.linalg.cholesky(P_pred)

        A = P_t @ jnp.swapaxes(F_t, -1, -2)
        B = jnp.swapaxes(A, -1, -2)
        Y = jax.lax.linalg.triangular_solve(L, B, left_side=True, lower=True)
        X = jax.lax.linalg.triangular_solve(
            L, Y, left_side=True, lower=True, transpose_a=True
        )
        Jgain = jnp.swapaxes(X, -1, -2)

        m_pred = (F_t @ m_t[..., None])[..., 0]
        if G_b is not None:
            m_pred = m_pred + jnp.einsum("bsc,bc->bs", G_b, u_t)

        mean_t = m_t + (Jgain @ (x_next - m_pred)[..., None])[..., 0]
        cov_t = P_t - Jgain @ P_pred @ jnp.swapaxes(Jgain, -1, -2)
        cov_t = 0.5 * (cov_t + jnp.swapaxes(cov_t, -1, -2)) + JITTER * I

        def sample_one(k, m, P):
            L = jnp.linalg.cholesky(P)
            z = random.normal(k, (S,), dtype=m.dtype)
            return m + L @ z

        x_t = jax.vmap(sample_one)(key_t, mean_t, cov_t)
        return x_t, x_t

    if G_b is None:
        scan_inputs = (m_in, P_in, F_in, Q_in, k_in)
    else:
        scan_inputs = (m_in, P_in, F_in, Q_in, u_in, k_in)

    _, x_rev = lax.scan(step, x_last, scan_inputs, unroll=unroll)
    x = jnp.concatenate(
        [jnp.flip(x_rev, axis=0).swapaxes(0, 1), x_last[:, None, :]], axis=1
    )
    return x


vsmooth = jax.vmap(rts, in_axes=(0, 0, 0, 0), out_axes=0)  # means,covs,F,Q per batch
