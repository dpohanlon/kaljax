from functools import partial
import jax
from jax import lax, jit
import jax.numpy as jnp
import jax.scipy.linalg as jsp
from jax.lax.linalg import triangular_solve

from .types import FilterResult
from .config import JITTER, eye

def _as_seq(A, T):
    return A if A.ndim >= 3 else jnp.broadcast_to(A, (T,) + A.shape)

def _as_seq_time(A, T):
    return A if A.ndim >= 4 else jnp.broadcast_to(A, (A.shape[0], T) + A.shape[1:])

@partial(jit, static_argnames=("method",))
def forward(y, F, H, Q, R, m0, P0, *, G=None, u=None, mask=None, method="chol"):
    T, obs_dim = y.shape
    state_dim = m0.shape[0]
    F_seq = _as_seq(F, T)
    H_seq = _as_seq(H, T)
    Q_seq = _as_seq(Q, T)
    R_seq = _as_seq(R, T)
    if G is not None:
        G_seq = _as_seq(G, T)
        u_seq = _as_seq(u, T)
    I = eye(state_dim, dtype=y.dtype)

    if mask is None:
        mask = jnp.ones((T,), dtype=bool)

    def step(carry, inputs):
        m_prev, P_prev = carry
        y_t, F_t, H_t, Q_t, R_t, obs_t = inputs
        m_pred = F_t @ m_prev
        P_pred = F_t @ P_prev @ F_t.T + Q_t + JITTER * I

        def do_update(_):
            S = H_t @ P_pred @ H_t.T + R_t + JITTER * eye(obs_dim, dtype=y.dtype)
            L = jsp.cholesky(S, lower=True)
            v = y_t - H_t @ m_pred
            Sinv_v = jsp.cho_solve((L, True), v)
            log_det_S = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
            ll_t = -0.5 * (log_det_S + v.T @ Sinv_v + obs_dim * jnp.log(2.0 * jnp.pi))

            A = H_t @ P_pred.T
            K = jsp.cho_solve((L, True), A).T
            m_upd = m_pred + K @ v
            I_KH = I - K @ H_t
            P_upd = I_KH @ P_pred @ I_KH.T + K @ R_t @ K.T
            return m_upd, P_upd, ll_t

        def skip_update(_):
            return m_pred, P_pred, jnp.array(0.0, dtype=y.dtype)

        if G is not None:
            m_pred = m_pred + G_seq[0] @ u_seq[0]  # aligned by scan input; see below

        m_upd, P_upd, ll_t = lax.cond(obs_t, do_update, skip_update, operand=None)
        return (m_upd, P_upd), (m_upd, P_upd, ll_t)

    if G is None:
        inputs = (y, F_seq, H_seq, Q_seq, R_seq, mask)
    else:
        # prepend G_t,u_t into y via closure; align with inputs by rotating through scan
        def pack(y, F_seq, H_seq, Q_seq, R_seq, mask, G_seq, u_seq):
            return (y, F_seq, H_seq, Q_seq, R_seq, mask, G_seq, u_seq)
        y, F_seq, H_seq, Q_seq, R_seq, mask, G_seq, u_seq = pack(y, F_seq, H_seq, Q_seq, R_seq, mask, G_seq, u_seq)
        inputs = (y, F_seq, H_seq, Q_seq, R_seq, mask)

    (mT, PT), (m_seq, P_seq, ll_seq) = lax.scan(step, (m0, P0), inputs)
    return FilterResult(m=m_seq, P=P_seq, ll=ll_seq)

@partial(jit, static_argnames=("method","unroll"))
def forward_batched(y, F, H, Q, R, m0, P0, *, G=None, u=None, mask=None, method="chol", unroll=1):
    B, T, O = y.shape
    S = m0.shape[-1]
    I_S = jnp.eye(S, dtype=y.dtype)
    I_O = jnp.eye(O, dtype=y.dtype)
    J = 1e-6
    BIG = 1e6

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

    F_bt = to_bt(F, 2)
    Q_bt = to_bt(Q, 2)
    H_bt = to_bt(H, 2)
    R_bt = to_bt(R, 2)
    G_bt = to_bt(G, 2) if (G is not None) else None
    u_bt = to_bt(u, 1) if (u is not None) else None

    if mask is None:
        mask_bt = jnp.ones((B, T, O), dtype=bool)
    elif mask.ndim == 2:
        mask_bt = mask[..., None]
    elif mask.ndim == 3:
        mask_bt = mask
    else:
        raise ValueError("bad mask rank")

    def step(carry, t):
        m_prev, P_prev = carry
        F_t = F_bt[:, t]
        Q_t = Q_bt[:, t]
        H_t = H_bt[:, t]
        R_t = R_bt[:, t]
        y_t = y[:, t]
        mskt = mask_bt[:, t].astype(y.dtype)

        m_pred = (F_t @ m_prev[..., None])[..., 0]
        if (G_bt is not None) and (u_bt is not None):
            m_pred = m_pred + (G_bt[:, t] @ u_bt[:, t][..., None])[..., 0]
        P_pred = F_t @ P_prev @ jnp.swapaxes(F_t, 1, 2) + Q_t + J * I_S

        Hm = H_t * mskt[..., :, None]
        R_eff = R_t + ((1.0 - mskt)[..., :, None] * (BIG * I_O))

        S_t = Hm @ P_pred @ jnp.swapaxes(Hm, 1, 2) + R_eff + J * I_O
        L = jnp.linalg.cholesky(S_t)
        v = y_t - (H_t @ m_pred[..., None])[..., 0]
        v_eff = v * mskt

        y1 = triangular_solve(L, v_eff[..., None], left_side=True, lower=True)
        x1 = triangular_solve(L, y1, left_side=True, lower=True, transpose_a=True)
        Sinv_v = x1[..., 0]
        log_det_S = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
        ll_t = -0.5 * (log_det_S + jnp.sum(v_eff * Sinv_v, axis=-1) + jnp.sum(mskt, axis=-1) * jnp.log(2.0 * jnp.pi))

        PHt = P_pred @ jnp.swapaxes(Hm, 1, 2)
        yK = triangular_solve(L, jnp.swapaxes(PHt, 1, 2), left_side=True, lower=True)
        xK = triangular_solve(L, yK, left_side=True, lower=True, transpose_a=True)
        K = jnp.swapaxes(xK, 1, 2)

        m_upd = m_pred + (K @ v_eff[..., None])[..., 0]
        I_KH = I_S[None, ...] - (K @ Hm)
        P_upd = I_KH @ P_pred @ jnp.swapaxes(I_KH, 1, 2) + K @ R_eff @ jnp.swapaxes(K, 1, 2)

        return (m_upd, P_upd), (m_upd, P_upd, ll_t)

    (mT, PT), (m_seq, P_seq, ll_seq) = lax.scan(step, (m0, P0), jnp.arange(T), unroll=unroll)
    return m_seq, P_seq, ll_seq

vfilter = jax.vmap(forward, in_axes=(0, 0, 0, 0, 0, 0, 0), out_axes=0)  # y,F,H,Q,R,m0,P0 per batch
