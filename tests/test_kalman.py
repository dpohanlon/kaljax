import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Ensure project root is on the Python path for local imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kaljax.filter import forward, forward_batched, vfilter
from kaljax.smooth import rts, rts_batched, vsmooth
from kaljax.model import ParametricSSM, vmapped_loglik


def _manual_filter(y, F, H, Q, R, m0, P0):
    T = y.shape[0]
    m_prev = m0
    P_prev = P0
    ms = []
    Ps = []
    lls = []
    I = jnp.eye(m0.shape[0])
    for t in range(T):
        m_pred = F @ m_prev
        P_pred = F @ P_prev @ F.T + Q
        v = y[t] - H @ m_pred
        S = H @ P_pred @ H.T + R
        Sinv = jnp.linalg.inv(S)
        K = P_pred @ H.T @ Sinv
        m_upd = m_pred + K @ v
        I_KH = I - K @ H
        P_upd = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
        ll_t = -0.5 * (jnp.log(jnp.linalg.det(S)) + (v.T @ Sinv @ v) + y.shape[1] * jnp.log(2.0 * jnp.pi))
        ms.append(m_upd)
        Ps.append(P_upd)
        lls.append(ll_t)
        m_prev, P_prev = m_upd, P_upd
    return jnp.stack(ms), jnp.stack(Ps), jnp.stack(lls)


def _manual_smoother(m_filt, P_filt, F, Q):
    T = m_filt.shape[0]
    ms = [None] * T
    Ps = [None] * T
    ms[-1] = m_filt[-1]
    Ps[-1] = P_filt[-1]
    for t in range(T - 2, -1, -1):
        P_pred = F @ P_filt[t] @ F.T + Q
        J = P_filt[t] @ F.T @ jnp.linalg.inv(P_pred)
        m_s = m_filt[t] + J @ (ms[t + 1] - F @ m_filt[t])
        P_s = P_filt[t] + J @ (Ps[t + 1] - P_pred) @ J.T
        ms[t] = m_s
        Ps[t] = P_s
    return jnp.stack(ms), jnp.stack(Ps)


def test_forward_matches_manual():
    F = jnp.array([[1.0]])
    H = jnp.array([[1.0]])
    Q = jnp.array([[0.1]])
    R = jnp.array([[0.2]])
    y = jnp.array([[1.0], [0.5], [-0.3]])
    m0 = jnp.array([0.0])
    P0 = jnp.array([[1.0]])

    res = forward(y, F, H, Q, R, m0, P0)
    m_exp, P_exp, ll_exp = _manual_filter(y, F, H, Q, R, m0, P0)

    np.testing.assert_allclose(res.m, m_exp, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(res.P, P_exp, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(res.ll, ll_exp, rtol=1e-5, atol=1e-8)


def test_forward_batched_matches_vmap():
    B, T = 2, 3
    F = jnp.array([[1.0]])
    H = jnp.array([[1.0]])
    Q = jnp.array([[0.1]])
    R = jnp.array([[0.2]])
    y = jnp.array([
        [[1.0], [0.5], [-0.3]],
        [[-0.2], [0.1], [0.0]],
    ])
    m0 = jnp.zeros((B, 1))
    P0 = jnp.broadcast_to(jnp.eye(1), (B, 1, 1))

    m_bt, P_bt, ll_bt = forward_batched(y, F, H, Q, R, m0, P0)

    F_b = jnp.broadcast_to(F, (B,) + F.shape)
    H_b = jnp.broadcast_to(H, (B,) + H.shape)
    Q_b = jnp.broadcast_to(Q, (B,) + Q.shape)
    R_b = jnp.broadcast_to(R, (B,) + R.shape)
    res_vm = vfilter(y, F_b, H_b, Q_b, R_b, m0, P0)
    m_vm = jnp.swapaxes(res_vm.m, 0, 1)
    P_vm = jnp.swapaxes(res_vm.P, 0, 1)
    ll_vm = jnp.swapaxes(res_vm.ll, 0, 1)

    np.testing.assert_allclose(m_bt, m_vm, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(P_bt, P_vm, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(ll_bt, ll_vm, rtol=1e-5, atol=1e-8)


def test_rts_matches_manual():
    F = jnp.array([[1.0]])
    Q = jnp.array([[0.1]])
    H = jnp.array([[1.0]])
    R = jnp.array([[0.2]])
    y = jnp.array([[1.0], [0.5], [-0.3]])
    m0 = jnp.array([0.0])
    P0 = jnp.array([[1.0]])

    filt_res = forward(y, F, H, Q, R, m0, P0)
    smooth_res = rts(filt_res.m, filt_res.P, F, Q)
    m_exp, P_exp = _manual_smoother(filt_res.m, filt_res.P, F, Q)

    np.testing.assert_allclose(smooth_res.m, m_exp, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(smooth_res.P, P_exp, rtol=1e-5, atol=1e-8)


def test_vfilter_with_controls_and_mask():
    B, T = 2, 3
    F = jnp.broadcast_to(jnp.array([[0.9]]), (B, 1, 1))
    H = jnp.ones((B, 1, 1))
    Q = jnp.full((B, 1, 1), 0.1)
    R = jnp.full((B, 1, 1), 0.2)
    y = jnp.array([[[1.0], [0.5], [-0.3]], [[-0.2], [0.1], [0.0]]])
    m0 = jnp.zeros((B, 1))
    P0 = jnp.ones((B, 1, 1))
    G = jnp.array([[0.5]])
    u = jnp.array([[0.2], [-0.1]])
    mask = jnp.array([[True, False, True], [False, True, True]])

    actual = vfilter(y, F, H, Q, R, m0, P0, u=u, mask=mask, G=G)
    expected = [
        forward(y[b], F[b], H[b], Q[b], R[b], m0[b], P0[b],
                G=G, u=u[b], mask=mask[b])
        for b in range(B)
    ]

    assert actual.m.shape == (B, T, 1)
    for field in ("m", "P", "ll"):
        np.testing.assert_allclose(
            getattr(actual, field),
            jnp.stack([getattr(result, field) for result in expected]),
            rtol=1e-5, atol=1e-8,
        )
    np.testing.assert_array_equal(np.asarray(actual.ll)[~np.asarray(mask)], 0.0)


def test_rts_batched_matches_vmap():
    B, T = 2, 3
    F = jnp.array([[1.0]])
    Q = jnp.array([[0.1]])
    H = jnp.array([[1.0]])
    R = jnp.array([[0.2]])
    y = jnp.array([
        [[1.0], [0.5], [-0.3]],
        [[-0.2], [0.1], [0.0]],
    ])
    m0 = jnp.zeros((B, 1))
    P0 = jnp.broadcast_to(jnp.eye(1), (B, 1, 1))

    m_bt, P_bt, _ = forward_batched(y, F, H, Q, R, m0, P0)
    # The batched filter returns time-first arrays; smoothers take batch-first arrays.
    m_bt = jnp.swapaxes(m_bt, 0, 1)
    P_bt = jnp.swapaxes(P_bt, 0, 1)
    m_s_bt, P_s_bt = rts_batched(m_bt, P_bt, F, Q)

    F_b = jnp.broadcast_to(F, (B,) + F.shape)
    Q_b = jnp.broadcast_to(Q, (B,) + Q.shape)
    res_vm = vsmooth(m_bt, P_bt, F_b, Q_b)
    m_s_vm = res_vm.m
    P_s_vm = res_vm.P

    assert m_s_bt.shape == (B, T, 1)
    assert P_s_bt.shape == (B, T, 1, 1)

    np.testing.assert_allclose(m_s_bt, m_s_vm, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(P_s_bt, P_s_vm, rtol=1e-5, atol=1e-8)


def test_parametric_vmapped_loglik():
    def build(theta):
        F = jnp.array([[theta]])
        H = jnp.array([[1.0]])
        Q = jnp.array([[1.0]])
        R = jnp.array([[1.0]])
        return F, H, Q, R

    model = ParametricSSM(theta=jnp.array(0.9), build_matrices=build)
    y = jnp.array([[0.0], [0.0]])
    m0 = jnp.array([0.0])
    P0 = jnp.array([[1.0]])

    # single theta log-likelihood matches manual forward
    F, H, Q, R = build(0.9)
    ll_manual = jnp.sum(forward(y, F, H, Q, R, m0, P0).ll)
    ll_model = model.loglik(y, m0, P0)
    np.testing.assert_allclose(ll_model, ll_manual, rtol=1e-5, atol=1e-8)

    thetas = jnp.array([0.8, 0.9, 1.1])
    vm_vals = vmapped_loglik(model, thetas, y, m0, P0)

    def manual(theta):
        F, H, Q, R = build(theta)
        return jnp.sum(forward(y, F, H, Q, R, m0, P0).ll)

    manual_vals = jax.vmap(manual)(thetas)
    np.testing.assert_allclose(vm_vals, manual_vals, rtol=1e-5, atol=1e-8)
