"""Lever-arm (sensor position in the IMU frame) from metric sensor motion + accelerometer.

A sensor that measures its own metric pose (camera against a target of known size, LiDAR
registration) gives p_s(t), R_s(t) in some fixed frame {t}. With R_tb = R_ts R_bs^T and the
rigid-body relation  p_s = p_b + R_tb t_bs  (so p_s'' = p_b'' + R_tb (w'^ + w^ w^) t_bs):

    p_s'' - R_tb a_m  =  [ R_tb (w'^ + w^ w^) ] t_bs  -  R_tb b_a  +  g_t

is linear in (t_bs, b_a, g_t). Gravity in {t} and the accelerometer bias are solved jointly, so
no knowledge of the target's orientation relative to gravity is needed.
"""
import cv2
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot, Slerp

from ..geom import hat, exp_so3
from .extrinsic_time import gyro_orientation


def pnp_poses(frames, obj, K, dist):
    """Camera pose in the target frame for each bench frame (IPPE + LM refinement)."""
    ts, Rs, ps = [], [], []
    for fr in frames:
        ok, rv, tv = cv2.solvePnP(obj.astype(np.float64), fr["uv"].astype(np.float64), K, dist,
                                  flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        rv, tv = cv2.solvePnPRefineLM(obj.astype(np.float64), fr["uv"].astype(np.float64), K, dist, rv, tv)
        R_cw, _ = cv2.Rodrigues(rv)             # x_c = R_cw x_t + t
        Rs.append(R_cw.T)
        ps.append(-R_cw.T @ tv.ravel())
        ts.append(fr["stamp"])
    return np.array(ts), np.array(Rs), np.array(ps)


def estimate_lever_arm(t_imu, R_ts, p_ts, R_bs, imu_t, gyro, acc, bg, win_len=1.0, hop=0.5, win=21,
                      max_gap=0.3, refine_rotation=False, local_bias=True):
    """Windowed double-integration (no numerical differentiation of the sensor positions).

    In every window the equation above is integrated twice over the IMU samples; the unknown
    window-start position and velocity of the sensor are eliminated by projecting out [1, tau].
    t_imu: sensor sample times on the IMU clock. Returns dict(t_bs, ba, g_t, R_bs, rms, n_windows).

    local_bias: treat accelerometer bias + gravity as constant *within* each window (projecting
    out [1, tau, tau^2] instead of [1, tau]); this absorbs bias random walk and leaves only t_bs.

    With refine_rotation the small residual rotation error of R_bs is a further unknown: writing
    R_bs_true = R_bs Exp(d), R_tb_true ~ R_tb Exp(-R_bs d), the accelerometer term gains the
    linear column R_tb [a_m]x d_b (d_b = R_bs d).
    """
    dt = imu_t[1] - imu_t[0]
    w = savgol_filter(gyro - bg, win, 3, axis=0)
    wd = savgol_filter(gyro - bg, win, 3, deriv=1, delta=dt, axis=0)
    a = savgol_filter(acc, win, 3, axis=0)
    # body orientation at IMU rate = (slowly varying anchor correction) * (integrated gyro): the
    # sensor fixes the orientation, the gyro supplies the shape between its (10-20 Hz) samples
    G = gyro_orientation(imu_t, gyro - bg)
    Gk = G(np.clip(t_imu, imu_t[0], imu_t[-1]))
    Ck = Rot.from_matrix(R_ts @ R_bs.T) * Gk.inv()
    slerp = Slerp(t_imu, Ck)
    m_ok = (imu_t >= t_imu[0]) & (imu_t <= t_imu[-1])
    ti = imu_t[m_ok]
    R_tb = (slerp(ti) * G(ti)).as_matrix()
    wk, wdk, ak = w[m_ok], wd[m_ok], a[m_ok]
    H = np.apply_along_axis(lambda v: hat(v), 1, wk)
    M = np.apply_along_axis(lambda v: hat(v), 1, wdk) + H @ H
    q = np.einsum('nij,nj->ni', R_tb, ak)                 # R a_m
    RM = R_tb @ M
    RA = R_tb @ np.apply_along_axis(lambda v: hat(v), 1, ak)    # R [a_m]x
    rows, rhs = [], []
    n_w = 0
    for t0 in np.arange(t_imu[0], t_imu[-1] - win_len, hop):
        ii = np.nonzero((ti >= t0) & (ti < t0 + win_len))[0]
        kk = np.nonzero((t_imu >= t0) & (t_imu < t0 + win_len))[0]
        if len(kk) < 8 or len(ii) < 20:
            continue
        if np.diff(np.r_[t0, t_imu[kk], t0 + win_len]).max() > max_gap:
            continue
        tau_i = ti[ii] - t0
        dbl = lambda y: cumulative_trapezoid(cumulative_trapezoid(y, dx=dt, initial=0, axis=0), dx=dt, initial=0, axis=0)
        Q2q = dbl(q[ii])
        Q2M = dbl(RM[ii])
        Q2R = dbl(R_tb[ii])
        Q2A = dbl(RA[ii])
        tau = t_imu[kk] - t0
        at = lambda Y: np.stack([np.interp(tau, tau_i, Y[:, c]) for c in range(Y.shape[1])], axis=1)
        y = p_ts[kk] - at(Q2q)                                # (k,3)
        XM = np.stack([at(Q2M.reshape(len(ii), 9))[:, :].reshape(len(kk), 3, 3)], 0)[0]
        XB = -at(Q2R.reshape(len(ii), 9)).reshape(len(kk), 3, 3)
        XG = (tau**2 / 2)[:, None, None] * np.eye(3)
        Phi = np.c_[np.ones_like(tau), tau, tau**2] if local_bias else np.c_[np.ones_like(tau), tau]
        P = np.eye(len(kk)) - Phi @ np.linalg.pinv(Phi)
        parts = [XM] if local_bias else [XM, XB, XG]
        if refine_rotation:
            parts.append(at(Q2A.reshape(len(ii), 9)).reshape(len(kk), 3, 3))
        Xw = np.concatenate(parts, axis=2)                    # (k,3,9 or 12)
        rows.append(np.einsum('kl,lij->kij', P, Xw).reshape(len(kk) * 3, -1))
        rhs.append((P @ y).reshape(-1))
        n_w += 1
    A, y = np.vstack(rows), np.concatenate(rhs)
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ x
    if local_bias:
        return dict(t_bs=x[:3], R_bs=R_bs, rms=float(np.sqrt(np.mean(res**2))), n_windows=n_w)
    R_new = R_bs @ exp_so3(R_bs.T @ x[9:12]) if refine_rotation else R_bs
    return dict(t_bs=x[:3], ba=x[3:6], g_t=x[6:9], R_bs=R_new, rms=float(np.sqrt(np.mean(res**2))), n_windows=n_w)
