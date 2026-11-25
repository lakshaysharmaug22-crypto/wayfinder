"""Sensor-to-IMU rotation extrinsic + time offset (+ gyro bias) from relative rotations.

Any exteroceptive sensor that can measure its own frame-to-frame rotation (camera via
the essential matrix, LiDAR via scan-to-scan ICP) can be aligned to the gyroscope:

    Log(dR_imu(t1 - td, t2 - td)) - b_g * (t2 - t1)  =  R_bs * Log(dR_sensor(t1, t2))

1. coarse time offset: cross-correlate rotation *magnitudes* (extrinsic-invariant),
2. closed-form rotation (Kabsch) at the coarse offset,
3. joint robust non-linear refinement of (R_bs, td, b_g).
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot, Slerp

from ..estimation.lidar_odom import prep_scan, icp_point_to_plane, deskew
from scipy.spatial import cKDTree


def gyro_orientation(t, gyro):
    # midpoint rule: forward Euler would lag the orientation by half a sample
    dt = np.diff(t)
    steps = Rot.from_rotvec(0.5 * (gyro[1:] + gyro[:-1]) * dt[:, None])
    q = [Rot.identity()]
    for k in range(len(t) - 1):
        q.append(q[-1] * steps[k])
    return Slerp(t, Rot.concatenate(q))


def _bearings(uv, K, dist):
    x = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    b = np.c_[x, np.ones(len(x))]
    return b / np.linalg.norm(b, axis=1, keepdims=True)


def camera_relative_rotations(frames, K, dist, step=1, min_common=20, thr_deg=0.4, rng=None):
    """Frame-to-frame rotation from bearing vectors (RANSAC + Kabsch).

    Over one frame interval the baseline is tiny w.r.t. scene depth, so the essential
    matrix is ill-conditioned; a rotation-only model is better posed and the remaining
    parallax is handled by the robust threshold.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    cth = np.cos(np.radians(thr_deg))
    t1s, t2s, phis = [], [], []
    for k in range(step, len(frames), step):
        a, b = frames[k - step], frames[k]
        common, ia, ib = np.intersect1d(a["ids"], b["ids"], return_indices=True)
        if len(common) < min_common:
            continue
        b1, b2 = _bearings(a["uv"][ia], K, dist), _bearings(b["uv"][ib], K, dist)
        best = None
        for _ in range(60):
            s = rng.choice(len(b1), 3, replace=False)
            R = kabsch(b1[s], b2[s])
            inl = np.einsum('ij,ij->i', b1 @ R.T, b2) > cth
            if best is None or inl.sum() > best.sum():
                best = inl
        if best.sum() < min_common * 0.6:
            continue
        R = kabsch(b1[best], b2[best])
        t1s.append(a["stamp"])
        t2s.append(b["stamp"])
        phis.append(Rot.from_matrix(R).as_rotvec())   # b2 = R b1, R = R_wc2^T R_wc1
    return np.array(t1s), np.array(t2s), np.array(phis)


def lidar_chain(prepped, twists):
    """Scan-to-scan point-to-plane ICP on de-skewed scans, chained into a trajectory.

    prepped: list of (pts, normals, tr); twists: per-scan (w_l, v_l) for the de-skew.
    Returns dict(phi (n-1,3) relative rotations in the camera-style convention, R, p: chain poses
    in the frame of scan 0 for scans 0..n-1)."""
    ds = [deskew(p, n, t, *tw)[:2] for (p, n, t), tw in zip(prepped, twists)]
    R_ch, p_ch, phis, good = [np.eye(3)], [np.zeros(3)], [], []
    for k in range(1, len(prepped)):
        (pa, na), (pb, _) = ds[k - 1], ds[k]
        T, info = icp_point_to_plane(pb, cKDTree(pa), pa, na, np.eye(4), iters=25)
        ok = info["ok"] and info.get("frac", 0) > 0.5
        if not ok:
            T = np.eye(4)
        good.append(ok)
        R_ch.append(R_ch[-1] @ T[:3, :3])
        p_ch.append(p_ch[-1] + R_ch[-2] @ T[:3, 3])
        phis.append(Rot.from_matrix(T[:3, :3].T).as_rotvec())
    return dict(phi=np.array(phis), good=np.array(good), R=np.array(R_ch), p=np.array(p_ch))


def calibrate_lidar_imu(scans, imu_t, gyro, R_bl0, bg0=None, td0=0.0, sweep=0.1, use_deskew=True,
                        prepped=None):
    """Two-pass LiDAR-IMU calibration. Pass 1 de-skews with the nominal rotation (and zero linear
    velocity), pass 2 with the pass-1 calibration and the velocity of the pass-1 chain."""
    from scipy.signal import savgol_filter
    if prepped is None:
        prepped = [prep_scan(s["pts"], s["tr"]) for s in scans]
    stamps = np.array([s["stamp"] for s in scans])
    gyro_at = lambda t: np.c_[[np.interp(t, imu_t, gyro[:, a]) for a in range(3)]].T
    bg = np.zeros(3) if bg0 is None else bg0
    R_bl, td = R_bl0, td0
    v_ch = np.zeros((len(scans), 3))
    out = None
    for it in range(2 if use_deskew else 1):
        w_b = gyro_at(stamps - td) - bg
        if use_deskew:
            tw = [(R_bl.T @ w_b[k], (out["R"][k].T @ v_ch[k]) if out else np.zeros(3)) for k in range(len(scans))]
        else:
            tw = [(np.zeros(3), np.zeros(3))] * len(scans)
        ch = lidar_chain(prepped, tw)
        ok = ch["good"]
        t1, t2 = stamps[:-1][ok], stamps[1:][ok]
        cal = calibrate_rotation_and_offset(t1, t2, ch["phi"][ok], imu_t, gyro)
        R_bl, td, bg = cal["R"], cal["td"], cal["bg"]
        if it == 0:
            # velocity of pass-1 chain; its positions refer to ~mid-sweep because v was not compensated
            v_ch = savgol_filter(ch["p"], 7, 3, deriv=1, delta=sweep, axis=0)
            out = ch
        else:
            out = ch
        cal["chain"] = ch
        cal["t1"], cal["t2"], cal["phi"] = t1, t2, ch["phi"][ok]
        cal["prepped"] = prepped
        cal["t_imu"] = stamps - td
        cal["frac_good"] = float(ok.mean())
    return cal


def _imu_rel(slerp, t1, t2, td, lo, hi):
    a = np.clip(t1 - td, lo, hi)
    b = np.clip(t2 - td, lo, hi)
    return (slerp(b).inv() * slerp(a)).as_rotvec()


def kabsch(src, dst, w=None):
    """R minimising sum w|dst - R src|^2."""
    w = np.ones(len(src)) if w is None else w
    H = (src * w[:, None]).T @ dst
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def calibrate_rotation_and_offset(t1, t2, phi_s, imu_t, gyro, td_range=0.06, slerp=None):
    slerp = gyro_orientation(imu_t, gyro) if slerp is None else slerp
    lo, hi = imu_t[0], imu_t[-1]
    mag_s = np.linalg.norm(phi_s, axis=1)

    grid = np.arange(-td_range, td_range + 1e-9, 0.0005)
    corr = []
    for td in grid:
        mag_i = np.linalg.norm(_imu_rel(slerp, t1, t2, td, lo, hi), axis=1)
        corr.append(np.corrcoef(mag_s, mag_i)[0, 1])
    corr = np.array(corr)
    i = int(np.argmax(corr))
    td0 = grid[i]
    if 0 < i < len(grid) - 1:   # parabolic sub-sample refinement
        y0, y1, y2 = corr[i - 1:i + 2]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            td0 += 0.5 * (y0 - y2) / den * (grid[1] - grid[0])

    phi_i = _imu_rel(slerp, t1, t2, td0, lo, hi)
    R0 = kabsch(phi_s, phi_i, w=mag_s)

    dts = (t2 - t1)[:, None]

    def res(x):
        R = Rot.from_rotvec(x[:3]).as_matrix()
        pi = _imu_rel(slerp, t1, t2, x[3], lo, hi) + x[4:7] * dts  # Log(R2^T R1) = -int(w)
        return (pi - phi_s @ R.T).ravel()

    x0 = np.r_[Rot.from_matrix(R0).as_rotvec(), td0, np.zeros(3)]
    sol = least_squares(res, x0, loss="soft_l1", f_scale=np.radians(0.3),
                        diff_step=np.r_[1e-6 * np.ones(3), 1e-3, 1e-6 * np.ones(3)])
    r = sol.fun.reshape(-1, 3)
    return dict(R=Rot.from_rotvec(sol.x[:3]).as_matrix(), td=float(sol.x[3]), bg=sol.x[4:7],
                td_coarse=float(td0), R_coarse=R0, corr_grid=grid, corr=corr,
                rms_deg=float(np.degrees(np.sqrt(np.mean(np.sum(r * r, axis=1))))), n=len(t1))
