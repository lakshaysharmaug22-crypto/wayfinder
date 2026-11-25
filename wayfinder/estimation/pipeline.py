"""Sequence generation and the multi-sensor estimator loop."""
from dataclasses import dataclass

import numpy as np

from ..geom import inv_T, make_T, exp_so3
from ..sensors import simulate_imu, simulate_camera, simulate_lidar
from .eskf import ESKF
from .lidar_odom import LocalMap, prep_scan, icp_point_to_plane, deskew, sensor_twist
from .msckf import MSCKF


def make_sequence(world, lms, traj, duration, ex, cam, lid, imu_p, seed,
                  cam_rate=20, lidar_rate=10, with_cam=True, with_lidar=True):
    rng = np.random.default_rng(seed)
    bg0 = rng.normal(0, 0.003, 3)
    ba0 = rng.normal(0, 0.04, 3)
    seq = dict(traj=traj, duration=duration,
               imu=simulate_imu(traj, duration, imu_p, rng, bg0, ba0))
    if with_cam:
        seq["frames"] = simulate_camera(lms, traj, ex["T_bc_true"], cam, cam_rate, duration, ex["td_cam"], rng)
    if with_lidar:
        scans = simulate_lidar(world, traj, ex["T_bl_true"], lid, lidar_rate, duration, ex["td_lidar"], rng)
        for s in scans:
            s["prep"] = prep_scan(s["pts"], s["tr"])
        seq["scans"] = scans
    return seq


@dataclass
class Calib:
    K: np.ndarray
    dist: np.ndarray
    T_bc: np.ndarray
    T_bl: np.ndarray
    td_cam: float = 0.0
    td_lidar: float = 0.0


def run(seq, calib: Calib, imu_p, use_cam=False, use_lidar=False, prior_map=None,
        init_T=None, log_every=10, lidar_sig=(0.05, 0.003), use_deskew=True, collect_scans=False,
        est_lever=False):
    imu = seq["imu"]
    t, gyro, acc = imu["t"], imu["gyro"], imu["acc"]
    traj = seq["traj"]
    p0, v0, _ = traj.pva(t[:1])
    R0 = traj.R(t[:1])[0]
    t_ext0 = calib.T_bl[:3, 3] if est_lever else None
    f = ESKF(p0[0], v0[0], R0, imu_p, t_ext0=t_ext0)
    if init_T is not None:     # started from a relocalised pose: velocity unknown
        f = ESKF(init_T[:3, 3], np.zeros(3), init_T[:3, :3], imu_p, t_ext0=t_ext0)
        f.P[0:3, 0:3] = np.eye(3) * 0.05**2
        f.P[3:6, 3:6] = np.eye(3) * 5.0**2
        f.P[6:9, 6:9] = np.eye(3) * np.radians(0.5)**2

    events = []
    if use_cam:
        vio = MSCKF(f, calib.K, calib.dist, calib.T_bc)
        events += [(fr["stamp"] - calib.td_cam, 0, k) for k, fr in enumerate(seq["frames"])]
    if use_lidar:
        lmap = LocalMap()
        events += [(s["stamp"] - calib.td_lidar, 1, k) for k, s in enumerate(seq["scans"])]
    events = [e for e in sorted(events) if t[0] < e[0] < t[-1]]
    stats = dict(cam_updates=0, lidar_updates=0, lidar_rejected=0)

    scan_log = []
    ext_trace = []

    def T_bl_now():
        return make_T(calib.T_bl[:3, :3], f.t_ext) if est_lever else calib.T_bl

    def on_lidar(k, w_b):
        pts, nrm, tr = seq["scans"][k]["prep"]
        if use_deskew:
            Tbl = T_bl_now()
            w_l, v_l = sensor_twist(f.R, f.v, w_b - f.bg, Tbl[:3, :3], Tbl[:3, 3])
            pts, nrm = deskew(pts, nrm, tr, w_l, v_l)
        T_init = make_T(f.R, f.p) @ T_bl_now()
        if prior_map is not None:
            T, info = prior_map.register(pts, T_init)
        else:
            if lmap.tree is None:
                lmap.add(pts, nrm, T_init)
                return
            T, info = lmap.register(pts, T_init)
        if info["ok"] and info.get("frac", 0) > 0.5:
            if est_lever:
                ok, _ = f.update_lidar_lever(T[:3, 3], T[:3, :3] @ calib.T_bl[:3, :3].T, *lidar_sig, gate=400.0)
            else:
                T_wb = T @ inv_T(calib.T_bl)
                ok, _ = f.update_pose(T_wb[:3, 3], T_wb[:3, :3], *lidar_sig, gate=400.0)
            stats["lidar_updates" if ok else "lidar_rejected"] += 1
        else:
            stats["lidar_rejected"] += 1
        T_post = make_T(f.R, f.p) @ T_bl_now()
        if collect_scans:
            scan_log.append((pts, nrm, T_post))
        if prior_map is None and lmap.need_kf(T_post):
            lmap.add(pts, nrm, T_post)

    def on_cam(k):
        vio.process(seq["frames"][k])

    out_t, out_p, out_R, out_v, out_sig = [], [], [], [], []
    ei = 0
    for i in range(len(t) - 1):
        w = 0.5 * (gyro[i] + gyro[i + 1])
        a = 0.5 * (acc[i] + acc[i + 1])
        tc = t[i]
        while ei < len(events) and events[ei][0] <= t[i + 1]:
            te, kind, k = events[ei]
            if te > tc:
                f.propagate(w, a, te - tc)
                tc = te
            on_cam(k) if kind == 0 else on_lidar(k, w)
            ei += 1
        if t[i + 1] > tc:
            f.propagate(w, a, t[i + 1] - tc)
        if (i + 1) % log_every == 0:
            out_t.append(t[i + 1])
            out_p.append(f.p.copy())
            out_R.append(f.R.copy())
            out_v.append(f.v.copy())
            d = np.sqrt(np.maximum(np.diag(f.P)[:9], 0.0))
            out_sig.append(np.r_[d[0:3], d[3:6], np.degrees(d[6:9])])
            if est_lever:
                ext_trace.append(f.t_ext.copy())
    if use_cam:
        stats["cam_updates"] = vio.n_updates
        stats["features_used"] = vio.n_feats_used
    return dict(t=np.array(out_t), p=np.array(out_p), R=np.array(out_R), v=np.array(out_v),
                sig=np.array(out_sig), stats=stats,
                bg=f.bg.copy(), ba=f.ba.copy(), scan_log=scan_log,
                t_ext=None if not est_lever else f.t_ext.copy(), ext_trace=np.array(ext_trace))


def global_relocalise(prior_map, pts, T_guess, xy=8.0, yaw=np.radians(45), step_xy=4.0,
                      step_yaw=np.radians(15)):
    """Kidnapped-robot initialisation: coarse grid over (x, y, yaw) + short ICP, then refine."""
    sub = pts[np.random.default_rng(0).choice(len(pts), min(600, len(pts)), replace=False)]
    best = None
    for dyaw in np.arange(-yaw, yaw + 1e-9, step_yaw):
        for dx in np.arange(-xy, xy + 1e-9, step_xy):
            for dy in np.arange(-xy, xy + 1e-9, step_xy):
                T0 = T_guess.copy()
                T0[:3, :3] = exp_so3([0, 0, dyaw]) @ T_guess[:3, :3]
                T0[:3, 3] = T_guess[:3, 3] + [dx, dy, 0]
                T, info = icp_point_to_plane(sub, prior_map.tree, prior_map.pts, prior_map.nrm, T0,
                                             iters=8, d_max=(4.0, 1.5))
                if not info["ok"]:
                    continue
                score = info["frac"] - 0.5 * info["rms"]
                if best is None or score > best[0]:
                    best = (score, T)
    if best is None:
        return None, None
    T, info = prior_map.register(pts, best[1], iters=30)
    return T, info
