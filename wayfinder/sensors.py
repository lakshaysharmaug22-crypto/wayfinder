"""Sensor models: MEMS IMU, pinhole camera with OpenCV distortion, multi-beam LiDAR."""
from dataclasses import dataclass, field

import numpy as np

from .geom import make_T, exp_so3


# ---------------------------------------------------------------- IMU
@dataclass
class ImuParams:
    rate: float = 200.0
    gyro_nd: float = 1.7e-4   # rad/s/sqrt(Hz)
    acc_nd: float = 2.0e-3    # m/s^2/sqrt(Hz)
    gyro_rw: float = 1.9e-5   # rad/s^2/sqrt(Hz)
    acc_rw: float = 3.0e-3    # m/s^3/sqrt(Hz)


def simulate_imu(traj, duration, p: ImuParams, rng, bg0=None, ba0=None):
    dt = 1.0 / p.rate
    t = np.arange(0.0, duration, dt)
    n = len(t)
    bg0 = np.zeros(3) if bg0 is None else bg0
    ba0 = np.zeros(3) if ba0 is None else ba0
    bg = bg0 + np.cumsum(rng.normal(0, p.gyro_rw * np.sqrt(dt), (n, 3)), axis=0)
    ba = ba0 + np.cumsum(rng.normal(0, p.acc_rw * np.sqrt(dt), (n, 3)), axis=0)
    gyro = traj.omega_body(t) + bg + rng.normal(0, p.gyro_nd / np.sqrt(dt), (n, 3))
    acc = traj.specific_force(t) + ba + rng.normal(0, p.acc_nd / np.sqrt(dt), (n, 3))
    return dict(t=t, gyro=gyro, acc=acc, bg=bg, ba=ba)


def simulate_static_imu(hours, p: ImuParams, rng):
    """Static log for Allan-variance analysis (gravity on z)."""
    dt = 1.0 / p.rate
    n = int(hours * 3600 * p.rate)
    gyro = (np.cumsum(rng.normal(0, p.gyro_rw * np.sqrt(dt), (n, 3)), axis=0)
            + rng.normal(0, p.gyro_nd / np.sqrt(dt), (n, 3)))
    acc = (np.cumsum(rng.normal(0, p.acc_rw * np.sqrt(dt), (n, 3)), axis=0)
           + rng.normal(0, p.acc_nd / np.sqrt(dt), (n, 3)) + np.array([0, 0, 9.81]))
    return dict(dt=dt, gyro=gyro, acc=acc)


# ---------------------------------------------------------------- camera
@dataclass
class Camera:
    fx: float = 420.0
    fy: float = 418.0
    cx: float = 322.0
    cy: float = 238.0
    dist: np.ndarray = field(default_factory=lambda: np.array([-0.18, 0.05, 8e-4, -6e-4, 0.0]))
    w: int = 640
    h: int = 480

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def distort(self, xn):
        k1, k2, p1, p2, k3 = self.dist
        x, y = xn[:, 0], xn[:, 1]
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.c_[xd, yd]

    def project(self, Pc):
        xn = Pc[:, :2] / Pc[:, 2:3]
        xd = self.distort(xn)
        return np.c_[self.fx * xd[:, 0] + self.cx, self.fy * xd[:, 1] + self.cy]


def default_extrinsics():
    """True and nominal (CAD) sensor-to-body extrinsics.

    Body = IMU frame (x fwd, y left, z up). Camera is OpenCV (x right, y down, z fwd),
    pitched 20 deg down. Ground truth carries a few-degree mounting error the
    estimator does not know about.
    """
    R_bc0 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    tilt = exp_so3([0, np.radians(20), 0])           # pitch down in body
    R_bc_nom = tilt @ R_bc0
    R_bc_true = R_bc_nom @ exp_so3(np.radians([1.6, -2.3, 1.1]))
    t_bc = np.array([0.12, 0.02, -0.06])
    t_bc_nom = t_bc + np.array([0.018, -0.022, 0.012])    # CAD lever-arm error ~3 cm

    R_bl_nom = np.eye(3)
    R_bl_true = exp_so3(np.radians([0.9, 1.4, -2.6]))
    t_bl = np.array([0.0, 0.0, 0.09])
    t_bl_nom = t_bl + np.array([0.015, -0.02, 0.012])
    return dict(
        T_bc_true=make_T(R_bc_true, t_bc), T_bc_nom=make_T(R_bc_nom, t_bc_nom),
        T_bl_true=make_T(R_bl_true, t_bl), T_bl_nom=make_T(R_bl_nom, t_bl_nom),
        td_cam=0.0123,    # camera stamp = capture time + td  [s]
        td_lidar=-0.0071,
    )


def simulate_camera(world_lms, traj, T_bc, cam: Camera, rate, duration, td, rng,
                    max_feats=120, px_noise=0.5, drop=0.05, outlier=0.02):
    """Feature tracks with persistent ids (ideal data association + noise/outliers)."""
    prio = rng.permutation(len(world_lms))
    t_cap = np.arange(0.0, duration, 1.0 / rate) + 0.013
    R_wb = traj.R(t_cap)
    p_wb, _, _ = traj.pva(t_cap)
    frames = []
    for k, tc in enumerate(t_cap):
        R_wc = R_wb[k] @ T_bc[:3, :3]
        p_wc = p_wb[k] + R_wb[k] @ T_bc[:3, 3]
        Pc = (world_lms - p_wc) @ R_wc
        m = (Pc[:, 2] > 4.0) & (Pc[:, 2] < 90.0)
        idx = np.nonzero(m)[0]
        uv = cam.project(Pc[idx])
        inim = (uv[:, 0] > 2) & (uv[:, 0] < cam.w - 2) & (uv[:, 1] > 2) & (uv[:, 1] < cam.h - 2)
        idx, uv = idx[inim], uv[inim]
        order = np.argsort(prio[idx])[:max_feats]
        idx, uv = idx[order], uv[order]
        keep = rng.random(len(idx)) > drop
        idx, uv = idx[keep], uv[keep] + rng.normal(0, px_noise, (keep.sum(), 2))
        bad = rng.random(len(idx)) < outlier
        uv[bad] += rng.uniform(15, 40, (bad.sum(), 2)) * rng.choice([-1, 1], (bad.sum(), 2))
        frames.append(dict(stamp=tc + td, t_true=tc, ids=idx, uv=uv))
    return frames


# ---------------------------------------------------------------- LiDAR
@dataclass
class Lidar:
    rings: int = 32
    el_min: float = np.radians(-60)
    el_max: float = np.radians(20)
    az_res: float = np.radians(1.0)
    rmin: float = 3.0
    rmax: float = 80.0
    range_noise: float = 0.02


def visible_points(points_w, T_wl, lid: Lidar):
    """Indices of world points that win the range-image z-buffer from pose T_wl, and their azimuth."""
    R, t = T_wl[:3, :3], T_wl[:3, 3]
    d = points_w - t
    near = np.nonzero(np.abs(d).max(axis=1) < lid.rmax)[0]
    P = d[near] @ R
    r = np.linalg.norm(P, axis=1)
    el = np.arcsin(np.clip(P[:, 2] / np.maximum(r, 1e-9), -1, 1))
    m = (r > lid.rmin) & (r < lid.rmax) & (el > lid.el_min) & (el < lid.el_max)
    near, P, r, el = near[m], P[m], r[m], el[m]
    az = np.arctan2(P[:, 1], P[:, 0])
    ring = np.round((el - lid.el_min) / (lid.el_max - lid.el_min) * (lid.rings - 1)).astype(np.int64)
    azb = np.floor((az + np.pi) / lid.az_res).astype(np.int64)
    cell = ring * 10000 + azb
    order = np.lexsort((r, cell))
    cs = cell[order]
    sel = order[np.r_[True, cs[1:] != cs[:-1]]]
    return near[sel], az[sel]


def simulate_lidar(world, traj, T_bl, lid: Lidar, rate, duration, td, rng, distort=True):
    """Spinning LiDAR: azimuth sweeps -pi..pi over one period, so every point is measured from the
    vehicle pose at its own time. A scan is stamped at the END of its sweep (+ td clock offset);
    `tr` is each point's time relative to that end (<= 0)."""
    T = 1.0 / rate
    t_end = (np.arange(int(duration * rate)) + 1) * T + 0.021
    t_end = t_end[t_end < duration - 0.05]
    R_b, p_b = traj.R(t_end), traj.pva(t_end)[0]
    scans = []
    for k, te in enumerate(t_end):
        T_end = make_T(R_b[k], p_b[k]) @ T_bl
        idx, az = visible_points(world.points, T_end, lid)
        X = world.points[idx]
        if distort:
            ti = te - T + (az + np.pi) / (2 * np.pi) * T
            Rb, pb = traj.R(ti), traj.pva(ti)[0]
            R_wl = Rb @ T_bl[:3, :3]
            p_wl = pb + Rb @ T_bl[:3, 3]
            P = np.einsum('nji,nj->ni', R_wl, X - p_wl)
            tr = ti - te
        else:
            P = (X - T_end[:3, 3]) @ T_end[:3, :3]
            tr = np.zeros(len(X))
        r = np.linalg.norm(P, axis=1)
        P = P * ((r + rng.normal(0, lid.range_noise, len(r))) / r)[:, None]
        scans.append(dict(stamp=te + td, t_true=te, pts=P.astype(np.float32), tr=tr.astype(np.float32)))
    return scans


def make_target(cols=11, rows=8, spacing=0.10):
    """Planar calibration target (corner grid) in its own frame, centred, z = 0."""
    obj = np.zeros((rows * cols, 3))
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * spacing
    obj[:, :2] -= obj[:, :2].mean(axis=0)
    return obj


def place_target(traj, T_bc, distance=1.6):
    """Target pose T_wt: `distance` in front of the camera at t=0, facing it."""
    R0, p0 = traj.R_nominal(), traj.c.copy()          # mean pose of the motion, not its t=0 sample
    R_wc, p_wc = R0 @ T_bc[:3, :3], p0 + R0 @ T_bc[:3, 3]
    z = R_wc[:, 2]                      # optical axis
    zt = -z
    xt = np.cross([0, 0, 1.0], zt)
    xt /= np.linalg.norm(xt)
    yt = np.cross(zt, xt)
    return make_T(np.c_[xt, yt, zt], p_wc + distance * z)


def simulate_bench_camera(traj, T_bc, cam: Camera, obj, T_wt, rate, duration, td, rng, px_noise=0.4):
    t_cap = np.arange(0.0, duration, 1.0 / rate) + 0.013
    R_wb, p_wb = traj.R(t_cap), traj.pva(t_cap)[0]
    Xw = obj @ T_wt[:3, :3].T + T_wt[:3, 3]
    frames = []
    for k, tc in enumerate(t_cap):
        R_wc = R_wb[k] @ T_bc[:3, :3]
        p_wc = p_wb[k] + R_wb[k] @ T_bc[:3, 3]
        Pc = (Xw - p_wc) @ R_wc
        if (Pc[:, 2] < 0.3).any():
            continue
        uv = cam.project(Pc)
        if (uv[:, 0] < 4).any() or (uv[:, 0] > cam.w - 4).any() or (uv[:, 1] < 4).any() or (uv[:, 1] > cam.h - 4).any():
            continue
        frames.append(dict(stamp=tc + td, t_true=tc, uv=uv + rng.normal(0, px_noise, uv.shape)))
    return frames
