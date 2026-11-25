"""Point-to-plane ICP, keyframe local map, and a static prior map."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

from ..geom import exp_so3, make_T


def voxel_downsample(pts, voxel, *extra):
    keys = np.floor(pts / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    lin = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, idx = np.unique(lin, return_index=True)
    return (pts[idx],) + tuple(e[idx] for e in extra)


def estimate_normals(pts, k=10, max_curv=0.06):
    tree = cKDTree(pts)
    d, idx = tree.query(pts, k=k)
    nb = pts[idx]
    c = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', c, c) / k
    evals, evecs = np.linalg.eigh(cov)
    nrm = evecs[:, :, 0]
    curv = evals[:, 0] / np.maximum(evals.sum(axis=1), 1e-12)
    good = (curv < max_curv) & (d[:, -1] < 4.0)
    return nrm, good


def prep_scan(pts, tr, voxel=0.8):
    """Normals on the raw scan, then voxel downsample. Returns sensor-frame pts, normals and the
    per-point time offset `tr` (<= 0, relative to the end of the sweep)."""
    pts = pts.astype(np.float64)
    nrm, good = estimate_normals(pts)
    p, n, t = voxel_downsample(pts[good], voxel, nrm[good], np.asarray(tr, np.float64)[good])
    return p, n, t


def deskew(pts, nrm, tr, w_l, v_l):
    """Constant-twist motion compensation to the end-of-sweep frame (exact SE(3) exponential).

    The sensor at time t_i = t_end + tr_i sat at Exp(tr_i * [w, v]) relative to its end pose, so
    P_end = Exp(w tr_i) P_i + J_l(w tr_i) v tr_i.  w, v are expressed in the end-of-sweep frame.
    """
    th = np.outer(tr, w_l)
    a = np.linalg.norm(th, axis=1)
    small = a < 1e-6
    a_s = np.where(small, 1.0, a)
    c1 = np.where(small, 0.5, (1 - np.cos(a_s)) / a_s**2)
    c2 = np.where(small, 1 / 6.0, (a_s - np.sin(a_s)) / a_s**3)
    # J_l v = v + c1 (th x v) + c2 th x (th x v)
    tv = np.outer(tr, v_l)
    c = np.cross(th, tv)
    p_rel = tv + c1[:, None] * c + c2[:, None] * np.cross(th, c)
    Rr = Rot.from_rotvec(th)
    return Rr.apply(pts) + p_rel, Rr.apply(nrm)


def sensor_twist(R_wb, v_w, w_b, R_bl, t_bl):
    """Angular / linear velocity of the LiDAR frame, in its own frame."""
    w_l = R_bl.T @ w_b
    v_world = v_w + R_wb @ np.cross(w_b, t_bl)
    return w_l, (R_wb @ R_bl).T @ v_world


def icp_point_to_plane(src, tree, map_pts, map_nrm, T_init, iters=20, d_max=(3.0, 1.0)):
    R = T_init[:3, :3].copy()
    t = T_init[:3, 3].copy()
    info = dict(ok=False, rms=np.inf, inliers=0)
    for it in range(iters):
        md = d_max[0] if it < 4 else d_max[1]
        pw = src @ R.T + t
        d, j = tree.query(pw, distance_upper_bound=md)
        ok = np.isfinite(d)
        if ok.sum() < 40:
            return make_T(R, t), info
        p, q, n = pw[ok], map_pts[j[ok]], map_nrm[j[ok]]
        r = np.einsum('ij,ij->i', n, p - q)
        s = 1.4826 * np.median(np.abs(r)) + 1e-3
        w = np.where(np.abs(r) < 1.5 * s, 1.0, 1.5 * s / np.abs(r))
        J = np.hstack([n, np.cross(p, n)])
        A = J.T @ (J * w[:, None])
        b = -J.T @ (w * r)
        dx = np.linalg.solve(A + 1e-6 * np.eye(6), b)
        dR = exp_so3(dx[3:])
        R = dR @ R
        t = dR @ t + dx[:3]
        info = dict(ok=True, rms=float(np.sqrt(np.mean(r * r))), inliers=int(ok.sum()),
                    frac=float(ok.mean()), H=A)
        if np.linalg.norm(dx[:3]) < 1e-4 and np.linalg.norm(dx[3:]) < 1e-5:
            break
    return make_T(R, t), info


class LocalMap:
    """Sliding window of keyframe scans in world frame."""

    def __init__(self, n_kf=12, kf_dist=1.5, kf_ang=np.radians(10)):
        self.n_kf, self.kf_dist, self.kf_ang = n_kf, kf_dist, kf_ang
        self.kfs = []
        self.last_T = None
        self.tree = None

    def need_kf(self, T):
        if self.last_T is None:
            return True
        dp = np.linalg.norm(T[:3, 3] - self.last_T[:3, 3])
        dR = self.last_T[:3, :3].T @ T[:3, :3]
        ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        return dp > self.kf_dist or ang > self.kf_ang

    def add(self, pts_s, nrm_s, T_wl):
        R, t = T_wl[:3, :3], T_wl[:3, 3]
        self.kfs.append((pts_s @ R.T + t, nrm_s @ R.T))
        self.kfs = self.kfs[-self.n_kf:]
        self.pts = np.vstack([k[0] for k in self.kfs])
        self.nrm = np.vstack([k[1] for k in self.kfs])
        self.tree = cKDTree(self.pts)
        self.last_T = T_wl.copy()

    def register(self, pts_s, T_init):
        return icp_point_to_plane(pts_s, self.tree, self.pts, self.nrm, T_init)


class PriorMap:
    """Static global map (built offline from a survey flight)."""

    def __init__(self, scans_and_poses, voxel=0.5):
        P, N = [], []
        for pts_s, nrm_s, T in scans_and_poses:
            P.append(pts_s @ T[:3, :3].T + T[:3, 3])
            N.append(nrm_s @ T[:3, :3].T)
        P, N = np.vstack(P), np.vstack(N)
        self.pts, self.nrm = voxel_downsample(P, voxel, N)
        self.tree = cKDTree(self.pts)

    def register(self, pts_s, T_init, **kw):
        return icp_point_to_plane(pts_s, self.tree, self.pts, self.nrm, T_init, **kw)
