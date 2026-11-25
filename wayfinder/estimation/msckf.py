"""MSCKF visual update (Mourikis & Roumeliotis, 2007).

Each camera frame adds a clone of the IMU pose. A feature track is used once it is
lost or about to fall out of the window: triangulate it from the clone poses, linearise
the reprojection residuals, project onto the left null space of the feature Jacobian
(so the 3-D point never enters the state) and run one EKF update.
"""
import cv2
import numpy as np
from scipy.linalg import null_space
from scipy.stats import chi2

from ..geom import hat


class MSCKF:
    def __init__(self, filt, K, dist, T_bc, window=15, px_sigma=1.0, min_track=4):
        self.f = filt
        self.K, self.dist = K, dist
        self.R_bc, self.t_bc = T_bc[:3, :3], T_bc[:3, 3]
        self.window = window
        self.sig = px_sigma / K[0, 0]
        self.min_track = min_track
        self.tracks = {}         # feature id -> list of (clone_id, xn)
        self.next_cid = 0
        self.n_updates = 0
        self.n_feats_used = 0

    def process(self, frame):
        f = self.f
        cid = self.next_cid
        self.next_cid += 1
        f.add_clone(cid)
        xn = cv2.undistortPoints(frame["uv"].reshape(-1, 1, 2).astype(np.float64),
                                 self.K, self.dist).reshape(-1, 2)
        seen = set()
        for fid, x in zip(frame["ids"].tolist(), xn):
            self.tracks.setdefault(fid, []).append((cid, x))
            seen.add(fid)

        use = [fid for fid in self.tracks if fid not in seen]           # lost tracks
        if len(f.clones) > self.window:
            oldest = f.clones[0][0]
            use += [fid for fid in self.tracks if fid in seen and self.tracks[fid][0][0] == oldest]
        self._update([self.tracks.pop(fid) for fid in use])

        if len(f.clones) > self.window:
            oldest = f.clones[0][0]
            f.drop_clone(0)
            for fid in list(self.tracks):
                tr = [o for o in self.tracks[fid] if o[0] != oldest]
                if tr:
                    self.tracks[fid] = tr
                else:
                    del self.tracks[fid]

    # -------------------------------------------------------------------------
    def _cam_pose(self, c):
        _, p, R = c
        R_wc = R @ self.R_bc
        return R_wc, p + R @ self.t_bc

    def _triangulate(self, obs, cidx):
        A, b = [], []
        for cid, (u, v) in obs:
            R_wc, p_wc = self._cam_pose(self.f.clones[cidx[cid]])
            R_cw = R_wc.T
            t = -R_cw @ p_wc
            A.append(u * R_cw[2] - R_cw[0]); b.append(t[0] - u * t[2])
            A.append(v * R_cw[2] - R_cw[1]); b.append(t[1] - v * t[2])
        A, b = np.array(A), np.array(b)
        X, *_ = np.linalg.lstsq(A, b, rcond=None)
        # a few Gauss-Newton steps on the reprojection error
        for _ in range(5):
            J, r = [], []
            for cid, z in obs:
                R_wc, p_wc = self._cam_pose(self.f.clones[cidx[cid]])
                Pc = R_wc.T @ (X - p_wc)
                if Pc[2] < 0.5:
                    return None
                Jp = np.array([[1 / Pc[2], 0, -Pc[0] / Pc[2]**2], [0, 1 / Pc[2], -Pc[1] / Pc[2]**2]])
                J.append(Jp @ R_wc.T)
                r.append(z - Pc[:2] / Pc[2])
            J, r = np.vstack(J), np.concatenate(r)
            dX = np.linalg.lstsq(J, r, rcond=None)[0]
            X = X + dX
            if np.linalg.norm(dX) < 1e-4:
                break
        return X

    def _update(self, tracks):
        f = self.f
        cidx = {c[0]: i for i, c in enumerate(f.clones)}
        n = f.P.shape[0]
        Hs, rs = [], []
        for obs in tracks:
            obs = [o for o in obs if o[0] in cidx]
            if len(obs) < self.min_track:
                continue
            p_first = self._cam_pose(f.clones[cidx[obs[0][0]]])[1]
            p_last = self._cam_pose(f.clones[cidx[obs[-1][0]]])[1]
            X = self._triangulate(obs, cidx)
            if X is None:
                continue
            depth = np.linalg.norm(X - p_first)
            if np.linalg.norm(p_last - p_first) / depth < 0.02:     # too little parallax
                continue
            Hx = np.zeros((2 * len(obs), n))
            Hf = np.zeros((2 * len(obs), 3))
            r = np.zeros(2 * len(obs))
            ok = True
            for k, (cid, z) in enumerate(obs):
                i = cidx[cid]
                _, p_i, R_i = f.clones[i]
                Pb = R_i.T @ (X - p_i)
                Pc = self.R_bc.T @ (Pb - self.t_bc)
                if Pc[2] < 0.5:
                    ok = False
                    break
                Jp = np.array([[1 / Pc[2], 0, -Pc[0] / Pc[2]**2], [0, 1 / Pc[2], -Pc[1] / Pc[2]**2]])
                s = f.base + 6 * i
                Hx[2 * k:2 * k + 2, s:s + 3] = Jp @ (-self.R_bc.T @ R_i.T)
                Hx[2 * k:2 * k + 2, s + 3:s + 6] = Jp @ (self.R_bc.T @ hat(Pb))
                Hf[2 * k:2 * k + 2] = Jp @ (self.R_bc.T @ R_i.T)
                r[2 * k:2 * k + 2] = z - Pc[:2] / Pc[2]
            if not ok or np.abs(r).max() * self.K[0, 0] > 6.0:
                continue
            A = null_space(Hf.T)
            Ho, ro = A.T @ Hx, A.T @ r
            # per-feature Mahalanobis gate
            S = Ho @ f.P @ Ho.T + self.sig**2 * np.eye(len(ro))
            if ro @ np.linalg.solve(S, ro) > chi2.ppf(0.95, len(ro)):
                continue
            Hs.append(Ho)
            rs.append(ro)
        if not Hs:
            return
        H, r = np.vstack(Hs), np.concatenate(rs)
        if H.shape[0] > n:      # QR compression
            Q, Rq = np.linalg.qr(H)
            H, r = Rq, Q.T @ r
        f.update(H, r, self.sig**2 * np.eye(len(r)))
        self.n_updates += 1
        self.n_feats_used += len(Hs)
