"""Error-state Kalman filter with optional stochastic clones (MSCKF-ready).

Nominal IMU state: p, v, R (world<-body), b_g, b_a.
Error state: [dp, dv, dtheta, db_g, db_a] (15) + 6 per clone [dp_i, dtheta_i],
with R_true = R * Exp(dtheta) (local perturbation).
"""
import numpy as np

from ..geom import G_W, exp_so3, hat, log_so3


class ESKF:
    def __init__(self, p, v, R, imu_params, P0=None, t_ext0=None, sig_ext=0.05):
        """t_ext0: if given, the LiDAR lever arm t_bl is added to the state (3 extra error states
        right after the 15 IMU states) and estimated online."""
        self.p, self.v, self.R = p.copy(), v.copy(), R.copy()
        self.t_ext = None if t_ext0 is None else np.asarray(t_ext0, float).copy()
        self.n_ext = 0 if t_ext0 is None else 3
        self.base = 15 + self.n_ext         # first clone's error-state index
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.ip = imu_params
        self.clones = []     # list of [id, p, R]
        self.P = np.diag(np.r_[[0.01] * 3, [0.01] * 3, [np.radians(0.5)**2] * 3,
                               [0.005**2] * 3, [0.05**2] * 3, [sig_ext**2] * self.n_ext]) if P0 is None else P0
        ip = imu_params
        self.Qc = np.r_[[0.0] * 3, [ip.acc_nd**2] * 3, [ip.gyro_nd**2] * 3,
                        [ip.gyro_rw**2] * 3, [ip.acc_rw**2] * 3]

    # ------------------------------------------------------------ propagation
    def propagate(self, w_m, a_m, dt):
        w = w_m - self.bg
        a = a_m - self.ba
        Rk = self.R
        acc_w = Rk @ a + G_W
        self.p = self.p + self.v * dt + 0.5 * acc_w * dt * dt
        self.v = self.v + acc_w * dt
        self.R = Rk @ exp_so3(w * dt)

        F = np.zeros((15, 15))
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -Rk @ hat(a)
        F[3:6, 12:15] = -Rk
        F[6:9, 6:9] = -hat(w)
        F[6:9, 9:12] = -np.eye(3)
        Fd = np.eye(15) + F * dt + 0.5 * (F @ F) * dt * dt
        G = np.eye(15)
        G[3:6, 3:6] = Rk
        Qd = (G * self.Qc) @ G.T * dt
        P = self.P
        P[:15, :15] = Fd @ P[:15, :15] @ Fd.T + Qd
        if P.shape[0] > 15:
            P[:15, 15:] = Fd @ P[:15, 15:]
            P[15:, :15] = P[:15, 15:].T

    # ------------------------------------------------------------ clones
    def add_clone(self, cid):
        n = self.P.shape[0]
        J = np.zeros((6, n))
        J[0:3, 0:3] = np.eye(3)
        J[3:6, 6:9] = np.eye(3)
        Pn = np.zeros((n + 6, n + 6))
        Pn[:n, :n] = self.P
        Pn[n:, :n] = J @ self.P
        Pn[:n, n:] = Pn[n:, :n].T
        Pn[n:, n:] = J @ self.P @ J.T
        self.P = Pn
        self.clones.append([cid, self.p.copy(), self.R.copy()])

    def drop_clone(self, idx):
        s = self.base + 6 * idx
        keep = np.r_[0:s, s + 6:self.P.shape[0]]
        self.P = self.P[np.ix_(keep, keep)]
        del self.clones[idx]

    # ------------------------------------------------------------ updates
    def update(self, H, r, Rn, gate=None):
        S = H @ self.P @ H.T + Rn
        try:
            Si = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False, np.inf
        d2 = float(r @ Si @ r)
        if gate is not None and d2 > gate:
            return False, d2
        K = self.P @ H.T @ Si
        dx = K @ r
        self.p += dx[0:3]
        self.v += dx[3:6]
        self.R = self.R @ exp_so3(dx[6:9])
        self.bg += dx[9:12]
        self.ba += dx[12:15]
        if self.n_ext:
            self.t_ext = self.t_ext + dx[15:18]
        for i, c in enumerate(self.clones):
            s = self.base + 6 * i
            c[1] = c[1] + dx[s:s + 3]
            c[2] = c[2] @ exp_so3(dx[s + 3:s + 6])
        IKH = np.eye(len(dx)) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rn @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True, d2

    def update_pose(self, p_m, R_m, sig_p, sig_th, gate=None):
        r = np.r_[p_m - self.p, log_so3(self.R.T @ R_m)]
        H = np.zeros((6, self.P.shape[0]))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 6:9] = np.eye(3)
        Rn = np.diag(np.r_[[sig_p**2] * 3, [sig_th**2] * 3])
        return self.update(H, r, Rn, gate)

    def update_lidar_lever(self, p_l, R_wb_meas, sig_p, sig_th, gate=None):
        """Pose update from a LiDAR pose p_l (world) with the lever arm in the state:
        p_l = p + R t_bl.  d p_l = dp - R [t_bl]x dtheta + R dt_bl."""
        r = np.r_[p_l - (self.p + self.R @ self.t_ext), log_so3(self.R.T @ R_wb_meas)]
        H = np.zeros((6, self.P.shape[0]))
        H[0:3, 0:3] = np.eye(3)
        H[0:3, 6:9] = -self.R @ hat(self.t_ext)
        H[0:3, 15:18] = self.R
        H[3:6, 6:9] = np.eye(3)
        Rn = np.diag(np.r_[[sig_p**2] * 3, [sig_th**2] * 3])
        return self.update(H, r, Rn, gate)
