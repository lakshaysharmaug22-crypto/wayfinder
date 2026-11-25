"""Ground-truth trajectories.

`FlightTrajectory`: quadrotor-style figure-eight; attitude follows from differential
flatness (thrust axis along a - g, yaw along velocity), so IMU data is physically
consistent.
`CalibrationTrajectory`: rig waved in front of the scene with rotation excitation on
all three axes, the motion you would perform for a Kalibr-style calibration.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .geom import G_W


class _Base:
    def pva(self, t):
        raise NotImplementedError

    def wobble(self, t):
        return np.zeros((len(t), 3))

    def R(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        _, v, a = self.pva(t)
        zb = a - G_W
        zb /= np.linalg.norm(zb, axis=1, keepdims=True)
        yaw = self.yaw(t, v)
        xc = np.c_[np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)]
        yb = np.cross(zb, xc)
        yb /= np.linalg.norm(yb, axis=1, keepdims=True)
        xb = np.cross(yb, zb)
        Rf = np.stack([xb, yb, zb], axis=2)
        Rw = Rot.from_rotvec(self.wobble(t)).as_matrix()
        return Rf @ Rw

    def yaw(self, t, v):
        return np.arctan2(v[:, 1], v[:, 0])

    def omega_body(self, t, h=1e-4):
        R0 = self.R(t - h)
        R1 = self.R(t + h)
        rel = np.einsum('nji,njk->nik', R0, R1)
        return Rot.from_matrix(rel).as_rotvec() / (2 * h)

    def specific_force(self, t):
        _, _, a = self.pva(t)
        R = self.R(t)
        return np.einsum('nji,nj->ni', R, a - G_W)


class FlightTrajectory(_Base):
    def __init__(self, lap=50.0, ax=32.0, ay=22.0, alt=24.0, phase=0.0, dz=3.0):
        self.w = 2 * np.pi / lap
        self.ax, self.ay, self.alt, self.ph, self.dz = ax, ay, alt, phase, dz

    def pva(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        w, ph = self.w, self.ph
        s1, c1 = np.sin(w * t + ph), np.cos(w * t + ph)
        s2, c2 = np.sin(2 * w * t + 2 * ph), np.cos(2 * w * t + 2 * ph)
        s3, c3 = np.sin(0.5 * w * t + ph), np.cos(0.5 * w * t + ph)
        p = np.c_[self.ax * s1, self.ay * s2, self.alt + self.dz * s3]
        v = np.c_[self.ax * w * c1, self.ay * 2 * w * c2, self.dz * 0.5 * w * c3]
        a = np.c_[-self.ax * w**2 * s1, -self.ay * 4 * w**2 * s2, -self.dz * 0.25 * w**2 * s3]
        return p, v, a


class CalibrationTrajectory(_Base):
    """Slow translation + strong roll/pitch/yaw oscillation around a hover point."""

    def __init__(self, center=(0.0, 0.0, 22.0), amp=0.5, rot_amp=(0.45, 0.40, 0.55),
                 freqs=(0.37, 0.29, 0.21), trans_freq=(0.31, 0.23, 0.17)):
        self.c = np.asarray(center, float)
        self.amp = amp
        self.ra = np.asarray(rot_amp)
        self.f = np.asarray(freqs)
        self.tf = np.asarray(trans_freq)

    def pva(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        w = self.tf * 2 * np.pi
        p = self.c + self.amp * np.sin(np.outer(t, w))
        v = self.amp * w * np.cos(np.outer(t, w))
        a = -self.amp * w**2 * np.sin(np.outer(t, w))
        return p, v, a

    def yaw(self, t, v):
        return 0.3 * np.ones_like(t)

    def wobble(self, t):
        return self.ra * np.sin(2 * np.pi * np.outer(t, self.f) + np.array([0.0, 1.1, 2.3]))


class HandheldTrajectory(CalibrationTrajectory):
    """Rig shaken by hand in front of a calibration target: orientation is just the commanded
    wobble (no thrust-tilt coupling), small fast oscillations keep the target in view."""

    def R_nominal(self):
        return Rot.from_euler("z", 0.3).as_matrix()

    def R(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        return self.R_nominal() @ Rot.from_rotvec(self.wobble(t)).as_matrix()
