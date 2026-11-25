"""Small SO(3)/SE(3) helpers."""
import numpy as np

G_W = np.array([0.0, 0.0, -9.81])


def hat(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_so3(w):
    w = np.asarray(w, float)
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3) + hat(w)
    K = hat(w / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def log_so3(R):
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    if np.pi - th < 1e-5:
        # near pi: use symmetric part
        A = (R + np.eye(3)) / 2.0
        v = np.sqrt(np.maximum(np.diag(A), 0.0))
        i = np.argmax(v)
        v = A[:, i] / max(v[i], 1e-12)
        return th * v / np.linalg.norm(v)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th / (2.0 * np.sin(th)) * w


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return make_T(R.T, -R.T @ t)


def rot_angle_deg(R):
    return np.degrees(np.linalg.norm(log_so3(R)))
