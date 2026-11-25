"""Camera intrinsic calibration from checkerboard views (Zhang's method via OpenCV)."""
import cv2
import numpy as np

from ..geom import exp_so3


def synth_checkerboard_views(cam, rng, n_views=25, cols=9, rows=6, square=0.08, px_noise=0.25):
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    obj -= obj.mean(axis=0)
    objs, imgs = [], []
    tries = 0
    while len(objs) < n_views and tries < 5000:
        tries += 1
        R = exp_so3(rng.uniform(-0.6, 0.6, 3))
        z = rng.uniform(0.5, 1.3)
        t = np.array([rng.uniform(-0.25, 0.25) * z, rng.uniform(-0.18, 0.18) * z, z])
        Pc = obj @ R.T + t
        if (Pc[:, 2] < 0.2).any():
            continue
        uv = cam.project(Pc)
        if (uv[:, 0] < 5).any() or (uv[:, 0] > cam.w - 5).any() or (uv[:, 1] < 5).any() or (uv[:, 1] > cam.h - 5).any():
            continue
        uv += rng.normal(0, px_noise, uv.shape)
        objs.append(obj.copy())
        imgs.append(uv.astype(np.float32).reshape(-1, 1, 2))
    return objs, imgs


def calibrate_intrinsics(objs, imgs, size):
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None)
    return dict(rms_px=float(rms), K=K, dist=dist.ravel()[:5])
