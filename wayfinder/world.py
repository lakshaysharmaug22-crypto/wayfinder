"""Procedural outdoor scene: rolling terrain, trees, buildings.

The scene is a dense surface point cloud. LiDAR scans are rendered from it with a range-image
z-buffer, camera features are sampled from it. Geometry (`seed`) and surface sampling
(`sample_seed`) are independent, so two `World`s with the same seed describe the same place but
share no sample points; `changes=True` additionally removes trees and adds parked boxes, as if
the survey were some weeks old.
"""
import numpy as np


def terrain_height(x, y):
    return (3.0 * np.sin(0.045 * x + 0.3) * np.cos(0.035 * y)
            + 2.0 * np.sin(0.09 * x + 0.1 * y + 1.0)
            + 1.2 * np.cos(0.13 * y - 0.5) + 5.0)


def _box(rng, w, d, h, n=1500):
    face = rng.integers(0, 5, n)
    u = rng.uniform(-0.5, 0.5, n)
    v = rng.uniform(0, 1, n)
    pts = np.zeros((n, 3))
    for f in range(5):
        m = face == f
        k = m.sum()
        if f == 0:
            pts[m] = np.c_[u[m] * w, np.full(k, -d / 2), v[m] * h]
        elif f == 1:
            pts[m] = np.c_[u[m] * w, np.full(k, d / 2), v[m] * h]
        elif f == 2:
            pts[m] = np.c_[np.full(k, -w / 2), u[m] * d, v[m] * h]
        elif f == 3:
            pts[m] = np.c_[np.full(k, w / 2), u[m] * d, v[m] * h]
        else:
            pts[m] = np.c_[u[m] * w, (v[m] - 0.5) * d, np.full(k, h)]
    return pts


class World:
    def __init__(self, seed=7, half=75.0, sample_seed=None, changes=False):
        geo = np.random.default_rng(seed)
        smp = np.random.default_rng(seed if sample_seed is None else sample_seed)
        trees = [(*geo.uniform(-60, 60, 2), geo.uniform(5.0, 8.0), geo.uniform(1.8, 2.8)) for _ in range(140)]
        blds = [(*geo.uniform(-55, 55, 2), *geo.uniform(6, 12, 2), geo.uniform(5, 9)) for _ in range(10)]
        extra = []
        if changes:
            ch = np.random.default_rng(seed + 1000)
            trees = [t for t in trees if ch.random() > 0.12]
            extra = [(*ch.uniform(-50, 50, 2), ch.uniform(4, 6), 2.4, 2.6) for _ in range(4)]

        g = np.arange(-half, half, 0.6)
        X, Y = np.meshgrid(g, g)
        X = X + smp.uniform(-0.25, 0.25, X.shape)
        Y = Y + smp.uniform(-0.25, 0.25, Y.shape)
        parts = [np.c_[X.ravel(), Y.ravel(), terrain_height(X, Y).ravel()]]
        for cx, cy, h, r in trees:
            gz = terrain_height(cx, cy)
            ang = smp.uniform(0, 2 * np.pi, 80)
            parts.append(np.c_[cx + 0.3 * np.cos(ang), cy + 0.3 * np.sin(ang), gz + smp.uniform(0, h, 80)])
            d = smp.normal(size=(150, 3))
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            parts.append(np.array([cx, cy, gz + h]) + r * d)
        for cx, cy, w, dd, h in blds + extra:
            parts.append(_box(smp, w, dd, h) + np.array([cx, cy, terrain_height(cx, cy)]))
        self.points = np.vstack(parts)
        self.half = half
