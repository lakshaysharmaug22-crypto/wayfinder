"""Trajectory metrics (ATE / RPE), following the usual TUM/KITTI conventions."""
import numpy as np

from .geom import log_so3


def align_se3(est, gt):
    """Umeyama without scale: R, t such that gt ~ R est + t."""
    me, mg = est.mean(0), gt.mean(0)
    H = (est - me).T @ (gt - mg)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mg - R @ me


def ate(est_p, gt_p, align=True):
    if align:
        R, t = align_se3(est_p, gt_p)
        est_p = est_p @ R.T + t
    e = np.linalg.norm(est_p - gt_p, axis=1)
    return float(np.sqrt(np.mean(e * e))), e


def rpe(est_p, est_R, gt_p, gt_R, step):
    et, er = [], []
    for i in range(len(est_p) - step):
        j = i + step
        dg = gt_R[i].T @ (gt_p[j] - gt_p[i])
        de = est_R[i].T @ (est_p[j] - est_p[i])
        et.append(np.linalg.norm(dg - de))
        er.append(np.degrees(np.linalg.norm(log_so3((gt_R[i].T @ gt_R[j]).T @ (est_R[i].T @ est_R[j])))))
    et, er = np.array(et), np.array(er)
    return float(np.sqrt(np.mean(et**2))), float(np.sqrt(np.mean(er**2)))


def rot_err_deg(est_R, gt_R):
    return np.array([np.degrees(np.linalg.norm(log_so3(g.T @ e))) for e, g in zip(est_R, gt_R)])


def summarise(est_p, est_R, gt_p, gt_R, rate=20, align=True):
    a, e = ate(est_p, gt_p, align)
    t5, r5 = rpe(est_p, est_R, gt_p, gt_R, step=5 * rate)
    path = float(np.sum(np.linalg.norm(np.diff(gt_p, axis=0), axis=1)))
    return dict(ate_rmse_m=a, rpe5s_trans_m=t5, rpe5s_rot_deg=r5,
                final_err_m=float(np.linalg.norm(est_p[-1] - gt_p[-1])),
                drift_pct=float(100 * np.linalg.norm(est_p[-1] - gt_p[-1]) / path),
                rot_rmse_deg=float(np.sqrt(np.mean(rot_err_deg(est_R, gt_R) ** 2))),
                path_m=path), e
