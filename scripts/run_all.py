"""Runs every experiment and writes results/metrics.json + web/data.json.

    python scripts/run_all.py            (~15 min on 2 CPU cores)
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wayfinder.world import World                                              # noqa: E402
from wayfinder.trajectory import FlightTrajectory, CalibrationTrajectory, HandheldTrajectory   # noqa: E402
from wayfinder.sensors import (Camera, Lidar, ImuParams, default_extrinsics, simulate_imu,    # noqa: E402
                               simulate_static_imu, simulate_camera, simulate_lidar,
                               make_target, place_target, simulate_bench_camera)
from wayfinder.geom import make_T, rot_angle_deg, exp_so3, inv_T                # noqa: E402
from wayfinder.calib.intrinsics import synth_checkerboard_views, calibrate_intrinsics   # noqa: E402
from wayfinder.calib.imu_noise import identify                                  # noqa: E402
from wayfinder.calib.extrinsic_time import (camera_relative_rotations, calibrate_lidar_imu,   # noqa: E402
                                            calibrate_rotation_and_offset, gyro_orientation)
from wayfinder.calib.lever_arm import pnp_poses, estimate_lever_arm              # noqa: E402
from wayfinder.estimation.pipeline import make_sequence, run, Calib, global_relocalise      # noqa: E402
from wayfinder.estimation.lidar_odom import PriorMap, prep_scan                 # noqa: E402
from wayfinder.eval import summarise                                            # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
T0 = time.time()
SMOKE = bool(os.environ.get("SMOKE"))        # SMOKE=1: short sequences, to check the plumbing only
OUT = os.path.join(ROOT, "results" if not SMOKE else "results/smoke")
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def r3(x):
    return np.round(np.asarray(x, float), 3).tolist()


world = World(seed=7)
rng = np.random.default_rng(3)
lms = world.points[rng.choice(len(world.points), 6000, replace=False)]
ex = default_extrinsics()
cam, lid, ip = Camera(), Lidar(), ImuParams()
M = dict(calibration={}, odometry={}, localisation={}, ablations={})
C = M["calibration"]

# =========================================================== 1. calibration
log("intrinsics")
objs, imgs = synth_checkerboard_views(cam, np.random.default_rng(5))
ci = calibrate_intrinsics(objs, imgs, (cam.w, cam.h))
K_hat, d_hat = ci["K"], ci["dist"]
C["intrinsics"] = dict(
    views=len(objs), reproj_rms_px=ci["rms_px"],
    true=dict(fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy, dist=r3(cam.dist)),
    est=dict(fx=float(K_hat[0, 0]), fy=float(K_hat[1, 1]), cx=float(K_hat[0, 2]),
             cy=float(K_hat[1, 2]), dist=np.round(d_hat, 4).tolist()))

log("allan variance (2 h static log)")
nid = identify(simulate_static_imu(2.0, ip, np.random.default_rng(6)))
C["imu_noise"] = dict(
    gyro=dict(N_true=ip.gyro_nd, N_est=nid["gyro"]["N"], K_true=ip.gyro_rw, K_est=nid["gyro"]["K"]),
    acc=dict(N_true=ip.acc_nd, N_est=nid["acc"]["N"], K_true=ip.acc_rw, K_est=nid["acc"]["K"]))
ip_hat = ImuParams(gyro_nd=nid["gyro"]["N"], acc_nd=nid["acc"]["N"],
                   gyro_rw=nid["gyro"]["K"], acc_rw=nid["acc"]["K"])

log("calibration sequence (60 s excitation)")
crng = np.random.default_rng(8)
ctr = CalibrationTrajectory()
D = 25.0 if SMOKE else 60.0
cimu = simulate_imu(ctr, D, ip, crng, bg0=np.array([0.003, -0.002, 0.0015]))
cframes = simulate_camera(lms, ctr, ex["T_bc_true"], cam, 20, D, ex["td_cam"], crng)
cscans = simulate_lidar(world, ctr, ex["T_bl_true"], lid, 10, D, ex["td_lidar"], crng)
slerp = gyro_orientation(cimu["t"], cimu["gyro"])
bg_true = cimu["bg"].mean(0)

conv = {}


def convergence(name, t1, t2, ph, R_true, td_true):
    conv[name] = []
    for secs in (5, 10, 20, 30, 45, 60):
        m = t2 < secs
        if m.sum() < 10:
            continue
        rr = calibrate_rotation_and_offset(t1[m], t2[m], ph[m], cimu["t"], cimu["gyro"], slerp=slerp)
        conv[name].append(dict(secs=secs, td_err_ms=1e3 * abs(rr["td"] - td_true),
                               rot_err_deg=rot_angle_deg(rr["R"].T @ R_true)))


# ---- camera <-> IMU: rotation + time offset + gyro bias
t1, t2, ph = camera_relative_rotations(cframes, K_hat, d_hat)
cal_c = calibrate_rotation_and_offset(t1, t2, ph, cimu["t"], cimu["gyro"], slerp=slerp)
C["camera_imu"] = dict(
    pairs=cal_c["n"], td_true_ms=1e3 * ex["td_cam"], td_est_ms=1e3 * cal_c["td"],
    td_err_ms=1e3 * abs(cal_c["td"] - ex["td_cam"]),
    rot_err_nominal_deg=rot_angle_deg(ex["T_bc_nom"][:3, :3].T @ ex["T_bc_true"][:3, :3]),
    rot_err_calibrated_deg=rot_angle_deg(cal_c["R"].T @ ex["T_bc_true"][:3, :3]),
    gyro_bias_est=r3(cal_c["bg"]), gyro_bias_true=r3(bg_true), residual_rms_deg=cal_c["rms_deg"])
convergence("camera", t1, t2, ph, ex["T_bc_true"][:3, :3], ex["td_cam"])
log("camera-IMU", C["camera_imu"])

# ---- LiDAR <-> IMU: rotation + time offset, with and without motion compensation
log("LiDAR-IMU calibration (de-skewed, two passes)")
cal_l = calibrate_lidar_imu(cscans, cimu["t"], cimu["gyro"], ex["T_bl_nom"][:3, :3])
log("LiDAR-IMU calibration (no de-skew)")
cal_l0 = calibrate_lidar_imu(cscans, cimu["t"], cimu["gyro"], ex["T_bl_nom"][:3, :3], use_deskew=False,
                             prepped=cal_l["prepped"])
R_bl_true = ex["T_bl_true"][:3, :3]
C["lidar_imu"] = dict(
    pairs=len(cal_l["t1"]), td_true_ms=1e3 * ex["td_lidar"], td_est_ms=1e3 * cal_l["td"],
    td_err_ms=1e3 * abs(cal_l["td"] - ex["td_lidar"]),
    rot_err_nominal_deg=rot_angle_deg(ex["T_bl_nom"][:3, :3].T @ R_bl_true),
    rot_err_calibrated_deg=rot_angle_deg(cal_l["R"].T @ R_bl_true),
    gyro_bias_est=r3(cal_l["bg"]), gyro_bias_true=r3(bg_true), residual_rms_deg=cal_l["rms_deg"],
    no_deskew=dict(td_est_ms=1e3 * cal_l0["td"], td_err_ms=1e3 * abs(cal_l0["td"] - ex["td_lidar"]),
                   rot_err_deg=rot_angle_deg(cal_l0["R"].T @ R_bl_true)))
convergence("lidar", cal_l["t1"], cal_l["t2"], cal_l["phi"], R_bl_true, ex["td_lidar"])
log("LiDAR-IMU", C["lidar_imu"])

# ---- camera lever arm: bench sequence, large near target, windowed double integration
log("camera lever arm (bench, target-based)")
brng = np.random.default_rng(12)
btr = HandheldTrajectory(center=(0, 0, 1.5), amp=0.03, rot_amp=(0.10, 0.09, 0.10),
                         freqs=(1.2, 0.9, 1.0), trans_freq=(1.0, 0.8, 0.7))
BD = 20.0 if SMOKE else 40.0
bimu = simulate_imu(btr, BD, ip, brng, bg0=np.array([0.003, -0.002, 0.0015]))
tgt = make_target(12, 9, 0.10)
T_wt = place_target(btr, ex["T_bc_true"], 1.2)
bframes = simulate_bench_camera(btr, ex["T_bc_true"], cam, tgt, T_wt, 20, BD, ex["td_cam"], brng)
bt, bR, bp = pnp_poses(bframes, tgt, K_hat, d_hat)
lv = estimate_lever_arm(bt - cal_c["td"], bR, bp, cal_c["R"], bimu["t"], bimu["gyro"], bimu["acc"],
                        cal_c["bg"])
t_bc_true = ex["T_bc_true"][:3, 3]
C["camera_lever_arm"] = dict(
    frames=len(bframes), windows=lv["n_windows"], true=r3(t_bc_true), cad=r3(ex["T_bc_nom"][:3, 3]),
    est=r3(lv["t_bs"]), err_cm=100 * float(np.linalg.norm(lv["t_bs"] - t_bc_true)),
    cad_err_cm=100 * float(np.linalg.norm(ex["T_bc_nom"][:3, 3] - t_bc_true)),
    method="windowed double integration, per-window accel bias/gravity")
log("camera lever arm", C["camera_lever_arm"])

# ---- LiDAR lever arm: online extrinsic state in the ESKF (FAST-LIO2 style) - reported as measured
log("LiDAR lever arm (online ESKF state)")
cseq = dict(traj=ctr, duration=D, imu=cimu, scans=cscans)
for s, pp in zip(cscans, cal_l["prepped"]):
    s["prep"] = pp
cal_for_ext = Calib(K_hat, d_hat, ex["T_bc_nom"], make_T(cal_l["R"], ex["T_bl_nom"][:3, 3]), 0.0, cal_l["td"])
r_ext = run(cseq, cal_for_ext, ip_hat, use_lidar=True, est_lever=True)
t_bl_true = ex["T_bl_true"][:3, 3]
C["lidar_lever_arm"] = dict(
    true=r3(t_bl_true), cad=r3(ex["T_bl_nom"][:3, 3]), est=r3(r_ext["t_ext"]),
    err_cm=100 * float(np.linalg.norm(r_ext["t_ext"] - t_bl_true)),
    cad_err_cm=100 * float(np.linalg.norm(ex["T_bl_nom"][:3, 3] - t_bl_true)),
    adopted=False)
log("LiDAR lever arm", C["lidar_lever_arm"])

# ---- assemble calibrations used downstream
t_bc_est = lv["t_bs"]
T_bc_est = make_T(cal_c["R"], t_bc_est)
T_bl_est = make_T(cal_l["R"], ex["T_bl_nom"][:3, 3])        # LiDAR lever arm stays at CAD (see above)
calib_full = Calib(K_hat, d_hat, T_bc_est, T_bl_est, cal_c["td"], cal_l["td"])
K_nom = np.array([[420.0, 0, 320], [0, 420.0, 240], [0, 0, 1]])
calib_nom = Calib(K_nom, np.zeros(5), ex["T_bc_nom"], ex["T_bl_nom"], 0.0, 0.0)
calib_no_td = Calib(K_hat, d_hat, T_bc_est, T_bl_est, 0.0, 0.0)
calib_cad_lever = Calib(K_hat, d_hat, make_T(cal_c["R"], ex["T_bc_nom"][:3, 3]),
                        make_T(cal_l["R"], ex["T_bl_nom"][:3, 3]), cal_c["td"], cal_l["td"])

# =========================================================== 2. odometry
log("flight sequence (100 s)")
ftr = FlightTrajectory()
FD = 25.0 if SMOKE else 100.0
seq = make_sequence(world, lms, ftr, FD, ex, cam, lid, ip, seed=11)


def evaluate(r, traj, align=True):
    gt_p = traj.pva(r["t"])[0]
    gt_R = traj.R(r["t"])
    s, e = summarise(r["p"], r["R"], gt_p, gt_R, align=align)
    s.update(r["stats"])
    return s, e


runs = {}
for name, kw in [("IMU only", dict(calib=calib_full)),
                 ("VIO (MSCKF)", dict(calib=calib_full, use_cam=True)),
                 ("LIO", dict(calib=calib_full, use_lidar=True)),
                 ("VIO + LIO", dict(calib=calib_full, use_cam=True, use_lidar=True)),
                 ("VIO, uncalibrated", dict(calib=calib_nom, use_cam=True)),
                 ("LIO, uncalibrated", dict(calib=calib_nom, use_lidar=True)),
                 ("VIO, no time sync", dict(calib=calib_no_td, use_cam=True)),
                 ("LIO, no time sync", dict(calib=calib_no_td, use_lidar=True)),
                 ("VIO, CAD lever arm", dict(calib=calib_cad_lever, use_cam=True)),
                 ("LIO, no de-skew", dict(calib=calib_full, use_lidar=True, use_deskew=False))]:
    c = kw.pop("calib")
    r = run(seq, c, ip_hat, **kw)
    s, e = evaluate(r, ftr)
    runs[name] = (r, e)
    M["odometry"][name] = s
    log(name, {k: round(v, 3) for k, v in s.items() if isinstance(v, float)})

log("time-offset sensitivity (VIO)")
sens = []
for err_ms in ((0, 10) if SMOKE else (0, 5, 10, 20, 40)):
    if err_ms == 0:
        s = M["odometry"]["VIO (MSCKF)"]
    else:
        c = Calib(K_hat, d_hat, T_bc_est, T_bl_est, calib_full.td_cam + err_ms / 1e3, 0)
        s, _ = evaluate(run(seq, c, ip_hat, use_cam=True), ftr)
    sens.append(dict(td_err_ms=err_ms, ate_m=s["ate_rmse_m"], rot_deg=s["rot_rmse_deg"]))
    log(err_ms, s["ate_rmse_m"])
M["ablations"]["vio_time_offset_error"] = sens

# =========================================================== 3. map-based localisation
# Survey: an earlier flight (own surface sample of the scene) flown with VIO+LIO; the map is built
# from the ESTIMATED poses and de-skewed scans, so it carries realistic drift and noise.
# Live flight: a different trajectory, a new surface sample and scene changes (trees removed, boxes added).
log("survey flight -> prior map (from VIO+LIO estimated poses)")
world_s = World(seed=7, sample_seed=101)
lms_s = world_s.points[np.random.default_rng(4).choice(len(world_s.points), 6000, replace=False)]
mtr = FlightTrajectory(lap=55, ax=34, ay=24, alt=27, phase=0.9)
mseq = make_sequence(world_s, lms_s, mtr, 30.0 if SMOKE else 110.0, ex, cam, lid, ip, seed=21, lidar_rate=5)
rs = run(mseq, calib_full, ip_hat, use_cam=True, use_lidar=True, collect_scans=True)
s_survey, _ = evaluate(rs, mtr)
M["localisation"]["survey_flight"] = s_survey
pm = PriorMap(rs["scan_log"])
log("prior map points", len(pm.pts), "survey ATE", round(s_survey["ate_rmse_m"], 3))

world_l = World(seed=7, sample_seed=202, changes=True)
ltr = FlightTrajectory(lap=45, ax=28, ay=25, alt=23, phase=2.1, dz=2.0)
lseq = make_sequence(world_l, None, ltr, 25.0 if SMOKE else 90.0, ex, cam, lid, ip, seed=31, with_cam=False)

t_first = lseq["scans"][0]["t_true"]
T_true0 = make_T(ltr.R([t_first])[0], ltr.pva([t_first])[0][0])
T_guess = T_true0.copy()
T_guess[:3, 3] += [5.0, -3.5, 0.8]
T_guess[:3, :3] = exp_so3([0, 0, np.radians(35)]) @ T_guess[:3, :3]
T_wl, info = global_relocalise(pm, lseq["scans"][0]["prep"][0], T_guess @ calib_full.T_bl)
T_init_b = T_wl @ inv_T(calib_full.T_bl)
M["localisation"]["kidnapped_start"] = dict(
    guess_err_m=float(np.linalg.norm(T_guess[:3, 3] - T_true0[:3, 3])),
    guess_err_deg=rot_angle_deg(T_guess[:3, :3].T @ T_true0[:3, :3]),
    recovered_err_m=float(np.linalg.norm(T_init_b[:3, 3] - T_true0[:3, 3])),
    recovered_err_deg=rot_angle_deg(T_init_b[:3, :3].T @ T_true0[:3, :3]))
log("kidnapped start", M["localisation"]["kidnapped_start"])

# start the filter at the time of the relocalised scan (velocity unknown, large prior)
k0 = np.searchsorted(lseq["imu"]["t"], t_first)
lseq_trim = dict(lseq, imu={k: v[k0:] for k, v in lseq["imu"].items() if k in ("t", "gyro", "acc")})
lseq_trim["scans"] = lseq["scans"][1:]
rl = run(lseq_trim, calib_full, ip_hat, use_lidar=True, prior_map=pm, init_T=T_init_b)
s, e_loc = evaluate(rl, ltr, align=False)
M["localisation"]["map_based"] = s
log("map-based localisation", {k: round(v, 3) for k, v in s.items() if isinstance(v, float)})
rl_odo = run(lseq_trim, calib_full, ip_hat, use_lidar=True, init_T=T_init_b)
s2, e_odo = evaluate(rl_odo, ltr, align=False)
M["localisation"]["odometry_same_flight_unaligned"] = s2
M["localisation"]["map_info"] = dict(points=int(len(pm.pts)), survey_world_sample=101, live_world_sample=202,
                                     live_changes="12% of trees removed, 4 boxes added")

log("relocalisation trials")
trng = np.random.default_rng(41)
trials = []
for _ in range(3 if SMOKE else 25):
    k = int(trng.integers(0, len(lseq["scans"])))
    sc = lseq["scans"][k]
    Tt = make_T(ltr.R([sc["t_true"]])[0], ltr.pva([sc["t_true"]])[0][0]) @ ex["T_bl_true"]
    off = np.r_[trng.uniform(-7, 7, 2), trng.uniform(-1, 1)]
    yaw = np.radians(trng.uniform(-40, 40))
    Tg = Tt.copy()
    Tg[:3, 3] += off
    Tg[:3, :3] = exp_so3([0, 0, yaw]) @ Tt[:3, :3]
    T, inf = global_relocalise(pm, sc["prep"][0], Tg)
    pe = float(np.linalg.norm(T[:3, 3] - Tt[:3, 3])) if T is not None else 99.0
    re = rot_angle_deg(T[:3, :3].T @ Tt[:3, :3]) if T is not None else 99.0
    trials.append(dict(off_m=float(np.linalg.norm(off)), off_deg=float(abs(np.degrees(yaw))),
                       err_m=pe, err_deg=re, success=bool(pe < 0.5 and re < 2.0)))
M["localisation"]["relocalisation_trials"] = dict(
    n=len(trials), criterion="< 0.5 m and < 2 deg", success_rate=float(np.mean([t["success"] for t in trials])),
    median_err_m=float(np.median([t["err_m"] for t in trials])), trials=trials)
log("relocalisation success", M["localisation"]["relocalisation_trials"]["success_rate"])

# =========================================================== 4. export
json.dump(M, open(os.path.join(OUT, "metrics.json"), "w"), indent=2, default=float)

sub = lambda a, k=2: a[::k]
gt_t = runs["LIO"][0]["t"]
web = dict(
    meta=dict(duration=FD, generated=time.strftime("%Y-%m-%d")),
    metrics=M,
    world=r3(world.points[np.random.default_rng(0).choice(len(world.points), 7000, replace=False)]),
    gt=r3(sub(ftr.pva(gt_t)[0])),
    t=r3(sub(gt_t)),
    runs={n: dict(p=r3(sub(r["p"])), err=r3(sub(e))) for n, (r, e) in runs.items() if n != "IMU only"},
    imu_only_err=r3(sub(runs["IMU only"][1])),
    calib=dict(
        allan=dict(taus=r3(nid["gyro"]["taus"]), gyro=np.round(nid["gyro"]["adev"][0], 8).tolist(),
                   acc=np.round(nid["acc"]["adev"][0], 7).tolist(), acc_taus=r3(nid["acc"]["taus"])),
        corr={"camera": dict(grid=r3(1e3 * cal_c["corr_grid"]), corr=r3(cal_c["corr"])),
              "lidar": dict(grid=r3(1e3 * cal_l["corr_grid"]), corr=r3(cal_l["corr"]))},
        convergence=conv, sensitivity=sens),
    loc=dict(t=r3(sub(rl["t"])), gt=r3(sub(ltr.pva(rl["t"])[0])), p=r3(sub(rl["p"])),
             err=r3(sub(e_loc)), odo_err=r3(sub(e_odo)), odo_p=r3(sub(rl_odo["p"])),
             map=r3(pm.pts[np.random.default_rng(2).choice(len(pm.pts), min(7000, len(pm.pts)), replace=False)])),
)
WEB = os.path.join(ROOT, "web") if not SMOKE else OUT
os.makedirs(WEB, exist_ok=True)
json.dump(web, open(os.path.join(WEB, "data.json"), "w"), separators=(",", ":"), default=float)
log("done")
