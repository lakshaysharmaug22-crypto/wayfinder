import numpy as np

from wayfinder.geom import exp_so3, log_so3, rot_angle_deg
from wayfinder.sensors import (ImuParams, Camera, Lidar, simulate_imu, simulate_static_imu,
                               default_extrinsics, simulate_camera, simulate_lidar, make_target,
                               place_target, simulate_bench_camera)
from wayfinder.trajectory import FlightTrajectory, CalibrationTrajectory, HandheldTrajectory
from wayfinder.world import World
from wayfinder.estimation.eskf import ESKF
from wayfinder.estimation.lidar_odom import prep_scan, deskew, sensor_twist
from wayfinder.calib.imu_noise import identify
from wayfinder.calib.extrinsic_time import calibrate_rotation_and_offset, camera_relative_rotations
from wayfinder.calib.intrinsics import synth_checkerboard_views, calibrate_intrinsics
from wayfinder.calib.lever_arm import pnp_poses, estimate_lever_arm


def test_so3_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = rng.uniform(-3, 3, 3)
        assert np.allclose(log_so3(exp_so3(w)), w, atol=1e-8) or np.linalg.norm(w) > np.pi


def test_eskf_dead_reckoning_noise_free():
    tr = FlightTrajectory()
    ip = ImuParams(gyro_nd=0, acc_nd=0, gyro_rw=0, acc_rw=0)
    imu = simulate_imu(tr, 5.0, ip, np.random.default_rng(0))
    p, v, _ = tr.pva(imu["t"][:1])
    f = ESKF(p[0], v[0], tr.R(imu["t"][:1])[0], ImuParams())
    for i in range(len(imu["t"]) - 1):
        f.propagate(0.5 * (imu["gyro"][i] + imu["gyro"][i + 1]),
                    0.5 * (imu["acc"][i] + imu["acc"][i + 1]), imu["t"][i + 1] - imu["t"][i])
    p_gt = tr.pva(imu["t"][-1:])[0][0]
    assert np.linalg.norm(f.p - p_gt) < 0.05


def test_allan_recovers_noise():
    ip = ImuParams()
    out = identify(simulate_static_imu(0.5, ip, np.random.default_rng(1)))
    assert abs(out["gyro"]["N"] / ip.gyro_nd - 1) < 0.1
    assert abs(out["acc"]["N"] / ip.acc_nd - 1) < 0.1


def test_intrinsics():
    cam = Camera()
    o, i = synth_checkerboard_views(cam, np.random.default_rng(2))
    r = calibrate_intrinsics(o, i, (cam.w, cam.h))
    assert abs(r["K"][0, 0] - cam.fx) < 2.0 and r["rms_px"] < 0.5


def test_camera_imu_time_offset_and_rotation():
    rng = np.random.default_rng(3)
    lms = rng.uniform([-60, -60, 0], [60, 60, 15], (4000, 3))
    ex, cam, ip = default_extrinsics(), Camera(), ImuParams()
    tr = CalibrationTrajectory()
    imu = simulate_imu(tr, 30.0, ip, rng)
    fr = simulate_camera(lms, tr, ex["T_bc_true"], cam, 20, 30.0, ex["td_cam"], rng)
    t1, t2, ph = camera_relative_rotations(fr, cam.K, cam.dist)
    r = calibrate_rotation_and_offset(t1, t2, ph, imu["t"], imu["gyro"])
    assert abs(r["td"] - ex["td_cam"]) < 1e-3
    assert rot_angle_deg(r["R"].T @ ex["T_bc_true"][:3, :3]) < 0.5


def test_camera_lever_arm_beats_cad():
    rng = np.random.default_rng(12)
    ex, cam, ip = default_extrinsics(), Camera(), ImuParams()
    tr = HandheldTrajectory(center=(0, 0, 1.5), amp=0.03, rot_amp=(0.10, 0.09, 0.10),
                            freqs=(1.2, 0.9, 1.0), trans_freq=(1.0, 0.8, 0.7))
    imu = simulate_imu(tr, 25.0, ip, rng, bg0=np.array([0.003, -0.002, 0.0015]))
    obj = make_target(12, 9, 0.10)
    fr = simulate_bench_camera(tr, ex["T_bc_true"], cam, obj, place_target(tr, ex["T_bc_true"], 1.2),
                               20, 25.0, ex["td_cam"], rng)
    t, R, p = pnp_poses(fr, obj, cam.K, cam.dist)
    est = estimate_lever_arm(t - ex["td_cam"], R, p, ex["T_bc_true"][:3, :3], imu["t"], imu["gyro"],
                             imu["acc"], imu["bg"].mean(0))
    err = np.linalg.norm(est["t_bs"] - ex["T_bc_true"][:3, 3])
    cad = np.linalg.norm(ex["T_bc_nom"][:3, 3] - ex["T_bc_true"][:3, 3])
    assert err < 0.5 * cad


def test_deskew_undoes_sweep_distortion():
    w = World(seed=7)
    ex, lid = default_extrinsics(), Lidar()
    tr = FlightTrajectory()
    d = simulate_lidar(w, tr, ex["T_bl_true"], lid, 10, 2.0, 0.0, np.random.default_rng(0), distort=True)[8]
    c = simulate_lidar(w, tr, ex["T_bl_true"], lid, 10, 2.0, 0.0, np.random.default_rng(0), distort=False)[8]
    assert d["pts"].shape == c["pts"].shape
    t = np.array([d["t_true"]])
    R, p, v = tr.R(t)[0], tr.pva(t)[0][0], tr.pva(t)[1][0]
    w_b = tr.omega_body(t)[0]
    w_l, v_l = sensor_twist(R, v, w_b, ex["T_bl_true"][:3, :3], ex["T_bl_true"][:3, 3])
    pts, _ = deskew(d["pts"].astype(float), d["pts"].astype(float), d["tr"], w_l, v_l)
    before = np.linalg.norm(d["pts"] - c["pts"], axis=1)
    after = np.linalg.norm(pts - c["pts"], axis=1)
    assert np.median(after) < 0.2 * np.median(before)
