"""IMU noise identification from a static log with overlapping Allan deviation."""
import numpy as np
from scipy.optimize import least_squares


def allan_deviation(x, dt, n_taus=40, max_frac=0.1):
    """Overlapping Allan deviation of a rate signal x (n,) sampled at dt."""
    n = len(x)
    theta = np.cumsum(x) * dt
    ms = np.unique(np.logspace(0, np.log10(int(n * max_frac)), n_taus).astype(int))
    taus, adev = [], []
    for m in ms:
        d = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        avar = np.sum(d * d) / (2.0 * (m * dt) ** 2 * len(d))
        taus.append(m * dt)
        adev.append(np.sqrt(avar))
    return np.array(taus), np.array(adev)


def fit_noise(taus, adev):
    """Fit sigma^2(tau) = N^2/tau + K^2 tau/3 (white noise + random walk)."""
    def res(x):
        N, K = np.exp(x)
        model = N**2 / taus + K**2 * taus / 3.0
        return np.log(np.sqrt(model)) - np.log(adev)
    N0 = adev[np.argmin(np.abs(taus - 1.0))]
    sol = least_squares(res, np.log([N0, N0 * 0.1]))
    N, K = np.exp(sol.x)
    return float(N), float(K)


def identify(static, axis_avg=True):
    out = {}
    for key in ("gyro", "acc"):
        sig = static[key]
        curves, Ns, Ks = [], [], []
        for a in range(3):
            taus, ad = allan_deviation(sig[:, a], static["dt"])
            N, K = fit_noise(taus, ad)
            curves.append(ad)
            Ns.append(N)
            Ks.append(K)
        out[key] = dict(taus=taus, adev=np.array(curves), N=float(np.mean(Ns)), K=float(np.mean(Ks)))
    return out
