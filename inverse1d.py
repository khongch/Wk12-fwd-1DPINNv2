"""Inverse 1-D scaffold for parameter identification from LIMITED, NOISY observations.

Forward model (non-dimensional, periodic, exact in Fourier space):

    dT/dFo + Pe dT/dx = d2T/dx2 + Q(x,Fo;A,sq) - Bi*T,     T(x,0)=0
    Q = A * exp(-(x - x0 - Pe*Fo)^2 / (2 sq^2))            (moving Gaussian source)

Unknown theta = (A, sq, Bi), with Pe KNOWN (travel speed is commanded in WAAM).

    A   <->  eta * U I     absorbed power
    sq  <->  c_f, c_r      source size
    Bi  <->  h_eff         heat loss

The A/sq pair is expected to be confounded when only the diffused far field is
observed, because a Gaussian source deposits A*sq*sqrt(2pi) regardless of how that
product is split. This is the 1-D analogue of the eta/b degeneracy.
"""
import math, time
import numpy as np

XQ = 0.30          # source launch position
NFFT = 256
FO_END = 0.016
PE_TRUE = 20.0
THETA_TRUE = np.array([1.0, 0.040, 30.0])     # A, sq, Bi
PNAMES = ["A", "sq", "Bi"]


# ------------------------------------------------------------------ forward
def _qhat(A, sq, N=NFFT):
    xg = np.arange(N) / N
    q0 = A * sum(np.exp(-(xg - XQ + m) ** 2 / (2 * sq ** 2)) for m in (-1, 0, 1))
    return np.fft.fft(q0) / N


def T_exact(x, fo, theta, pe=PE_TRUE, N=NFFT):
    """Exact solution at arbitrary (x, fo). x, fo broadcast together."""
    A, sq, Bi = theta
    Qh = _qhat(A, sq, N)
    n = np.fft.fftfreq(N, d=1.0 / N)
    k = 2 * np.pi * n
    d = k ** 2 + Bi
    x = np.asarray(x, dtype=float)
    fo = np.asarray(fo, dtype=float)
    out = np.zeros(np.broadcast(x, fo).shape, dtype=complex)
    for j in range(N):
        if abs(d[j]) < 1e-14:
            coef = Qh[j] * fo
        else:
            coef = Qh[j] * np.exp(-1j * k[j] * pe * fo) * (-np.expm1(-d[j] * fo)) / d[j]
        out = out + coef * np.exp(1j * k[j] * x)
    return out.real


# ------------------------------------------------------------------ sensors
def layout(kind):
    """Observation designs, all deliberately sparse -- 'limited experimental data'."""
    if kind == "TC3":            # three fixed thermocouples
        xs = np.array([0.25, 0.50, 0.75]); ts = np.linspace(0.002, FO_END, 12)
    elif kind == "TC8":          # eight fixed thermocouples
        xs = np.linspace(0.10, 0.90, 8);   ts = np.linspace(0.002, FO_END, 12)
    elif kind == "SCAN":         # IR line scan: dense in space, four frames
        xs = np.linspace(0.02, 0.98, 40);  ts = np.array([0.004, 0.008, 0.012, 0.016])
    elif kind == "NEAR":         # three sensors placed close to the source path
        xs = np.array([0.32, 0.42, 0.52]); ts = np.linspace(0.002, FO_END, 12)
    else:
        raise ValueError(kind)
    X, T = np.meshgrid(xs, ts, indexing="ij")
    return X.ravel(), T.ravel()


def comoving(nx=12, nt=12):
    """Sensor that travels with the source (co-moving pyrometer)."""
    xi = np.linspace(-0.10, 0.10, nx)
    ts = np.linspace(0.002, FO_END, nt)
    XI, T = np.meshgrid(xi, ts, indexing="ij")
    return (XQ + PE_TRUE * T + XI).ravel() % 1.0, T.ravel()


def synth_data(kind, noise_pct, seed=0, theta=THETA_TRUE):
    """Noiseless truth -> add Gaussian sensor noise scaled to the signal peak."""
    xs, ts = comoving() if kind == "COMOV" else layout(kind)
    clean = T_exact(xs, ts, theta)          # vectorised
    sigma = noise_pct * np.abs(clean).max()
    rng = np.random.default_rng(seed)
    return xs, ts, clean + rng.normal(0.0, sigma, clean.shape), clean, sigma


# ------------------------------------------------------- identifiability gate
def jacobian(xs, ts, theta, rel=1e-6):
    """dT/dtheta at the sensor points, by central differences on the exact model."""
    J = np.zeros((len(xs), len(theta)))
    base = T_exact(xs, ts, theta)
    for j in range(len(theta)):
        h = rel * max(abs(theta[j]), 1e-8)
        tp = theta.copy(); tp[j] += h
        tm = theta.copy(); tm[j] -= h
        up = T_exact(xs, ts, tp); dn = T_exact(xs, ts, tm)
        J[:, j] = (up - dn) / (2 * h)
    return J, base


def fisher(xs, ts, theta, sigma):
    """FIM, its conditioning, Cramer-Rao bounds and the degenerate direction."""
    J, base = jacobian(xs, ts, theta)
    F = J.T @ J / sigma ** 2
    U, S, Vt = np.linalg.svd(J)
    cond = S[0] / S[-1]
    try:
        C = np.linalg.inv(F)
        crb = np.sqrt(np.diag(C))
        corr = C / np.outer(np.sqrt(np.diag(C)), np.sqrt(np.diag(C)))
    except np.linalg.LinAlgError:
        crb = np.full(len(theta), np.inf); corr = np.full((3, 3), np.nan)
    return dict(J=J, F=F, sv=S, cond=cond, crb=crb, corr=corr,
                worst_dir=Vt[-1], rel_crb=crb / np.abs(theta), n_obs=len(xs))


# --------------------------------------------------------------- LM baseline
def lm_fit(xs, ts, obs, sigma, theta0, bounds=None):
    from scipy.optimize import least_squares
    def resid(lp):
        th = np.exp(lp)
        return (T_exact(xs, ts, th) - obs) / sigma
    t0 = time.time()
    r = least_squares(resid, np.log(theta0), method="lm", xtol=1e-12, ftol=1e-12)
    th = np.exp(r.x)
    # covariance from the Gauss-Newton approximation, in log space -> delta method
    JtJ = r.jac.T @ r.jac
    try:
        Clog = np.linalg.inv(JtJ)
        se = np.sqrt(np.diag(Clog)) * th          # delta method
    except np.linalg.LinAlgError:
        se = np.full(3, np.inf)
    return dict(theta=th, se=se, nfev=r.nfev, cost=r.cost,
                wall=time.time() - t0, success=r.success)


def fisher_scaled(xs, ts, theta, sigma):
    """Scale-free identifiability: use RELATIVE sensitivities theta_j * dT/dtheta_j,
    otherwise the SVD just reports which parameter has the largest units."""
    J, base = jacobian(xs, ts, theta)
    Js = J * theta[None, :]
    U, S, Vt = np.linalg.svd(Js, full_matrices=False)
    F = Js.T @ Js / sigma ** 2
    try:
        C = np.linalg.inv(F)
        rel_crb = np.sqrt(np.diag(C))
        corr = C / np.outer(np.sqrt(np.diag(C)), np.sqrt(np.diag(C)))
    except np.linalg.LinAlgError:
        rel_crb = np.full(3, np.inf); corr = np.full((3, 3), np.nan)
    return dict(sv=S, cond=S[0] / S[-1], rel_crb=rel_crb, corr=corr,
                worst_dir=Vt[-1], n_obs=len(xs), Js=Js)


def lm_multistart(xs, ts, obs, sigma, starts=None):
    """LM from several starting points; keep the lowest-cost fit. Averaging raw fits
    across seeds hides non-convergence, so callers should report medians."""
    if starts is None:
        starts = [np.array([0.5, 0.020, 10.0]),
                  np.array([2.0, 0.080, 60.0]),
                  np.array([1.0, 0.040, 30.0]) * 1.5]
    best = None
    for s0 in starts:
        try:
            r = lm_fit(xs, ts, obs, sigma, s0)
        except Exception:
            continue
        if best is None or r["cost"] < best["cost"]:
            best = r
    return best
