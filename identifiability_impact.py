"""Does the speed correction change the identifiability conclusions?

Recomputes the scaled Fisher analysis for every sensor layout in the scaffold
under the double-counted operator and under the corrected laboratory-frame
operator, using identical layouts, noise level and parameters. Anything that
moves here has to be rerun in Chapter 2.
"""
import numpy as np

from inverse1d import layout, comoving, THETA_TRUE, PE_TRUE, FO_END
from inverse1d_frames import T_exact_frame

NOISE = 0.05
PN = ["A", "s_q", "L_h"]


def jac_scaled(xs, ts, theta, frame, rel=1e-6):
    """Relative sensitivities theta_j * dT/dtheta_j by central differences on
    the exact solution, so the SVD reports information rather than units."""
    J = np.zeros((len(xs), 3))
    for j in range(3):
        h = rel * abs(theta[j])
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        up = T_exact_frame(xs, ts, tp, PE_TRUE, frame=frame)
        dn = T_exact_frame(xs, ts, tm, PE_TRUE, frame=frame)
        J[:, j] = (up - dn) / (2 * h) * theta[j]
    return J


def analyse(xs, ts, frame):
    base = T_exact_frame(xs, ts, THETA_TRUE, PE_TRUE, frame=frame)
    sigma = NOISE * np.abs(base).max()
    J = jac_scaled(xs, ts, THETA_TRUE, frame)
    s = np.linalg.svd(J, compute_uv=False)
    F = J.T @ J / sigma ** 2
    try:
        C = np.linalg.inv(F)
        crb = np.sqrt(np.diag(C))
    except np.linalg.LinAlgError:
        crb = np.full(3, np.inf)
    return s[0] / s[-1], crb


LAYOUTS = ["TC3", "TC8", "SCAN", "NEAR", "COMOV"]

print(f"{'layout':<8}{'cond (double-counted)':>23}{'cond (corrected lab)':>22}"
      f"{'change':>10}")
print("-" * 63)
store = {}
for name in LAYOUTS:
    xs, ts = comoving() if name == "COMOV" else layout(name)
    c_old, crb_old = analyse(xs, ts, "comoving")
    c_new, crb_new = analyse(xs, ts, "lab")
    store[name] = (crb_old, crb_new)
    print(f"{name:<8}{c_old:>23.4g}{c_new:>22.4g}{c_new/c_old:>9.2f}x")

print(f"\nrelative Cramer-Rao bounds, % (double-counted -> corrected)")
print(f"{'layout':<8}" + "".join(f"{p:>22}" for p in PN))
print("-" * 74)
for name in LAYOUTS:
    old, new = store[name]
    cells = "".join(f"{o*100:>9.1f} ->{n*100:>9.1f}" for o, n in zip(old, new))
    print(f"{name:<8}{cells}")

print("\nrank ordering of layouts by conditioning")
for tag, fr in (("double-counted", "comoving"), ("corrected     ", "lab")):
    conds = {}
    for name in LAYOUTS:
        xs, ts = comoving() if name == "COMOV" else layout(name)
        conds[name] = analyse(xs, ts, fr)[0]
    order = sorted(conds, key=conds.get)
    print(f"  {tag}: " + "  <  ".join(order))
