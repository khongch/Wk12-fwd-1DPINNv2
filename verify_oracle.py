"""Independent checks on the exact solutions used as ground truth in Week 11.

Nothing in this week's results is allowed to rest on a loss curve, so the
reference solutions are themselves verified two ways before any network is
trained:

  [1] spectral method-of-lines: integrate the PDE numerically with an
      independent solver (Radau, spectral x-derivatives) and compare to the
      closed-form solution.
  [2] residual substitution: push the closed-form solution through the PDE
      operator (spectral in x, 5-point central difference in Fo). A correct
      solution returns zero.

Check [2] is the important one: it tests the OPERATOR, which is the same
operator the PINN's autodiff residual implements.
"""
import numpy as np
from scipy.integrate import solve_ivp

from inverse1d import T_exact, THETA_TRUE, PE_TRUE, XQ, FO_END

NX = 256


def _grid(nx=NX):
    x = np.arange(nx) / nx
    k = 2 * np.pi * np.fft.fftfreq(nx, d=1.0 / nx)
    return x, k


def source_C(x, fo):
    """Week 10's moving Gaussian, periodised over three images."""
    A, sq, _ = THETA_TRUE
    c = XQ + PE_TRUE * fo
    return A * sum(np.exp(-(x - c + m) ** 2 / (2 * sq ** 2)) for m in (-1, 0, 1))


# ----------------------------------------------------------------- the cases
# Each entry: (name, Pe, Bi, Fo_end, T0(x), exact(x,fo), Q(x,fo) or None)
def case_table():
    x, _ = _grid()

    def A_T0(xx):
        return np.sin(2 * np.pi * xx) + 0.5 * np.sin(6 * np.pi * xx)

    def A_exact(xx, fo):
        return (np.exp(-((2 * np.pi) ** 2) * fo) * np.sin(2 * np.pi * xx)
                + 0.5 * np.exp(-((6 * np.pi) ** 2) * fo) * np.sin(6 * np.pi * xx))

    def B_T0(xx):
        return np.sin(2 * np.pi * xx)

    def B_exact(xx, fo):
        lam = (2 * np.pi) ** 2 + 30.0
        return np.exp(-lam * fo) * np.sin(2 * np.pi * (xx - 20.0 * fo))

    def C_T0(xx):
        return np.zeros_like(xx)

    def C_exact(xx, fo):
        return T_exact(xx, np.full_like(xx, fo), THETA_TRUE)

    return [
        ("A  diffusion    ", 0.0, 0.0, 0.020, A_T0, A_exact, None),
        ("B  adv-diff-reac", 20.0, 30.0, 0.016, B_T0, B_exact, None),
        ("C  moving source", 20.0, 30.0, FO_END, C_T0, C_exact, source_C),
    ]


# --------------------------------------------------------- [1] method of lines
def check_mol(pe, bi, fo_end, t0_fn, exact_fn, q_fn, nx=NX):
    x, k = _grid(nx)

    def rhs(fo, T):
        Th = np.fft.fft(T)
        Tx = np.fft.ifft(1j * k * Th).real
        Txx = np.fft.ifft(-(k ** 2) * Th).real
        out = -pe * Tx + Txx - bi * T
        if q_fn is not None:
            out = out + q_fn(x, fo)
        return out

    sol = solve_ivp(rhs, (0.0, fo_end), t0_fn(x), method="Radau",
                    rtol=1e-11, atol=1e-13, dense_output=True)
    num = sol.sol(fo_end)
    ref = exact_fn(x, fo_end)
    return np.linalg.norm(num - ref) / np.linalg.norm(ref)


# ------------------------------------------------------ [2] residual of exact
def check_residual(pe, bi, fo_end, t0_fn, exact_fn, q_fn, nx=NX, nfo=9):
    x, k = _grid(nx)
    h = 1e-7
    fos = np.linspace(0.15 * fo_end, 0.9 * fo_end, nfo)
    worst = 0.0
    for fo in fos:
        def T_at(f):
            return exact_fn(x, f)
        T = T_at(fo)
        # 5-point central difference in Fo
        Tf = (T_at(fo - 2 * h) - 8 * T_at(fo - h)
              + 8 * T_at(fo + h) - T_at(fo + 2 * h)) / (12 * h)
        Th = np.fft.fft(T)
        Tx = np.fft.ifft(1j * k * Th).real
        Txx = np.fft.ifft(-(k ** 2) * Th).real
        r = Tf + pe * Tx - Txx + bi * T
        if q_fn is not None:
            r = r - q_fn(x, fo)
        scale = max(np.abs(Tf).max(), np.abs(Txx).max(), 1.0)
        worst = max(worst, np.abs(r).max() / scale)
    return worst


if __name__ == "__main__":
    print(f"{'case':<18}{'[1] MoL rel-L2':>18}{'[2] residual':>18}   verdict")
    print("-" * 72)
    all_ok = True
    for (name, pe, bi, fe, t0, ex, q) in case_table():
        e1 = check_mol(pe, bi, fe, t0, ex, q)
        e2 = check_residual(pe, bi, fe, t0, ex, q)
        ok = (e1 < 1e-8) and (e2 < 1e-5)
        all_ok &= ok
        print(f"{name:<18}{e1:>18.3e}{e2:>18.3e}   {'PASS' if ok else 'FAIL'}")
    print("-" * 72)
    print("ORACLE VERIFIED" if all_ok else "ORACLE CHECK FAILED")
