"""Corrected forward oracle: one travel speed, used once.

The scaffold as written solves

    T_Fo + Pe T_x = T_xx - L T + Q(x - x0 - Pe Fo)

which carries material advection at Pe *and* a source translating at Pe. For a
stationary substrate that is the same speed twice, and the two cancel in the
source frame. The physical problem admits exactly one of:

    frame="lab"     T_Fo              = T_xx - L T + Q(x - x0 - Pe_s Fo)
    frame="source"  W_Fo - Pe_s W_xi  = W_xi_xi - L W + Q(xi)

related by W(xi, Fo) = T(xi + Pe_s Fo, Fo). The double-counted form is retained
as frame="comoving" so archived results stay reproducible, but it is a
diagnostic, not the WAAM model.

Transfer functions, all with T(x,0) = 0 and d = k^2 + L - i k Pe_s:

    lab       T_k = q_k [ e^{-i k Pe_s Fo} - e^{-(k^2+L) Fo} ] / d
    source    W_k = q_k [ 1 - e^{-d Fo} ] / d
    comoving  T_k = q_k e^{-i k Pe_s Fo} [ 1 - e^{-(k^2+L) Fo} ] / (k^2 + L)
"""
import numpy as np

from inverse1d import _qhat, XQ, NFFT, FO_END, PE_TRUE, THETA_TRUE


def T_exact_frame(x, fo, theta, pe=PE_TRUE, N=NFFT, frame="source"):
    A, sq, L = theta
    Qh = _qhat(A, sq, N)
    k = 2 * np.pi * np.fft.fftfreq(N, d=1.0 / N)
    x = np.asarray(x, dtype=float)
    fo = np.asarray(fo, dtype=float)
    out = np.zeros(np.broadcast(x, fo).shape, dtype=complex)

    d_pure = k ** 2 + L                       # no speed in the denominator
    d_adv = k ** 2 + L - 1j * k * pe          # speed enters every mode

    for j in range(N):
        if frame == "source":
            dj = d_adv[j]
            coef = Qh[j] * (-np.expm1(-dj * fo)) / dj
        elif frame == "lab":
            dj = d_adv[j]
            coef = Qh[j] * (np.exp(-1j * k[j] * pe * fo)
                            - np.exp(-d_pure[j] * fo)) / dj
        elif frame == "comoving":             # the double-counted scaffold
            dj = d_pure[j]
            coef = (Qh[j] * np.exp(-1j * k[j] * pe * fo)
                    * (-np.expm1(-dj * fo)) / dj)
        else:
            raise ValueError(frame)
        out = out + coef * np.exp(1j * k[j] * x)
    return out.real


# ------------------------------------------------------------------ checks
def _grid(nx=NFFT):
    return np.arange(nx) / nx, 2 * np.pi * np.fft.fftfreq(nx, d=1.0 / nx)


def _src(x, theta, moving_at=None, fo=0.0):
    A, sq, _ = theta
    c = XQ + (moving_at * fo if moving_at else 0.0)
    return A * sum(np.exp(-(x - c + m) ** 2 / (2 * sq ** 2)) for m in (-1, 0, 1))


def check_mol(theta, pe, fo_end, frame):
    """Independent spectral method-of-lines integration of the same frame."""
    from scipy.integrate import solve_ivp
    xg, k = _grid()
    L = theta[2]

    def rhs(t, T):
        Th = np.fft.fft(T)
        Tx = np.fft.ifft(1j * k * Th).real
        Txx = np.fft.ifft(-(k ** 2) * Th).real
        if frame == "lab":
            q = _src(xg, theta, moving_at=pe, fo=t)
            return Txx - L * T + q                    # no advection
        if frame == "source":
            q = _src(xg, theta)
            return pe * Tx + Txx - L * T + q          # one advection term
        q = _src(xg, theta, moving_at=pe, fo=t)
        return -pe * Tx + Txx - L * T + q             # comoving: both
    s = solve_ivp(rhs, (0, fo_end), np.zeros(NFFT), method="Radau",
                  rtol=1e-11, atol=1e-13, dense_output=True)
    num = s.sol(fo_end)
    ref = T_exact_frame(xg, np.full(NFFT, fo_end), theta, pe, frame=frame)
    return np.linalg.norm(num - ref) / np.linalg.norm(ref)


if __name__ == "__main__":
    th, pe, fe = THETA_TRUE, PE_TRUE, FO_END
    print("solution verified against an independent integrator")
    for fr in ("lab", "source", "comoving"):
        print(f"   frame={fr:<9} rel-L2 = {check_mol(th, pe, fe, fr):.3e}")

    xg, _ = _grid()
    lab = T_exact_frame(xg, np.full(NFFT, fe), th, pe, frame="lab")
    src = T_exact_frame((xg - pe * fe) % 1.0, np.full(NFFT, fe), th, pe,
                        frame="source")
    print(f"\nlab and source frames agree under translation: "
          f"rel-L2 = {np.linalg.norm(lab - src)/np.linalg.norm(lab):.3e}")

    print("\nspeed sensitivity of the field (rel-L2 against Pe_s = 0)")
    for fr in ("source", "comoving"):
        ref0 = T_exact_frame(xg, np.full(NFFT, fe), th, 0.0, frame=fr)
        row = []
        for p in (5.0, 20.0, 50.0):
            w = T_exact_frame(xg, np.full(NFFT, fe), th, p, frame=fr)
            row.append(np.linalg.norm(w - ref0) / np.linalg.norm(ref0))
        print(f"   frame={fr:<9} Pe=5 {row[0]:.3f}   Pe=20 {row[1]:.3f}   "
              f"Pe=50 {row[2]:.3f}")
