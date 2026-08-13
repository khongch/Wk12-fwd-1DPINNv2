"""Forward PDE-based PINN in PyTorch — Week 11.

Governing equation (non-dimensional, periodic on x in [0,1)):

    dT/dFo + Pe dT/dx = d2T/dx2 - Bi*T + Q(x,Fo),      T(x,0) = T0(x)

Three cases of increasing difficulty, each with a closed-form solution:

    A  DIFF   Pe=0,  Bi=0,  Q=0,   T0 = sin(2pi x) + 0.5 sin(6pi x)
    B  ADR    Pe=20, Bi=30, Q=0,   T0 = sin(2pi x)
    C  SRC    Pe=20, Bi=30, Q=moving Gaussian, T0 = 0     <- Week 10's operator

The network is built in seven steps, in the order the tensors flow:

    1. Fourier feature embedding in x        -> periodic BC is EXACT
    2. hard initial condition by ansatz      -> L_IC is identically zero
    3. autodiff residual (T_Fo, T_x, T_xx)   -> the only loss term
    4. residual normalisation by D0          -> loss is O(1) at initialisation
    5. Sobol collocation in (x, Fo)
    6. Adam + cosine anneal -> L-BFGS/strong Wolfe, float64 throughout
    7. accuracy measured against the exact solution, never against the loss

Because steps 1 and 2 are exact by construction, L_total = L_PDE alone. There
is no boundary term and no initial-condition term, hence no weights to balance
between them: the Week 8 loss-balancing problem is REMOVED here rather than
tuned. That is the main structural point of this week.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn

from inverse1d import _qhat, THETA_TRUE, PE_TRUE, XQ, FO_END, NFFT

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


# =========================================================== 1. the test cases
class Case:
    """A forward problem: coefficients, initial condition, source, exact solution.

    `d0` is the magnitude of dT/dFo at Fo=0 implied by the PDE itself:

        dT/dFo|_0 = T0'' - Pe*T0' - Bi*T0 + Q(x,0)

    It is computable WITHOUT the exact solution, and is used both to normalise
    the residual (step 4) and to predict the amplitude the network must reach
    under the hard-IC ansatz (step 2).
    """

    def __init__(self, name, pe, bi, fo_end, t0, exact, q=None):
        self.name = name
        self.pe = pe
        self.bi = bi
        self.fo_end = fo_end
        self.t0 = t0
        self.exact = exact
        self.q = q
        self.d0 = self._compute_d0()

    def _compute_d0(self, nx=1024):
        x = torch.arange(nx, dtype=torch.float64).reshape(-1, 1) / nx
        x.requires_grad_(True)
        T0 = self.t0(x)

        def d_dx(u):
            """One derivative, returning exact zeros when u is constant in x."""
            if not u.requires_grad:
                return torch.zeros_like(u)
            return torch.autograd.grad(u, x, torch.ones_like(u),
                                       create_graph=True, allow_unused=True,
                                       materialize_grads=True)[0]

        g = d_dx(T0)
        gg = d_dx(g)
        rhs = gg.detach() - self.pe * g.detach() - self.bi * T0.detach()
        if self.q is not None:
            rhs = rhs + self.q(x.detach(), torch.zeros_like(x))
        return float(rhs.abs().max())

    def source(self, x, fo):
        return self.q(x, fo) if self.q is not None else torch.zeros_like(x)


def _q_moving(x, fo):
    """Week 10's periodised moving Gaussian, in torch."""
    A, sq, _ = THETA_TRUE
    c = XQ + PE_TRUE * fo
    return A * sum(torch.exp(-(x - c + m) ** 2 / (2 * sq ** 2)) for m in (-1, 0, 1))


def _exact_C(x, fo):
    """Week 10's Fourier solution, re-expressed in torch so it is differentiable.

    T = sum_j g_j(Fo) * [ Re(Qh_j) cos(phi_j) - Im(Qh_j) sin(phi_j) ],
        phi_j = k_j (x - Pe*Fo),   g_j = (1 - exp(-d_j Fo)) / d_j,  d_j = k_j^2 + Bi
    """
    A, sq, Bi = THETA_TRUE
    Qh = _qhat(A, sq, NFFT)
    k = torch.tensor(2 * np.pi * np.fft.fftfreq(NFFT, d=1.0 / NFFT))
    ar = torch.tensor(Qh.real)
    ai = torch.tensor(Qh.imag)
    d = k ** 2 + Bi
    phi = k * (x - PE_TRUE * fo)                       # (n, NFFT) by broadcasting
    g = -torch.expm1(-d * fo) / d
    return (g * (ar * torch.cos(phi) - ai * torch.sin(phi))).sum(1, keepdim=True)


def case_A():
    """Pure diffusion. Two modes, 9x separation in decay rate."""
    def t0(x):
        return torch.sin(2 * math.pi * x) + 0.5 * torch.sin(6 * math.pi * x)

    def ex(x, fo):
        return (torch.exp(-((2 * math.pi) ** 2) * fo) * torch.sin(2 * math.pi * x)
                + 0.5 * torch.exp(-((6 * math.pi) ** 2) * fo)
                * torch.sin(6 * math.pi * x))

    return Case("A_diffusion", 0.0, 0.0, 0.020, t0, ex)


def case_B():
    """Advection-diffusion-reaction. Same Pe and Bi as the Week 10 operator."""
    def t0(x):
        return torch.sin(2 * math.pi * x)

    def ex(x, fo):
        lam = (2 * math.pi) ** 2 + 30.0
        return torch.exp(-lam * fo) * torch.sin(2 * math.pi * (x - 20.0 * fo))

    return Case("B_adv_diff_reac", 20.0, 30.0, 0.016, t0, ex)


def case_C():
    """Moving Gaussian source: exactly the operator Week 10's inverse PINN solved."""
    # 0.0 * x rather than zeros_like(x): keeps T0 on the autograd graph so the
    # same d0 routine works for every case without a special branch.
    return Case("C_moving_source", PE_TRUE, 30.0, FO_END,
                lambda x: 0.0 * x, _exact_C, _q_moving)


CASES = {"A": case_A, "B": case_B, "C": case_C}


# ================================================================ 2. the model
def fourier_features(x, K):
    """Step 1. gamma(x) = [sin(2pi k x), cos(2pi k x)] for k = 1..K.

    Every feature is 1-periodic, so ANY network of these features is periodic.
    The boundary condition is satisfied identically, not penalised.
    """
    ks = torch.arange(1, K + 1, dtype=x.dtype)
    a = 2 * math.pi * x * ks
    return torch.cat([torch.sin(a), torch.cos(a)], 1)


class ForwardPINN(nn.Module):
    """T(x,Fo) = T0(x) + (Fo/Fo_end) * N(gamma(x), Fo/Fo_end)

    The prefactor vanishes at Fo=0, so T(x,0) = T0(x) exactly for any weights:
    that is step 2, the hard initial condition.

    Consequence worth noting: the network does not represent T. Differentiating
    the ansatz at Fo=0 and matching the PDE gives

        N(x, 0) = Fo_end * [ T0'' - Pe*T0' - Bi*T0 + Q(x,0) ] = Fo_end * O(d0)

    so the amplitude the network must reach is set by the PDE, not by the size
    of T. `predicted_amplitude` below states that number before training.
    """

    def __init__(self, case, K=8, width=48, depth=4, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.case = case
        self.K = K
        layers, d = [], 2 * K + 1
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def predicted_amplitude(self):
        return self.case.fo_end * self.case.d0

    def forward(self, x, fo):
        s = fo / self.case.fo_end
        z = torch.cat([fourier_features(x, self.K), s], 1)
        return self.case.t0(x) + s * self.net(z)

    def residual(self, x, fo):
        """Step 3. Exact derivatives of the ansatz by reverse-mode autodiff."""
        x = x.requires_grad_(True)
        fo = fo.requires_grad_(True)
        T = self.forward(x, fo)
        Tx, Tfo = torch.autograd.grad(T, [x, fo], torch.ones_like(T),
                                      create_graph=True)
        Txx = torch.autograd.grad(Tx, x, torch.ones_like(Tx), create_graph=True)[0]
        return (Tfo + self.case.pe * Tx - Txx
                + self.case.bi * T - self.case.source(x, fo))


# =========================================================== 3. training utils
def sobol(n, fo_end, seed=0):
    """Step 5. Low-discrepancy collocation in (x, Fo)."""
    from scipy.stats import qmc
    m = int(math.ceil(math.log2(n)))
    p = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(m)[:n]
    return (torch.tensor(p[:, 0:1]), torch.tensor(p[:, 1:2]) * fo_end)


def rel_l2(model, case, nx=200, nfo=40):
    """Step 7. Accuracy against the exact solution on a dense grid."""
    xg = torch.arange(nx, dtype=torch.float64).reshape(-1, 1) / nx
    fg = torch.linspace(0.0, case.fo_end, nfo, dtype=torch.float64).reshape(-1, 1)
    X = xg.repeat(nfo, 1)
    F = fg.repeat_interleave(nx, 0)
    with torch.no_grad():
        pred = model(X, F)
    ref = case.exact(X, F)
    return float(torch.linalg.norm(pred - ref) / torch.linalg.norm(ref))


def train(case, K=8, width=48, depth=4, seed=0, n_col=2048,
          n_adam=1500, n_lbfgs=800, lr=3e-3, normalise=True,
          hard_ic=True, resample=False, verbose=False):
    """Step 6. Adam with cosine annealing, then L-BFGS with strong Wolfe."""
    model = ForwardPINN(case, K=K, width=width, depth=depth, seed=seed)
    if not hard_ic:
        model.forward = _soft_forward.__get__(model)     # ablation, see below
    xc, fc = sobol(n_col, case.fo_end, seed)
    rn = case.d0 if normalise else 1.0

    xi = torch.rand(512, 1, generator=torch.Generator().manual_seed(seed + 7))
    fi = torch.zeros(512, 1)

    def loss():
        L = ((model.residual(xc, fc) / rn) ** 2).mean()
        if not hard_ic:                                   # soft IC needs a term
            L = L + 100.0 * ((model(xi, fi) - case.t0(xi)) ** 2).mean()
        return L

    t0 = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_adam, eta_min=1e-4)
    hist = []
    for i in range(n_adam):
        if resample and i % 100 == 0 and i > 0:
            xc, fc = sobol(n_col, case.fo_end, seed + i)
        opt.zero_grad()
        L = loss()
        L.backward()
        opt.step()
        sch.step()
        if i % 250 == 0:
            hist.append((i, float(L.detach()), rel_l2(model, case, 100, 20)))
            if verbose:
                print(f"      adam {i:>5}  L={float(L):.3e}  "
                      f"relL2={hist[-1][2]:.3e}", flush=True)

    lb = torch.optim.LBFGS(model.parameters(), max_iter=n_lbfgs, history_size=60,
                           tolerance_grad=1e-14, tolerance_change=1e-16,
                           line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad()
        L = loss()
        L.backward()
        return L

    lb.step(closure)

    with torch.no_grad():
        s = torch.rand(400, 1)
        amp = float(model.net(torch.cat([fourier_features(s, model.K),
                                         torch.zeros_like(s)], 1)).abs().max())
    return dict(case=case.name, K=K, width=width, depth=depth, seed=seed,
                normalise=normalise, hard_ic=hard_ic, resample=resample,
                n_col=n_col,
                rel_l2=rel_l2(model, case),
                loss=float(loss().detach()),
                amp_measured=amp,
                amp_predicted=case.fo_end * case.d0,
                wall=time.time() - t0, hist=hist), model


def _soft_forward(self, x, fo):
    """Ablation: no hard IC. T = N(gamma(x), Fo/Fo_end), IC enforced by penalty."""
    s = fo / self.case.fo_end
    z = torch.cat([fourier_features(x, self.K), s], 1)
    return self.net(z)


# ==================================================== 4. operator verification
def verify_operator(case, n=512, seed=0):
    """Substitute the EXACT solution into the autodiff residual.

    This tests the PyTorch operator that the PINN minimises, independently of
    any training. A correct operator returns zero to machine precision. Two
    bugs were caught this way rather than by reading the code.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g, dtype=torch.float64).requires_grad_(True)
    fo = (torch.rand(n, 1, generator=g, dtype=torch.float64)
          * 0.9 * case.fo_end + 0.05 * case.fo_end).requires_grad_(True)
    T = case.exact(x, fo)
    Tx, Tfo = torch.autograd.grad(T, [x, fo], torch.ones_like(T), create_graph=True)
    Txx = torch.autograd.grad(Tx, x, torch.ones_like(Tx), create_graph=True)[0]
    r = Tfo + case.pe * Tx - Txx + case.bi * T - case.source(x, fo)
    scale = max(float(Tfo.detach().abs().max()),
                float(Txx.detach().abs().max()), 1.0)
    return float(r.detach().abs().max()) / scale


if __name__ == "__main__":
    print(f"{'case':<18}{'Pe':>6}{'Bi':>6}{'Fo_end':>9}"
          f"{'d0':>12}{'amp_pred':>12}{'operator':>13}")
    print("-" * 76)
    for key in "ABC":
        c = CASES[key]()
        e = verify_operator(c)
        print(f"{c.name:<18}{c.pe:>6.0f}{c.bi:>6.0f}{c.fo_end:>9.3f}"
              f"{c.d0:>12.4f}{c.fo_end * c.d0:>12.4f}{e:>13.2e}")
    print("-" * 76)
