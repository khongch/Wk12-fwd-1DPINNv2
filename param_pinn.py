"""Parametric (operator) PINN: T_NN(x, Fo, theta).

Because theta is an INPUT, it appears inside T_NN and therefore inside L_data.
The gradient pathway that the standard inverse PINN lacks is restored by
construction. Trained on physics alone over a theta prior, then theta is fitted
to the noisy observations with the network frozen.
"""
import math, time
import numpy as np, torch, torch.nn as nn
from inverse1d import T_exact, synth_data, THETA_TRUE, PE_TRUE, XQ, FO_END

torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
LO = torch.log(torch.tensor([0.30, 0.015,  5.0]))
HI = torch.log(torch.tensor([3.00, 0.100, 100.0]))


def embed(x, K=16):
    ks = torch.arange(1, K + 1, dtype=x.dtype)
    a = 2 * math.pi * x * ks
    return torch.cat([torch.sin(a), torch.cos(a)], 1)


class ParamPINN(nn.Module):
    def __init__(self, width=64, depth=4, K=16, seed=0):
        super().__init__(); torch.manual_seed(seed); self.K = K
        L, d = [], 2 * K + 1 + 3
        for _ in range(depth):
            L += [nn.Linear(d, width), nn.Tanh()]; d = width
        L += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*L)

    def forward(self, x, fo, logth):
        th = (logth - LO) / (HI - LO) * 2 - 1               # normalise to [-1,1]
        if th.dim() == 1:
            th = th.expand(x.shape[0], 3)
        z = torch.cat([embed(x, self.K), fo / FO_END, th], 1)
        return (fo / FO_END) * self.net(z)

    def residual(self, x, fo, logth):
        x = x.requires_grad_(True); fo = fo.requires_grad_(True)
        T = self.forward(x, fo, logth)
        Tx, Tfo = torch.autograd.grad(T, [x, fo], torch.ones_like(T), create_graph=True)
        Txx = torch.autograd.grad(Tx, x, torch.ones_like(Tx), create_graph=True)[0]
        A, sq, Bi = torch.exp(logth)
        c = XQ + PE_TRUE * fo
        Q = A * sum(torch.exp(-(x - c + m) ** 2 / (2 * sq ** 2)) for m in (-1, 0, 1))
        return Tfo + PE_TRUE * Tx - Txx - Q + Bi * T


def pretrain(steps=1200, ncol=768, seed=0, log_every=300):
    m = ParamPINN(seed=seed)
    o = torch.optim.Adam(m.parameters(), lr=3e-3)
    s = torch.optim.lr_scheduler.CosineAnnealingLR(o, steps, eta_min=1e-4)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for i in range(steps):
        x = torch.rand(ncol, 1, generator=g)
        fo = torch.rand(ncol, 1, generator=g) * FO_END
        lt = LO + torch.rand(3, generator=g) * (HI - LO)
        A = float(torch.exp(lt[0])); sq = float(torch.exp(lt[1]))
        rn = A                      # natural residual scale: the source is O(A).
                                    # rn = A/FO_END (earlier) shrank the loss ~60x
                                    # and made an undertrained net look converged.
        o.zero_grad()
        L = ((m.residual(x, fo, lt) / rn) ** 2).mean()
        L.backward(); o.step(); s.step()
        if log_every and i % log_every == 0:
            print(f"    step {i:>5}  physics loss {L.item():.3e}  {time.time()-t0:5.0f}s",
                  flush=True)
    return m, time.time() - t0


def surrogate_error(m, ntest=6, seed=99):
    """How good is the frozen operator surrogate, independent of any inversion?"""
    g = torch.Generator().manual_seed(seed)
    errs = []
    for _ in range(ntest):
        lt = LO + torch.rand(3, generator=g) * (HI - LO)
        th = np.exp(lt.numpy())
        xg = np.linspace(0, 1, 120, endpoint=False)
        fg = np.linspace(0.002, FO_END, 8)
        X, F = np.meshgrid(xg, fg, indexing="ij")
        ref = T_exact(X.ravel(), F.ravel(), th)
        with torch.no_grad():
            pred = m(torch.tensor(X.ravel()).reshape(-1, 1),
                     torch.tensor(F.ravel()).reshape(-1, 1), lt).numpy().ravel()
        errs.append(np.linalg.norm(pred - ref) / np.linalg.norm(ref))
    return float(np.median(errs)), errs


def invert(m, layout="COMOV", pct=0.05, seed=0, steps=400):
    """Fit theta to noisy data with the network frozen. L_data is now a function
    of theta, so the gradient pathway exists."""
    xs, ts, obs, clean, sig = synth_data(layout, pct, seed=seed)
    xo = torch.tensor(xs).reshape(-1, 1); to = torch.tensor(ts).reshape(-1, 1)
    yo = torch.tensor(obs).reshape(-1, 1)
    for p in m.parameters():
        p.requires_grad_(False)
    lt = torch.tensor(np.log([0.5, 0.020, 10.0]), requires_grad=True)
    o = torch.optim.Adam([lt], lr=2e-2)
    gnorm0 = None
    for i in range(steps):
        o.zero_grad()
        L = (((m(xo, to, lt) - yo) / sig) ** 2).mean()
        L.backward()
        if i == 0:
            gnorm0 = float(lt.grad.norm())
        o.step()
        with torch.no_grad():
            lt.clamp_(LO, HI)
    th = torch.exp(lt).detach().numpy()
    return th, gnorm0, float(L.detach())
