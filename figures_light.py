"""Week 11 figures, light theme, to sit on the Week 10 white content slides.

Palette taken from the Week 10 theme: accent1 teal, accent2 orange, accent4
cyan, body text 232323. The heavy case-C field is cached to .npz so this can be
re-run without retraining.
"""
import json
import os

import matplotlib
try:                       # inside IPython, keep the inline backend
    get_ipython           # noqa: F821
except NameError:          # plain script: render headless
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from forward_pinn import CASES, train

TEAL, ORANGE, CYAN = "#156082", "#E97132", "#0F9ED5"
INK, MUTE, GRID = "#232323", "#6E6E6E", "#D8D8D8"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTE, "ytick.color": MUTE, "axes.edgecolor": "#B9B9B9",
    "grid.color": GRID, "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "legend.frameon": False,
})

def _finish(fig, save):
    """Save for the deck, or hand the figure back for inline display."""
    if save:
        fig.savefig(save, dpi=200)
        plt.close(fig)
        return None
    return fig


R = json.load(open("results.json"))
NAME = {"A": "A_diffusion", "B": "B_adv_diff_reac", "C": "C_moving_source"}
LABEL = {"A": "A \u00b7 pure diffusion",
         "B": "B \u00b7 advection\u2013diffusion\u2013reaction",
         "C": "C \u00b7 moving source (Week 10 operator)"}
COLOR = {"A": CYAN, "B": TEAL, "C": ORANGE}


def agg(case, field="rel_l2"):
    d = {}
    for r in R["E1"]:
        if r["case"] != NAME[case]:
            continue
        v = (r["amp_measured"] / r["amp_predicted"]) if field == "amp_ratio" \
            else r[field]
        d.setdefault(r["K"], []).append(v)
    return {k: (float(np.median(v)), float(np.min(v)), float(np.max(v)))
            for k, v in sorted(d.items())}


def field_grid(case, nx=220, nf=180):
    c = CASES[case]()
    x = torch.arange(nx, dtype=torch.float64).reshape(-1, 1) / nx
    f = torch.linspace(0, c.fo_end, nf, dtype=torch.float64).reshape(-1, 1)
    X, F = x.repeat(nf, 1), f.repeat_interleave(nx, 0)
    return c, X, F, c.exact(X, F).detach().numpy().reshape(nf, nx)


# ----------------------------------------------------------------- fig cases
def fig_cases(save='L_fig_cases.png'):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.25))
    for ax, key in zip(axes, "ABC"):
        c, _, _, T = field_grid(key)
        m = np.abs(T).max()
        im = ax.imshow(T, origin="lower", aspect="auto", cmap="magma",
                       extent=[0, 1, 0, c.fo_end],
                       vmin=0 if key == "C" else -m, vmax=m)
        ax.set_title(LABEL[key], color=COLOR[key], fontsize=11.5)
        ax.set_xlabel("x")
        # A and B share a scale, so only B carries a colourbar
        if key != "A":
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(colors=MUTE, labelsize=8)
    axes[0].set_ylabel("Fo")
    fig.tight_layout(w_pad=2.2)
    return _finish(fig, save)


# ------------------------------------------------------------- fig bandwidth
def fig_bandwidth(save='L_fig_bandwidth.png'):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.35))
    for key in "ABC":
        e, a = agg(key), agg(key, "amp_ratio")
        ks = sorted(e)
        a1.plot(ks, [e[k][0] for k in ks], "o-", color=COLOR[key], lw=2.2,
                ms=6, label=LABEL[key])
        a1.fill_between(ks, [e[k][1] for k in ks], [e[k][2] for k in ks],
                        color=COLOR[key], alpha=0.16)
        a2.plot(ks, [a[k][0] for k in ks], "s-", color=COLOR[key], lw=2.2, ms=6)
    a1.set_xscale("log", base=2)
    a1.set_yscale("log")
    a1.set_xlabel("Fourier bandwidth  K")
    a1.set_ylabel("relative $L^2$ error vs exact")
    a1.grid(alpha=0.55, lw=0.7)
    a1.axvline(16, color=MUTE, ls="--", lw=1.2)
    a1.annotate("Week 10 used K = 16", xy=(16, a1.get_ylim()[1]),
                xytext=(-6, -14), textcoords="offset points", ha="right",
                color=MUTE, fontsize=9.5)
    a1.set_title("Error rises monotonically for K \u2265 2", color=INK,
                 loc="left", fontweight="bold")
    a2.axhline(1.0, color=INK, ls="--", lw=1.2)
    a2.annotate("amplitude the PDE requires", xy=(1, 1.0), xytext=(4, 6),
                textcoords="offset points", color=INK, fontsize=9.5)
    a2.set_xscale("log", base=2)
    a2.set_ylim(0, 1.35)
    a2.set_xlabel("Fourier bandwidth  K")
    a2.set_ylabel("measured / required output amplitude")
    a2.grid(alpha=0.55, lw=0.7)
    a2.set_title("The same story, without touching the loss", color=INK,
                 loc="left", fontweight="bold")
    # one legend for both panels, placed below the axes so it cannot sit on
    # top of the curves it describes
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    return _finish(fig, save)


# -------------------------------------------------------------- fig ablation
def fig_ablation(save='L_fig_ablation.png'):
    d = {}
    for r in R["E2"]:
        d.setdefault(r["tag"], []).append(r["rel_l2"])
    order = ["baseline", "no_normalise", "resample_colloc", "wider_deeper",
             "no_hard_IC"]
    lab = {"baseline": "baseline\n(K=2, hard IC, normalised)",
           "no_normalise": "residual not\nnormalised",
           "resample_colloc": "collocation\nresampled",
           "wider_deeper": "wider + deeper\n(96\u00d76)",
           "no_hard_IC": "IC by penalty\ninstead of ansatz"}
    med = [float(np.median(d[o])) for o in order]
    lo = [med[i] - float(np.min(d[o])) for i, o in enumerate(order)]
    hi = [float(np.max(d[o])) - med[i] for i, o in enumerate(order)]
    fig, ax = plt.subplots(figsize=(11.4, 4.0))
    cols = [TEAL] + [ORANGE if m > med[0] * 1.5 else CYAN for m in med[1:]]
    ax.bar(range(len(order)), med, yerr=[lo, hi], color=cols, width=0.58,
           ecolor=MUTE, capsize=4)
    # linear, not log: four of the five bars sit within 10% of each other and a
    # log axis flattens them into slivers, hiding the only comparison that matters
    for i, m in enumerate(med):
        ax.text(i, m + max(med) * 0.045, f"{m:.2f}", ha="center", color=INK,
                fontsize=11, fontweight="bold")
    ax.axhline(med[0], color=MUTE, ls="--", lw=1.1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([lab[o] for o in order], fontsize=9.5)
    ax.set_ylim(0, max(med) * 1.22)
    ax.set_ylabel("relative $L^2$ error")
    ax.grid(axis="y", alpha=0.55, lw=0.7)
    ax.set_title("Case C ablations \u2014 median of 3 seeds, whiskers show the "
                 "seed range", color=INK, loc="left", fontweight="bold")
    fig.tight_layout()
    return _finish(fig, save)


# -------------------------------------------------------------- fig solution
CACHE = "solution_field.npz"


def fig_solution(save='L_fig_solution.png'):
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        P, E, fo_end, rel = z["P"], z["E"], float(z["fo_end"]), float(z["rel"])
        print(f"  (cached field, rel-L2 = {rel:.4e})")
    else:
        r, model = train(CASES["C"](), K=2, seed=0, n_adam=3000, n_lbfgs=1500,
                         n_col=2048)
        c, X, F, E = field_grid("C", 200, 160)
        with torch.no_grad():
            P = model(X, F).numpy().reshape(160, 200)
        fo_end, rel = c.fo_end, r["rel_l2"]
        np.savez(CACHE, P=P, E=E, fo_end=fo_end, rel=rel)
        print(f"  (trained, rel-L2 = {rel:.4e}, {r['wall']:.0f}s)")
    return solution_panels(P, E, fo_end, save=save), rel


def solution_panels(P, E, fo_end, save="L_fig_solution.png"):
    """The exact / PINN / difference triptych, given fields computed elsewhere."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.2))
    vm = E.max()
    for i, (ax, (Dm, t, cm, kw)) in enumerate(zip(axes, [
            (E, "exact", "magma", dict(vmin=0, vmax=vm)),
            (P, "PINN", "magma", dict(vmin=0, vmax=vm)),
            (np.abs(P - E), "|PINN \u2212 exact|", "inferno", {})])):
        im = ax.imshow(Dm, origin="lower", aspect="auto", cmap=cm,
                       extent=[0, 1, 0, fo_end], **kw)
        ax.set_title(t, color=INK, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("x")
        if i != 0:            # panels 0 and 1 share a scale
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(colors=MUTE, labelsize=8)
    axes[0].set_ylabel("Fo")
    fig.tight_layout(w_pad=2.2)
    return _finish(fig, save)


if __name__ == "__main__":
    fig_cases(); print("L_fig_cases.png")
    fig_bandwidth(); print("L_fig_bandwidth.png")
    fig_ablation(); print("L_fig_ablation.png")
    _, rel = fig_solution(); print("L_fig_solution.png", rel)
