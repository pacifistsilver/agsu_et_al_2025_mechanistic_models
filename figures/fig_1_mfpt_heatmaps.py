import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from stochtf.plotting import use_paper_style
import stochtf.analytical.heterodimer as h
import stochtf.analytical.monomer as m

use_paper_style(sans_serif="Arial")

BS, BN = 0.06, 0.24         # fixed unbinding rates
AS0, AN0 = 0.5, 0.2      # reference binding rates

SHOW_FIXED = True            # annotate the parameters held constant in each panel
SHOW_FLAT = True             # annotate axes along which tau_ON is exactly constant
ROW_LABELS = ("HD", "M")   

N = 400
betas = np.logspace(-2.5, 0.0, N)
alphas = np.logspace(-2.5, 2.5, N)
X1, Y1 = np.meshgrid(alphas, betas)     # (alpha_s, beta_s)
X2, Y2 = np.meshgrid(alphas, alphas)    # (alpha_s, alpha_n)
X3, Y3 = np.meshgrid(betas, betas)      # (beta_s,  beta_n)

RATE = r"(M$^{-1}$s$^{-1}$)"
fx = lambda **kw: ", ".join(rf"${k}={v:g}$" for k, v in kw.items())


def field(Z, shape):
    """t_on may be independent of one axis and come back with reduced shape."""
    return np.broadcast_to(np.asarray(Z, dtype=float), shape).copy()


def panel(ax, Z, X, Y, xsym, ysym, mark, fixed, lo, hi, levels, tag):
    pcm = ax.pcolormesh(X, Y, Z, norm=LogNorm(vmin=lo, vmax=hi),
                        cmap="magma", shading="auto", rasterized=True)

    if np.isclose(Z.min(), Z.max()):
        ax.text(0.5, 0.5, rf"constant, $\tau_{{\rm ON}}={Z.flat[0]:.3g}$ s",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7.5, color="w")
    else:
        cs = ax.contour(X, Y, Z, levels=levels, colors="w",
                        linewidths=1.5, alpha=0.6)
        ax.clabel(cs, inline=False, inline_spacing=1, fontsize=8, fmt="%g")

    # flag the axes along which tau_ON does not vary -- this is the signature
    # that separates the two models, so say it rather than leave it to the eye
    if SHOW_FLAT:
        notes = []
        if np.allclose(Z, Z[:, :1]):
            notes.append(rf"independent of ${xsym}$")
        if np.allclose(Z, Z[:1, :]):
            notes.append(rf"independent of ${ysym}$")
        if notes:
            ax.text(0.5, 0.95, "; ".join(notes), transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.5, color="w")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    if mark:
        ax.plot(*mark, "*", mfc="yellow", mec="black", mew=0.9, ms=11,
                clip_on=False, zorder=10)
    # 1. Handle the X-axis label independently
    if xsym in ["k_{on,s}", "k_{on,n}"]:
        ax.set_xlabel(rf"${xsym}$" + r" (M$^{-1}$s$^{-1}$)")
    elif xsym in ["k_{off,s}", "k_{off,n}"]:
        ax.set_xlabel(rf"${xsym}$" + r" (s$^{-1}$)")

    # 2. Handle the Y-axis label independently
    if ysym in ["k_{on,s}", "k_{on,n}"]:
        ax.set_ylabel(rf"${ysym}$" + r" (M$^{-1}$s$^{-1}$)")
    elif ysym in ["k_{off,s}", "k_{off,n}"]:
        ax.set_ylabel(rf"${ysym}$" + r" (s$^{-1}$)")    
    return pcm


panels = [
    # HD
    (field(h.t_on(X1, Y1, AN0, BN), X1.shape), X1, Y1,
     r"k_{on,s}", r"k_{off,s}", (),
     fx(**{r"k_{on,n}": AN0, r"k_{off,n}": BN})),
    (field(h.t_on(X2, BS, Y2, BN), X2.shape), X2, Y2,
     r"k_{on,s}", r"k_{on,n}", (),
     fx(**{r"k_{off,s}": BS, r"k_{on,n}": BN})),
    (field(h.t_on(AS0, X3, AN0, Y3), X3.shape), X3, Y3,
     r"k_{off,s}", r"k_{off,n}", (BS, BN),
     fx(**{r"k_{on,s}": AS0, r"k_{on,n}": AN0})),
    # SEQ
    (field(m.t_on(X1, Y1, AN0, BN), X1.shape), X1, Y1,
     r"k_{on,s}", r"k_{off,s}", (),
     fx(**{r"k_{on,n}": AN0, r"k_{off,n}": BN})),
    (field(m.t_on(X2, BS, Y2, BN), X2.shape), X2, Y2,
     r"k_{on,s}", r"k_{on,n}", (),
     fx(**{r"k_{off,s}": BS, r"k_{off,n}": BN})),
    (field(m.t_on(AS0, X3, AN0, Y3), X3.shape), X3, Y3,
     r"k_{off,s}", r"k_{off,n}", (BS, BN),
     fx(**{r"k_{on,s}": AS0, r"k_{on,n}": AN0})),
]

# one colour scale for every panel, snapped to whole decades so the
# colourbar ticks and the contour levels line up
LO = 10.0 ** np.floor(np.log10(min(float(p[0].min()) for p in panels)))
HI = 10.0 ** np.ceil(np.log10(max(float(p[0].max()) for p in panels)))
LEVELS =  5 ** np.arange(np.log10(LO), np.log10(HI) + 5)
LEVELS = [5, 15, 25, 30, 100, 200, 500]
fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6),
                         constrained_layout=True)

tags = "abcdef"
for k, (ax, spec) in enumerate(zip(axes.flat, panels)):
    Z, X, Y, xsym, ysym, mark, fixed = spec
    pcm = panel(ax, Z, X, Y, xsym, ysym, mark, fixed, LO, HI, LEVELS, tags[k])

cb = fig.colorbar(pcm, ax=axes, shrink=0.85, pad=0.015)
cb.set_label(r"$\tau_{\rm ON}$  (s)")
cb.ax.tick_params(labelsize=6.5)

os.makedirs("./figures/output", exist_ok=True)
out = "./figures/output/fig_1_mfpt"
fig.savefig(out + ".svg", bbox_inches="tight", facecolor="w")

for k, spec in enumerate(panels):
    Z = spec[0]
    print(f"{tags[k]}: min={Z.min():.4g}  max={Z.max():.4g}  "
          f"flat_x={np.allclose(Z, Z[:, :1])}  flat_y={np.allclose(Z, Z[:1, :])}")
print(f"colour scale {LO:g} to {HI:g} s; contour levels {LEVELS}")