"""Plots of burst frequency and burst size for the two-site OR-gate promoter."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from stochtf.analytical import heterodimer as ts
from stochtf.plotting import PALETTE, output_path, use_paper_style

use_paper_style(sans_serif="Arial")
C = PALETTE
gamma, k_y = 1.0, 20.0          # time in mRNA lifetimes

## Figure 1 plots

fig, ax = plt.subplots(2, 3, figsize=(11.5, 6.6))
al = np.logspace(-2.5, 2.5, 600)
betas = [(1, 0.00001), (0.05, 0.2), (0.05, 0.04)]

# (a) burst frequency along the symmetric line alpha_s = alpha_n
for i, (bs, bn) in enumerate(betas):
    f = ts.burst_frequency(al, bs, al, bn)
    ax[0, 0].loglog(al, f / gamma, color=C[i], lw=1.7,
                    label=r"$\beta_s=%.2f,\ \beta_n=%.2f$" % (bs, bn))
    astar = np.sqrt(bs * bn)
    ax[0, 0].plot(astar, ts.burst_frequency(astar, bs, astar, bn) / gamma,
                  "o", ms=5, mfc="w", mec=C[i], mew=1.4, zorder=5)
    ax[0, 0].axhline(max(bs, bn) / gamma, color=C[i], ls=":", lw=0.9, alpha=0.6)
ax[0, 0].set_xlabel(r"$\alpha_s=\alpha_n=\alpha$")
ax[0, 0].set_ylabel(r"burst frequency  $f/\gamma$")
ax[0, 0].legend(fontsize=7, loc="lower center")
ax[0, 0].text(0.03, 0.93, "o : $\\alpha^*=\\sqrt{\\beta_s\\beta_n}$\n"
              "$\\cdots$ : sup $=\\max(\\beta_s,\\beta_n)$",
              transform=ax[0, 0].transAxes, fontsize=7, va="top")

# (b) burst size along the same line
for i, (bs, bn) in enumerate(betas):
    ax[0, 1].loglog(al, ts.burst_size(al, bs, al, bn, k_y), color=C[i], lw=1.7)
    ax[0, 1].axhline(k_y * (1 / bs + 1 / bn) / 2, color=C[i], ls=":", lw=0.9, alpha=0.6)
ax[0, 1].set_xlabel(r"$\alpha_s=\alpha_n=\alpha$")
ax[0, 1].set_ylabel(r"mean burst size  $b$")
ax[0, 1].text(0.04, 0.93,
              r"$\cdots$ : $\alpha\!\to\!0$ limit $=\frac{k_y}{2}(\beta_s^{-1}+\beta_n^{-1})$",
              transform=ax[0, 1].transAxes, fontsize=7, va="top")

# (c) f-b trade-off, parametric in alpha, with iso-expression hyperbolae
al = np.logspace(-3, 3, 1000)
for i, (bs, bn) in enumerate(betas):
    f = ts.burst_frequency(al, bs, al, bn) / gamma
    b = ts.burst_size(al, bs, al, bn, k_y)
    ax[0, 2].loglog(f, b, color=C[i], lw=1.7)
bb = np.logspace(1, 3, 1000)
for yv in [0.3, 3, 30]:
    ax[0, 2].loglog(yv / bb, bb, color="0.55", ls="--", lw=0.7, zorder=0)
    ax[0, 2].text(yv / 1.6e3, 1.6e3, r"$\langle y\rangle=%g$" % yv,
                  fontsize=6.5, color="0.45", rotation=-45,
                  ha="left", va="top")
ax[0, 2].set_xlabel(r"$f/\gamma$"); ax[0, 2].set_ylabel(r"$b$")

# (d,e) full (alpha_s, alpha_n) plane at fixed betas
bs, bn = 0.05, 0.2
g = np.logspace(-2.5, 2.5, 260)
As, An = np.meshgrid(g, g, indexing="ij")
F = ts.burst_frequency(As, bs, An, bn) / gamma
B = ts.burst_size(As, bs, An, bn, k_y)
for j, (Z, lab, ttl, cm) in enumerate(
        [(F, r"$f/\gamma$", "(d) burst frequency", "viridis"),
         (B, r"$b$", "(e) mean burst size", "magma")]):
    a = ax[1, j]
    pc = a.pcolormesh(g, g, Z.T, norm=LogNorm(Z.min(), Z.max()),
                      cmap=cm, shading="auto", rasterized=True)
    a.contour(g, g, Z.T, levels=np.logspace(np.log10(Z.min()), np.log10(Z.max()), 9),
              colors="w", linewidths=0.4, alpha=0.5)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"$\alpha_s$"); a.set_ylabel(r"$\alpha_n$")
    a.set_title(ttl + r"   ($\beta_s=%.1f,\beta_n=%.1f$)" % (bs, bn))
    a.grid(False)
    plt.colorbar(pc, ax=a, label=lab, pad=0.02)
ax[1, 0].axvline(bn, color="w", ls="--", lw=0.9)
ax[1, 0].axhline(bs, color="w", ls="--", lw=0.9)
ax[1, 0].text(0.04, 0.05, r"$\partial f/\partial\alpha_s\propto\beta_s-\alpha_n$",
              transform=ax[1, 0].transAxes, fontsize=7, color="w")

# (f) how much of the expression is burst-like
al2 = np.logspace(-2.5, 2.5, 400)
bs, bn = 3, 8
ax[1, 2].loglog(al2, ts.mean_y(al2, bs, al2, bn, k_y, gamma), color=C[0], lw=1.7,
                label=r"$\langle y\rangle$")
ax[1, 2].loglog(al2, ts.fano(al2, bs, al2, bn, k_y, gamma), color=C[1], lw=1.7,
                label=r"$F=\mathrm{Var}/\langle y\rangle$")
ax[1, 2].loglog(al2, ts.burst_frequency(al2, bs, al2, bn) / gamma, color=C[2],
                lw=1.7, label=r"$f/\gamma$")
ax[1, 2].axhline(1, color="k", lw=0.8, ls="--")
ax[1, 2].fill_between(al2, 1e-3, 1e4,
                      where=ts.burst_frequency(al2, bs, al2, bn) / gamma < 1,
                      color="0.85", alpha=0.5, zorder=0)
ax[1, 2].set_xlabel(r"$\alpha_s=\alpha_n=\alpha$")
ax[1, 2].set_ylim(1e-2, 1e3); ax[1, 2].legend(fontsize=7.5)

fig.tight_layout()
fig.savefig(output_path("fig1_burst_parameters.svg"), bbox_inches="tight")
