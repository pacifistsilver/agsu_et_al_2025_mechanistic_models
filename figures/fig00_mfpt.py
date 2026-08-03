"""
Sensitivity of the ON-state MFPT (tbound) to binding rates (alpha_s, alpha_n)
and unbinding rates (beta_s, beta_n) in the heterodimer promoter model.

Baseline: beta_s = 0.05, beta_n = 0.2 (Agsu et al.); alpha_s, alpha_n inferred.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import stochtf.analytical.heterodimer as h
import stochtf.analytical.mfpt as fx

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 160,
    "mathtext.fontset": "dejavusans",
})

# ---- baseline ----------------------------------------------------------
BS, BN = 0.05, 0.20          # fixed unbinding rates
AS0, AN0 = 0.05, 0.20        # reference binding rates (tau_ON ~ order 1/beta)
grid = np.logspace(-3, 1, 400)

C_S, C_N = "#1b6ca8", "#c8553d"   # SOX2 / NANOG colours

fig = plt.figure(figsize=(11, 7.2))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.34)

# ======================================================================
# A. tau_ON vs alpha_s, family of beta_s
# ======================================================================
ax = fig.add_subplot(gs[0, 0])
for bs, ls in zip([0.2, 0.05, 1.0], ["-", "-",  ":"]):
    y = h.tbound(grid, bs, AN0, BN)
    ax.loglog(grid, y, ls, color=C_S, alpha=0.45 + 0.55 * (bs == BS), lw=1.8,
              label=rf"$\beta_s={bs}$")
ax.set_xlabel(r"$\alpha_s$  (SOX2 binding rate)")
ax.set_ylabel(r"$\tau_{\rm ON}$  (MFPT, bound states)")
ax.legend(frameon=False, fontsize=7.5)

# ======================================================================
# B. tau_ON vs beta_s, family of alpha_s
# ======================================================================
ax = fig.add_subplot(gs[0, 1])
for a_s, ls in zip([0.01, 0.5, 1.0], ["-", "-",  ":"]):
    y = h.tbound(a_s, grid, AN0, BN)
    ax.loglog(grid, y, ls, color=C_S, alpha=0.45 + 0.55 * (a_s == AS0), lw=1.8,
              label=rf"$\alpha_s={a_s}$")
ax.set_xlabel(r"$\beta_s$  (SOX2 unbinding rate)")
ax.set_ylabel(r"$\tau_{\rm ON}$")
ax.legend(frameon=False, fontsize=7.5)

# ======================================================================
# C. cross-site effect: vary NANOG rates, SOX2 fixed
# ======================================================================
ax = fig.add_subplot(gs[0, 2])
ax.loglog(grid, h.tbound(AS0, BS, grid, BN), color=C_N, lw=2,
          label=r"vary $\alpha_n$  ($\beta_n=0.2$)")
ax.loglog(grid, h.tbound(AS0, BS, AN0, grid), color=C_N, lw=2, ls="--",
          label=r"vary $\beta_n$  ($\alpha_n=0.2$)")
ax.text(1.2e-3, 1 / BS * 1.15, r"$1/\beta_s$  (SOX2 alone)", fontsize=7, color="0.35")
ax.set_xlabel(r"NANOG rate")
ax.set_ylabel(r"$\tau_{\rm ON}$")
ax.legend(frameon=False, fontsize=7.5)

# ======================================================================
# D-F. heatmaps
# ======================================================================
g2 = np.logspace(-2.5, 2.5, 220)
X, Y = np.meshgrid(g2, g2)


def panel(ax, Z, xlab, ylab, title, mark):
    pcm = ax.pcolormesh(X, Y, Z, norm=LogNorm(vmin=Z.min(), vmax=Z.max()),
                        cmap="magma", shading="auto", rasterized=True)
    cs = ax.contour(X, Y, Z, levels=np.logspace(np.log10(Z.min()), np.log10(Z.max()), 8),
                    colors="w", linewidths=0.5, alpha=0.55)
    ax.clabel(cs, inline=True, fontsize=5.5, fmt="%.3g")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(*mark, "o", mfc="none", mec="cyan", mew=1.4, ms=7)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r"$\tau_{\rm ON}$", fontsize=8)
    cb.ax.tick_params(labelsize=6.5)


panel(fig.add_subplot(gs[1, 0]),
      h.tbound(X, Y, AN0, BN), r"$\alpha_s$", r"$\beta_s$",
      "D  SOX2 site: $\\tau_{\\rm ON}$ set by\nthe ratio $\\alpha_s/\\beta_s$", (AS0, BS))

panel(fig.add_subplot(gs[1, 1]),
      h.tbound(X, BS, Y, BN), r"$\alpha_s$", r"$\alpha_n$",
      "E  Both binding rates\n(the inferred plane)", (BS, BN))

panel(fig.add_subplot(gs[1, 2]),
      h.tbound(AS0, X, AN0, Y), r"$\beta_s$", r"$\beta_n$",
      "F  Both unbinding rates:\nthe slower site dominates", (BS, BN))

fig.savefig("./figures/output/mfpt_alpha_beta.png", bbox_inches="tight")
print("saved main figure")