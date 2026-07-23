"""Plots of burst frequency and burst size for the two-site OR-gate promoter."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import twosite as ts

mpl.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130, "savefig.dpi": 160, "legend.frameon": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "font.sans-serif": "Arial"
})
C = ["#2a6f97", "#c1440e", "#3d8168", "#8a5cb8", "#b08000"]
gamma, k_y = 1.0, 20.0          # time in mRNA lifetimes

## Figure 1 plots


fig, ax = plt.subplots(2, 3, figsize=(11.5, 6.6))
al = np.logspace(-2.5, 0, 600)
betas = [(1.0, 1.0), (0.05, 0.2), (1.0, 0.001)]

# (a) burst frequency along the symmetric line alpha_s = alpha_n
for i, (bs, bn) in enumerate(betas):
    f = ts.burst_frequency(al, bs, al, bn)
    ax[0, 0].loglog(al, f / gamma, color=C[i], lw=1.7,
                    label=r"$\beta_s=%.1f,\ \beta_n=%.1f$" % (bs, bn))
    astar = np.sqrt(bs * bn)
    ax[0, 0].plot(astar, ts.burst_frequency(astar, bs, astar, bn) / gamma,
                  "o", ms=5, mfc="w", mec=C[i], mew=1.4, zorder=5)
    ax[0, 0].axhline(max(bs, bn) / gamma, color=C[i], ls=":", lw=0.9, alpha=0.6)
ax[0, 0].set_xlabel(r"$\alpha_s=\alpha_n=\alpha$")
ax[0, 0].set_ylabel(r"burst frequency  $f/\gamma$")
ax[0, 0].set_ylim(1e-3, 20)
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
g = np.logspace(-2.5, 0, 260)
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
    a.grid(False)
    plt.colorbar(pc, ax=a, label=lab, pad=0.02)
ax[1, 0].axvline(bn, color="w", ls="--", lw=0.9)
ax[1, 0].axhline(bs, color="w", ls="--", lw=0.9)
ax[1, 0].text(0.04, 0.05, r"$\partial f/\partial\alpha_s\propto\beta_s-\alpha_n$",
              transform=ax[1, 0].transAxes, fontsize=7, color="w")

# (f) how much of the expression is burst-like
al2 = np.logspace(-2.5, 2.5, 400)
bs, bn = 0.05, 0.2
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
fig.savefig("./fig1_burst_parameters.svg", bbox_inches="tight")
"""
# ======================================================================
# FIGURE 2 -- the burst-size distribution
# ======================================================================
fig2, ax2 = plt.subplots(1, 3, figsize=(11.5, 3.4))
a_s, b_s, a_n, b_n = 0.7, 0.05, 0.35, 0.2

pmf = ts.burst_size_pmf(a_s, b_s, a_n, b_n, k_y, mmax=250)
m = np.arange(pmf.size)
b = ts.burst_size(a_s, b_s, a_n, b_n, k_y)
geo = (1 / (1 + b)) * (b / (1 + b)) ** m           # geometric, same mean
_, sz, _ = ts.gillespie_bursts(a_s, b_s, a_n, b_n, k_y, n_bursts=40000, seed=2)

ax2[0].bar(m, pmf, width=1.0, color=C[0], alpha=0.35, label="exact (phase-type)")
ax2[0].plot(m, pmf, color=C[0], lw=1.4)
ax2[0].plot(m, geo, color=C[1], lw=1.4, ls="--", label="geometric, same mean")
hb = np.bincount(sz, minlength=pmf.size)[:pmf.size] / sz.size
ax2[0].plot(m, hb, ".", ms=2.5, color="k", label="Gillespie")
ax2[0].set_xlim(0, 90); ax2[0].set_xlabel("burst size $m$")
ax2[0].set_ylabel(r"$P(m)$"); ax2[0].legend(fontsize=7)
ax2[0].set_title("(a) not a single geometric")

ax2[1].semilogy(m, 1 - np.cumsum(pmf), color=C[0], lw=1.6, label="exact")
ax2[1].semilogy(m, (b / (1 + b)) ** (m + 1), color=C[1], lw=1.4, ls="--",
                label="geometric")
for r, c in zip(ts.burst_size_geometric_rates(a_s, b_s, a_n, b_n, k_y), ["0.3", "0.5", "0.7"]):
    ax2[1].semilogy(m, r ** m, color=c, lw=0.8, ls=":")
ax2[1].set_ylim(1e-6, 2); ax2[1].set_xlim(0, 200)
ax2[1].set_xlabel("burst size $m$"); ax2[1].set_ylabel(r"$P(N>m)$")
ax2[1].set_title("(b) three geometric components (dotted)")
ax2[1].legend(fontsize=7)

# bursty limit with unequal beta: two-component geometric mixture
sc = 40.0
a_s2, a_n2 = 0.5, 0.5
b_s2, b_n2, k2 = 2.0 * sc, 0.5 * sc, 1.0 * sc     # b_s=0.5, b_n=2 -> mean sizes 0.5,2
pmf2 = ts.burst_size_pmf(a_s2, b_s2, a_n2, b_n2, k2, mmax=400)
m2 = np.arange(pmf2.size)
bs_, bn_ = k2 / b_s2, k2 / b_n2
mix = 0.5 * (1 / (1 + bs_)) * (bs_ / (1 + bs_)) ** m2 \
    + 0.5 * (1 / (1 + bn_)) * (bn_ / (1 + bn_)) ** m2
bbar = (a_s2 * bs_ + a_n2 * bn_) / (a_s2 + a_n2)
single = (1 / (1 + bbar)) * (bbar / (1 + bbar)) ** m2
ax2[2].semilogy(m2, pmf2, color=C[0], lw=1.6, label="exact")
ax2[2].semilogy(m2, mix, color=C[2], lw=1.3, ls="--", label="2-geometric mixture")
ax2[2].semilogy(m2, single, color=C[1], lw=1.3, ls=":", label="1 geometric, same mean")
ax2[2].set_xlim(0, 20); ax2[2].set_ylim(1e-5, 1)
ax2[2].set_xlabel("burst size $m$"); ax2[2].set_ylabel(r"$P(m)$")
ax2[2].set_title(r"(c) bursty limit, $\beta_s\neq\beta_n$")
ax2[2].legend(fontsize=7)

fig2.suptitle("Burst-size distribution is discrete phase-type with 3 phases", y=1.02, fontsize=11)
fig2.tight_layout()
fig2.savefig("./fig2_burst_size_distribution.png", bbox_inches="tight")

# ======================================================================
# FIGURE 3 -- mechanistic vs measured burst parameters
# ======================================================================
fig3, ax3 = plt.subplots(1, 3, figsize=(11.5, 3.4))
a_s, b_s, a_n, b_n = 0.7, 0.05, 0.35, 0.2
gam = np.logspace(-2, 2, 400)

bmech = ts.burst_size(a_s, b_s, a_n, b_n, k_y) * np.ones_like(gam)
beff = ts.burst_size_eff(a_s, b_s, a_n, b_n, k_y, gam)
fmech = ts.burst_frequency(a_s, b_s, a_n, b_n) * np.ones_like(gam)
feff = ts.burst_frequency_eff(a_s, b_s, a_n, b_n, k_y, gam)

ax3[0].loglog(gam, bmech, color=C[0], lw=1.7, label=r"mechanistic $b=k_y\langle T_{on}\rangle$")
ax3[0].loglog(gam, beff, color=C[1], lw=1.7, label=r"measured $b_{\rm eff}=F-1$")
ax3[0].axvline(b_s + a_s, color="0.5", ls=":", lw=0.9)
ax3[0].axvline(b_n + a_n, color="0.5", ls=":", lw=0.9)
ax3[0].set_xlabel(r"$\gamma$"); ax3[0].set_ylabel("burst size")
ax3[0].set_title(r"(a) degradation filters the burst"); ax3[0].legend(fontsize=7)

ax3[1].loglog(gam, fmech / gam, color=C[0], lw=1.7, label=r"$f/\gamma$")
ax3[1].loglog(gam, feff / gam, color=C[1], lw=1.7, label=r"$f_{\rm eff}/\gamma$")
ax3[1].set_xlabel(r"$\gamma$"); ax3[1].set_ylabel(r"burst frequency / $\gamma$")
ax3[1].set_title("(b) inferred frequency is over-estimated"); ax3[1].legend(fontsize=7)

# size-bias inflation in the bursty limit
rat = np.logspace(-2, 2, 300)                     # b_n / b_s
bs0 = 1.0
bser, bapp = [], []
for r in rat:
    bs_i, bn_i = bs0, bs0 * r
    sc = 400.0
    A_s, A_n = 0.5, 0.5
    B_s, B_n, K = sc / bs_i, sc / bn_i, sc
    bt = (A_s * bs_i + A_n * bn_i) / (A_s + A_n)
    bser.append(bt)
    bapp.append(ts.burst_size_eff(A_s, B_s, A_n, B_n, K, 1e-4))
bser, bapp = np.array(bser), np.array(bapp)
ax3[2].loglog(rat, bapp / bser, color=C[3], lw=1.8)
ax3[2].axhline(1, color="k", lw=0.8, ls="--")
ax3[2].set_xlabel(r"$b_n/b_s$  (burst-size heterogeneity)")
ax3[2].set_ylabel(r"$b_{\rm eff}/b$")
ax3[2].set_title(r"(c) size bias: $b_{\rm eff}=b\,(1+\mathrm{CV}_b^2)$")

fig3.suptitle("Mechanistic burst parameters vs what a Fano-factor fit reports",
              y=1.02, fontsize=11)
fig3.tight_layout()
fig3.savefig("./fig3_measured_vs_mechanistic.png", bbox_inches="tight")
print("figures written")
"""