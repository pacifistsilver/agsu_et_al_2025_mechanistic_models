"""The burst plane by FSP: mean mRNA over (burst frequency, burst size).

Outputs: fig10_burst_plane_fsp.svg / .pdf
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from stochtf.plotting import PALETTE, output_path, use_paper_style

use_paper_style()
def output_path(name):
    d = os.environ.get("STOCHTF_FIGURE_DIR", ".")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def fsp_stationary(k_on, k_off, k_y, gamma, ymax):
    """Stationary P(y) for the two-state telegraph.

    The stationary CME couples only neighbouring counts, so with
    P_y = (P(OFF, y), P(ON, y)) it is block-tridiagonal in 2x2 blocks

        K P_{y-1} + (Q^T - K - gamma y I) P_y + gamma (y+1) P_{y+1} = 0,

    with Q the promoter generator and K = diag(0, k_y).  Writing
    P_{y+1} = R_y P_y and sweeping down from R_ymax = 0 costs O(ymax) tiny
    solves instead of one big sparse solve.  K has rank one, so R_y has a
    single non-zero column and the sweep collapses to two scalars per level:

        r0_y = u_{y+1}/v_y,   r1_y = v_{y+1}/v_y.

    Production out of y = ymax is switched off, so the truncation is reflecting
    and probability is conserved.  Verified against a full sparse-LU FSP to
    2e-16 and against NB(f/gamma, b) in the bursty limit.

    Args:
        k_on: Promoter activation rate.
        k_off: Promoter deactivation rate.
        k_y: Transcription rate in the active state.
        gamma: mRNA degradation rate.
        ymax: Count grid bound.

    Returns:
        The stationary distribution over counts 0..ymax.
    """
    n = ymax + 1
    r0 = np.zeros(n)
    r1 = np.zeros(n)
    for y in range(ymax, 0, -1):
        ky = k_y if y < ymax else 0.0                # reflecting top boundary
        a = -k_on - gamma * y
        c = k_off + gamma * (y + 1) * r0[y]
        d = -k_off - ky - gamma * y + gamma * (y + 1) * r1[y]
        det = a * d - c * k_on
        r0[y - 1] = k_y * c / det
        r1[y - 1] = -k_y * a / det

    v = np.empty(n)                                  # ON compartment
    v[0] = 1.0
    np.cumprod(r1[:n - 1], out=v[1:])
    u = np.empty(n)                                  # OFF compartment
    u[0] = (k_off + gamma * r0[0]) / k_on
    u[1:] = r0[:n - 1] * v[:n - 1]
    p = u + v
    return p / p.sum()


def auto_ymax(k_on, k_off, k_y, gamma, nsig=6.0, pad=40):
    """Truncation level from the negative-binomial moment estimates."""
    mu = k_y * k_on / ((k_on + k_off) * gamma)
    var = mu * (1.0 + k_y / k_off)
    return int(mu + nsig * np.sqrt(var)) + pad


def fsp_moments(k_on, k_off, k_y, gamma=1.0, tol=1e-9, grow=6):
    """(mean, variance) with the truncation audited against the exact mean.

    <y> = k_y <sigma>/gamma needs no truncation, so the gap between it and the
    FSP sum measures directly how much of the distribution the projection is
    missing.  ymax is doubled until that gap falls below tol.

    Args:
        k_on: Promoter activation rate.
        k_off: Promoter deactivation rate.
        k_y: Transcription rate in the active state.
        gamma: mRNA degradation rate.
        tol: Largest acceptable gap against the exact mean.
        grow: Factor by which ymax is grown each round.

    Returns:
        A tuple (mean, variance).
    """
    exact = k_y * k_on / ((k_on + k_off) * gamma)
    ymax = auto_ymax(k_on, k_off, k_y, gamma)
    for _ in range(grow):
        P = fsp_stationary(k_on, k_off, k_y, gamma, ymax)
        y = np.arange(P.size)
        m1 = (y * P).sum()
        if abs(m1 - exact) <= tol * max(exact, 1e-30):
            break
        ymax *= 2
    m2 = (y * y * P).sum()
    return m1, m2 - m1 * m1


GAMMA = 1.0                       # time in mRNA lifetimes
K_Y = 1.0 * GAMMA                # initiation rate, when it is the held rate
K_OFF = 300.0 * GAMMA             # OFF rate, when it is the held rate
NF, NB = 72, 72                   # grid

fg = np.logspace(-2, 2, NF)       # f / gamma
bb = np.logspace(0, 2, NB)        # mean burst size


def sweep(hold):
    """<y>, F over the (f, b) grid, under one of the two rate conventions."""
    Y = np.empty((NF, NB))
    F = np.empty((NF, NB))
    for i, f in enumerate(fg):
        for j, b in enumerate(bb):
            k_on = f * GAMMA
            if hold == "k_y":
                k_y, k_off = K_Y, K_Y / b
            else:
                k_off, k_y = K_OFF, b * K_OFF
            m, v = fsp_moments(k_on, k_off, k_y, GAMMA)
            Y[i, j], F[i, j] = m, v / m
    return Y, F


def or_track(bs, bn, k_y=K_Y, n=400):
    al = np.logspace(-3, 3, n)
    r = (bs / (al + bs)) * (bn / (al + bn))
    f = r * 2 * al
    return f / GAMMA, k_y * (1 - r) / f


use_paper_style()
C = PALETTE

Yk, Fk = sweep("k_y")
Yo, Fo = sweep("k_off")

BURSTY = np.outer(fg, bb)          # b f / gamma, the bursty-limit prediction
LEVELS = [0.1, 1, 10, 100, 1000]

fig, ax = plt.subplots(2, 3, figsize=(13.0, 7.4))
rows = [("k_y", Yk, Fk, r"$k_y=%g\gamma$ held  (initiation-limited)" % K_Y),
        ("k_off", Yo, Fo, r"$k_{\rm off}=%g\gamma$ held  (duration-limited)" % K_OFF)]

ymin = min(Yk.min(), Yo.min())
ymax_ = max(Yk.max(), Yo.max())

for row, (hold, Y, F, tag) in enumerate(rows):

    a = ax[row, 0]
    pc = a.pcolormesh(fg, bb, Y.T, norm=LogNorm(ymin, ymax_), cmap="viridis",
                      shading="auto", rasterized=True)
    cs = a.contour(fg, bb, Y.T, levels=LEVELS, colors="w", linewidths=1.2)
    a.clabel(cs, fmt=r"%g", fontsize=6.5)
    a.contour(fg, bb, BURSTY.T, levels=LEVELS, colors="#ff7f4d",
              linewidths=0.9, linestyles="--")
    ft, bt = or_track(1.0, 1.0)
    a.plot(ft, bt, color="k", lw=1.4, alpha=0.9)
    cb = fig.colorbar(pc, ax=a, label=r"$\langle y\rangle$")
    if hold == "k_y":
        cb.ax.axhline(K_Y / GAMMA, color="#ff7f4d", lw=1.4)
    a.set_title(r"(%s) mean mRNA, %s" % ("ad"[row], tag))

    a = ax[row, 1]
    pc = a.pcolormesh(fg, bb, F.T, norm=LogNorm(1.0, max(Fk.max(), Fo.max())),
                      cmap="magma", shading="auto", rasterized=True)
    lv = [l for l in (1.04, 1.5, 3, 10, 50) if F.min() < l < F.max()]
    cs = a.contour(fg, bb, F.T, levels=lv, colors="w", linewidths=1.1)
    a.clabel(cs, fmt=r"%g", fontsize=6.5)
    fig.colorbar(pc, ax=a, label=r"$F=\sigma^2/\langle y\rangle$")
    a.set_title(r"(%s) Fano factor" % "be"[row])

    a = ax[row, 2]
    R = Y / BURSTY
    pc = a.pcolormesh(fg, bb, R.T, norm=Normalize(0, 1), cmap="cividis",
                      shading="auto", rasterized=True)
    cs = a.contour(fg, bb, R.T, levels=[0.1, 0.5, 0.9, 0.99], colors="w",
                   linewidths=1.1)
    a.clabel(cs, fmt=r"%.2f", fontsize=6.5)
    fig.colorbar(pc, ax=a, label=r"$\gamma\langle y\rangle/(bf)$")
    a.set_title(r"(%s) burst-limit error" % "cf"[row])

for a in ax.ravel():
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel(r"burst frequency  $f/\gamma$")
    a.set_ylabel(r"mean burst size  $b$")
    a.set_xlim(fg[0], fg[-1])
    a.set_ylim(bb[0], bb[-1])
    a.grid(False)

# annotations
for r in (0, 1):
    ax[r, 0].text(0.035, 0.05,
                  "white: exact $\\langle y\\rangle$ (FSP)\n"
                  "orange $--$: bursty limit $bf/\\gamma$\n"
                  "black: two-site OR track ($\\beta_s=\\beta_n=\\gamma$)",
                  transform=ax[r, 0].transAxes, fontsize=6.3, va="bottom",
                  color="w")
ax[0, 0].text(0.52, 0.72,
              r"ceiling: $\langle y\rangle\to k_y/\gamma=%g$" % (K_Y / GAMMA),
              transform=ax[0, 0].transAxes, fontsize=7.5, color="w",
              rotation=-38, ha="center")
ax[0, 1].text(0.035, 0.06, r"$F=1.04$: the Poisson floor of the mESC fits",
              transform=ax[0, 1].transAxes, fontsize=6.5, color="w")
ax[1, 1].text(0.035, 0.06,
              r"no $F\!<\!1.04$ anywhere: $k_{\rm off}\!\gg\!\gamma$ forces $F\approx 1+b$",
              transform=ax[1, 1].transAxes, fontsize=6.5, color="w")
ax[1, 2].text(0.035, 0.10,
              r"depends on $f$ only: $\gamma\langle y\rangle/(bf)=1-p_{\rm on}$",
              transform=ax[1, 2].transAxes, fontsize=6.5, color="k")

fig.suptitle(r"The burst plane by FSP: $(f,b)$ alone does not determine "
             r"$\langle y\rangle$", y=1.0, fontsize=11)
fig.tight_layout()
fig.savefig(output_path("fig10_burst_plane_fsp.svg"), bbox_inches="tight")
fig.savefig(output_path("fig10_burst_plane_fsp.pdf"), bbox_inches="tight")

print("ceiling under fixed k_y:  max <y> = %.4f   (k_y/gamma = %.1f)"
      % (Yk.max(), K_Y / GAMMA))
print("max <y> under fixed k_off: %.1f" % Yo.max())
r_ky = (Yk / BURSTY)
r_ko = (Yo / BURSTY)
print("gamma<y>/(bf):  fixed k_y   %.4f -- %.4f" % (r_ky.min(), r_ky.max()))
print("                fixed k_off %.4f -- %.4f" % (r_ko.min(), r_ko.max()))
print("fraction of plane within 1%% of the bursty limit: k_y %.3f, k_off %.3f"
      % ((r_ky > 0.99).mean(), (r_ko > 0.99).mean()))
print("Fano <= 1.04 (Poisson-indistinguishable) fraction: k_y %.3f, k_off %.3f"
      % ((Fk <= 1.04).mean(), (Fo <= 1.04).mean()))
print("largest disagreement between the two conventions at the same (f,b): %.1fx"
      % np.max(Yo / Yk))
print("wrote fig10")