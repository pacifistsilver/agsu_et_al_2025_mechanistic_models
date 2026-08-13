
"""
Noise and count distributions for HD vs M, everything from the exact
stationary solution of the joint (promoter, mRNA) CME.

HD : four-state square 00-10-01-11, independent sites, OR emission,
     active {10, 01, 11}
M  : three-state star  S <-> 0 <-> N, exclusive (competitive) binding --
     SOX2 and NANOG contest the same site and are never co-bound.
     OR emission, active {S, N}.

Two sweeps are produced from the same machinery:
  fig_4_noise_alpha : (alpha_s, alpha_n) plane, unbinding rates held fixed
  fig_4_noise_beta  : (beta_s,  beta_n)  plane, binding rates held fixed

Heatmaps use finite state projection (FSP) at every grid point.  Exact
moment equations are solved alongside as a validation of the truncation.
"""
import os

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from stochtf.plotting import use_paper_style
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.sparse import diags, identity, bmat, csc_matrix
from scipy.sparse.linalg import spsolve

use_paper_style(sans_serif="Arial")

BS0, BN0 = 0.05, 0.20        # reference unbinding rates
AS0, AN0 = 10.0, 0.01       # reference binding rates
GAM, KY = 1.0, 20.0
NGRID = 48
LBL_M = "M"                  # exclusive-binding monomer
RATE = r"(s$^{-1}$)"


# ------------------------------------------------------------- generators
def excl_Q(a_s, b_s, a_n, b_n):
    """Exclusive binding:  S <-> 0 <-> N,  no co-bound state.

    State order [0, S, N].  A bound factor must leave before the other can
    bind, so there is no S <-> N edge and the ON set is two states each
    with a single exit.
    """
    Q = np.array([[0., a_s, a_n],
                  [b_s, 0., 0.],
                  [b_n, 0., 0.]])
    np.fill_diagonal(Q, -Q.sum(1))
    return Q, np.array([0., 1., 1.])


def hd_Q(a_s, b_s, a_n, b_n):
    """00, 10, 01, 11 with independent sites."""
    Q = np.zeros((4, 4))
    Q[0,1]=a_s; Q[0,2]=a_n; Q[1,0]=b_s; Q[1,3]=a_n
    Q[2,0]=b_n; Q[2,3]=a_s; Q[3,1]=b_n; Q[3,2]=b_s
    np.fill_diagonal(Q, -Q.sum(1))
    return Q, np.array([0., 1., 1., 1.])


# ------------------------------------------- analytic kinetics, M (exact)
def excl_kinetics(a_s, b_s, a_n, b_n, ky=KY):
    """(tau_on, tau_off, f, b, CV_on) for the exclusive-binding model.

    An ON period is a single sojourn in whichever state was entered, so
        tau_on = phi_s/beta_s + phi_n/beta_n,  phi_j = alpha_j/(alpha_s+alpha_n)
    a mixture of two exponentials -- NOT 1/(beta_s + beta_n).
    """
    tot = a_s + a_n
    ph_s, ph_n = a_s/tot, a_n/tot
    mu_s, mu_n = 1.0/b_s, 1.0/b_n
    tau_on = ph_s*mu_s + ph_n*mu_n
    tau_off = 1.0/tot
    m2 = 2.0*(ph_s*mu_s**2 + ph_n*mu_n**2)
    cv_on = np.sqrt(m2 - tau_on**2)/tau_on
    f = 1.0/(tau_on + tau_off)
    return tau_on, tau_off, f, ky*tau_on, cv_on


# ---------------------------------------------------------- exact moments
def exact_moments(Q, act, ky, gamma):
    """(<y>, Fano) from the stationary moment equations -- no truncation.

    (Q^T - gamma I) m = -lambda*p
    (Q^T - 2 gamma I) q = -[lambda*(2m + p) + gamma m]
    """
    ns = Q.shape[0]
    lam = ky * np.asarray(act, dtype=float)
    A = np.vstack([Q.T[:-1], np.ones(ns)])
    rhs = np.zeros(ns); rhs[-1] = 1.0
    p = np.linalg.lstsq(A, rhs, rcond=None)[0]
    m = np.linalg.solve(Q.T - gamma*np.eye(ns), -(lam*p))
    q = np.linalg.solve(Q.T - 2*gamma*np.eye(ns), -(lam*(2*m + p) + gamma*m))
    mean = m.sum()
    return mean, (q.sum() - mean**2)/mean


# ------------------------------------------------------------------- FSP
def fsp(Q, act, ky, gamma, ymax):
    """Exact stationary P(y) on the truncated state space, marginalised
    over the promoter.  Block-tridiagonal build (one block per promoter
    state) rather than an element-wise loop."""
    ns, ny = Q.shape[0], ymax + 1
    y = np.arange(ny)
    out = -np.diag(Q)
    blocks = [[None]*ns for _ in range(ns)]
    for s in range(ns):
        birth = ky * act[s]
        d = -(birth*(y < ny-1) + gamma*y + out[s])
        blocks[s][s] = diags([d, birth*np.ones(ny-1), gamma*y[1:]],
                             offsets=[0, 1, -1], format="csr")
    for u in range(ns):
        for v in range(ns):
            if u != v and Q[u, v] > 0:
                blocks[u][v] = Q[u, v] * identity(ny, format="csr")
    M = bmat(blocks, format="csc").T.tolil()
    M[0, :] = 1.0
    rhs = np.zeros(ns*ny); rhs[0] = 1.0
    return spsolve(csc_matrix(M), rhs).reshape(ns, ny).sum(0)


def ymax_for(*moment_pairs, floor=60):
    """Truncation that covers mean + 10 sd for every model at this point."""
    hi = max(mu + 10.0*np.sqrt(max(F*mu, 0.0)) for mu, F in moment_pairs)
    return int(max(floor, np.ceil(hi) + 20))


def stats(P):
    y = np.arange(P.size)
    mean = P @ y
    var = P @ (y - mean)**2
    return mean, var/mean, np.sqrt(var)/mean


def both_models(rates, floor=60):
    """FSP stats for HD and M at one parameter point."""
    Qh, Ah = hd_Q(*rates)
    Qm, Am = excl_Q(*rates)
    mh, Fh = exact_moments(Qh, Ah, KY, GAM)
    mm, Fm = exact_moments(Qm, Am, KY, GAM)
    ym = ymax_for((mh, Fh), (mm, Fm), floor=floor)
    Ph = fsp(Qh, Ah, KY, GAM, ym)
    Pm = fsp(Qm, Am, KY, GAM, ym)
    return Ph, Pm, (Fh, Fm)


# --------------------------------------------------------------- figure
def make_figure(build, xvals, yvals, xlab, ylab, xref, yref,
                fixed_note, dist_pts, dist_title, outname):
    """build(x, y) -> (a_s, b_s, a_n, b_n) defines which plane is swept."""
    X, Y = np.meshgrid(xvals, yvals)
    F_hd = np.zeros_like(X); F_m = np.zeros_like(X)
    CV2_hd = np.zeros_like(X); CV2_m = np.zeros_like(X)
    TV = np.zeros_like(X)
    err = 0.0

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Ph, Pm, (Fh, Fm) = both_models(build(X[i, j], Y[i, j]))
            _, f1, c1 = stats(Ph)
            _, f2, c2 = stats(Pm)
            F_hd[i, j], CV2_hd[i, j] = f1, c1**2
            F_m[i, j], CV2_m[i, j] = f2, c2**2
            TV[i, j] = 0.5*np.abs(Ph - Pm).sum()
            err = max(err, abs(f1 - Fh)/Fh, abs(f2 - Fm)/Fm)
    print(f"[{outname}] max |FSP - exact| relative error in Fano: {err:.3e}")

    dists = []
    for v in dist_pts:
        Ph, Pm, _ = both_models(build(v, v), floor=120)
        dists.append((v, Ph, Pm, 0.5*np.abs(Ph - Pm).sum()))

    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.linewidth": .7, "axes.edgecolor": "#3a3a3a",
                         "legend.frameon": False, "figure.dpi": 160})
    C_M, C_HD = "#c0562b", "#1f5f8b"
    fig, ax = plt.subplots(3, 3, figsize=(11.0, 9.4))

    def heat(a, Z, title, cmap, norm, cb_label, levels=None, note=True):
        im = a.pcolormesh(X, Y, Z, cmap=cmap, norm=norm,
                          shading="auto", rasterized=True)
        if levels is not None:
            cs = a.contour(X, Y, Z, levels=levels, colors="w",
                           linewidths=.4, alpha=.7)
            a.clabel(cs, fmt="%g", fontsize=5.5, inline=True)
        a.set_xscale("log"); a.set_yscale("log")
        a.axvline(xref, color="w", lw=.5, ls=":", alpha=.5)
        a.axhline(yref, color="w", lw=.5, ls=":", alpha=.5)
        a.set_xlabel(xlab); a.set_ylabel(ylab)
        if note:
            a.text(0.97, 0.04, fixed_note, transform=a.transAxes,
                   ha="right", va="bottom", fontsize=5.8, color="w",
                   bbox=dict(fc="k", ec="none", alpha=0.35, pad=1.4))
        cb = fig.colorbar(im, ax=a, pad=.02, fraction=.046)
        cb.set_label(cb_label, fontsize=7)
        cb.ax.tick_params(labelsize=6)

    # ---- row 1: distributions
    for k, (v, Ph, Pm, tv) in enumerate(dists):
        a = ax[0, k]
        yy = np.arange(Ph.size)
        a.plot(yy, Pm, color=C_M, lw=1.6, label=LBL_M)
        a.fill_between(yy, Pm, color=C_M, alpha=.18)
        a.plot(yy, Ph, color=C_HD, lw=1.6, label="HD")
        a.fill_between(yy, Ph, color=C_HD, alpha=.18)
        a.set_yscale("log"); a.set_ylim(1e-5, 1.0)
        a.set_xlim(0, min(Ph.size-1, 110))
        a.set_xlabel("mRNA count $y$"); a.set_ylabel("$P(y)$")
        mM, FM, _ = stats(Pm); mH, FH, _ = stats(Ph)
        a.text(.40, .95,
               f"{LBL_M:<4} <y>={mM:5.1f}  F={FM:5.2f}\n"
               f"HD   <y>={mH:5.1f}  F={FH:5.2f}\n"
               f"TV  = {tv:.3f}",
               transform=a.transAxes, fontsize=6.2, va="top",
               family="monospace")
        a.legend(fontsize=7, loc="lower left")

    # ---- row 2: Fano
    flo = min(F_hd.min(), F_m.min()); fhi = max(F_hd.max(), F_m.max())
    fnorm = LogNorm(vmin=flo, vmax=fhi)
    flev = 10.0**np.arange(np.floor(np.log10(flo)), np.ceil(np.log10(fhi))+1)
    heat(ax[1, 0], F_hd, "HD:  Fano factor", "magma", fnorm, r"$F$", flev)
    heat(ax[1, 1], F_m, f"{LBL_M}:  Fano factor", "magma", fnorm, r"$F$", flev)
    R = np.log2(F_hd/F_m)
    rmax = np.abs(R).max()
    heat(ax[1, 2], R, rf"$\log_2 (F_{{\rm HD}}/F_{{\rm {LBL_M}}})$", "RdBu_r",
         TwoSlopeNorm(vcenter=0.0, vmin=-rmax, vmax=rmax),
         r"$\log_2$ ratio", levels=[-1, 0, 1])

    # ---- row 3: CV^2 and distinguishability
    clo = min(CV2_hd.min(), CV2_m.min()); chi = max(CV2_hd.max(), CV2_m.max())
    cnorm = LogNorm(vmin=clo, vmax=chi)
    clev = 10.0**np.arange(np.floor(np.log10(clo)), np.ceil(np.log10(chi))+1)
    heat(ax[2, 0], CV2_hd, "HD:  $CV^2$", "magma", cnorm, r"$CV^2$", clev)
    heat(ax[2, 1], CV2_m, f"{LBL_M}:  $CV^2$", "magma", cnorm, r"$CV^2$", clev)
    heat(ax[2, 2], TV,
         rf"total variation $d_{{\rm TV}}(P_{{\rm HD}}, P_{{\rm {LBL_M}}})$",
         "RdBu_r", None, r"$d_{\rm TV}$", levels=[0.02, 0.05, 0.1, 0.25])

    fig.tight_layout(w_pad=2.2, h_pad=2.6)
    os.makedirs("./figures/output", exist_ok=True)
    out = f"./figures/output/{outname}"
    fig.savefig(out + ".svg", bbox_inches="tight", facecolor="w")
    plt.close(fig)

    frac = lambda t: float(np.mean(TV < t))
    print(f"  peak d_TV = {TV.max():.3f}   "
          f"< 0.05 on {frac(0.05):.3f} of grid   < 0.10 on {frac(0.10):.3f}")
    print(f"  |log2 Fano ratio| max = {rmax:.3f}   "
          f"within 2x on {float(np.mean(np.abs(R) < 1)):.3f} of grid")
    print(f"  wrote {out}.svg / .png")
    return TV, R


# --------------------------------------------------------------- alpha plane
al = np.logspace(-2.5,2.5, NGRID)
fx_a = rf"$\beta_s={BS0:g}$, $\beta_n={BN0:g}$"
make_figure(lambda x, y: (x, BS0, y, BN0), al, al,
            rf"$k_{{on,s}}$ {RATE}", rf"$k_{{on,n}}$ {RATE}", BS0, BN0, fx_a,
            [0.01, 0.1, 1.0], "",
            "fig_4_noise_alpha")


# ------------------------------------------------- M kinetics on the beta plane
print(f"\n{LBL_M} burst kinetics along beta_s = beta_n "
      f"(alpha_s={AS0:g}, alpha_n={AN0:g}); mass balance is exact")
print(f"{'beta':>8}{'tau_on':>10}{'tau_off':>9}{'f':>9}{'b':>10}"
      f"{'CV_on':>8}{'b f/gam':>10}{'<y> exact':>11}")
for b in [0.005, 0.02, 0.05, 0.2, 1.0]:
    ton, toff, f_, b_, cv = excl_kinetics(AS0, b, AN0, b)
    my, _ = exact_moments(*excl_Q(AS0, b, AN0, b), KY, GAM)
    print(f"{b:8.3g}{ton:10.3f}{toff:9.3f}{f_:9.4f}{b_:10.2f}"
          f"{cv:8.3f}{b_*f_/GAM:10.4f}{my:11.4f}")
print(f"  bounds: 1/beta_n <= tau_on <= 1/beta_s always, so exclusive binding "
      f"cannot extend residence")