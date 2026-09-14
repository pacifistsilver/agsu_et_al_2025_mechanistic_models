"""
Noise and count distributions for HD vs M, everything from the exact stationary solution of the joint (promoter, mRNA) CME.
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from stochtf.plotting import use_paper_style
from stochtf.cme import stationary as st
from matplotlib.colors import LogNorm, TwoSlopeNorm

use_paper_style(sans_serif="Arial")

BS0, BN0 = 0.06, 0.24        # reference unbinding rates
AS, AN = 1.0, 0.2
GAM, KY = 1.0, 20.0
NGRID = 48
LBL_M = "M"                  # exclusive-binding monomer
RATE = r"(s$^{-1}$)"


def excl_Q(a_s, b_s, a_n, b_n):
    """Exclusive binding:  S <-> 0 <-> N,  no co-bound state.

    State order [0, S, N].  A bound factor must leave before the other can
    bind, so there is no S <-> N edge and the ON set is two states each
    with a single exit.

    Args:
        a_s: SOX2 binding rate.
        b_s: SOX2 unbinding rate.
        a_n: NANOG binding rate.
        b_n: NANOG unbinding rate.

    Returns:
        The three-state generator, in state order [0, S, N].
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


def excl_kinetics(a_s, b_s, a_n, b_n, ky=KY):
    """(tau_on, tau_off, f, b, CV_on) for the exclusive-binding model.

    An ON period is a single sojourn in whichever state was entered, so
        tau_on = phi_s/beta_s + phi_n/beta_n,  phi_j = alpha_j/(alpha_s+alpha_n)
    a mixture of two exponentials -- NOT 1/(beta_s + beta_n).

    Args:
        a_s: SOX2 binding rate.
        b_s: SOX2 unbinding rate.
        a_n: NANOG binding rate.
        b_n: NANOG unbinding rate.
        ky: Transcription rate in the active states.

    Returns:
        A tuple (tau_on, tau_off, f, b, CV_on).
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


TOL = 1e-9


def stats(P):
    y = np.arange(P.size)
    mean = P @ y
    var = P @ (y - mean)**2
    return mean, var/mean, np.sqrt(var)/mean


def pad(P, n):
    """Zero-extend to n grid points, which the certificate says is safe."""
    out = np.zeros(n)
    out[:P.size] = P
    return out


def both_models(rates, floor=0):
    """sFSP distributions for HD and M at one parameter point.

    Each model stops at whatever grid satisfies its own certificate, so the
    two can differ by a few counts; padding to a common length costs only
    what the certificate already bounds, and is what makes P_HD and P_M
    comparable pointwise.  ``floor`` only widens the display panels.

    Returns:
        The two distributions, the exact truncation-free Fano factors for
        validation, and the worse of the two convergence factors.

    Args:
        rates: The four switching rates.
        floor: Probability floor, which only widens the display.
    """
    Qh, Ah = hd_Q(*rates)
    Qm, Am = excl_Q(*rates)
    Ph, dh = st.stationary_pmf(Qh, Ah, KY, GAM, tol=TOL,
                               return_diagnostics=True)
    Pm, dm = st.stationary_pmf(Qm, Am, KY, GAM, tol=TOL,
                               return_diagnostics=True)
    n = max(Ph.size, Pm.size, floor)
    Fh = st.moments(Qh, Ah, KY, GAM)[2]
    Fm = st.moments(Qm, Am, KY, GAM)[2]
    return (pad(Ph, n), pad(Pm, n), (Fh, Fm),
            max(dh["convergence_factor"], dm["convergence_factor"]))


def make_figure(build, xvals, yvals, xlab, ylab, xref, yref,
                fixed_note, dist_pts, dist_title, outname):
    """build(x, y) -> (a_s, b_s, a_n, b_n) defines which plane is swept."""
    X, Y = np.meshgrid(xvals, yvals)
    F_hd = np.zeros_like(X); F_m = np.zeros_like(X)
    CV2_hd = np.zeros_like(X); CV2_m = np.zeros_like(X)
    TV = np.zeros_like(X)
    err = cert = 0.0

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Ph, Pm, (Fh, Fm), g = both_models(build(X[i, j], Y[i, j]))
            _, f1, c1 = stats(Ph)
            _, f2, c2 = stats(Pm)
            F_hd[i, j], CV2_hd[i, j] = f1, c1**2
            F_m[i, j], CV2_m[i, j] = f2, c2**2
            TV[i, j] = 0.5*np.abs(Ph - Pm).sum()
            err = max(err, abs(f1 - Fh)/Fh, abs(f2 - Fm)/Fm)
            cert = max(cert, g)

    dists = []
    for v in dist_pts:
        Ph, Pm, _, _ = both_models(build(v, v), floor=120)
        dists.append((v, Ph, Pm, 0.5*np.abs(Ph - Pm).sum()))

    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.linewidth": .7, "axes.edgecolor": "#3a3a3a",
                         "legend.frameon": False, "figure.dpi": 160})
    C_M, C_HD = "#c0562b", "#1f5f8b"
    fig, ax = plt.subplots(2, 3, figsize=(11.5, 6.6))

    def heat(a, Z, title, cmap, norm, cb_label, levels=None, note=True):
        """Draws one shaded panel with contours and a colour bar.

        Args:
            a: Axes to draw on.
            Z: Values to shade.
            title: Panel title.
        """
        im = a.pcolormesh(X, Y, Z, cmap=cmap, norm=norm,
                          shading="auto", rasterized=True)
        if levels is not None:
            levels = np.geomspace(Z.min(), Z.max(), num=5) 
            cs = a.contour(X, Y, Z, levels=levels, colors="w",
                           linewidths=1.5, alpha=0.6)
            a.clabel(cs, fmt="%g", fontsize=8, inline=True)
        a.set_xscale("log"); a.set_yscale("log")
        a.axvline(xref, color="w", lw=.5, ls=":", alpha=.5)
        a.axhline(yref, color="w", lw=.5, ls=":", alpha=.5)
        a.set_xlabel(xlab); a.set_ylabel(ylab)
        cb = fig.colorbar(im, ax=a, pad=.02, fraction=.046)
        cb.set_label(cb_label, fontsize=7)
        cb.ax.tick_params(labelsize=6)


    flo = min(F_hd.min(), F_m.min()); fhi = max(F_hd.max(), F_m.max())
    fnorm = LogNorm(vmin=flo, vmax=fhi)
    flev = 10.0**np.arange(np.floor(np.log10(flo)), np.ceil(np.log10(fhi))+1)
    heat(ax[0, 0], F_hd, "HD:  Fano factor", "magma", fnorm, r"$F$", flev)
    heat(ax[0, 1], F_m, f"{LBL_M}:  Fano factor", "magma", fnorm, r"$F$", flev)
    R = np.log2(F_hd/F_m)
    rmax = np.abs(R).max()
    heat(ax[0, 2], R, rf"$\log_2 (F_{{\rm HD}}/F_{{\rm {LBL_M}}})$", "RdBu_r",
         TwoSlopeNorm(vcenter=0.0, vmin=-rmax, vmax=rmax),
         r"$\log_2$ ratio", levels=[-1, 0, 1])

    # ---- row 3: CV^2 and distinguishability
    clo = min(CV2_hd.min(), CV2_m.min()); chi = max(CV2_hd.max(), CV2_m.max())
    cnorm = LogNorm(vmin=clo, vmax=chi)
    clev = 10.0**np.arange(np.floor(np.log10(clo)), np.ceil(np.log10(chi))+1)
    heat(ax[1, 0], CV2_hd, "HD:  $CV^2$", "magma", cnorm, r"$CV^2$", clev)
    heat(ax[1, 1], CV2_m, f"{LBL_M}:  $CV^2$", "magma", cnorm, r"$CV^2$", clev)
    R = np.log2(CV2_hd/CV2_m)
    rmax = np.abs(R).max()
    heat(ax[1, 2], R, rf"$\log_2 (F_{{\rm HD}}/F_{{\rm {LBL_M}}})$", "RdBu_r",
         TwoSlopeNorm(vcenter=0.0, vmin=-rmax, vmax=rmax),
         r"$\log_2$ ratio", levels=[-1, 0, 1])
    """
        heat(ax[2, 2], TV,
            rf"total variation $d_{{\rm TV}}(P_{{\rm HD}}, P_{{\rm {LBL_M}}})$",
            "RdBu_r", None, r"$d_{\rm TV}$", levels=[0.02, 0.05, 0.1, 0.25])
    """
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
    print(f"  wrote {out}.svg")
    return TV, R

fx = lambda **kw: ", ".join(rf"${k}={v:g}$" for k, v in kw.items())

al = np.logspace(-2.5,2.5, NGRID)
fx_a = fx(**{r"k_{off,s}": BS0, r"k_{off,n}": BN0})
make_figure(lambda x, y: (x, BS0, y, BN0), al, al,
            rf"$k_{{on,s}}$ {RATE}", rf"$k_{{on,n}}$ {RATE}", BS0, BN0, fx_a,
            [0.01, 0.1, 1.0], "",
            "fig_4_noise")
