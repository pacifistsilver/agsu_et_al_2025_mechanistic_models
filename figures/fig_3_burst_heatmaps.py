"""Burst frequency and size over the (k_on_s, k_on_n) plane, HD against M."""

import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Patch
from stochtf.plotting import use_paper_style
from matplotlib.ticker import LogLocator
use_paper_style(sans_serif="Arial")

BS, BN = 0.05, 0.20            # unbinding rates (s^-1), held fixed
GAM, KY = 1.0, 20.0
NGRID = 220
RATE = r"(M$^{-1}$s$^{-1}$)"
B_LO, B_HI = KY/BN, KY/BS      # M's burst-size bounds
fx = lambda **kw: ", ".join(rf"${k}={v:g}$" for k, v in kw.items())


# --------------------------------------------------------------- kinetics
def m_kin(a_s, a_n):
    """(tau_on, tau_off, f, b) for the exclusive three-state model."""
    tot = a_s + a_n
    ph = a_s/tot
    tau_on = ph/BS + (1.0 - ph)/BN
    tau_off = 1.0/tot
    return tau_on, tau_off, 1.0/(tau_on + tau_off), KY*tau_on


def hd_kin(a_s, a_n):
    """(tau_on, tau_off, f, b) for the four-state OR model.

    tau_on by renewal-reward: the ON:OFF time ratio equals the stationary
    probability ratio, and P_00 = q_s q_n exactly.

    Args:
        a_s: SOX2 binding rate.
        a_n: NANOG binding rate.

    Returns:
        A tuple (tau_on, tau_off, f, b).
    """
    q_s = BS/(a_s + BS)
    q_n = BN/(a_n + BN)
    p00 = q_s*q_n
    tau_off = 1.0/(a_s + a_n)
    tau_on = tau_off*(1.0 - p00)/p00
    return tau_on, tau_off, 1.0/(tau_on + tau_off), KY*tau_on


# ------------------------------------------------------------------- grid
al = np.logspace(-2.5, 2.5, NGRID)
AS, AN = np.meshgrid(al, al)
TON_M, TOFF, F_M, B_M = m_kin(AS, AN)
TON_H, _, F_H, B_H = hd_kin(AS, AN)
MU_M, MU_H = B_M*F_M/GAM, B_H*F_H/GAM


# ----------------------------------------------------------------- figure
XL = r"$k_{on,s}$  " + RATE
YL = r"$k_{on,n}$  " + RATE
C_M, C_HD = "#c0562b", "#1f5f8b"

fig, ax = plt.subplots(2, 2, figsize=(8.4, 7.2))


def tag(a, t):
    a.text(-0.24, 1.10, t, transform=a.transAxes, fontsize=11,
           fontweight="bold", va="top")


def heat(a, Z, title, cmap, norm, cb_label, levels=None, lev_c="w",
         lev_fmt="%g", note=None):
    """Draws one shaded panel with contours and a colour bar.

    Args:
        a: Axes to draw on.
        Z: Values to shade.
        title: Panel title.
    """
    im = a.pcolormesh(AS, AN, Z, cmap=cmap, norm=norm, shading="auto",
                      rasterized=True)
    if levels is None:
        levels = np.geomspace(Z.min(), Z.max(), num=12) 
        
    if levels is not None:
        inside = [L for L in levels if Z.min() < L < Z.max()]
        if inside:
            cs = a.contour(AS, AN, Z, levels=inside, colors=lev_c,
                       linewidths=1.5, alpha=0.6)
            a.clabel(cs, fmt=lev_fmt, fontsize=8, inline=False, inline_spacing=0)
    a.plot([al[0], al[-1]], [al[0], al[-1]], color="w", lw=.6, ls="--",
           alpha=.5)
    a.axvline(BS, color="w", lw=.5, ls=":", alpha=.4)
    a.axhline(BN, color="w", lw=.5, ls=":", alpha=.4)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(XL, fontsize=7.5); a.set_ylabel(YL, fontsize=7.5)
    
    cb = fig.colorbar(im, ax=a, pad=.02, fraction=.046)
    cb.set_label(cb_label, fontsize=7)
    cb.ax.tick_params(labelsize=6)


# ---- row 1: burst frequency -------------------------------------------
flo = min(F_M.min(), F_H.min()); fhi = max(F_M.max(), F_H.max())
fnorm = LogNorm(vmin=flo, vmax=fhi)
FLEV = [0.0001, 0.001,  0.01, 0.045, 0.1]
heat(ax[0, 0], F_M, r"M:  burst frequency $f$", "viridis", fnorm,
     r"Burst Frequency", FLEV,
     note=fx(**{r"k_{off,s}": BS, r"k_{off,n}": BN}))
heat(ax[0, 1], F_H, r"HD:  burst frequency $f$", "viridis", fnorm,
     r"Burst Frequency", FLEV,
     note=fx(**{r"k_{off,s}": BS, r"k_{off,n}": BN}))
# ---- row 2: burst size -------------------------------------------------
blo = min(B_M.min(), B_H.min()); bhi = max(B_M.max(), B_H.max())
bnorm = LogNorm(vmin=blo, vmax=bhi)
FLEV = [150, 250, 500, 750, 1000, 5000, 10000]
heat(ax[1, 0], B_M, r"M:  burst size $b=k_y\tau_{\rm on}$", "magma", bnorm,
     "Burst Size", FLEV,
     note=fx(**{r"k_{off,s}": BS, r"k_{off,n}": BN}))
heat(ax[1, 1], B_H, r"HD:  burst size $b=k_y\tau_{\rm on}$", "magma", bnorm,
     "Burst Size", FLEV,
     note=fx(**{r"k_{off,s}": BS, r"k_{off,n}": BN}))

fig.tight_layout(w_pad=2.4, h_pad=2.9)
os.makedirs("./figures/output", exist_ok=True)
out = "./figures/output/fig_3_burst_kinetics_heatmap"
fig.savefig(out + ".svg", bbox_inches="tight", facecolor="w")
plt.close(fig)

print(f"f    M : {F_M.min():.3e} - {F_M.max():.3e} s^-1")
print(f"f   HD : {F_H.min():.3e} - {F_H.max():.3e} s^-1")
print(f"b    M : {B_M.min():9.3f} - {B_M.max():11.3f}   "
      f"bounds [{B_LO:g}, {B_HI:g}]  BOUNDED")
print(f"b   HD : {B_H.min():9.3f} - {B_H.max():11.3e}   unbounded")
print(f"max f*b: M {(F_M*B_M).max():.6f}, HD {(F_H*B_H).max():.6f} "
      f"(both < k_y = {KY:g})")
print(f"<y>  M : {MU_M.min():.3e} - {MU_M.max():.3e}")
print(f"<y> HD : {MU_H.min():.3e} - {MU_H.max():.3e}")
print(f"wrote {out}.svg / .png")