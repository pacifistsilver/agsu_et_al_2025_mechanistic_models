import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Patch
from stochtf.plotting import use_paper_style

use_paper_style(sans_serif="Arial")

BS, BN = 0.05, 0.20            # unbinding rates (s^-1), held fixed
GAM, KY = 1.0, 20.0
NGRID = 220
RATE = r"(s$^{-1}$)"
B_LO, B_HI = KY/BN, KY/BS      # M's burst-size bounds


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

assert B_M.min() >= B_LO - 1e-9 and B_M.max() <= B_HI + 1e-9
assert (F_M*B_M).max() < KY and (F_H*B_H).max() <= KY + 1e-9


# ----------------------------------------------------------------- figure
XL = r"$k_{on,s}$  " + RATE
YL = r"$k_{on,n}$  " + RATE
C_M, C_HD = "#c0562b", "#1f5f8b"

fig, ax = plt.subplots(3, 3, figsize=(11.4, 10.2))


def tag(a, t):
    a.text(-0.24, 1.10, t, transform=a.transAxes, fontsize=11,
           fontweight="bold", va="top")


def heat(a, Z, title, cmap, norm, cb_label, levels=None, lev_c="w",
         lev_fmt="%g", note=None):
    im = a.pcolormesh(AS, AN, Z, cmap=cmap, norm=norm, shading="auto",
                      rasterized=True)
    if levels is not None:
        inside = [L for L in levels if Z.min() < L < Z.max()]
        if inside:
            cs = a.contour(AS, AN, Z, levels=inside, colors=lev_c,
                           linewidths=.7, alpha=.9)
            a.clabel(cs, fmt=lev_fmt, fontsize=5.8, inline=True)
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
FLEV = list(10.0**np.arange(np.floor(np.log10(flo)), np.ceil(np.log10(fhi))+1))
heat(ax[0, 0], F_M, r"M:  burst frequency $f$", "viridis", fnorm,
     r"Burst Frequency", FLEV,
     note=r"$f\to\beta$-limited as $\alpha\to\infty$")
heat(ax[0, 1], F_H, r"HD:  burst frequency $f$", "viridis", fnorm,
     r"Burst Frequency", FLEV,
     note=r"$f\to0$ as $\alpha\to\infty$: one endless burst")
Rf = np.log2(F_H/F_M); rf_ = np.abs(Rf).max()
heat(ax[0, 2], Rf, r"$\log_2(f_{\rm HD}/f_{\rm M})$", "RdBu_r",
     TwoSlopeNorm(vcenter=0.0, vmin=-rf_, vmax=rf_), r"$\log_2$ ratio",
     [-8, -4, -2, -1, 0], lev_c="0.25",
     note=r"HD $\leq$ M everywhere")

# ---- row 2: burst size -------------------------------------------------
blo = min(B_M.min(), B_H.min()); bhi = max(B_M.max(), B_H.max())
bnorm = LogNorm(vmin=blo, vmax=bhi)
heat(ax[1, 0], B_M, r"M:  burst size $b=k_y\tau_{\rm on}$", "magma", bnorm,
     "Burst Size", [6, 8, 10, 12, 14, 16, 18, 19.5],
     note=rf"bounded: ${B_LO:g}\leq b\leq{B_HI:g}$, contours are rays")
ax[1, 0].text(0.04, 0.95, rf"$b\in[{B_LO:g},\,{B_HI:g}]$ for" "\n"
              r"every $(\alpha_s,\alpha_n)$",
              transform=ax[1, 0].transAxes, fontsize=6.4, va="top", color="w",
              bbox=dict(fc="#c0392b", ec="none", alpha=.75, pad=2.0))
heat(ax[1, 1], B_H, r"HD:  burst size $b=k_y\tau_{\rm on}$", "magma", bnorm,
     "Burst Size", [20, 100, 1e3, 1e4, 1e5],
     note=r"unbounded: state 11 traps the ON set")
Rb = np.log2(B_H/B_M); rb_ = np.abs(Rb).max()
heat(ax[1, 2], Rb, r"$\log_2(b_{\rm HD}/b_{\rm M})$", "RdBu_r",
     TwoSlopeNorm(vcenter=0.0, vmin=-rb_, vmax=rb_), r"$\log_2$ ratio",
     [0, 1, 2, 4, 8, 12], lev_c="0.25",
     note=r"HD $\geq$ M everywhere")

# ---- (g) mean expression ratio ----------------------------------------
Rm = np.log2(MU_H/MU_M); rm_ = np.abs(Rm).max()
heat(ax[2, 0], Rm, r"$\log_2(\langle y\rangle_{\rm HD}/"
     r"\langle y\rangle_{\rm M})$", "RdBu_r",
     TwoSlopeNorm(vcenter=0.0, vmin=-rm_, vmax=rm_), r"$\log_2$ ratio",
     [0, 0.25, 0.5, 1.0], lev_c="0.25",
     note=r"$\langle y\rangle = bf/\gamma$: the $b$ and $f$"
          "\n" r"differences largely cancel")

# ---- (h) attainable region in the (f, b) plane ------------------------
a = ax[2, 1]
bb = np.logspace(np.log10(blo), np.log10(bhi), 400)
hyp = KY/bb                                     # f*b = k_y boundary
a.fill_betweenx(bb, 1e-6, hyp, color=C_HD, alpha=.22, lw=0)
strip = (bb >= B_LO) & (bb <= B_HI)
a.fill_betweenx(bb[strip], 1e-6, hyp[strip], color=C_M, alpha=.55, lw=0)
a.loglog(hyp, bb, color="k", lw=1.2)
a.axhline(B_LO, color=C_M, lw=.9, ls=":")
a.axhline(B_HI, color=C_M, lw=.9, ls=":")
a.text(2e-6, B_LO*1.15, rf"$b=k_y/\beta_n={B_LO:g}$", fontsize=6, color=C_M)
a.text(2e-6, B_HI*1.15, rf"$b=k_y/\beta_s={B_HI:g}$", fontsize=6, color=C_M)
a.text(0.30, 0.13, r"$fb=k_y$" "\n" r"$(\langle y\rangle=k_y/\gamma)$",
       transform=a.transAxes, fontsize=6.4, rotation=-38)
a.set_xlim(1e-6, 1.0); a.set_ylim(blo, bhi)
a.set_xlabel(r"Burst Frequency", fontsize=7.5)
a.set_ylabel(r"Burst Size", fontsize=7.5)
a.set_title(r"(f, b) attainable sets:  M $\subset$ HD", fontsize=8.5, pad=4)
a.legend(handles=[Patch(fc=C_M, alpha=.55, label="M and HD both reachable"),
                  Patch(fc=C_HD, alpha=.22, label="HD only")],
         fontsize=6.3, loc="upper right")

# ---- (i) 1-D cuts through the plane -----------------------------------
a = ax[2, 2]
for R, ls in [(1.0, "-"), (100.0, "--")]:
    an = al
    a_s = R*an
    _, _, fm, bm = m_kin(a_s, an)
    _, _, fh, bh = hd_kin(a_s, an)
    a.loglog(an, bm, color=C_M, lw=1.7, ls=ls,
             label=rf"M,  $R={R:g}$")
    a.loglog(an, bh, color=C_HD, lw=1.7, ls=ls,
             label=rf"HD, $R={R:g}$")
a.axhspan(B_LO, B_HI, color=C_M, alpha=.13, lw=0)
a.axhline(B_HI, color=C_M, lw=.8, ls=":")
a.set_xlabel(r"$\alpha_n$  " + RATE, fontsize=7.5)
a.set_ylabel(r"burst size $b$ (transcripts)", fontsize=7.5)
a.set_title(rf"cuts at $R=\alpha_s/\alpha_n$: M saturates at ${B_HI:g}$",
            fontsize=8.5, pad=4)
a.legend(fontsize=6.3, loc="upper left", ncol=2)

fig.text(0.5, -0.004,
         rf"$\beta_s={BS:g}$, $\beta_n={BN:g}$ s$^{{-1}}$, $k_y={KY:g}$ "
         rf"s$^{{-1}}$, $\gamma={GAM:g}$ s$^{{-1}}$;  "
         rf"dotted guides at $\alpha_j=\beta_j$;  dashed diagonal "
         rf"$\alpha_s=\alpha_n$;  $\tau_{{\rm off}}=1/(\alpha_s+\alpha_n)$ "
         rf"is identical in both models", ha="center", fontsize=6.6,
         color="0.35")
fig.tight_layout(w_pad=2.4, h_pad=2.9)
os.makedirs("./figures/output", exist_ok=True)
out = "./figures/output/fig_2_burst_f_b"
fig.savefig(out + ".svg", bbox_inches="tight", facecolor="w")
plt.close(fig)

# ------------------------------------------------------------ diagnostics
print(f"f    M : {F_M.min():.3e} - {F_M.max():.3e} s^-1")
print(f"f   HD : {F_H.min():.3e} - {F_H.max():.3e} s^-1")
print(f"b    M : {B_M.min():9.3f} - {B_M.max():11.3f}   "
      f"bounds [{B_LO:g}, {B_HI:g}]  BOUNDED")
print(f"b   HD : {B_H.min():9.3f} - {B_H.max():11.3e}   unbounded")
print(f"max f*b: M {(F_M*B_M).max():.6f}, HD {(F_H*B_H).max():.6f} "
      f"(both < k_y = {KY:g})")
print(f"<y>  M : {MU_M.min():.3e} - {MU_M.max():.3e}")
print(f"<y> HD : {MU_H.min():.3e} - {MU_H.max():.3e}")
print(f"log2(b_HD/b_M) max = {rb_:.2f}   log2(f_HD/f_M) max = {rf_:.2f}   "
      f"log2(<y> ratio) max = {rm_:.2f}")
print(f"wrote {out}.svg / .png")