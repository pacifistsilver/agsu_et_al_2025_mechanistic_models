"""Where the fitted genes sit in the burst plane.

Each gene in the panel was fitted independently by ``scripts/run_gene_panel.py``
with the topology as a sampled parameter. That fit is summarised in
``results/panel_joint_fits.json`` as a posterior median and a 94% interval for
each free rate; this figure takes the *median* of each -- the one point estimate
the checkpoint carries -- and maps the three rates onto the two numbers that
describe a promoter's output: how often it fires, and how much it makes when it
does.

    f = 1 / (tau_on + tau_off)      bursts per mRNA lifetime
    b = k_y tau_on                  molecules per burst

computed in closed form by :func:`stochtf.analytical.bursts.rate_burst_stats`
under whichever topology the fit favours, since tau_on differs between them.
Both are in units of gamma, as is everything the stationary counts identify.

The dashed diagonals are lines of constant b f / gamma, which for these
promoters is *exactly* the model's mean count -- not the bursty-limit
approximation to it -- so the diagonal a gene sits on reads off the mean its
fitted rates predict. Panel (b) puts that prediction against the mean actually
observed. They are not the same, and systematically so: plugging in the median
of each marginal is not the same as taking a point in the joint posterior, and
for the high-Fano genes, whose k_y posteriors run long-tailed, the plug-in
overshoots. Read the plane as where the fits place each gene relative to the
others, and panel (b) as what that point estimate costs.

Usage
-----
    python scripts/run_gene_panel.py     # produce the fits
    python figures/fig_11_panel_burst_plane.py
"""

import argparse
import json
import os

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from stochtf import constants, paths
from stochtf.analytical.bursts import rate_burst_stats
from stochtf.plotting import output_path, use_paper_style

#: Index of the point estimate inside each [3%, median, 97%] triple.
MEDIAN = 1

#: Constant-mean diagonals, b f / gamma = level.
LEVELS = [1, 3, 10, 30, 100, 300, 1000]

#: Marker per favoured topology.
MARKERS = {"monomer": "o", "heterodimer": "^"}


def burst_plane(records):
    """Point estimates mapped to (gene, topology, f, b, observed mean, Fano)."""
    rows = []
    for r in records:
        if r.get("status") != "fitted":
            continue
        favoured = "heterodimer" if r["p_heterodimer"] >= 0.5 else "monomer"
        _, f, b = rate_burst_stats(favoured,
                                   r["alpha_s"][MEDIAN], r["alpha_n"][MEDIAN],
                                   r["k_y"][MEDIAN],
                                   constants.BETA_S, constants.BETA_N)
        rows.append((r["gene"], favoured, f, b, r["mean"], r["fano"]))
    return rows


def label_set(rows, n_top=12):
    """Genes worth naming: the most expressed, plus the four extremes."""
    by_mean = sorted(rows, key=lambda t: -t[4])[:n_top]
    extremes = [min(rows, key=lambda t: t[2]), max(rows, key=lambda t: t[2]),
                min(rows, key=lambda t: t[3]), max(rows, key=lambda t: t[3])]
    return {t[0] for t in by_mean + extremes}


def label_offset(i, f, b, k=6):
    """Point the label for gene ``i`` away from where its neighbours are.

    The panel is crowded in the middle, so a fixed offset stacks names on top
    of each other. Distances are taken in log space, which is the space the
    axes are drawn in.
    """
    x, y = np.log10(f), np.log10(b)
    d = np.hypot(x - x[i], y - y[i])
    near = np.argsort(d)[1:k + 1]
    away_x = x[i] - x[near].mean()
    away_y = y[i] - y[near].mean()
    norm = np.hypot(away_x, away_y)
    if norm == 0:
        return (5, 3), "left"
    away_x, away_y = away_x / norm, away_y / norm
    return (7 * away_x, 7 * away_y), ("left" if away_x >= 0 else "right")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", default=paths.results("panel_joint_fits.json"))
    ap.add_argument("--label-top", type=int, default=12,
                    help="name this many of the most expressed genes")
    args = ap.parse_args()

    if not os.path.exists(args.fits):
        raise SystemExit(f"{args.fits} not found. Run "
                         "scripts/run_gene_panel.py first.")
    with open(args.fits, encoding="utf-8") as fh:
        rows = burst_plane(json.load(fh))
    if not rows:
        raise SystemExit(f"{args.fits} has no completed fits yet")

    f = np.array([t[2] for t in rows])
    b = np.array([t[3] for t in rows])
    observed = np.array([t[4] for t in rows])
    fano = np.array([t[5] for t in rows])
    predicted = b * f                      # <y> = b f / gamma, with gamma = 1

    use_paper_style()
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.4),
                           gridspec_kw={"width_ratios": [1.6, 1]})

    # ---- (a) the burst plane ------------------------------------------
    a = ax[0]
    norm = LogNorm(observed.min(), observed.max())
    grid = np.logspace(np.log10(f.min()) - 0.35, np.log10(f.max()) + 0.35, 64)
    xlim = (grid[0], grid[-1])
    ylim = (b.min() / 2.2, b.max() * 2.6)
    for level in LEVELS:
        a.plot(grid, level / grid, color="0.78", lw=0.8, ls="--", zorder=0)
        # Label where the line leaves the axes: the right edge for the low
        # levels, the top edge for the ones that run off the top first.
        if ylim[0] < level / xlim[1] < ylim[1]:
            pos, ha, va = (xlim[1], level / xlim[1]), "right", "bottom"
        else:
            pos, ha, va = (level / ylim[1], ylim[1]), "left", "top"
        a.annotate(f"{level:g}", pos, textcoords="offset points",
                   xytext=(-3 if ha == "right" else 3, -2), fontsize=6,
                   color="0.5", ha=ha, va=va, zorder=1)

    for topology, marker in MARKERS.items():
        keep = [i for i, t in enumerate(rows) if t[1] == topology]
        if not keep:
            continue
        a.scatter(f[keep], b[keep], c=observed[keep], norm=norm, cmap="viridis",
                  marker=marker, s=44, lw=0.5, edgecolor="k", zorder=3)
        a.scatter([], [], marker=marker, s=44, lw=0.5, edgecolor="k",
                  facecolor="0.85", label=f"{topology} ({len(keep)})")

    named = label_set(rows, args.label_top)
    for i, (gene, _, fi, bi, _, _) in enumerate(rows):
        if gene in named:
            (dx, dy), ha = label_offset(i, f, b)
            a.annotate(gene, (fi, bi), textcoords="offset points",
                       xytext=(dx, dy), fontsize=6, ha=ha, va="center",
                       zorder=4)

    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(*xlim)
    a.set_ylim(*ylim)
    a.set_xlabel(r"burst frequency  $f/\gamma$")
    a.set_ylabel(r"mean burst size  $b$")
    a.set_title(f"(a) {len(rows)} genes at the posterior median")
    a.legend(fontsize=7, loc="lower left")
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=a,
                      label="observed mean count")
    cb.ax.tick_params(labelsize=7)
    a.text(0.985, 0.97,
           r"dashed: constant $bf/\gamma = \langle y\rangle$",
           transform=a.transAxes, fontsize=6.5, color="0.4",
           ha="right", va="top")

    # ---- (b) what that point estimate predicts ------------------------
    a = ax[1]
    lim = [min(observed.min(), predicted.min()) / 1.7,
           max(observed.max(), predicted.max()) * 1.7]
    a.plot(lim, lim, color="0.6", lw=1.0, ls="--", zorder=1)
    sc = a.scatter(observed, predicted, c=fano, norm=LogNorm(), cmap="magma",
                   s=40, lw=0.5, edgecolor="k", zorder=3)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.set_xlabel("observed mean count")
    a.set_ylabel(r"predicted  $bf/\gamma$")
    a.set_title("(b) plug-in estimate against the data")
    cb = fig.colorbar(sc, ax=a, label="observed Fano factor")
    cb.ax.tick_params(labelsize=7)

    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(output_path(f"fig_11_panel_burst_plane.{ext}"),
                    bbox_inches="tight", facecolor="w")

    ratio = predicted / observed
    print(f"{len(rows)} genes, off-rates pinned at "
          f"beta_s={constants.BETA_S:g}, "
          f"beta_n={constants.BETA_N:g}; rates in units of gamma\n")
    print(f"burst frequency f: {f.min():.4f} to {f.max():.4f} per lifetime")
    print(f"burst size      b: {b.min():.1f} to {b.max():.1f} molecules")
    print(f"predicted/observed mean: min {ratio.min():.2f}  "
          f"median {np.median(ratio):.2f}  max {ratio.max():.2f}  "
          f"({int(((ratio > 0.8) & (ratio < 1.2)).sum())}/{len(rows)} "
          f"within 20%)")
    worst = sorted(zip((t[0] for t in rows), ratio), key=lambda p: -p[1])[:5]
    print("furthest off: " + ", ".join(f"{g} ({x:.1f}x)" for g, x in worst))
    print(f"\nWrote {output_path('fig_11_panel_burst_plane.svg')}")


if __name__ == "__main__":
    main()
