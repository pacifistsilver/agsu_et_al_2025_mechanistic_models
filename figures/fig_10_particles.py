"""Accepted SMC particles for each promoter, fitted separately.

Usage
-----
    python figures/fig_10_particles.py
    python figures/fig_10_particles.py --point 5 0
    python figures/fig_10_particles.py --gene Klf4
"""

import argparse

import arviz as az
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from stochtf import paths
from stochtf.inference.models import MODELS
from stochtf.plotting import output_path, use_paper_style

RATES = ["alpha_s", "alpha_n", "k_y"]
LABELS = {"alpha_s": r"$k_{on,s}$", "alpha_n": r"$k_{on,n}$", "k_y": r"$k_y$"}
PROMOTERS = ["monomer", "heterodimer"]

#: Reported interval width.
CI_PROB = 0.95

#: Grid points used for the smooth densities.
RESOLUTION = 140


def fit_grid_cell(name, row, col, draws, chains):
    """Fit one cell of the synthetic sweep, pinned as the sweep pinned it."""
    source = np.load(paths.synthetic("heterodimer_grid.npz"), allow_pickle=False)
    counts, alpha = source["counts"], source["alpha"]
    beta_s, beta_n, _, _ = (float(v) for v in source["params"])
    config = {**MODELS[name].get_default_model_config(),
              "beta_s_fixed": beta_s, "beta_n_fixed": beta_n}
    model = MODELS[name](model_config=config)
    model.fit(counts[row, col].astype(float),
              sampler_config={"draws": draws, "chains": chains},
              progressbar=False, sample_prior=False)
    truth = {"alpha_s": float(alpha[col]), "alpha_n": float(alpha[row])}
    return model.idata, truth


def fit_gene(name, gene, draws, chains):
    """Fit one gene from the extracted panel."""
    panel = np.load(paths.processed("ochiai_panel_counts.npz"),
                    allow_pickle=False)
    where = np.flatnonzero(panel["genes"] == gene)
    if where.size == 0:
        raise SystemExit(f"{gene} is not in the panel")
    row = int(where[0])
    counts = np.concatenate([panel["counts_129"][row],
                             panel["counts_cast"][row]]).astype(float)
    model = MODELS[name]()
    model.fit(counts, sampler_config={"draws": draws, "chains": chains},
              progressbar=False, sample_prior=False)
    return model.idata, {}


def summarise(idata):
    """Posterior median and credible interval per rate, via arviz.

    ``round_to="auto"`` would collapse a narrow interval such as k_y's onto a
    single repeated value, so full precision is asked for.

    Args:
        idata: ``InferenceData`` from a completed fit.

    Returns:
        One (median, lower, upper) triple per rate.
    """
    stats = az.summary(idata, var_names=RATES, kind="stats", ci_prob=CI_PROB,
                       ci_kind="hdi", round_to=8)
    low = next(c for c in stats.columns if c.endswith("_lb"))
    high = next(c for c in stats.columns if c.endswith("_ub"))
    draws = {n: np.asarray(idata["posterior"][n]).ravel() for n in RATES}
    estimate = {n: float(np.median(draws[n])) for n in RATES}
    interval = {n: (float(stats.loc[n, low]), float(stats.loc[n, high]))
                for n in RATES}
    return draws, estimate, interval


def draw_triangle(draws, estimate, interval, truth, limits, title):
    """Lower triangle: marginals on the diagonal, smooth density below."""
    n = len(RATES)
    fig, axes = plt.subplots(n, n, figsize=(7.6, 7.0))

    for r, y_name in enumerate(RATES):
        for c, x_name in enumerate(RATES):
            ax = axes[r, c]
            if r < c:                       # upper triangle is unused
                ax.axis("off")
                continue

            if r == c:
                support = np.linspace(*limits[x_name], RESOLUTION)
                density = gaussian_kde(draws[x_name])(support)
                ax.plot(support, density, color="#2a6f97", lw=1.4)
                ax.fill_between(support, density, color="#2a6f97", alpha=.25)

                low, high = interval[x_name]
                top = float(density.max())
                ax.plot([low, high], [top * .06] * 2, color="k", lw=2.4,
                        solid_capstyle="butt", zorder=6)
                ax.plot([estimate[x_name]], [top * .06], "o", ms=4.5,
                        color="k", zorder=7)
                if x_name in truth:
                    ax.axvline(truth[x_name], color="#d62728", lw=1.1,
                               ls="--", zorder=5)
                ax.set_ylim(0, top * 1.18)
                ax.set_yticks([])
                ax.text(0.5, 1.02,
                        f"{estimate[x_name]:.3g}  [{low:.3g}, {high:.3g}]",
                        transform=ax.transAxes, ha="center", fontsize=6.5)
            else:
                xg = np.linspace(*limits[x_name], RESOLUTION)
                yg = np.linspace(*limits[y_name], RESOLUTION)
                mesh_x, mesh_y = np.meshgrid(xg, yg)
                kde = gaussian_kde(np.vstack([draws[x_name], draws[y_name]]))
                surface = kde(np.vstack([mesh_x.ravel(),
                                         mesh_y.ravel()])).reshape(mesh_x.shape)
                ax.contourf(mesh_x, mesh_y, surface, levels=14, cmap="viridis")
                ax.contour(mesh_x, mesh_y, surface, levels=14, colors="w",
                           linewidths=.25, alpha=.5)
                if x_name in truth and y_name in truth:
                    ax.plot(truth[x_name], truth[y_name], marker="*", ms=16,
                            mfc="#ffd700", mec="k", mew=.9, ls="none",
                            zorder=8)
                ax.set_ylim(*limits[y_name])

            ax.set_xlim(*limits[x_name])
            if r == n - 1:
                ax.set_xlabel(LABELS[x_name])
            if c == 0 and r != 0:
                ax.set_ylabel(LABELS[y_name])
            ax.tick_params(labelsize=6.5)

    handles = [plt.Line2D([], [], marker="*", ls="none", ms=13, mfc="#ffd700",
                          mec="k", label="generating value"),
               plt.Line2D([], [], color="k", lw=2.4,
                          label=f"median and {CI_PROB:.0%} HDI")]
    axes[0, -1].legend(handles=handles, fontsize=7, loc="center",
                       frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def main():
    """Builds the particle-grid figure and writes it out."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--point", nargs=2, type=int, default=None,
                    metavar=("ROW", "COL"))
    ap.add_argument("--gene", default=None)
    ap.add_argument("--draws", type=int, default=800)
    ap.add_argument("--chains", type=int, default=2)
    args = ap.parse_args()

    point = None
    if args.gene is None:
        if args.point is None:
            matrix = np.load(paths.results("joint_grid_pheterodimer.npz"),
                             allow_pickle=False)
            bayes, stride = matrix["bayes_factor"], int(matrix["stride"])
            a, b = np.unravel_index(np.nanargmax(bayes), bayes.shape)
            rows = list(range(0, 50, stride))
            point = (rows[int(a)], rows[int(b)])
        else:
            point = tuple(args.point)

    use_paper_style(sans_serif="Arial")

    fits = {}
    for name in PROMOTERS:
        if args.gene:
            idata, truth = fit_gene(name, args.gene, args.draws, args.chains)
            label = args.gene
        else:
            idata, truth = fit_grid_cell(name, *point, args.draws, args.chains)
            label = f"grid cell {point}"
        fits[name] = (summarise(idata), truth)

    # Shared limits, so the two figures are directly comparable.
    limits = {}
    for rate in RATES:
        pooled = np.concatenate([fits[n][0][0][rate] for n in PROMOTERS])
        span = pooled.max() - pooled.min()
        limits[rate] = (pooled.min() - .04 * span, pooled.max() + .04 * span)

    stem = f"fig_10_particles_{(args.gene or 'grid').lower()}"
    for name in PROMOTERS:
        (draws, estimate, interval), truth = fits[name]
        fig = draw_triangle(draws, estimate, interval, truth, limits,
                            f"{name} fit - {label}")
        for suffix in ("svg", "png"):
            fig.savefig(output_path(f"{stem}_{name}.{suffix}"),
                        bbox_inches="tight", facecolor="w")
        plt.close(fig)

        print(f"{name}: {draws[RATES[0]].size} accepted particles")
        for rate in RATES:
            low, high = interval[rate]
            line = (f"  {rate:>8}  median {estimate[rate]:9.4f}  "
                    f"{CI_PROB:.0%} HDI [{low:9.4f}, {high:9.4f}]")
            if rate in truth:
                line += f"   generating {truth[rate]:.4g}"
            print(line)
        print(f"  wrote {output_path(stem + '_' + name + '.png')}")


if __name__ == "__main__":
    main()
