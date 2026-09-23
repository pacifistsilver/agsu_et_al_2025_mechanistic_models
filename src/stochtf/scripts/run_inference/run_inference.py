"""Fit promoter models to count data by SMC, singly or across a parameter grid.

Usage
-----
    python -m stochtf.scripts.run_inference --gene esrrb --model heterodimer
    python -m stochtf.scripts.run_inference --gene sox2 --model joint
    python -m stochtf.scripts.run_inference --grid --stride 10 --draws 300
"""

import argparse
import os
import time

import numpy as np

from stochtf import paths
from stochtf.inference.models import MODELS, model_probabilities

GENES = ["sox2", "nanog", "rex1", "esrrb", "synthetic_heterodimer_data",
         "synthetic_monomer_data", "synthetic_telegraph_data"]

GRID_FILE = "heterodimer_grid.npz"


def load_counts(gene):
    """Per-cell counts for one gene: real ones live under data/processed,
    the ``synthetic_*`` ones under data/synthetic."""
    if gene.startswith("synthetic_"):
        return np.load(paths.synthetic(f"{gene}.npy"))
    return np.load(paths.processed(f"{gene}.npy"))


def fit_counts(counts, model_name, draws, chains, progressbar=False,
               model_config=None, sample_prior=True):
    """Fit one dataset and return the fitted ModelBuilder."""
    model = MODELS[model_name](model_config=model_config)
    sampler_config = dict(model.get_default_sampler_config())
    if draws is not None:
        sampler_config["draws"] = draws
    if chains is not None:
        sampler_config["chains"] = chains
    model.fit(counts, sampler_config=sampler_config, progressbar=progressbar,
              sample_prior=sample_prior)
    return model


def run_single(args):
    data = load_counts(args.gene)
    print(f"Fitting {args.model} to {args.gene}: {data.shape} observations "
          f"(mean {data.mean():.1f}, Fano {data.var() / data.mean():.1f})")

    model = fit_counts(data, args.model, args.draws, args.chains,
                       progressbar=True)

    if args.model == "joint":
        probs = model_probabilities(model.idata)
        print(f"\nP(monomer)     = {probs['monomer']:.4f}")
        print(f"P(heterodimer) = {probs['heterodimer']:.4f}")
        print(f"Bayes factor (heterodimer / monomer) = "
              f"{probs['bayes_factor_het_over_mono']:.3f}")

    out = args.out or paths.results(f"{args.gene}_{args.model}.nc")
    model.save(out)
    print(f"Wrote {out}")


def run_grid(args):
    """Fit the joint model at every grid point and return P(heterodimer)."""
    path = args.grid_file or paths.synthetic(GRID_FILE)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run "
                         "python -m stochtf.scripts.generate_synthetic_grid first.")
    store = np.load(path)
    counts, alpha = store["counts"], store["alpha"]
    n_grid = alpha.size

    beta_s, beta_n, k_y, gamma = (float(v) for v in store["params"])
    model_config = {**MODELS["joint"].get_default_model_config(),
                    "beta_s_fixed": beta_s, "beta_n_fixed": beta_n}

    rows = list(range(0, n_grid, args.stride))
    alpha_used = alpha[rows]
    n = len(rows)

    print(f"{path}\n  full grid {n_grid}x{n_grid}, stride {args.stride} "
          f"-> {n}x{n} = {n * n} joint fits, "
          f"{counts.shape[2]} cells each")
    print(f"  alpha from {alpha_used[0]:.4g} to {alpha_used[-1]:.4g}")
    print(f"  generating rates: beta_s={beta_s:g}, beta_n={beta_n:g}, "
          f"k_y={k_y:g}, gamma={gamma:g}  (betas pinned to match)")

    p_het = np.full((n, n), np.nan)
    bayes = np.full((n, n), np.nan)
    start = time.perf_counter()

    for a, i in enumerate(rows):          # alpha_n along rows, as generated
        for b, j in enumerate(rows):      # alpha_s along columns
            data = counts[i, j].astype(float)
            model = fit_counts(data, "joint", args.draws, args.chains,
                               model_config=dict(model_config),
                               sample_prior=False)
            probs = model_probabilities(model.idata)
            p_het[a, b] = probs["heterodimer"]
            bayes[a, b] = probs["bayes_factor_het_over_mono"]
        done = (a + 1) * n
        rate = (time.perf_counter() - start) / done
        print(f"  row {a + 1}/{n}  ({rate * done:.0f}s elapsed, "
              f"~{rate * (n * n - done):.0f}s left)  "
              f"P(het) this row: {np.nanmin(p_het[a]):.2f}-"
              f"{np.nanmax(p_het[a]):.2f}")

    elapsed = time.perf_counter() - start
    print(f"\n{n * n} fits in {elapsed:.0f}s")
    print(f"P(heterodimer): min {np.nanmin(p_het):.3f}  "
          f"median {np.nanmedian(p_het):.3f}  max {np.nanmax(p_het):.3f}")
    decisive = np.nansum(p_het > 0.95)
    print(f"  decisive for the true topology (>0.95): {decisive}/{n * n}")
    print(f"  near the 0.5 prior (0.4-0.6): "
          f"{np.nansum((p_het > 0.4) & (p_het < 0.6))}/{n * n}")

    print("\nP(heterodimer), rows = alpha_n, columns = alpha_s:")
    header = "".join(f"{v:>7.3g}" for v in alpha_used)
    print(f"{'a_n \\ a_s':>10}{header}")
    for a, value in enumerate(alpha_used):
        cells = "".join(f"{v:>7.2f}" for v in p_het[a])
        print(f"{value:>10.3g}{cells}")

    out = args.out or paths.results("joint_grid_pheterodimer.npz")
    np.savez_compressed(out, p_heterodimer=p_het, bayes_factor=bayes,
                        alpha=alpha_used, stride=args.stride,
                        n_cells=counts.shape[2], draws=args.draws,
                        note=np.array("p_heterodimer[a, b] has "
                                      "alpha_n=alpha[a], alpha_s=alpha[b]"))
    print(f"\nWrote {out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene", choices=GENES, default="esrrb",
                    help="gene whose counts to fit (default: esrrb)")
    ap.add_argument("--model", choices=sorted(MODELS), default="heterodimer",
                    help="promoter model, or 'joint' to fit both at once "
                         "(default: heterodimer)")
    ap.add_argument("--grid", action="store_true",
                    help="sweep the joint fit over the synthetic grid instead")
    ap.add_argument("--grid-file", default=None,
                    help=f"grid .npz (default: data/synthetic/{GRID_FILE})")
    ap.add_argument("--stride", type=int, default=10,
                    help="thin the grid by this factor (default 10)")
    ap.add_argument("--draws", type=int, default=None,
                    help="SMC draws; defaults to the model's sampler config")
    ap.add_argument("--chains", type=int, default=None,
                    help="SMC chains; defaults to the model's sampler config")
    ap.add_argument("--out", default=None, help="output path")
    args = ap.parse_args()

    if args.grid:
        run_grid(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
