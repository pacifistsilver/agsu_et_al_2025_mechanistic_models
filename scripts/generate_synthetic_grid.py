"""Simulate heterodimer counts across the binding-rate grid used by fig_5.

``figures/fig_5_distributions`` sweeps the two on-rates over
``logspace(-2.5, 2.5, 50)`` at fixed off-rates and compares the two promoters'
*exact* distributions on that grid. This script produces the sampled
counterpart: a synthetic cell population simulated at every one of those grid
points, so anything the CME predicts there can be checked against data of the
size an experiment would actually deliver.

The simulator
-------------
``stochtf.inference.models.simulate_counts`` is used, which is a Gillespie
simulation of the promoter jump chain with the mRNA layer integrated out --
conditional on the promoter path, the stationary count is Poisson with mean
k_y * integral act(sigma(s)) exp(-gamma s) ds, so the transcription and
degradation events need not be simulated one by one. Each cell is an
independent draw from the exact stationary law: there is no burn-in to choose
and no correlation between cells, which is what the earlier
``fast_ssa_dimer`` route got wrong (it recorded ten correlated points along a
single not-yet-stationary trajectory).

Output
------
``data/synthetic/heterodimer_grid.npz`` holding

    counts        (ngrid, ngrid, n_cells) int32, indexed [a_n, a_s, cell]
                  to match fig_5's meshgrid orientation
    alpha         the shared on-rate axis
    mean_exact    stationary mean at each grid point, from sFSP
    fano_exact    stationary Fano factor at each grid point, from sFSP

Usage
-----
    python scripts/generate_synthetic_grid.py
    python scripts/generate_synthetic_grid.py --n-cells 2000 --ngrid 50
"""

import argparse
import os
import time

import numpy as np

from stochtf.cme import stationary as cme_stationary
from stochtf.inference.models import chain_generator, simulate_counts
from stochtf.paths import SYNTHETIC_DATA_DIR, synthetic

#: Fixed rates, matching figures/fig_5_distributions.
BS0, BN0 = 0.05, 0.20
GAM, KY = 1.0, 20.0

#: The on-rate axis swept by fig_5, shared by both sites.
GRID_LO, GRID_HI = -2.5, 2.5


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ngrid", type=int, default=50,
                    help="points per axis (default 50, as in fig_5)")
    ap.add_argument("--n-cells", type=int, default=1000,
                    help="cells simulated per grid point (default 1000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output .npz (default data/synthetic/heterodimer_grid.npz)")
    args = ap.parse_args()

    alpha = np.logspace(GRID_LO, GRID_HI, args.ngrid)
    rng = np.random.default_rng(args.seed)

    counts = np.zeros((args.ngrid, args.ngrid, args.n_cells), dtype=np.int32)
    mean_exact = np.zeros((args.ngrid, args.ngrid))
    fano_exact = np.zeros((args.ngrid, args.ngrid))

    print(f"heterodimer, {args.ngrid}x{args.ngrid} grid of "
          f"(alpha_s, alpha_n) over 1e{GRID_LO:g}..1e{GRID_HI:g}")
    print(f"beta_s={BS0}, beta_n={BN0}, k_y={KY}, gamma={GAM}; "
          f"{args.n_cells} cells per point")

    start = time.perf_counter()
    for i, a_n in enumerate(alpha):          # rows, as in fig_5's meshgrid
        for j, a_s in enumerate(alpha):      # columns
            counts[i, j] = simulate_counts(rng, a_s, a_n, BS0, BN0, KY,
                                           promoter="heterodimer",
                                           size=args.n_cells, gamma=GAM)
            Q, act = chain_generator("heterodimer", a_s, BS0, a_n, BN0)
            mean_exact[i, j], _, fano_exact[i, j] = cme_stationary.moments(
                Q, act, KY, GAM)
        if (i + 1) % 10 == 0:
            print(f"  row {i + 1}/{args.ngrid}  "
                  f"({time.perf_counter() - start:.0f}s)")

    elapsed = time.perf_counter() - start

    # The simulator should reproduce the CME it was never told about.
    sample_mean = counts.mean(axis=2)
    sample_var = counts.var(axis=2)
    sample_fano = np.divide(sample_var, sample_mean,
                            out=np.zeros_like(sample_var),
                            where=sample_mean > 0)
    live = mean_exact > 1e-6
    z = ((sample_mean[live] - mean_exact[live])
         / np.sqrt(sample_var[live] / args.n_cells + 1e-300))
    fano_rel = np.abs(sample_fano[live] - fano_exact[live]) / fano_exact[live]

    print(f"\nsimulated {args.ngrid ** 2 * args.n_cells:,} cells in "
          f"{elapsed:.0f}s")
    print(f"  mean vs CME: |z| median {np.median(np.abs(z)):.2f}, "
          f"95th {np.percentile(np.abs(z), 95):.2f}, max {np.abs(z).max():.2f}")
    print(f"  Fano vs CME: relative error median {np.median(fano_rel):.3f}, "
          f"95th {np.percentile(fano_rel, 95):.3f}")
    print(f"  counts: min {counts.min()}, max {counts.max()}, "
          f"mean {counts.mean():.2f}")

    os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)
    out = args.out or synthetic("heterodimer_grid.npz")
    np.savez_compressed(
        out, counts=counts, alpha=alpha, mean_exact=mean_exact,
        fano_exact=fano_exact,
        params=np.array([BS0, BN0, KY, GAM]),
        param_names=np.array(["beta_s", "beta_n", "k_y", "gamma"]),
        note=np.array("counts[i, j, :] has alpha_n=alpha[i], alpha_s=alpha[j]"))
    print(f"\nWrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
