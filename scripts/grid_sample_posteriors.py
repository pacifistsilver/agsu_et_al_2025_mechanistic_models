"""Refit a few grid points and keep the full posterior draws.

``run_inference.py --grid`` stores only the summary matrices, which is what the
maps in ``fig_9_model_selection`` need but not enough to show what a posterior
at any single point actually looks like. This refits a handful of representative
points -- the strongest evidence each way, the most undecided, and a typical one
-- and saves their draws so the figure can show the marginals against the values
that generated the data.

The refits use the same configuration as the sweep, including off-rates pinned
to the ones the synthetic data was generated with, so the posteriors are
comparable to the matrix rather than to a differently specified model.

Usage
-----
    python scripts/grid_sample_posteriors.py
    python scripts/grid_sample_posteriors.py --draws 600 --point 0 5 --point 5 0
"""

import argparse
import os

import numpy as np

from stochtf import paths
from stochtf.inference.models import MODELS, model_probabilities

TRACKED = ("alpha_s", "alpha_n", "k_y", "model_index")


def choose_points(bayes):
    """(row, column, label) for the points worth showing."""
    finite = np.isfinite(bayes)
    flat = np.argsort(np.where(finite, bayes, np.nan), axis=None)
    flat = [k for k in flat if finite.ravel()[k]]
    undecided = int(np.nanargmin(np.abs(np.log(np.where(finite, bayes,
                                                        np.nan))).ravel()))
    picks = [(flat[-1], "strongest for heterodimer"),
             (flat[len(flat) // 2], "typical"),
             (undecided, "most undecided"),
             (flat[0], "strongest for monomer")]

    seen, out = set(), []
    for k, label in picks:
        if k in seen:
            continue
        seen.add(k)
        a, b = np.unravel_index(k, bayes.shape)
        out.append((int(a), int(b), label))
    return out


def main():
    """Parses arguments and refits the selected grid points."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=paths.synthetic("heterodimer_grid.npz"))
    ap.add_argument("--matrix",
                    default=paths.results("joint_grid_pheterodimer.npz"))
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--point", nargs=2, type=int, action="append", default=[],
                    metavar=("ROW", "COL"),
                    help="matrix cell to sample; repeatable, overrides the "
                         "automatic choice")
    ap.add_argument("--out", default=paths.results("grid_sample_posteriors.npz"))
    args = ap.parse_args()

    source = np.load(args.grid, allow_pickle=False)
    counts, alpha_full = source["counts"], source["alpha"]
    beta_s, beta_n, k_y_true, gamma = (float(v) for v in source["params"])

    matrix = np.load(args.matrix, allow_pickle=False)
    bayes, alpha = matrix["bayes_factor"], matrix["alpha"]
    stride = int(matrix["stride"])
    rows = list(range(0, alpha_full.size, stride))

    picks = ([(r, c, "requested") for r, c in args.point] if args.point
             else choose_points(bayes))

    config = {**MODELS["joint"].get_default_model_config(),
              "beta_s_fixed": beta_s, "beta_n_fixed": beta_n}

    store = {"labels": [], "true_alpha_s": [], "true_alpha_n": [],
             "bayes_factor": [], "p_heterodimer": [], "cell": []}
    draws = {name: [] for name in TRACKED}

    print(f"generating rates: beta_s={beta_s:g} beta_n={beta_n:g} "
          f"k_y={k_y_true:g} gamma={gamma:g}\n")

    for a, b, label in picks:
        i, j = rows[a], rows[b]
        true_a_n, true_a_s = float(alpha_full[i]), float(alpha_full[j])
        data = counts[i, j].astype(float)

        model = MODELS["joint"](model_config=dict(config))
        model.fit(data, sampler_config={"draws": args.draws,
                                        "chains": args.chains},
                  progressbar=False, sample_prior=False)
        post = model.idata["posterior"]
        probs = model_probabilities(model.idata)

        for name in TRACKED:
            draws[name].append(np.asarray(post[name]).ravel())
        store["labels"].append(label)
        store["true_alpha_s"].append(true_a_s)
        store["true_alpha_n"].append(true_a_n)
        store["bayes_factor"].append(probs["bayes_factor_het_over_mono"])
        store["p_heterodimer"].append(probs["heterodimer"])
        store["cell"].append([a, b])

        print(f"  ({a},{b}) {label:26s} true k_on,s={true_a_s:9.4g} "
              f"k_on,n={true_a_n:9.4g}  BF={probs['bayes_factor_het_over_mono']:7.3f}"
              f"  P(het)={probs['heterodimer']:.3f}")

    n = min(d.size for name in TRACKED for d in draws[name])
    payload = {f"draws_{name}": np.array([d[:n] for d in draws[name]])
               for name in TRACKED}
    payload.update({k: np.array(v) for k, v in store.items()})
    payload["k_y_true"] = k_y_true
    payload["beta_s"] = beta_s
    payload["beta_n"] = beta_n

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"\nWrote {args.out} ({os.path.getsize(args.out) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
