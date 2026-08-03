"""Fit a promoter model to allele-resolved counts by SMC over the exact posterior.

This was ABC-SMC against a Gillespie simulator compared through a two-number
summary statistic. It now uses the exact stationary likelihood from
:mod:`stochtf.inference.likelihood`, so the whole distribution is scored and
there is no ABC tolerance to tune.

Rates are inferred in units of the mRNA degradation rate gamma, which
stationary counts cannot identify separately; see stochtf.inference.models.

Usage
-----
    python scripts/run_inference.py --gene esrrb --model dimer
    python scripts/run_inference.py --gene sox2 --model monomer --draws 200

Traces are written to ``results/<gene>_<model>.nc`` and are the input to
``notebooks/05_abc_diagnostics.ipynb``.
"""

import argparse

import numpy as np

from stochtf import paths
from stochtf.inference.models import MODELS

GENES = ["sox2", "nanog", "rex1", "esrrb", "synthetic_heterodimer_data",
         "synthetic_monomer_data", "synthetic_telegraph_data"]


def load_counts(gene):
    """Load the processed per-cell count vector for one gene."""
    return np.load(paths.processed(f"{gene}.npy"))

def load_synthetic(gene):
    """Load the processed per-cell count vector for one gene."""
    return np.load(paths.synthetic(f"{gene}.npy"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene", choices=GENES, default="esrrb",
                    help="gene whose counts to fit (default: esrrb)")
    ap.add_argument("--model", choices=sorted(MODELS), default="dimer",
                    help="promoter model to fit (default: dimer)")
    ap.add_argument("--draws", type=int, default=None,
                    help="SMC draws; defaults to the model's sampler config")
    ap.add_argument("--out", default=None,
                    help="output .nc path (default: results/<gene>_<model>.nc)")
    args = ap.parse_args()

    data = load_synthetic(args.gene)
    print(f"Fitting {args.model} model to {args.gene}: {data.shape} observations "
          f"(mean {data.mean():.1f}, Fano {data.var() / data.mean():.1f})")

    model = MODELS[args.model]()
    sampler_config = None
    if args.draws is not None:
        sampler_config = dict(model.get_default_sampler_config())
        sampler_config["draws"] = args.draws

    model.fit(data, sampler_config=sampler_config)

    out = args.out or paths.results(f"{args.gene}_{args.model}.nc")
    model.save(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
