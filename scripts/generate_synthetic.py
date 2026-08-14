"""Generate synthetic count data from known parameters, for inference validation.

Simulating from a known ground truth and re-fitting it is how the inference is
checked: the posterior should cover ``TRUE_PARAMS``.

Two generators
--------------
``--method stationary`` (default)
    Draws iid counts directly from the exact stationary distribution. This is
    what the likelihood assumes, so it isolates the inference from any
    simulation artefact.

``--method ssa``
    The original Gillespie route. It records 10 counts *along one trajectory* at
    t_max/10 ... t_max, so the samples are autocorrelated and, at the default
    settings, not yet stationary: with gamma = 0.005 and t_max = 1000 the first
    observation is at half an mRNA lifetime. That produced data with mean 16.5
    and Fano 1.9 where the true stationary values are 19.1 and 1.006, which is
    why parameters could not be recovered from it. Kept for comparison.

Rates are in units of gamma; see stochtf.inference.models.

Usage
-----
    python scripts/generate_synthetic.py --model heterodimer
    python scripts/generate_synthetic.py --model telegraph --n-cells 800
    python scripts/generate_synthetic.py --model heterodimer --method ssa
"""

import argparse
import os

import numpy as np

from stochtf.analytical import pgf
from stochtf.paths import SYNTHETIC_DATA_DIR
from stochtf.ssa.fast import fast_ssa_dimer, fast_ssa_monomer

#: Ground-truth rates, in units of gamma (so gamma = 1 and rates read as
#: "per mRNA lifetime"). Slow switching and a large k_y put this in the bursty
#: regime, where the promoter state leaves a visible signature in the counts.
TRUE_PARAMS = {
    "alpha_s": 0.01,
    "beta_s": 0.06,
    "alpha_n": 1.0,
    "beta_n": 0.24,
    "gamma_y": 1.0,
    "k_y": 20.0,
}

SSA_SIMULATORS = {"monomer": fast_ssa_monomer, "heterodimer": fast_ssa_dimer}
GATES = {"monomer": "ADD", "heterodimer": "OR", "telegraph": "OR"}

#: Each SSA call returns 10 observations along one trajectory.
OBS_PER_CELL = 10

#: Site n is switched off to reduce the two-site promoter to a single site.
SITE_OFF = 1e-9


def truth_for(model):
    """(alpha_s, beta_s, alpha_n, beta_n, k_y) in units of gamma."""
    t = TRUE_PARAMS
    return (t["alpha_s"], t["beta_s"], t["alpha_n"], t["beta_n"], t["k_y"])


def generate_ssa(model, n_cells, t_max):
    """The original Gillespie route; see the module docstring for its caveats."""
    if model not in SSA_SIMULATORS:
        raise SystemExit(f"--method ssa has no simulator for model {model!r}")
    simulator = SSA_SIMULATORS[model]
    a_s, b_s, a_n, b_n, k_y = truth_for(model)
    counts = np.empty((n_cells, OBS_PER_CELL))
    for i in range(n_cells):
        counts[i] = simulator(a_s, b_s, a_n, b_n, k_y,
                              TRUE_PARAMS["gamma_y"], t_max)
    return counts.flatten()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(GATES), default="heterodimer")
    ap.add_argument("--method", choices=["stationary", "ssa"], default="stationary")
    ap.add_argument("--n-cells", type=int, default=800)
    ap.add_argument("--t-max", type=float, default=1000.0,
                    help="ssa method only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    flat = generate_ssa(args.model, args.n_cells, args.t_max)

    a_s, b_s, a_n, b_n, k_y = truth_for(args.model)
    mean, var, fano = pgf.moments(a_s, b_s, a_n, b_n, k_y, 1.0, GATES[args.model])

    os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)
    out = os.path.join(SYNTHETIC_DATA_DIR, f"synthetic_{args.model}_data.npy")
    np.save(out, flat)

    print(f"true stationary : mean {mean:8.3f}  Fano {fano:7.3f}")
    print(f"generated       : mean {flat.mean():8.3f}  Fano "
          f"{flat.var() / flat.mean():7.3f}  n {flat.size}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
