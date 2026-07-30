"""Generate synthetic count data from known parameters, for ABC validation.

Simulating from a known ground truth and re-fitting it is how the ABC-SMC setup
is checked: the posterior should cover ``TRUE_PARAMS``.

Usage
-----
    python scripts/generate_synthetic.py --model heterodimer
    python scripts/generate_synthetic.py --model monomer --n-cells 40
"""

import argparse
import os

import numpy as np

from stochtf.inference.abc_smc import fast_ssa_dimer, fast_ssa_monomer
from stochtf.paths import SYNTHETIC_DATA_DIR

#: Ground-truth rates the synthetic data is generated from.
TRUE_PARAMS = {
    "alpha_s": 0.5,
    "beta_s": 0.06,
    "alpha_n": 0.3,
    "beta_n": 0.2,
    "gamma_y": 0.005,
    "k_y": 0.1,
}

SIMULATORS = {"monomer": fast_ssa_monomer, "heterodimer": fast_ssa_dimer}

#: Each SSA call returns 10 observations, one per sampling time.
OBS_PER_CELL = 10


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(SIMULATORS), default="heterodimer")
    ap.add_argument("--n-cells", type=int, default=40)
    ap.add_argument("--t-max", type=float, default=1000.0)
    args = ap.parse_args()

    simulator = SIMULATORS[args.model]
    counts = np.empty((args.n_cells, OBS_PER_CELL))
    for i in range(args.n_cells):
        counts[i] = simulator(
            TRUE_PARAMS["alpha_s"],
            TRUE_PARAMS["beta_s"],
            TRUE_PARAMS["alpha_n"],
            TRUE_PARAMS["beta_n"],
            TRUE_PARAMS["k_y"],
            TRUE_PARAMS["gamma_y"],
            args.t_max,
        )

    flat = counts.flatten()
    os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)
    out = os.path.join(SYNTHETIC_DATA_DIR, f"synthetic_{args.model}_data.npy")
    np.save(out, flat)
    print(f"mean {flat.mean():.3f}  var {flat.var():.3f}  n {flat.size}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
