"""Simulate one promoter model by Gillespie SSA and plot the time course.

Replaces monomer_ssa.py / homodimer_ssa.py / heterodimer_ssa.py, which carried
three near-identical copies of this reporting and plotting block.

Usage
-----
    python scripts/run_ssa.py --model heterodimer
    python scripts/run_ssa.py --model monomer --t-max 100 --out-dir /tmp
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from stochtf.paths import RESULTS_DIR
from stochtf.plotting import use_paper_style
from stochtf.ssa import gillespie as gil
from stochtf.ssa.models import MODELS

#: Default simulated duration per model, matching the original scripts.
DEFAULT_T_MAX = {"monomer": 100.0, "homodimer": 100.0, "heterodimer": 500.0}
#: Default step cap per model, matching the original scripts.
DEFAULT_MAX_STEPS = {"monomer": 1000000, "homodimer": 50000, "heterodimer": 50000}
#: ON/OFF extraction window per model, matching the original scripts.
DEFAULT_ACTIVE_WINDOW = {"monomer": (1, 100), "homodimer": (1, 200),
                         "heterodimer": (1, 200)}


def summarise(times, states, promoter_idx, mrna_idx, active_window):
    """Prints the ON/OFF dwell times and the mRNA moments."""
    on_times, off_times = gil.extract_on_off(
        times, states, promoter_idx, active_window[0], active_window[1]
    )
    print(f"Mean ON time: {on_times.mean()}")
    print(f"Mean OFF time: {off_times.mean()}")

    mrna = states[:, mrna_idx]
    mean_mrna = np.mean(mrna)
    fano_mrna = np.var(mrna) / mean_mrna if mean_mrna > 0 else 0
    print(f"Mean mRNA: {mean_mrna:.2f}")
    print(f"Fano factor: {fano_mrna:.2f}")


def plot(times, states, name, promoter_idx, mrna_idx, out_dir):
    """Plots one simulated trajectory and writes it to disk.

    Args:
        times: Reaction times returned by the simulator.
        states: State after each reaction.
        name: Model name, used in the title and filename.
        promoter_idx: Column holding the promoter state.
        mrna_idx: Column holding the mRNA count.
        out_dir: Directory to write the figure into.
    """
    use_paper_style()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))

    if name == "monomer":
        ax1.step(times, states[:, 2], label="Bound NANOG ($n_n$)", color="green")
        ax1.step(times, states[:, 3], label="Bound SOX2 ($n_s$)", color="blue")
    else:
        ax1.step(times, states[:, 3], label="Both bound ($n_{11}$)",
                 color="purple", alpha=0.7)
    ax1.set_ylabel("Bound TFs")
    ax1.legend(loc="upper right")
    ax1.set_title("Time course of SOX2/NANOG binding and mRNA expression")

    ax2.step(times, states[:, mrna_idx], label="mRNA ($y$)", color="red")
    ax2.set_ylabel("mRNA count")
    ax2.legend(loc="upper right")

    mrna_max = int(np.max(states[:, mrna_idx]))
    bins = np.arange(mrna_max + 2) - 0.5
    ax3.hist(states[:, mrna_idx], bins=bins, density=True, color="gray",
             edgecolor="black", alpha=0.7)
    ax3.set_xlabel("mRNA count")
    ax3.set_ylabel("Probability density")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"cme_gillespie_{name}_simulation.png")
    fig.savefig(path, dpi=300)
    print(f"Saved plot to {path}")


def main():
    """Parses arguments and runs one SSA simulation."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), default="heterodimer")
    ap.add_argument("--t-max", type=float, default=None,
                    help="simulated duration (default: per-model, see DEFAULT_T_MAX)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    args = ap.parse_args()

    name = args.model
    params, initial_state, stoichiometry, propensity_fn, promoter_idx, mrna_idx = \
        MODELS[name]
    t_max = args.t_max if args.t_max is not None else DEFAULT_T_MAX[name]
    max_steps = args.max_steps if args.max_steps is not None else DEFAULT_MAX_STEPS[name]

    print(f"Running Gillespie simulation for the {name} model...")
    times, states = gil.gillespie(
        initial_state, stoichiometry, propensity_fn,
        t_max=t_max, parameters=params, max_steps=max_steps,
    )
    print("Simulation complete.")

    summarise(times, states, promoter_idx, mrna_idx, DEFAULT_ACTIVE_WINDOW[name])
    plot(times, states, name, promoter_idx, mrna_idx, args.out_dir)


if __name__ == "__main__":
    main()
